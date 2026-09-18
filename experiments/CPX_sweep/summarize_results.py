#!/usr/bin/env python3
"""Generate a local CPX sweep summary from the reduced CSVs."""
import argparse
import csv
import json
import math
import statistics as st
from pathlib import Path
from collections import Counter, defaultdict

import compare_results as cr

def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))

def span(row,left,right,digits=2):
    return f"{float(row[left]):.{digits}f}--{float(row[right]):.{digits}f}"

def markdown(header,body):
    return '\n'.join(['| '+' | '.join(header)+' |','| '+' | '.join(['---']*len(header))+' |']+['| '+' | '.join(row)+' |' for row in body])

def write_csv(path, data):
    with path.open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)

def metric_summary(analysis):
    devices=defaultdict(list)
    for row in rows(analysis/'per_device.csv'):
        devices[(row['node'],row['package'],row['metric'])].append(float(row['raw_mean']))
    grouped=defaultdict(list)
    for row in rows(analysis/'device_effects.csv'):
        key=row['node'],row['package'],row['metric']
        means=devices[key]
        assert len(means)==6
        grouped[row['metric']].append((row,max(means)-min(means)))
    result=[]
    for metric in cr.METRICS:
        groups=grouped[metric.name]
        assert len(groups)==6
        result.append(dict(metric=metric.name,unit=metric.unit,
            significant_groups=sum(float(row['holm_p'])<.05 for row,_ in groups),
            group_mean_min=min(float(row['raw_group_mean']) for row,_ in groups),
            group_mean_max=max(float(row['raw_group_mean']) for row,_ in groups),
            device_range_min_pp=min(float(row['device_mean_range_percent']) for row,_ in groups),
            device_range_max_pp=max(float(row['device_mean_range_percent']) for row,_ in groups),
            absolute_device_range_min=min(spread for _,spread in groups),
            absolute_device_range_max=max(spread for _,spread in groups)))
    return result

def read_grid(result_path, metric):
    level=metric.path[1]
    prefix={'l3':'L3','main':'MainMemory'}[level]
    direction='Read' if 'read' in metric.name else 'Write'
    files=list((Path(result_path).parent/'results').rglob(f'*__{prefix}_{direction}_BW_Grid.csv'))
    assert len(files)==1,(result_path,metric.name,files)
    grid={}
    for row in rows(files[0]):
        config=int(row['blocks']),int(row['threads']),int(row['reps'])
        value=float(row['bandwidth'])
        assert config not in grid and math.isfinite(value) and value>0
        grid[config]=value
    return grid

def fixed_bandwidth(analysis):
    metrics={m.name:m for m in cr.METRICS if m.path[1] in ('l3','main') and 'bandwidth' in m.name}
    groups=defaultdict(list)
    for row in rows(analysis/'observations.csv'):
        if row['metric'] in metrics:
            groups[(row['node'],row['package'],row['metric'])].append(row)
    all_points=[]
    selected=[]
    for (node,package,name),records in sorted(groups.items()):
        records.sort(key=lambda row:(int(row['repeat']),int(row['logical_device'])))
        assert len(records)==72
        metric=metrics[name]
        grids=[read_grid(row['result'],metric) for row in records]
        common=set.intersection(*(set(grid) for grid in grids))
        assert common
        winners=Counter()
        for record in records:
            peak=cr.dig(json.loads(Path(record['result']).read_text()),metric.path[:-1])
            winners[tuple(int(peak[key]) for key in ('numBlocks','numThreads','numReps'))]+=1
        modal=min(common,key=lambda config:(-winners[config],config))
        for config in sorted(common):
            values=[grid[config] for grid in grids]
            matrix=[values[index:index+6] for index in range(0,72,6)]
            raw_means=[st.mean(column) for column in zip(*matrix)]
            normalized=[[100*(value/st.median(sweep)-1) for value in sweep] for sweep in matrix]
            effects=[st.mean(column) for column in zip(*normalized)]
            point=dict(node=node,package=package,metric=name,blocks=config[0],threads=config[1],
                reps=config[2],modal=config==modal,raw_device_min=min(raw_means),
                raw_device_max=max(raw_means),device_range_pp=max(effects)-min(effects),
                common_configurations=len(common))
            all_points.append(point)
            if config==modal:
                selected.append(point)
    assert len(selected)==24
    return all_points,selected

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis',type=Path,default=Path(__file__).resolve().parent/'analysis'/'details')
    a=p.parse_args()
    metrics=metric_summary(a.analysis)
    write_csv(a.analysis/'metric_summary.csv',metrics)
    points,selected=fixed_bandwidth(a.analysis)
    write_csv(a.analysis/'fixed_configurations.csv',points)
    write_csv(a.analysis/'fixed_summary.csv',selected)
    primary=[r for r in metrics if r['metric'].startswith(('L3','Main-memory'))]
    bw=[]
    for r in primary:
        if 'bandwidth' in r['metric']:
            bw.append([r['metric'].replace(' bandwidth',''),span(r,'group_mean_min','group_mean_max',1),span(r,'absolute_device_range_min','absolute_device_range_max',2)])
    fixed=[]
    for r in selected:
        if (r['node'],r['package']) in [('vipa1020','0'),('vipa1099','1')] and r['metric'].startswith('L3'):
            fixed.append([r['node']+'/'+r['package'],r['metric'].split()[1],f"{r['blocks']} × {r['threads']}",span(r,'raw_device_min','raw_device_max',1)])
    # Average within each package first, then give both packages equal node weight.
    grouped=defaultdict(list)
    for r in rows(a.analysis/'observations.csv'):
        if r['metric'].startswith(('L3','Main-memory')):
            grouped[(r['node'],int(r['package']),r['metric'])].append(float(r['value']))
    node_rows=[]
    for node in sorted({key[0] for key in grouped}):
        for metric in sorted({key[2] for key in grouped}):
            node_rows.append(dict(node=node,metric=metric,unit='cycles' if 'latency' in metric else 'GiB/s',mean=st.mean(st.mean(grouped[(node,pkg,metric)]) for pkg in (0,1))))
    with (a.analysis/'node_averages.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(node_rows[0]));writer.writeheader();writer.writerows(node_rows)
    bw_names=['L3 read bandwidth','L3 write bandwidth','Main-memory read bandwidth','Main-memory write bandwidth']
    node_body=[]
    for node in sorted({r['node'] for r in node_rows}):
        lookup={r['metric']:r['mean'] for r in node_rows if r['node']==node}
        node_body.append([node]+[f'{lookup[m]:.2f}' for m in bw_names])
    overview=[]
    for r in primary:
        overview.append([r['metric'],span(r,'group_mean_min','group_mean_max',2)+' '+r['unit'],span(r,'device_range_min_pp','device_range_max_pp',3),str(r['significant_groups'])+'/6'])
    text=['# CPX device sweep: L3 and main-memory performance',
    '', '**Conclusion:** No practical device preference is supported for the measured main-memory adaptive peaks. Larger, repeatable L3 bandwidth differences occur in two of six node/package groups, but no universal preferred device or physical-route explanation is established. Neither L3 nor main memory shows a convincing stable latency ranking.',
    '', '## What was measured?', '', 'Only `cpx_11360825`, clean commit `4b6e88d`: three nodes, two packages per node, six CPX devices per package. One sweep runs each of twelve devices sequentially. Twelve sweeps per node give **3 × 12 × 12 = 432 runs**, or twelve measurements per device. CPX, NPS1, hipMalloc and XNACK=1 stay fixed.',
    '', '## Average behavior across nodes', '', markdown(['Node','L3 read','L3 write','Main read','Main write'],node_body), '', 'GiB/s, averages of per-device adaptive peaks with equal package weights—not summed node throughput. L3 node averages differ by about 0.64% for reads and 1.05% for writes; main-memory node averages differ by about 0.013%. These are descriptive comparisons, not tests of node equivalence.',
    '', '## Peak bandwidth across packages', '', markdown(['Metric','Package averages (GiB/s)','Device difference (GiB/s)'],bw), '', 'Ranges cover six packages. The device difference is the highest minus lowest twelve-run device average within a package.',
    '', '## Differences among the six devices within each package', '', markdown(['Metric','Package-average interval','Device spread (pp)','Significant groups'],overview),
    '', 'Package-average intervals span six groups. Device spreads compare the largest and smallest twelve-run device averages after normalization to each package/sweep median. Significance is a supporting check, not a statement of practical importance.',
    '', '- **Main memory:** all adaptive bandwidth spreads are below 0.1% and 0.64 GiB/s. The one significant write result is only 0.051% (0.332 GiB/s), practically negligible for this measured peak. At shared launch settings, spreads can reach 1.78% read and 2.87% write; peak uniformity does not mean identical behavior at every launch.',
    '- **L3:** vipa1020/package 0 and vipa1099/package 1 show read spreads of 12.88% and 13.82%, and write spreads of 11.87% and 11.89%. The other four groups do not show statistically convincing differences. Ordinals 4 and 6 respectively are slower; they are host-local labels, not physical coordinates.',
    '- **Common launches:** the large L3 contrast remains at fixed launches (1,216 blocks; 256 read or 1,024 write threads; 2,048 repetitions; 64 MiB). Read spreads are 20.38–22.00% and write spreads 10.72–10.74%. This rules out independent peak selection as the sole explanation, but not other implementation effects.',
    '', '## L3 bandwidth at common launch settings', '', markdown(['Node/package','Direction','Blocks × threads','Device averages (GiB/s)'],fixed), '', 'All listed configurations use a 64 MiB working set and 2,048 repetitions.',
    '', '## Latency', '', 'L3 package averages are 494.3–496.8 cycles; main memory 858.4–870.2 cycles. Neither has a significant device screen or a convincing stable ranking. Lack of significance is not proof of equality.',
    '', '## Supporting controls and limits', '', 'L1/L2 are supporting controls, not the question’s focus. Their small repeatable bandwidth differences provide context for the larger L3 contrasts; they are not dismissed as noise. The two devices with low L3 bandwidth are also slower for L2 writes, so an exclusively L3 explanation is not supported.',
    '', 'The near-uniform main-memory peaks are consistent with NPS1 interleaving; the experiment does not independently verify address-to-stack mapping or its causal effect. Before/after snapshots record the worker bound to CPUs 0 and 48 in NUMA node 0. MT4G inherits this binding at launch for devices from either package; the benchmark code does not explicitly change it. Individual runtime threads were not continuously monitored, and CPU placement was not varied.',
    '', '**Source limitation:** every vector-L1 write search executed unsafe allocation configurations, although its selected winner was within bounds. This metric is excluded from hardware interpretation. Later device resets cannot retrospectively certify unaffected results. Shared read sinks and unverified main-memory assembly lowering further prevent an intrinsic-hardware claim. A focused follow-up omitting the unsafe search, with instruction/traffic validation, is necessary only before making a stronger hardware claim; none was run.',
    '', '## Reproducibility and detailed evidence', '', 'Run `python3 refresh_analysis.py` from CPX_sweep to validate and regenerate the analysis. The original fifteen-metric Holm family per group (and all-90 sensitivity) is preserved; narrowing the question does not select a more favorable statistical threshold.',
    '', '- [Device averages and variation](details/per_device.csv)', '- [Statistical definitions and checks](details/comparison.md)', '- [All common launch configurations](details/fixed_configurations.csv)', '- [Node averages, including latency](details/node_averages.csv)']
    (a.analysis.parent/'SUMMARY.md').write_text('\n'.join(text)+'\n')
    print('Updated analysis/SUMMARY.md and details/node_averages.csv.')

if __name__=='__main__':main()
