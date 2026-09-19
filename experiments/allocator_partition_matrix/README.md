# Allocator and partition matrix

This experiment compares MT4G L3 and main-memory latency and bandwidth across SPX, TPX, and CPX; XNACK on and off; and four allocators: `hipmalloc`, `hipmallocmanaged`, `hiphostmalloc`, and `malloc`. Vector L1, L2, and static LDS provide allocator-independent controls.

## Submit on the cluster

The scripts expect the checkout at `/u/halba/mt4g_extension` and the executable at `build/mt4g`. Build instructions are in [build.md](../../build.md), then run from this directory:

```bash
mkdir -p logs
for mode in spx tpx cpx; do
    for xnack in 1 0; do
        sbatch --job-name="mt4g_${mode}_x${xnack}" \
            --mi300-partition="$mode" \
            --export="ALL,MODE=$mode,XNACK=$xnack" \
            run_allocator_partition_matrix.sh
    done
done
```

This submits six three-node jobs, one for each partition and XNACK setting. Each worker verifies the active partition before measuring. For a short smoke test, run `export REPETITIONS=1` before the loop; the full campaign uses five repetitions. `MT4G_BIN` and `ROCM_MODULE` can override the executable and ROCm module.

## Measurement protocol

On each node, the worker measures logical device 0 and cycles through all four allocators in each of five rounds. Three nodes give 15 attempts per matrix case, or 360 invocations in total. The 45 `malloc + XNACK=0` attempts are recorded as expected failures; the other 315 are successful measurements in the retained campaign.

The allocator option affects L3 and main-memory measurements. SPX and TPX use MT4G's default AMD benchmark groups. CPX selects the stable groups explicitly and omits scalar L1, which did not complete reliably in CPX. All modes include static LDS. Every invocation uses `--optimal --static`; its saved `command.txt` records the exact flags and selected groups.

Results are stored under `runs/<mode>_xnack<0|1>_<job-id>/nodes/<hostname>/`, with per-invocation results and statuses beneath each `repeat_*/<allocator>/` directory. Slurm logs go to `logs/`. These generated files are ignored by Git.

## Analyze saved runs

```bash
./compare_results.py --strict --md comparison.md --csv-dir analysis
```

`compare_results.py` selects the latest job for each partition/XNACK pair, checks completeness, averages repetitions within each node before averaging the three nodes, and calculates the partition, allocator, and XNACK comparisons. `--strict` accepts the recorded `malloc + XNACK=0` failures but rejects missing supported measurements.

The command creates `comparison.md`, a readable report, and four CSV files under `analysis/`: all extracted measurements, summaries for each matrix cell, pooled allocator-independent controls, and calculated effects. These are generated files and can be recreated by running the command again. Keep the six retained jobs as the latest runs when reproducing their comparison.
