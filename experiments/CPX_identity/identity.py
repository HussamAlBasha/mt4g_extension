#!/usr/bin/env python3
"""Record and compare CPX device identities without running a GPU workload."""
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone

BASE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def capture():
    job = os.environ.get('SLURM_JOB_ID', '')
    require(job.isdigit(), 'Capture must run inside a Slurm job.')
    host = socket.gethostname().split('.')[0]
    out = BASE / 'runs' / f'{host}_{job}'
    out.mkdir(parents=True, exist_ok=False)
    record = dict(host=host, job_id=job, timestamp=datetime.now(timezone.utc).isoformat(),
                  boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                  rocm_module=os.environ.get('ROCM_MODULE', 'rocm/7.2'))
    for name, command in [('rocminfo', ['rocminfo']),
                          ('partition', ['amd-smi', 'static', '--partition'])]:
        result = subprocess.run(command, text=True, capture_output=True, check=True)
        (out / f'{name}.txt').write_text(result.stdout + result.stderr)
    partition = (out / 'partition.txt').read_text()
    modes = re.findall(r'(?:ACCELERATOR|COMPUTE)_PARTITION:\s*(\S+)', partition)
    modes = [m.upper() for m in modes if m.upper() != 'N/A']
    memory = re.findall(r'MEMORY_PARTITION:\s*(\S+)', partition)
    memory = [m.upper() for m in memory if m.upper() != 'N/A']
    require(len(modes) >= 2 and set(modes) == {'CPX'}, 'Expected both packages in CPX.')
    require(len(memory) >= 2 and set(memory) == {'NPS1'}, 'Expected NPS1.')
    record.update(mode='CPX', memory_partition='NPS1')

    # HIP ordinal is queried directly, rather than inferred from rocminfo order.
    hip = ct.CDLL('libamdhip64.so')
    signatures = {'hipGetDeviceCount': [ct.POINTER(ct.c_int)],
                  'hipSetDevice': [ct.c_int], 'hipGetDevice': [ct.POINTER(ct.c_int)],
                  'hipDeviceGetUuid': [ct.c_void_p, ct.c_int],
                  'hipDeviceGetPCIBusId': [ct.c_void_p, ct.c_int, ct.c_int],
                  'hipRuntimeGetVersion': [ct.POINTER(ct.c_int)]}
    for name, args in signatures.items():
        getattr(hip, name).argtypes = args
        getattr(hip, name).restype = ct.c_int

    def call(name, *args):
        status = getattr(hip, name)(*args)
        require(status == 0, f'{name} failed with HIP error {status}')

    count = ct.c_int()
    call('hipGetDeviceCount', ct.byref(count))
    require(count.value == 12, f'Expected 12 HIP devices, got {count.value}')
    version = ct.c_int()
    call('hipRuntimeGetVersion', ct.byref(version))
    record['hip_runtime_version'] = version.value
    kfd = []
    for p in sorted(Path('/sys/class/kfd/kfd/topology/nodes').glob('*/properties')):
        props = dict(line.split(maxsplit=1) for line in p.read_text().splitlines() if ' ' in line)
        if int(props.get('simd_count', '0')):
            kfd.append(dict(kfd_node=int(p.parent.name), **props))
    (out / 'kfd.json').write_text(json.dumps(kfd, indent=2) + '\n')
    devices = []
    for ordinal in range(count.value):
        uuid = (ct.c_ubyte * 16)()
        pci = ct.create_string_buffer(64)
        call('hipDeviceGetUuid', ct.byref(uuid), ordinal)
        call('hipDeviceGetPCIBusId', pci, len(pci), ordinal)
        domain, bus, dev, func = re.split('[:.]', pci.value.decode())
        bdfid = (int(bus, 16) << 8) | (int(dev, 16) << 3) | int(func, 16)
        matches = [k for k in kfd if int(k['location_id']) == bdfid
                   and int(k.get('domain', '0')) == int(domain, 16)]
        require(len(matches) == 1, f'Cannot uniquely match HIP device {ordinal} to KFD')
        k = matches[0]
        require(int(k['simd_count']) == 152 and int(k.get('num_xcc', '0')) == 1,
                'Expected one XCC and 152 SIMDs per CPX device')
        rocr_uuid = f"GPU-{int(k['unique_id']):016x}"
        require(rocr_uuid in (out / 'rocminfo.txt').read_text(), 'KFD UUID absent from rocminfo')
        devices.append(dict(ordinal=ordinal, hip_uuid=bytes(uuid).hex(),
                            rocr_uuid=rocr_uuid, pci=pci.value.decode(), bdfid=bdfid,
                            kfd_node=k['kfd_node'], render_minor=int(k['drm_render_minor'])))
    require(len({d['hip_uuid'] for d in devices}) == 12, 'HIP UUIDs are not unique')
    require(all(int(d['hip_uuid'], 16) for d in devices), 'Empty HIP UUID')
    call('hipSetDevice', 0)
    selected = ct.c_int()
    call('hipGetDevice', ct.byref(selected))
    require(selected.value == 0, 'HIP did not select device 0')
    record.update(selected_ordinal=0, devices=devices)
    (out / 'identity.json').write_text(json.dumps(record, indent=2) + '\n')
    print(f"Saved {out / 'identity.json'}\nRecorded {len(devices)} CPX devices.")


def compare(paths):
    records = [json.loads(Path(p).read_text()) for p in paths]
    require(len(records) >= 2, 'Provide records from at least two separate jobs.')
    require(len({r['host'] for r in records}) == 1, 'Records must come from the same node.')
    require(len({r['job_id'] for r in records}) == len(records), 'Job IDs must be distinct.')
    require(all(r['mode'] == 'CPX' and r['memory_partition'] == 'NPS1' for r in records),
            'All records must use CPX/NPS1.')
    require(len({r['hip_runtime_version'] for r in records}) == 1, 'HIP versions differ.')
    expected_ordinals = set(range(12))

    def by_ordinal(record):
        devices = record['devices']
        indexed = {device['ordinal']: device for device in devices}
        require(len(devices) == 12 and set(indexed) == expected_ordinals,
                f"Job {record['job_id']} must record each of the 12 CPX devices once.")
        return indexed

    def identity(device):
        return device['hip_uuid'], device['rocr_uuid'], device['pci']

    reference = by_ordinal(records[0])
    stable = True
    print(f"Host: {records[0]['host']}\nJob | matching ordinals | changed ordinals")
    for record in records:
        current = by_ordinal(record)
        changed = [ordinal for ordinal in range(12)
                   if identity(current[ordinal]) != identity(reference[ordinal])]
        print(f"{record['job_id']} | {12 - len(changed)}/12 | "
              f"{','.join(map(str, changed)) if changed else '-'}")
        stable &= not changed
    print(f"Boot IDs observed: {len({r['boot_id'] for r in records})}")
    if stable:
        print('MATCH: all 12 ordinals retain the same HIP UUID, ROCr UUID, and PCI address.')
        print('This supports repeatable selection across these jobs, not all future jobs.')
    else:
        print('MISMATCH: at least one ordinal identifies a different device.')
    return 0 if stable else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('capture')
    sub.add_parser('compare').add_argument('records', nargs='+')
    args = parser.parse_args()
    try:
        if args.action == 'capture':
            capture()
            return 0
        return compare(args.records)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
