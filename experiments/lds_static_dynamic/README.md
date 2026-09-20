# Static versus dynamic LDS bandwidth

This experiment compares two ways MT4G allocates a 32 KiB LDS array: dynamic `extern __shared__` storage and a compile-time `__shared__` array selected with `--static`. Both versions use the same 128-bit LDS accesses and search the same launch settings. The measured rate is per CU. LDS latency is identical in both runs and serves only as a control.

## Run on Viper

Run from this directory. Each mode uses three exclusive nodes; each node runs ten paired static and dynamic measurements on logical device 0. The order alternates within each node. XNACK is fixed at 1, and the worker checks the active partition before and after measurement.

```bash
mkdir -p logs
for mode in spx tpx cpx; do
    sbatch --job-name="mt4g_lds_${mode}" \
        --mi300-partition="$mode" \
        --export="ALL,MODE=$mode" \
        job.sh
done
```

This submits three jobs, one for each partition mode. To make a shorter run, export an even number of pairs before submitting so the execution order remains balanced:

```bash
export REPETITIONS=2
```

The default ten pairs produce 30 matched pairs per mode. `ALLOW_DIRTY=0` requires a clean benchmark source tree; use `ALLOW_DIRTY=1` only for a smoke test. `MT4G_BIN` and `ROCM_MODULE` can override the executable and ROCm module.

## Analyze

After the three jobs finish, generate the comparison and CSV tables:

```bash
./compare_results.py --strict --md comparison.md --csv-dir analysis
```

The analysis compares static and dynamic rates within each node and repetition, then combines the paired ratios across nodes. It also compares raw bandwidth grids at matching launch settings. Run records are saved under `runs/`, Slurm output under `logs/`, and generated analysis under `analysis/`; all remain outside Git.
