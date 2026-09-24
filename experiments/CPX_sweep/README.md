# CPX device sweep (RQ3)

This experiment compares L3 and main-memory latency and bandwidth across the twelve CPX devices on each of three MI300A nodes. Each node runs twelve sequential sweeps, giving 432 device runs.

## Submit on the cluster

The job scripts expect this checkout at `/u/halba/mt4g_extension` and an existing `build/mt4g` executable. Build instructions are in [build.md](../../build.md). From this directory on the cluster:

```bash
mkdir -p logs
sbatch --mi300-partition=cpx job.sh
```

This submits one Slurm job with one worker on each of three nodes. With the default twelve sweeps of twelve devices per node, a completed job makes **3 × 12 × 12 = 432 MT4G invocations**, measured sequentially on each node. Each worker checks CPX/NPS1 and device identity and generates balanced device orders. The defaults are `hipMalloc` and `HSA_XNACK=1`. New measurements go to `runs/cpx_<job-id>/`; the worker records Git HEAD and the executable's SHA-256.

## Regenerate the analysis

From `experiments/CPX_sweep`, analyze a specific completed run with:

```bash
python3 refresh_analysis.py --run runs/cpx_<job-id>
```

If `--run` is omitted, the script selects the `runs/cpx_<job-id>` directory with the highest numeric job ID. The command validates the 432 runs and rebuilds the descriptive results under `analysis/`. It calculates the twelve-run average for each device, within-package device ranges, package-balanced node averages, and common-launch comparisons. The analysis uses only the Python standard library.

## Python scripts

Normally, run only `refresh_analysis.py` for saved data. It runs validation followed by analysis. `worker.sh` calls `generate_orders.py` during collection.

| Script | Purpose and output |
| --- | --- |
| `refresh_analysis.py` | Selects, validates, and analyzes a completed run; produces the files under `analysis/`. |
| `generate_orders.py` | Creates the balanced device schedule in each node's `orders.tsv`. |
| `validate_results.py` | Checks run completeness, partition and device maps, and result files; reports pass or fail. |
| `analyze.py` | Calculates device and node averages, ranges, run variation, and common-launch comparisons; writes `analysis/SUMMARY.md` and supporting CSVs. |

The summary includes node averages, within-package device comparisons, and L3 bandwidth at common launch settings. Keep or archive a run directory to regenerate its exact summary; a new job produces new measurements. The `runs/`, `analysis/`, and `logs/` directories are generated locally and are not published.
