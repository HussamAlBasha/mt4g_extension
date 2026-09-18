# Topology snapshots

This experiment collects provenance for an AMD compute partition and stores it under `runs/<mode>/<run-id>/`.

Submit one job for each desired mode:

```bash
mkdir -p logs
sbatch --mi300-partition=spx job.sh spx
sbatch --mi300-partition=tpx job.sh tpx
sbatch --mi300-partition=cpx job.sh cpx
```

The `--mi300-partition` option asks the cluster to configure the requested GPU partition. The matching argument is used for validation and output naming. The worker checks the active partition with `amd-smi` and rejects a mismatch.

`job.sh` owns the Slurm allocation and launches `worker.sh` with `srun`, so all topology collection runs inside the Slurm step. Logs are written to `logs/`.

The worker records both the requested and observed mode in `summary.txt` and fails after preserving the snapshot if they differ. A single allocation cannot measure all three modes because the compute partition is selected when the job is submitted.

## Future per-XCD measurements

These topology runs do not measure bandwidth or latency. In CPX mode, each MI300A is exposed as six logical GPU agents, each with 38 compute units and one XCD. A two-APU Viper node therefore exposes 12 logical GPU agents.

For a future latency or bandwidth experiment, keep one CPX Slurm allocation and run all measurements inside it. Before benchmarking, capture `/sys/class/kfd/kfd/topology/nodes/*/properties` and `rocminfo` to record the mapping between the logical device index, `location_id`, `unique_id`, and render device. Then run the benchmark serially on logical devices 0 through 5 to compare the six XCDs of the first APU (or 6 through 11 for the second APU).

The [CPX identity experiment](../CPX_identity/README.md) found the same mapping for all twelve logical devices across three separate jobs on `vipa1069` within one boot. Device 0 kept the same HIP and ROCr UUIDs, PCI address, and KFD identity. This supports selecting the same physical XCD by logical index across repeated jobs under that tested CPX/NPS1 setup. The jobs did not test another node, a reboot, a deliberate partition reset, or changed device masks. Record device identities in each measurement job so its mapping can be checked.
