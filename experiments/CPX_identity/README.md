# Do CPX device ordinals identify the same XCDs across jobs?

This experiment records the identities of all 12 logical GPUs on a two-MI300A node in CPX mode. It compares each device ordinal's HIP UUID, ROCr UUID, and PCI address across separate jobs on the same node. It also saves KFD and render-device information for inspection. No MT4G benchmark or compilation is needed.

## Run on Viper

From this directory, submit the command below three times. Wait for each job to finish before submitting the next, and use the same node each time. Replace `vipa1008` with an available APU node.

```bash
cd /u/halba/mt4g_extension/experiments/CPX_identity
sbatch --mi300-partition=cpx --nodelist=vipa1008 job.sh
```

Compare the saved records after the jobs finish:

```bash
python3 identity.py compare runs/vipa1008_*/identity.json
```

`MATCH` means every ordinal identified the same device in the supplied jobs. `MISMATCH` lists the changed ordinals. The comparison requires distinct jobs on one host with the same HIP runtime version and reports how many boot IDs were observed. If the directory contains other campaigns, pass only the records you intend to compare.

The result describes the tested jobs. It does not establish that the mapping survives a reboot, partition change, device-mask change, or move to another node.
