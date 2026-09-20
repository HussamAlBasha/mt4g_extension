# MT4G partition experiments

These experiments use the extended MT4G implementation on the MPCDF Viper cluster to study MI300A topology and memory performance across SPX, TPX, and CPX compute partition modes.

| Experiment | Purpose |
| --- | --- |
| [`topology`](topology/README.md) | Records the GPU topology exposed in SPX, TPX, and CPX and verifies that each Slurm job received the requested partition mode. |
| [`CPX_identity`](CPX_identity/README.md) | Checks whether the twelve CPX device ordinals identify the same physical XCDs across repeated jobs on one node. |
| [`allocator_partition_matrix`](allocator_partition_matrix/README.md) | Compares L3 and main-memory latency and bandwidth across partition modes, XNACK settings, and four allocators. |
| [`lds_static_dynamic`](lds_static_dynamic/README.md) | Compares MT4G bandwidth with static and dynamic LDS declarations using paired measurements in all three partition modes. |
| [`CPX_sweep`](CPX_sweep/README.md) | Repeatedly measures all twelve CPX devices on three nodes to evaluate device-level variation and repeatability. |

The topology and identity experiments establish how logical devices map to the MI300A hardware. The allocator matrix and LDS experiment compare controlled configuration choices, while the CPX sweep examines repeated measurements of individual logical devices.

The performance experiments expect an existing `build/mt4g` executable. Build instructions are in [`build.md`](../build.md). Each experiment README provides the commands for submitting its jobs and analyzing its output.
