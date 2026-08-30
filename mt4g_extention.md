# MT4G Extension Change Log

This document tracks source-code changes made on the `cdna3-and-bw` branch.
Add a dated entry whenever the branch gains or changes functionality.

## 2026-08-30: Configurable memory allocator for bandwidth sweeps

Commit: `260a7fb` (`Add allocator selector for main-memory and L3 bandwidth sweeps`)

### Purpose

Allow the optimal-search L3 and main-memory bandwidth benchmarks to compare
different memory allocation mechanisms while preserving the existing benchmark
lifecycle and warm-up behavior.

### Supported allocators

The `util::AllocatorType` enum and its allocation/deallocation dispatch are
defined in `src/utils/hip/memory.hpp`.

| CLI value | Allocation | Deallocation | Notes |
|---|---|---|---|
| `hipmalloc` | `hipMalloc` | `hipFree` | Default device-memory allocator |
| `hipmallocmanaged` | `hipMallocManaged` | `hipFree` | Demand-paged managed memory |
| `hiphostmalloc` | `hipHostMalloc` | `hipHostFree` | Uses `hipHostMallocNonCoherent` on AMD and the portable default elsewhere |
| `malloc` | `::malloc` | `::free` | Intended for HMM-accessible CPU memory on MI300A; requires `XNACK=1` |

### Command-line interface

The new option is:

```text
--allocator hipmalloc|hipmallocmanaged|hiphostmalloc|malloc
```

Its default is `hipmalloc`. Invalid values terminate with an error that lists
the accepted values.

Examples:

```bash
./build/mt4g --optimal --l3 --allocator hipmallocmanaged
./build/mt4g --optimal --memory --allocator hiphostmalloc
./build/mt4g --optimal --l3 --memory --allocator malloc
```

The selected value is written to the JSON output as `bandwidthAllocator` under
the corresponding `memory.l3` or `memory.main` object.

### Affected benchmarks

Allocator selection applies only to these optimal-search sweep functions:

- `benchmark::amd::measureL3ReadBandwidthSweep`
- `benchmark::amd::measureL3WriteBandwidthSweep`
- `benchmark::measureMainMemoryReadBandwidthSweep`
- `benchmark::measureMainMemoryWriteBandwidthSweep`

The non-sweep bandwidth benchmarks and the L1/L2 benchmarks continue to use
`hipMalloc`.

### Buffer selection

For read bandwidth, only the large source working set uses the selected
allocator. The small destination buffer remains device memory because it only
stores the final value used to prevent dead-code elimination.

For write bandwidth, the large destination working set is the benchmarked
buffer, so it uses the selected allocator.

### Allocation and warm-up lifecycle

Each tested configuration gets a fresh allocation and its own warm-up:

```text
for each block/thread/repetition configuration:
    allocate with the selected allocator
    run the existing untimed warm-up
    run the timed measurement
    free with the matching deallocator
```

This preserves fair comparison between configurations by preventing allocation
state, residency, or earlier configurations from influencing later ones.

For main-memory sweeps, `repsTested` contains only `1`. This means one streaming
pass per block/thread configuration; it does not mean the launcher is called
only once for the complete sweep.

Warm-up remains mandatory and internal to the benchmark. No configurable
warm-up option was added.

### Source files changed

- `src/utils/hip/memory.hpp`
- `src/typedef/cliOptions.hpp`
- `src/utils/printing.cpp`
- `src/main.cpp`
- `src/benchmarks/bandwidth/mainMemoryReadBandwidth.cpp/.hpp`
- `src/benchmarks/bandwidth/mainMemoryWriteBandwidth.cpp/.hpp`
- `src/benchmarks/bandwidth/amd_l3ReadBandwidth.cpp/.hpp`
- `src/benchmarks/bandwidth/amd_l3WriteBandwidth.cpp/.hpp`
- `README.md`

### Validation

- Full CMake build completed successfully for `gfx942` with ROCm 7.2.
- `--help` displays the allocator option.
- Invalid allocator values are rejected during CLI parsing.
- Existing unrelated `cuIdx` unused-variable warnings remain.
- Runtime bandwidth measurements for every allocator are not yet recorded here.

### Deferred work

The following options from the develop-branch allocator work were deliberately
not ported in this step:

- managed-memory prefetch
- working-set size override
- CPU-side initialization
- configurable warm-up

If any of these are added later, document their measurement semantics and
affected benchmarks in a new dated section rather than silently changing this
benchmark lifecycle.

## 2026-08-30: Configurable allocator for main-memory latency

Commit: pending

### Purpose

Extend `--allocator` to the main-memory pointer-chase latency benchmark as the
first latency step. L3 latency is intentionally unchanged and will be evaluated
separately.

### Behavior and design decisions

The 1 GiB randomized pointer-chase array uses the selected allocator. All
allocator modes use the same GPU-side initialization path; there is no special
direct-copy path for `hipmalloc`. The timing-results buffer remains ordinary
device memory because it is only an output of the timing kernel and is not the
memory whose latency is measured.

The cache-safe initialization, scrub, and partial warm-up sequence in the next
dated section is the authoritative lifecycle for this benchmark. No
configurable warm-up, managed-memory prefetch, or CPU-first-touch option was
added.

The selected allocator is written to JSON as
`memory.main.latency.allocator`. The existing
`memory.main.bandwidthAllocator` field remains specific to the bandwidth
sweeps.

### Source files changed

- `src/benchmarks/latency/mainMemoryLatency.cpp/.hpp`
- `src/utils/hip/memory.hpp`
- `src/typedef/cliOptions.hpp`
- `src/utils/printing.cpp`
- `src/main.cpp`
- `README.md`

### Validation

- CMake rebuild completed successfully for `gfx942` with ROCm 7.2.
- Existing unrelated unused-variable warnings remain.
- Final cache-safety and runtime validation are recorded in the next section.

## 2026-08-31: Cache-safe main-memory latency initialization

Commit: pending

### Purpose

Prevent the main-memory latency benchmark from measuring MALL hits introduced
by either a complete pointer-chain warm-up or allocator-specific initialization.

### Behavior and design decisions

Every allocator now uses the same initialization sequence:

```text
generate the randomized pointer chain on the CPU
allocate the measured array with the selected allocator
copy the pointer chain into it from an untimed GPU kernel
read one word from every 64-byte line of the separate 1 GiB source buffer
perform 2,048 untimed pointer-chase hops
continue from that index and time the next 2,048 distinct hops
```

The source-buffer traversal uses ordinary cache-allocating loads. Its 1 GiB
active cache-line footprint is four times the MI300A's 256 MiB MALL, displacing
measured-array lines left by initialization. The source is then freed before
measurement. The final pointer index is written to a separate result slot, so
all 2,048 timing samples remain valid.

With a 1 KiB chain stride, the 1 GiB array contains 1,048,576 pointer-chain
nodes. Warm-up and measurement together visit only 4,096 nodes, so the chain
does not wrap and a timed hop cannot revisit a line touched by the warm-up.
Both phases use `__forceBypassAllCacheReads`. Each measured hop is bracketed by
its own `__timer()` calls and is reported in cycles; no batched timing or
assumed shader-clock conversion is part of this benchmark.

### Validation

- The Release build completed successfully for `gfx942` with ROCm 7.2.
- Disassembly confirms that the scrub kernel uses ordinary
  `global_load_dword` instructions without non-temporal or cache-bypass flags.
- The allocator/partition matrix completed for SPX, TPX, and CPX with XNACK on
  and off. Device, managed, and host allocations completed in every mode;
  ordinary `malloc` completed with XNACK enabled and produced the expected
  failure with XNACK disabled.
- Runtime results keep main-memory latency clearly above L3 latency across the
  successful allocator and partition combinations.

## 2026-08-30: Configurable allocator for L3 latency

Commit: pending

### Purpose

Extend `--allocator` to the AMD L3 pointer-chase latency benchmark while
preserving its existing L3 cache warm-up behavior.

### Behavior and design decisions

The pointer-chase working set uses the selected allocator. The timing-results
buffer remains ordinary device memory. For `hipmalloc`, initialization retains
the original host-to-device copy path. For the other allocators, a temporary
`hipMalloc` buffer receives the generated pointer chain and an untimed GPU
kernel copies it into the selected allocation.

The existing first loop in `l3LatencyKernel` remains unchanged. It traverses
the complete pointer cycle before timing, evicting the working set from L2 and
placing it in L3. Removing this traversal would measure cold first-access,
mapping, and migration costs instead of L3-hit latency.

The selected allocator is written to JSON as
`memory.l3.latency.allocator`. The existing
`memory.l3.bandwidthAllocator` field remains specific to the bandwidth sweeps.

### Source files changed

- `src/benchmarks/latency/amd_l3Latency.cpp/.hpp`
- `src/utils/hip/memory.hpp`
- `src/typedef/cliOptions.hpp`
- `src/utils/printing.cpp`
- `src/main.cpp`
- `README.md`

### Validation

- Full CMake build completed successfully for `gfx942` with ROCm 7.2.
- Runtime L3-latency measurements for all allocators are not yet recorded.

## Entry template

```markdown
## YYYY-MM-DD: Change title

Commit: `<commit>`

### Purpose

### Behavior and design decisions

### Source files changed

### Validation

### Follow-up work
```
