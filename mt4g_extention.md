# Configurable Allocator Extension for MT4G Memory Benchmarks

This document explains how MT4G selects the allocator for the measured working
set in its L3 and main-memory bandwidth and latency benchmarks. It describes
which buffers use the selected allocator, how each benchmark prepares and
measures them, and how the choice appears in the results.

Allocator selection changes the backing memory while keeping each benchmark's
access pattern and timed metric fixed. Auxiliary buffers continue to use device
memory. The bandwidth and L3-latency benchmarks retain their existing
measurement sequences. Main-memory latency uses the same initialization and
cache preparation for every allocator to reduce cache effects before timing.

## Scope

The extension covers these benchmark implementations:

- `src/benchmarks/bandwidth/mainMemoryReadBandwidth.cpp/.hpp`
- `src/benchmarks/bandwidth/mainMemoryWriteBandwidth.cpp/.hpp`
- `src/benchmarks/bandwidth/amd_l3ReadBandwidth.cpp/.hpp`
- `src/benchmarks/bandwidth/amd_l3WriteBandwidth.cpp/.hpp`
- `src/benchmarks/latency/mainMemoryLatency.cpp/.hpp`
- `src/benchmarks/latency/amd_l3Latency.cpp/.hpp`

For bandwidth, allocator selection applies specifically to the four
optimal-search sweep functions:

- `benchmark::measureMainMemoryReadBandwidthSweep`
- `benchmark::measureMainMemoryWriteBandwidthSweep`
- `benchmark::amd::measureL3ReadBandwidthSweep`
- `benchmark::amd::measureL3WriteBandwidthSweep`

The non-sweep bandwidth functions, and all L1 and L2 benchmarks, continue to
use `hipMalloc`. The allocator applies to the two latency benchmarks whenever
their corresponding L3 or main-memory benchmark group runs; latency does not
depend on `--optimal`.

## Unified allocator path

The command-line interface exposes one option for all six benchmarks:

```text
--allocator hipmalloc|hipmallocmanaged|hiphostmalloc|malloc
```

`hipmalloc` is the default. The parser is case-insensitive and rejects an
unknown value with the list of accepted values. The selected value is stored as
`util::AllocatorType` in `CLIOptions` and passed through `main.cpp` to the
public benchmark function declared in each `.hpp` file. Those declarations use
`HipMalloc` as a default argument, which preserves the previous behavior for
callers that do not provide an allocator.

Within each `.cpp` file, the selection reaches the launcher that owns the
measured working set. Allocator-dependent paths create that set through
`util::allocateMemory`, run the benchmark-specific preparation and timed
access, and release it with the matching `util::freeMemory` operation:

```text
--allocator
    -> CLIOptions::allocType
    -> public benchmark function
    -> benchmark launcher
    -> allocate measured working set
    -> prepare/warm up and measure
    -> matching deallocation
```

The common dispatch is defined in `src/utils/hip/memory.hpp`:

| CLI value | Allocation | Deallocation | Intended memory path |
|---|---|---|---|
| `hipmalloc` | `hipMalloc` | `hipFree` | Device memory; the backward-compatible default |
| `hipmallocmanaged` | `hipMallocManaged` | `hipFree` | HIP managed memory |
| `hiphostmalloc` | `hipHostMalloc` | `hipHostFree` | Host allocation visible to the GPU; AMD uses `hipHostMallocNonCoherent` |
| `malloc` | `::malloc` | `::free` | CPU heap accessed by the GPU through HMM on MI300A; requires `XNACK=1` |

The only equivalent shortcut is the `hipmalloc` branch of L3 latency, which
retains the original `allocateGPUMemory(hostChaseArray)` host-to-device copy.
That helper also uses `hipMalloc`; the other three L3-latency modes use the
common allocator dispatch and an explicit GPU initialization kernel.

No managed-memory prefetch, CPU-first-touch mode, configurable warm-up, or
working-set-size override is introduced. Consequently, the allocator is the
independent variable while each benchmark's access pattern remains fixed.

## Effect on each benchmark

| Benchmark | Selectable measured allocation | Allocation that stays on `hipMalloc` | Measurement lifecycle |
|---|---|---|---|
| Main-memory read bandwidth sweep | Large source array | Small destination/sink array | Warm-up followed by one timed streaming pass per block/thread configuration |
| Main-memory write bandwidth sweep | Large destination array | No auxiliary data array | Warm-up followed by one timed streaming pass per block/thread configuration |
| AMD L3 read bandwidth sweep | Source working set | Small destination/sink array | Existing L3 warm-up and block/thread/repetition search |
| AMD L3 write bandwidth sweep | Destination working set | No auxiliary data array | Existing L3 warm-up and block/thread/repetition search |
| Main-memory latency | 1 GiB randomized pointer-chase array | Initialization source, scrub sink, and timing-results buffers | Cache scrub, partial pointer-chase warm-up, then per-hop timing with cache-bypassing loads |
| AMD L3 latency | Pointer-chase array sized to exceed L2 | Initialization source when needed and timing-results buffer | Complete untimed traversal to establish L3 residency, then per-hop timing of L3 reads |

This separation is deliberate. A read benchmark selects the allocator for its
large source because that is the region supplying the measured traffic. A
write benchmark selects it for the large destination because that region
receives the measured traffic. Latency benchmarks select it for the
pointer-chase array. Small output and timing buffers do not represent the
memory path under test, so changing their allocator would add an unrelated
variable.

### Bandwidth sweep lifecycle

Every tested sweep configuration invokes its launcher independently. The
lifecycle is therefore:

```text
for each block/thread/repetition configuration:
    allocate a fresh working set with the selected allocator
    run the existing untimed warm-up on that allocation
    run the timed measurement on the same allocation
    free it with the matching deallocator
```

This gives each configuration a fresh allocation and warm-up rather than
reusing the previous configuration's working set. The AMD L3 sweeps retain
their repetition axis and 512-repetition warm-up. The main-memory sweeps retain
a large working set of up to 1 GiB and four warm-up passes, but `repsTested`
contains only `1`.
That value means one timed streaming pass for each block/thread configuration,
not one launcher call for the entire sweep. Repeating the timed pass over the
same bounded region could turn the main-memory test into a cache-bandwidth
measurement.

### Main-memory latency lifecycle

Main-memory latency uses GPU initialization followed by cache preparation to
reduce the chance that initialization leaves measured-array data in the MI300A
MALL. The same sequence is used for all four allocators:

1. Generate a randomized pointer chain on the CPU. The array is 1 GiB, the
   stride is 1 KiB, and the resulting chain has 1,048,576 nodes.
2. Allocate the pointer-chase array with the selected allocator.
3. Copy the host-generated chain to a temporary `hipMalloc` buffer, then use an
   untimed GPU kernel to initialize the selected allocation.
4. Reuse the separate 1 GiB source buffer as a scrubber by reading one word from
   each 64-byte cache line with ordinary cache-allocating loads. The touched
   lines span four times the MI300A's 256 MiB MALL capacity and are intended to
   displace measured-array lines loaded during initialization. This is not a
   guaranteed cache flush.
5. Free the initialization and scrub buffers.
6. Perform 2,048 untimed dependent loads with
   `__forceBypassAllCacheReads`, continue from the resulting pointer index, and
   time the next 2,048 dependent loads with the same intrinsic.

Warm-up and measurement therefore visit 4,096 distinct nodes, well below the
1,048,576-node chain length, and the timed traversal does not wrap to lines
visited by the warm-up. Every measured hop is bracketed by its own `__timer()`
calls and reported in cycles. A separate extra result slot makes the final
pointer index observable without overwriting any of the 2,048 timing samples.

### AMD L3 latency lifecycle

The L3 latency benchmark preserves the original host-to-device initialization
path for `hipmalloc`. For the other allocators, it allocates the pointer-chase
array with the selected mechanism, copies the generated chain first to a
temporary `hipMalloc` buffer, and initializes the selected allocation with an
untimed GPU copy kernel.

After initialization, the original first loop in `l3LatencyKernel` still walks
the complete pointer cycle with `__l3Read`. This traversal evicts the working
set from L2 while establishing it in L3. Only the subsequent dependent loads
are timed. Removing the traversal would mix cold-access, mapping, or migration
costs into a benchmark intended to report L3-hit latency.

## Result metadata

The chosen allocator is recorded in the JSON result so measurements remain
self-describing:

| Result path | Applies to |
|---|---|
| `memory.l3.bandwidthAllocator` | L3 read and write bandwidth sweeps |
| `memory.main.bandwidthAllocator` | Main-memory read and write bandwidth sweeps |
| `memory.l3.latency.allocator` | L3 latency |
| `memory.main.latency.allocator` | Main-memory latency |

The two bandwidth fields are emitted only when optimal search runs, matching
the scope of allocator selection for bandwidth. The latency fields are nested
inside their respective latency results.

## Cross-cutting source changes

The twelve benchmark files implement the per-benchmark portion of the
extension: each header accepts `util::AllocatorType`, and each implementation
uses it for the measured allocation and its matching deallocation. The shared
plumbing is implemented by these files:

- `src/utils/hip/memory.hpp`: defines `AllocatorType` and the allocation/free
  dispatch.
- `src/typedef/cliOptions.hpp`: stores the selected allocator.
- `src/utils/printing.cpp`: declares and validates `--allocator`.
- `src/main.cpp`: forwards the selection to all six benchmarks and records it
  in the JSON result.
- `README.md`: documents the user-facing option and its scope.

## Preserved invariants and limitations

- `hipmalloc` remains the default, preserving earlier runs and direct API calls.
- Timed regions are not made allocator-specific; main-memory latency uses one
  common preparation path for every allocator.
- Only the measured working set uses the selected allocator; control and result
  buffers remain device allocations.
- Main-memory bandwidth still measures one timed streaming pass per tested
  launch configuration.
- L3 latency still measures a warmed L3 working set. Main-memory latency uses
  cache preparation and cache-bypassing dependent loads to limit cache effects
  from initialization and warm-up.
- Ordinary `malloc` depends on GPU access to CPU heap memory through HMM and is
  therefore expected to fail on the target MI300A configuration when XNACK is
  disabled.

## Recorded validation

- A Release build for `gfx942` completed with ROCm 7.2.
- `--help` displayed the allocator option, and invalid values were rejected by
  command-line parsing.
- Disassembly confirmed that the main-memory scrub kernel uses ordinary
  `global_load_dword` instructions without cache-bypass or non-temporal flags.
- Main-memory latency was exercised for SPX, TPX, and CPX with XNACK enabled
  and disabled. Device, managed, and host allocations completed in every mode;
  `malloc` completed with XNACK enabled and failed as expected when it was
  disabled.
- Successful main-memory latency results remained clearly above L3 latency,
  consistent with the intended cache preparation.
- A complete runtime bandwidth matrix for every allocator has been recorded.
