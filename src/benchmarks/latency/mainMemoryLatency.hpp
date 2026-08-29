#pragma once

#include <cstddef>
#include <cstdint>
#include "utils/hip/memory.hpp"

namespace benchmark {
    /**
     * @brief Measure the latency of main memory accesses.
     *
     * @param allocType  Memory allocator for the pointer-chase array (default: HipMalloc).
     * @param warmup     If true, traverse the pointer-chase array once (untimed) before
     *                   the timed 2048-sample pass, to warm up TLB and page mappings.
     * @param prefetch   Prefetch managed memory after pointer-chain initialization and before
     *                   optional warm-up or timing (HipMallocManaged only).
     * @param cpuInit    If true, initialize from the CPU; otherwise initialize from the GPU
     *                   in an untimed kernel (USM allocators only). Consequently, GPU-first
     *                   page establishment is outside the timed pointer chase.
     * @return Latency distribution. Unit field is CYCLE; convert to ns via clock frequency.
     */
    CacheLatencyResult measureMainMemoryLatency(
        util::AllocatorType allocType = util::AllocatorType::HipMalloc,
        bool warmup = false,
        bool prefetch = false,
        bool cpuInit = false);
}
