#pragma once

#include <cstddef>
#include "utils/hip/memory.hpp"
#include "typedef/bandwidthResult.hpp"

namespace benchmark {
    /**
     * @brief Measure peak main memory write bandwidth.
     *
     * @param mainMemorySizeBytes Working set size in bytes.
     * @param allocType           Memory allocator to use for the destination buffer.
     * @param warmup              Run one untimed kernel pass before timing rounds.
     * @param prefetch            Prefetch to device after alloc (HipMallocManaged only).
     * @param cpuInit             If true, host memset before any GPU access (USM allocators only).
     * @return BandwidthResult with per-round GiB/s values and their average.
     */
    BandwidthResult measureMainMemoryWriteBandwidth(
        size_t mainMemorySizeBytes,
        util::AllocatorType allocType = util::AllocatorType::HipMalloc,
        bool warmup = false,
        bool prefetch = false,
        bool cpuInit = false);
}
