#pragma once

#include <cstddef>
#include "utils/hip/memory.hpp"
#include "typedef/bandwidthResult.hpp"

namespace benchmark {
    namespace amd {
        /**
         * @brief Measure achievable L3 read bandwidth on AMD GPUs.
         *
         * @param l3SizeBytes Size of the working set in bytes.
         * @param allocType   Memory allocator for the source buffer (default: HipMalloc).
         * @param warmup      Run one untimed kernel pass before timing rounds.
         * @param prefetch    Prefetch managed memory before any GPU kernel access.
         * @param cpuInit     Touch the source buffer from the CPU before GPU access.
         * @return BandwidthResult with per-round GiB/s values and their average.
         */
        BandwidthResult measureL3ReadBandwidth(size_t l3SizeBytes,
                                               util::AllocatorType allocType = util::AllocatorType::HipMalloc,
                                               bool warmup = false,
                                               bool prefetch = false,
                                               bool cpuInit = false);
    }
}
