#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

#include <vector>
#include <map>
#include <numeric>
#include <optional>
#include <cstring>

static constexpr auto MS_PER_SECOND = 1000.0; // ms
static constexpr auto ROUNDS = DEFAULT_ROUNDS; // rounds

__global__ void l3WriteBandwidthKernel(uint32v4* __restrict__ dst, size_t n) {
    uint32_t tid = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    uint32_t stride = static_cast<uint32_t>(gridDim.x * blockDim.x);

    uint32v4 dummy = {tid, tid + 1, tid + 2, tid + 3};

    for (size_t j = 0; j < blockDim.x; ++j) {
        tid = (((blockIdx.x + j) * blockDim.x) + threadIdx.x) % stride;

        for (size_t i = tid; i < n; i += stride) {
            #ifdef __HIP_PLATFORM_AMD__
            asm volatile(
                "flat_store_dwordx4 %0, %1\n"
                :
                : "v"(dst + i), "v"(dummy)
                : "memory"
            );
            #endif
        }
    }
}

BandwidthResult l3WriteBandwidthLauncher(size_t arraySizeBytes,
                                         util::AllocatorType allocType,
                                         bool warmup,
                                         bool prefetch,
                                         bool cpuInit) {
    util::hipDeviceReset();

    uint32_t maxThreadsPerBlock = util::min(util::getMaxThreadsPerBlock(), util::getWarpSize() * util::getSIMDsPerCU());
    uint32_t maxBlocks = util::getNumberOfComputeUnits() * util::getDeviceProperties().maxBlocksPerMultiProcessor;

    // Allocate once and reuse across all timed rounds. Round 0 is the first measured
    // kernel, but is cold only when no preceding preparation established access state.
    size_t dstElems = arraySizeBytes / sizeof(uint32v4);
    size_t allocatedBytes = dstElems * sizeof(uint32v4);
    uint32v4* d_dstArr = util::allocateMemory<uint32v4>(dstElems, allocType);

    if (cpuInit) {
        // Touch the complete destination allocation from the CPU. Effective mapping and
        // physical placement remain allocator- and runtime-dependent.
        std::memset(d_dstArr, 0, allocatedBytes);
    }

    if (prefetch && allocType == util::AllocatorType::HipMallocManaged) {
        int device;
        util::hipCheck(hipGetDevice(&device));
        util::hipCheck(hipMemPrefetchAsync(d_dstArr, allocatedBytes, device, 0));
        util::hipCheck(hipDeviceSynchronize());
    }

    if (warmup) {
        // Untimed complete repeated-access kernel. This can establish mappings and exercise
        // cache/translation state without guaranteeing that every later access hits L3.
        l3WriteBandwidthKernel<<<maxBlocks, maxThreadsPerBlock>>>(d_dstArr, dstElems);
        util::hipCheck(hipDeviceSynchronize());
    }

    auto start = util::createHipEvent();
    auto end = util::createHipEvent();

    std::vector<double> results(ROUNDS);
    for (uint32_t i = 0; i < ROUNDS; ++i) {
        util::hipCheck(hipDeviceSynchronize());
        util::hipCheck(hipEventRecord(start));
        l3WriteBandwidthKernel<<<maxBlocks, maxThreadsPerBlock>>>(d_dstArr, dstElems);
        util::hipCheck(hipEventRecord(end));
        util::hipCheck(hipDeviceSynchronize());
        results[i] = (util::getElapsedTimeMs(start, end) / maxThreadsPerBlock) / MS_PER_SECOND;
    }

    util::hipCheck(hipEventDestroy(start));
    util::hipCheck(hipEventDestroy(end));

    util::freeMemory(d_dstArr, allocType);

    double testSizeGiB = static_cast<double>(allocatedBytes) / (1 * GiB);
    BandwidthResult br;
    br.rounds.reserve(ROUNDS);
    for (uint32_t i = 0; i < ROUNDS; ++i) {
        br.rounds.push_back(testSizeGiB / results[i]);
    }
    br.average = testSizeGiB / util::average(results);
    return br;
}

namespace benchmark {
    namespace amd {
        BandwidthResult measureL3WriteBandwidth(size_t l3SizeBytes,
                                                util::AllocatorType allocType,
                                                bool warmup,
                                                bool prefetch,
                                                bool cpuInit) {
            return l3WriteBandwidthLauncher(
                l3SizeBytes, allocType, warmup, prefetch, cpuInit);
        }
    }
}
