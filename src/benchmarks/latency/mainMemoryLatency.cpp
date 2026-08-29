#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

#include <vector>
#include <map>
#include <numeric>
#include <optional>
#include <cstring>

static constexpr auto SAMPLE_SIZE = 2048;// 2048 Loads should suffice to rule out random flukes

__global__ void mainMemoryLatencyKernel(uint32_t *pChaseArray, uint32_t *timingResults) {
    uint32_t index = 0;
    __shared__ uint64_t s_timings[SAMPLE_SIZE];

    // Do not load from caches
    for (uint32_t i = 0; i < SAMPLE_SIZE; ++i) {
        uint64_t start = __timer();
        index = __forceBypassAllCacheReads(pChaseArray, index); 
        uint64_t end = __timer(); 

        s_timings[i] = end - start;
    }

    for (uint32_t i = 0; i < SAMPLE_SIZE; ++i) {
        timingResults[i] = s_timings[i];
    }

    timingResults[0] += index >> util::min(index / 2, 32);
}

// Warm-up kernel: traverse the full array once without timing.
// The result is written to a single-element sink to prevent dead-code elimination.
__global__ void mainMemoryLatencyWarmupKernel(uint32_t *pChaseArray, size_t numHops, uint32_t *sink) {
    uint32_t index = 0;
    for (size_t i = 0; i < numHops; ++i) {
        index = __forceBypassAllCacheReads(pChaseArray, index);
    }
    sink[0] = index;
}

__global__ void mainMemoryLatencyInitKernel(uint32_t *destination, const uint32_t *source, size_t count) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    for (; index < count; index += stride) {
        destination[index] = source[index];
    }
}

std::vector<uint32_t> mainMemoryLatencyLauncher(size_t arraySizeBytes, size_t strideBytes,
                                                util::AllocatorType allocType,
                                                bool warmup,
                                                bool prefetch,
                                                bool cpuInit) {
    util::hipDeviceReset();

    std::vector<uint32_t> hostChaseArray =
        util::generateRandomizedPChaseArray(arraySizeBytes, strideBytes);
    size_t bytes = hostChaseArray.size() * sizeof(uint32_t);

    uint32_t *d_pChaseArray = nullptr;
    if (allocType == util::AllocatorType::HipMalloc) {
        // Baseline: hipMalloc + hipMemcpy (unchanged behaviour)
        d_pChaseArray = util::allocateGPUMemory(hostChaseArray);
    } else {
        d_pChaseArray = util::allocateMemory<uint32_t>(
            hostChaseArray.size(), allocType);

        // Populate the pointer chain from the selected processor. GPU initialization is
        // deliberately untimed: it establishes the chain and any associated GPU mappings
        // before the pointer-chase latency samples are collected.
        if (cpuInit) {
            std::memcpy(d_pChaseArray, hostChaseArray.data(), bytes);
        } else {
            uint32_t *d_initArray = util::allocateGPUMemory(hostChaseArray);
            uint32_t threads = util::min(
                util::getMaxThreadsPerBlock(), util::getWarpSize() * util::getSIMDsPerCU());
            uint32_t blocks = util::getNumberOfComputeUnits()
                * util::getDeviceProperties().maxBlocksPerMultiProcessor;
            mainMemoryLatencyInitKernel<<<blocks, threads>>>(
                d_pChaseArray, d_initArray, hostChaseArray.size());
            util::hipCheck(hipDeviceSynchronize());
            util::hipCheck(hipFree(d_initArray));
        }

        if (prefetch && allocType == util::AllocatorType::HipMallocManaged) {
            // Prefetch the initialized managed allocation to the selected GPU. On MI300A
            // this need not imply migration between distinct physical memory pools; the
            // effective mapping and residency behavior remains runtime-dependent.
            int device;
            util::hipCheck(hipGetDevice(&device));
            util::hipCheck(hipMemPrefetchAsync(d_pChaseArray, bytes, device, 0));
            util::hipCheck(hipDeviceSynchronize());
        }
    }

    uint32_t *d_timingResults = util::allocateGPUMemory(SAMPLE_SIZE);

    // Untimed complete dependent traversal. This can establish mappings and exercise
    // translation state, but does not guarantee that the complete chain remains TLB-resident.
    if (warmup) {
        uint32_t *d_sink = util::allocateGPUMemory(static_cast<size_t>(1));
        size_t numHops = arraySizeBytes / strideBytes;
        mainMemoryLatencyWarmupKernel<<<1, 1>>>(d_pChaseArray, numHops, d_sink);
        util::hipCheck(hipDeviceSynchronize());
        util::hipCheck(hipFree(d_sink));
    }

    util::hipCheck(hipDeviceSynchronize());
    mainMemoryLatencyKernel<<<1, 1>>>(d_pChaseArray, d_timingResults);

    std::vector<uint32_t> timingResultBuffer = util::copyFromDevice(d_timingResults, SAMPLE_SIZE);

    util::freeMemory(d_pChaseArray, allocType);
    util::hipCheck(hipFree(d_timingResults));
    
    timingResultBuffer.erase(timingResultBuffer.begin());

    return timingResultBuffer;
}

namespace benchmark {
    CacheLatencyResult measureMainMemoryLatency(util::AllocatorType allocType,
                                                bool warmup,
                                                bool prefetch,
                                                bool cpuInit) {

        auto timings = mainMemoryLatencyLauncher(1 * GiB, 1 * KiB, allocType, warmup, prefetch,
                                                  cpuInit);

        CacheLatencyResult result {
            timings,
            util::average(timings),
            util::percentile(timings, 0.5),
            util::percentile(timings, 0.95),
            util::stdev(timings),
            timings.size(),
            SAMPLE_SIZE,
            CYCLE,
            PCHASE
        };

        return result;
    }
}
