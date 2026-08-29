#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

#include <vector>
#include <map>
#include <numeric>
#include <optional>
#include <cstring>

static constexpr auto SIZE_DOWN = DEFAULT_SIZE_DOWN_FACTOR;// Factor
static constexpr auto MS_PER_SECOND = 1000.0;// ms
static constexpr auto ROUNDS = DEFAULT_ROUNDS;// rounds

__global__ void mainMemoryWriteBandwidthKernel(uint32v4* __restrict__ dst, size_t n) {
    // thread unique global index
    uint32_t tid = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    // Total number of threads in the grid
    uint32_t stride = static_cast<uint32_t>(gridDim.x * blockDim.x);

    // The 16-byte value being written 
    uint32v4 dummy = { tid, tid + 1, tid + 2, tid + 3 }; 

    for (size_t i = tid; i < n; i += stride) {
        #ifdef __HIP_PLATFORM_NVIDIA__
        asm volatile(
            "st.global.wt.v4.u32 [%0], {%1, %2, %3, %4};\n"
            :
            : "l"(dst + i) // uint32v4*
            , "r"(dummy.x) // int
            , "r"(dummy.y) // int
            , "r"(dummy.z) // int
            , "r"(dummy.w) // int
        );
        #endif

        #ifdef __HIP_PLATFORM_AMD__
        asm volatile(
            "flat_store_dwordx4 %0, %1\n"
            :
            : "s"(dst + i) // uint32v4*
            , "v"(dummy) // uint32v4
            : "memory"
        );
        #endif
    }
}

BandwidthResult mainMemoryWriteBandwidthLauncher(
    size_t arraySizeBytes,
    util::AllocatorType allocType,
    bool warmup,
    bool prefetch,
    bool cpuInit)
{
    util::hipDeviceReset();

    uint32_t maxThreadsPerBlock = util::min(util::getMaxThreadsPerBlock(), util::getWarpSize() * util::getSIMDsPerCU());
    uint32_t maxBlocks = util::getNumberOfComputeUnits() * util::getDeviceProperties().maxBlocksPerMultiProcessor;

    size_t dstElems = arraySizeBytes / sizeof(uint32v4);
    size_t allocatedBytes = dstElems * sizeof(uint32v4);
    uint32v4 *d_dstArr = util::allocateMemory<uint32v4>(dstElems, allocType);

    // Touch the complete destination allocation from the CPU. Effective mapping and physical
    // placement remain allocator- and runtime-dependent.
    if (cpuInit) {
        std::memset(d_dstArr, 0, allocatedBytes);
    }

    if (prefetch && allocType == util::AllocatorType::HipMallocManaged) {
        int device;
        util::hipCheck(hipGetDevice(&device));
        util::hipCheck(hipMemPrefetchAsync(d_dstArr, allocatedBytes, device, 0));
        util::hipCheck(hipDeviceSynchronize());
    }

    // Untimed complete GPU write pass. This can establish mappings and exercise translation
    // state, but does not guarantee that the complete working set remains TLB-resident.
    if (warmup) {
        mainMemoryWriteBandwidthKernel<<<maxBlocks, maxThreadsPerBlock>>>(d_dstArr, dstElems);
        util::hipCheck(hipDeviceSynchronize());
    }

    // Use events to measure timings
    auto start = util::createHipEvent();
    auto end = util::createHipEvent();

    std::vector<double> results(ROUNDS);
    for (uint32_t i = 0; i < ROUNDS; ++i) {
        util::hipCheck(hipDeviceSynchronize());
        util::hipCheck(hipEventRecord(start));
        mainMemoryWriteBandwidthKernel<<<maxBlocks, maxThreadsPerBlock>>>(d_dstArr, dstElems);
        util::hipCheck(hipEventRecord(end));
        util::hipCheck(hipDeviceSynchronize());
        results[i] = util::getElapsedTimeMs(start, end) / MS_PER_SECOND;
    }

    util::hipCheck(hipEventDestroy(start));
    util::hipCheck(hipEventDestroy(end));

    util::freeMemory(d_dstArr, allocType);

    double testSizeGiB = (double)allocatedBytes / (double)(1 * GiB); // Convert to GiB
    BandwidthResult br;
    br.rounds.reserve(ROUNDS);
    for (uint32_t i = 0; i < ROUNDS; ++i) {
        br.rounds.push_back(testSizeGiB / results[i]);
    }
    br.average = testSizeGiB / util::average(results);
    return br;
}

namespace benchmark {
    BandwidthResult measureMainMemoryWriteBandwidth(
        size_t mainMemorySizeBytes,
        util::AllocatorType allocType,
        bool warmup,
        bool prefetch,
        bool cpuInit)
    {
        size_t testSizeBytes = mainMemorySizeBytes / SIZE_DOWN; // Divide by SIZE_DOWN to avoid too large memory allocations
        return mainMemoryWriteBandwidthLauncher(testSizeBytes, allocType, warmup, prefetch,
                                                cpuInit);
    }
}