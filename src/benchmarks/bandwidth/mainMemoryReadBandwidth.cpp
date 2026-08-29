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

__global__ void mainMemoryReadBandwidthKernel(uint32v4* __restrict__ dst, uint32v4* __restrict__ src, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    uint32v4 dummy = {0, 0, 0, 0}; 

    for (size_t i = tid; i < n; i += stride) {
        uint32v4 loaded; 
        
        #ifdef __HIP_PLATFORM_NVIDIA__
        asm volatile(
            "ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(loaded.x) // int
            , "=r"(loaded.y) // int
            , "=r"(loaded.z) // int
            , "=r"(loaded.w) // int
            : "l"(src + i) // uint32v4*
        );
        #endif

        #ifdef __HIP_PLATFORM_AMD__
        asm volatile(
            "flat_load_dwordx4 %0, %1\n" 
            : "=v"(loaded) // uint32v4
            : "s"(src + i) // uint32v4*
            :
        );
        #endif

        // XOR is efficient
        dummy.x ^= loaded.x;
    }

    dst[tid % blockDim.x] = dummy; // prevent dead code elimination
}

BandwidthResult mainMemoryReadBandwidthLauncher(
    size_t arraySizeBytes,
    util::AllocatorType allocType,
    bool warmup,
    bool prefetch,
    bool cpuInit)
{
    util::hipDeviceReset();

    uint32_t maxThreadsPerBlock = util::min(util::getMaxThreadsPerBlock(), util::getWarpSize() * util::getSIMDsPerCU());
    uint32_t maxBlocks = util::getNumberOfComputeUnits() * util::getDeviceProperties().maxBlocksPerMultiProcessor;

    // sizeof(uint32v4) = 16 bytes -> allows us to load 4 integers with one instruction
    size_t srcElems = arraySizeBytes / sizeof(uint32v4);
    size_t allocatedBytes = srcElems * sizeof(uint32v4);
    uint32v4 *d_srcArr = util::allocateMemory<uint32v4>(srcElems, allocType);
    uint32v4 *d_dstArr = util::allocateMemory<uint32v4>(maxThreadsPerBlock, util::AllocatorType::HipMalloc); // total threads

    // Touch the complete source allocation from the CPU. Effective mapping and physical
    // placement remain allocator- and runtime-dependent.
    if (cpuInit) {
        std::memset(d_srcArr, 0, allocatedBytes);
    }

    if (prefetch && allocType == util::AllocatorType::HipMallocManaged) {
        int device;
        util::hipCheck(hipGetDevice(&device));
        util::hipCheck(hipMemPrefetchAsync(d_srcArr, allocatedBytes, device, 0));
        util::hipCheck(hipDeviceSynchronize());
    }

    // Untimed complete GPU read pass. This can establish mappings and exercise translation
    // state, but does not guarantee that the complete working set remains TLB-resident.
    if (warmup) {
        mainMemoryReadBandwidthKernel<<<maxBlocks, maxThreadsPerBlock>>>(d_dstArr, d_srcArr, srcElems);
        util::hipCheck(hipDeviceSynchronize());
    }
    
    // Use events to measure timings
    auto start = util::createHipEvent();
    auto end = util::createHipEvent();

    std::vector<double> results(ROUNDS);
    for (uint32_t i = 0; i < ROUNDS; ++i) {
        util::hipCheck(hipDeviceSynchronize());
        util::hipCheck(hipEventRecord(start));
        mainMemoryReadBandwidthKernel<<<maxBlocks, maxThreadsPerBlock>>>(d_dstArr, d_srcArr, srcElems);
        util::hipCheck(hipEventRecord(end));
        util::hipCheck(hipDeviceSynchronize());
        results[i] = util::getElapsedTimeMs(start, end) / MS_PER_SECOND;
    }

    util::hipCheck(hipEventDestroy(start));
    util::hipCheck(hipEventDestroy(end));

    util::freeMemory(d_srcArr, allocType);
    util::freeMemory(d_dstArr, util::AllocatorType::HipMalloc);

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
    BandwidthResult measureMainMemoryReadBandwidth(
        size_t mainMemorySizeBytes,
        util::AllocatorType allocType,
        bool warmup,
        bool prefetch,
        bool cpuInit)
    {
        size_t testSizeBytes = mainMemorySizeBytes / SIZE_DOWN; // Divide by SIZE_DOWN to avoid too large memory allocations
        return mainMemoryReadBandwidthLauncher(testSizeBytes, allocType, warmup, prefetch,
                                               cpuInit);
    }
}
