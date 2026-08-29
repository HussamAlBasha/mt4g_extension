#pragma once

#include <hip/hip_runtime.h>
#include <filesystem>
#include <string>
#include <optional>

#include "utils/hip/memory.hpp"

struct CLIOptions {
    std::string fileName;             // Name of output files
    std::filesystem::path location;   // Location of output files
    int  deviceId;                    // GPU ID via -d / --device-id
    bool graphs;                      // Generate graphs if true
    bool rawData;                     // Output raw measurement data
    bool fullReport;                  // Write README with summary and graphs
    bool useStdout;                   // Dump final JSON result to stdout
    bool randomize;                   // Randomize P-Chase arrays if true
    bool runSilently;                 // Do not print progress information if true

    // Main-memory and AMD L3 benchmark options
    bool warmup;                      // Run one untimed warm-up kernel before timed rounds
    util::AllocatorType allocType;    // Memory allocator for main-memory and AMD L3 benchmarks
    bool prefetch;                    // Prefetch to device after allocation (HipMallocManaged only)
    std::optional<size_t> testSizeBytes; // Override main-memory and AMD L3 bandwidth working-set size
    bool cpuInit;                      // --cpu-init: CPU-first initialization (USM allocators only)

    // Benchmark groups
    bool runL3;
    bool runL2;
    bool runL1;
    bool runScalar;
    bool runConstant;
    bool runReadOnly;
    bool runTexture;
    bool runSharedMemory;
    bool runMainMemory;
    bool runDepartureDelay;
    bool runResourceSharing;

    hipFuncCache_t cachePreference; // Cache config preference
};
