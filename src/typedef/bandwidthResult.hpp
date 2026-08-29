#pragma once

#include <vector>
#include <nlohmann/json.hpp>

// Return type for bandwidth launchers that expose per-round timing.
// average: working-set GiB divided by the arithmetic mean of all timed-round
//          durations; it is not necessarily the arithmetic mean of rounds.
// rounds:  GiB/s for each timed round in execution order. rounds[0] is the first
//          measured access and is cold only when no preceding preparation established
//          the relevant mappings or access state.
struct BandwidthResult {
    double average;
    std::vector<double> rounds;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(BandwidthResult, average, rounds)
};
