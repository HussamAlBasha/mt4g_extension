#!/usr/bin/env python3
"""Analyze repeated randomized CPX device sweeps without pooling node identity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RUN_PATTERN = re.compile(r"^cpx_(.+)$")


@dataclass(frozen=True)
class Metric:
    name: str
    path: tuple[str, ...]
    unit: str


METRICS = (
    Metric("Vector L1 latency", ("memory", "l1", "latency", "mean"), "cycles"),
    Metric("Vector L1 read bandwidth", ("memory", "l1", "readBandwidthPerCU", "measuredBandwidth"), "GiB/s per CU"),
    Metric("Vector L1 write bandwidth", ("memory", "l1", "writeBandwidthPerCU", "measuredBandwidth"), "GiB/s per CU"),
    Metric("L2 latency", ("memory", "l2", "latency", "mean"), "cycles"),
    Metric("L2 read bandwidth", ("memory", "l2", "readBandwidth", "measuredBandwidth"), "GiB/s"),
    Metric("L2 write bandwidth", ("memory", "l2", "writeBandwidth", "measuredBandwidth"), "GiB/s"),
    Metric("L3 latency", ("memory", "l3", "latency", "mean"), "cycles"),
    Metric("L3 read bandwidth", ("memory", "l3", "readBandwidth", "measuredBandwidth"), "GiB/s"),
    Metric("L3 write bandwidth", ("memory", "l3", "writeBandwidth", "measuredBandwidth"), "GiB/s"),
    Metric("LDS latency", ("memory", "shared", "latency", "mean"), "cycles"),
    Metric("LDS read bandwidth", ("memory", "shared", "readBandwidthPerCU", "measuredBandwidth"), "GiB/s per CU"),
    Metric("LDS write bandwidth", ("memory", "shared", "writeBandwidthPerCU", "measuredBandwidth"), "GiB/s per CU"),
    Metric("Main-memory latency", ("memory", "main", "latency", "mean"), "cycles"),
    Metric("Main-memory read bandwidth", ("memory", "main", "readBandwidth", "measuredBandwidth"), "GiB/s"),
    Metric("Main-memory write bandwidth", ("memory", "main", "writeBandwidth", "measuredBandwidth"), "GiB/s"),
)


@dataclass
class Observation:
    node: str
    repeat: int
    position: int
    device: int
    package: int
    package_device: int
    result_path: Path
    result: dict[str, Any]


def run_rank(path: Path) -> tuple[int, int]:
    suffix = path.name.rsplit("_", 1)[-1]
    return (int(suffix) if suffix.isdigit() else -1, path.stat().st_mtime_ns)


def latest_run(runs_dir: Path) -> Path | None:
    candidates = [
        path for path in runs_dir.iterdir()
        if path.is_dir() and RUN_PATTERN.match(path.name)
    ] if runs_dir.is_dir() else []
    return max(candidates, key=run_rank) if candidates else None


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def dig(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def load_observations(run: Path) -> tuple[list[Observation], list[str]]:
    observations: list[Observation] = []
    problems: list[str] = []
    nodes_dir = run / "nodes"
    node_dirs = sorted(path for path in nodes_dir.iterdir() if path.is_dir()) \
        if nodes_dir.is_dir() else []

    for node_dir in node_dirs:
        orders_path = node_dir / "orders.tsv"
        if not orders_path.exists():
            problems.append(f"{node_dir}: missing orders.tsv")
            continue
        try:
            orders = read_tsv(orders_path)
        except (OSError, csv.Error) as exc:
            problems.append(f"{orders_path}: {exc}")
            continue

        for row in orders:
            try:
                repeat = int(row["repeat"])
                position = int(row["position"])
                device = int(row["device"])
                package = int(row["package"])
                package_device = int(row["package_device"])
            except (KeyError, ValueError) as exc:
                problems.append(f"{orders_path}: malformed row: {exc}")
                continue

            base = node_dir / f"repeat_{repeat:02d}" / f"device_{device:02d}"
            # New runs write directly into device_XX. The fallback keeps the
            # analysis usable for results produced by the original full/
            # layout before the campaign was simplified.
            result_path = base / "result.json"
            if not result_path.exists():
                result_path = base / "full" / "result.json"
            if not result_path.exists():
                problems.append(f"{result_path}: missing")
                continue
            try:
                result = json.loads(result_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                problems.append(f"{result_path}: {exc}")
                continue
            observations.append(
                Observation(
                    node=node_dir.name,
                    repeat=repeat,
                    position=position,
                    device=device,
                    package=package,
                    package_device=package_device,
                    result_path=result_path,
                    result=result,
                )
            )
    return observations, problems


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def sample_sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else math.nan


def ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    output = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for position in range(index, end):
            output[ordered[position][0]] = average_rank
        index = end
    return output


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return math.nan
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_ss = sum((x - left_mean) ** 2 for x in left)
    right_ss = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_ss * right_ss)
    return numerator / denominator if denominator else math.nan


def rank_stability(sweeps: list[dict[int, float]], devices: list[int]) -> float:
    rank_vectors = [ranks([sweep[device] for device in devices]) for sweep in sweeps]
    correlations = [
        pearson(rank_vectors[left], rank_vectors[right])
        for left in range(len(rank_vectors))
        for right in range(left + 1, len(rank_vectors))
    ]
    finite = [value for value in correlations if math.isfinite(value)]
    return statistics.mean(finite) if finite else math.nan


def slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return math.nan
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if not denominator:
        return math.nan
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator


def effect_statistic(sweeps: list[dict[int, float]], devices: list[int]) -> float:
    means = [sum(sweep[device] for sweep in sweeps) / len(sweeps) for device in devices]
    return sum(value * value for value in means)


def permutation_p_value(
    sweeps: list[dict[int, float]],
    devices: list[int],
    permutations: int,
    seed_material: str,
) -> float:
    observed = effect_statistic(sweeps, devices)
    seed = int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8], "big")
    rng = random.Random(seed)
    exceedances = 0
    for _ in range(permutations):
        permuted: list[dict[int, float]] = []
        for sweep in sweeps:
            values = [sweep[device] for device in devices]
            rng.shuffle(values)
            permuted.append(dict(zip(devices, values)))
        if effect_statistic(permuted, devices) >= observed - 1e-15:
            exceedances += 1
    return (exceedances + 1) / (permutations + 1)


def holm_adjust(p_values: list[float]) -> list[float]:
    adjusted = [math.nan] * len(p_values)
    ordered = sorted(range(len(p_values)), key=lambda index: p_values[index])
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(ordered):
        candidate = min(1.0, (count - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        + ["| " + " | ".join(row) + " |" for row in rows]
    )


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if math.isfinite(value) else "—"


def analyze(
    run: Path,
    observations: list[Observation],
    permutations: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
]:
    problems: list[str] = []
    observation_rows: list[dict[str, Any]] = []
    for observation in observations:
        for metric in METRICS:
            value = dig(observation.result, metric.path)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                problems.append(
                    f"{observation.result_path}: missing {metric.name} ({'.'.join(metric.path)})"
                )
                continue
            observation_rows.append(
                {
                    "node": observation.node,
                    "package": observation.package,
                    "package_device": observation.package_device,
                    "logical_device": observation.device,
                    "repeat": observation.repeat,
                    "position": observation.position,
                    "metric": metric.name,
                    "unit": metric.unit,
                    "value": float(value),
                    "normalized_percent": math.nan,
                    "result": str(observation.result_path),
                }
            )

    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in observation_rows:
        grouped[(row["node"], row["package"], row["metric"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    per_device_rows: list[dict[str, Any]] = []
    for (node, package, metric_name), rows in sorted(grouped.items()):
        metric = next(item for item in METRICS if item.name == metric_name)
        by_repeat: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_repeat[row["repeat"]].append(row)

        normalized_sweeps: list[dict[int, float]] = []
        normalized_rows: list[tuple[dict[str, Any], float]] = []
        for repeat, repeat_rows in sorted(by_repeat.items()):
            if len(repeat_rows) != 6 or len({row["logical_device"] for row in repeat_rows}) != 6:
                problems.append(
                    f"{node}/package {package}/{metric_name}: repeat {repeat} is incomplete"
                )
                continue
            center = median([row["value"] for row in repeat_rows])
            if center == 0:
                problems.append(
                    f"{node}/package {package}/{metric_name}: repeat {repeat} has zero median"
                )
                continue
            normalized: dict[int, float] = {}
            for row in repeat_rows:
                relative = (row["value"] / center - 1.0) * 100.0
                row["normalized_percent"] = relative
                normalized[row["logical_device"]] = relative
                normalized_rows.append((row, relative))
            normalized_sweeps.append(normalized)

        devices = sorted({row["logical_device"] for row in rows})
        if len(devices) != 6 or len(normalized_sweeps) < 2:
            problems.append(f"{node}/package {package}/{metric_name}: insufficient complete sweeps")
            continue

        if any(sum(row["logical_device"] == d for row in rows) != len(normalized_sweeps) for d in devices):
            problems.append(f"{node}/{package}/{metric_name}: unequal repetition counts")
            continue
        device_means: dict[int, float] = {}
        device_noise: list[float] = []
        for device in devices:
            raw_values = [row["value"] for row in rows if row["logical_device"] == device]
            relative_values = [
                relative for row, relative in normalized_rows
                if row["logical_device"] == device
            ]
            device_means[device] = statistics.mean(relative_values)
            device_noise.append(sample_sd(relative_values))
            per_device_rows.append(
                {
                    "node": node,
                    "package": package,
                    "package_device": device % 6,
                    "logical_device": device,
                    "metric": metric_name,
                    "unit": metric.unit,
                    "n": len(raw_values),
                    "raw_mean": statistics.mean(raw_values),
                    "raw_sd": sample_sd(raw_values),
                    "normalized_mean_percent": statistics.mean(relative_values),
                    "normalized_sd_percent": sample_sd(relative_values),
                    "raw_min": min(raw_values),
                    "raw_max": max(raw_values),
                }
            )

        order_x = [float(row["position"]) for row, _ in normalized_rows]
        order_y = [relative for _, relative in normalized_rows]
        raw_p = permutation_p_value(
            normalized_sweeps,
            devices,
            permutations,
            f"{run.name}\0{node}\0{package}\0{metric_name}",
        )
        summary_rows.append(
            {
                "node": node,
                "package": package,
                "metric": metric_name,
                "unit": metric.unit,
                "complete_sweeps": len(normalized_sweeps),
                "observations": len(normalized_rows),
                "raw_group_mean": statistics.mean(row["value"] for row in rows),
                "raw_device_mean_min": min(statistics.mean(row["value"] for row in rows if row["logical_device"] == d) for d in devices),
                "raw_device_mean_max": max(statistics.mean(row["value"] for row in rows if row["logical_device"] == d) for d in devices),
                "device_mean_range_percent": max(device_means.values()) - min(device_means.values()),
                "mean_within_device_sd_percent": statistics.mean(
                    value for value in device_noise if math.isfinite(value)
                ),
                "mean_spearman_rank_correlation": rank_stability(normalized_sweeps, devices),
                "order_slope_percent_per_position": slope(order_x, order_y),
                "permutation_p": raw_p,
                "holm_p": math.nan,
            }
        )

    adjustment_groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(summary_rows):
        adjustment_groups[(row["node"], row["package"])].append(index)
    for indexes in adjustment_groups.values():
        adjusted = holm_adjust([summary_rows[index]["permutation_p"] for index in indexes])
        for index, value in zip(indexes, adjusted):
            summary_rows[index]["holm_p"] = value

    for row, adjusted in zip(summary_rows, holm_adjust([r["permutation_p"] for r in summary_rows])):
        row["holm_p_all_90"] = adjusted
    return observation_rows, per_device_rows, summary_rows, problems


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(run: Path, summary_rows: list[dict[str, Any]], permutations: int) -> str:
    rows = [
        [
            row["node"],
            str(row["package"]),
            row["metric"],
            str(row["complete_sweeps"]),
            fmt(row["device_mean_range_percent"], 2),
            fmt(row["mean_within_device_sd_percent"], 2),
            fmt(row["mean_spearman_rank_correlation"], 3),
            fmt(row["order_slope_percent_per_position"], 3),
            fmt(row["permutation_p"], 4),
            fmt(row["holm_p"], 4),
        ]
        for row in summary_rows
    ]
    significant = sum(1 for row in summary_rows if row["holm_p"] < 0.05)
    return "\n".join(
        [
            "# Randomized CPX device-sweep analysis",
            "",
            f"Run: `{run.name}`",
            "",
            "Each value is normalized to the median of its MI300A package in the same sweep. "
            "The permutation test shuffles device labels within those sweep/package blocks. "
            f"P-values use {permutations:,} deterministic permutations and are Holm-adjusted "
            "within each node/package across the reported core metrics.",
            "",
            "Device range is the maximum minus minimum of six device means of "
            "100*(value/package-sweep median-1), in percentage points. Within-device SD "
            "is the mean of six sample SDs over repeated normalized observations. Rank rho "
            "is mean pairwise Spearman correlation over all 66 sweep pairs, with average "
            "ranks for ties and undefined constant-vector pairs omitted. Order slope is "
            "OLS normalized percentage points per execution position (0..11). The test "
            "statistic is the sum of squared normalized device means; its common grand "
            "mean is invariant under permutation. p=(exceedances+1)/(permutations+1). "
            "Holm p controls the 15-metric family within one node/package, not all 90 "
            "screens; holm_p_all_90 is an additional campaign-wide sensitivity check.",
            "",
            "Independent within-sweep shuffles assume exchangeable device errors under "
            "the null; they are not an exact re-randomization of the constrained Williams "
            "design. Balanced positions and predecessors guard against confounding; "
            "order slopes remain descriptive diagnostics, not proof of absent drift.",
            "",
            "Vector-L1 write remains in the screening family for transparency but is "
            "excluded from hardware conclusions because the executed search contains "
            "out-of-allocation writes.",
            "",
            "Logical ordinals are interpreted only within their recorded node and package. "
            "A significant screening result is not, by itself, a physical-route explanation.",
            "",
            f"Holm-adjusted screening results below 0.05: {significant} of {len(summary_rows)}.",
            "",
            markdown_table(
                [
                    "Node", "Package", "Metric", "Sweeps", "Device range (%)",
                    "Within-device SD (%)", "Rank rho", "Order slope", "p", "Holm p",
                ],
                rows,
            ),
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path(__file__).resolve().parent / "runs",
    )
    parser.add_argument("--run", type=Path, help="specific cpx_<job-id> directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "analysis" / "details",
    )
    parser.add_argument("--permutations", type=int, default=9999)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    if args.permutations < 99:
        parser.error("--permutations must be at least 99")

    run = args.run or latest_run(args.runs)
    if run is None or not run.is_dir():
        print("No CPX sweep run found", file=sys.stderr)
        return 1

    if args.strict:
        from validate_results import validate_node
        for node in sorted((run / "nodes").iterdir()):
            if node.is_dir():
                errors, *_ = validate_node(node, 12)
                if errors:
                    print("\n".join(errors), file=sys.stderr)
                    return 1
    observations, problems = load_observations(run)
    observation_rows, per_device_rows, summary_rows, analysis_problems = analyze(
        run, observations, args.permutations
    )
    problems.extend(analysis_problems)

    if args.strict and problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "observations.csv", observation_rows)
    write_csv(args.output_dir / "per_device.csv", per_device_rows)
    write_csv(args.output_dir / "device_effects.csv", summary_rows)
    markdown = report(run, summary_rows, args.permutations)
    (args.output_dir / "comparison.md").write_text(markdown)
    print(markdown)
    print(f"Analysis written to {args.output_dir}")

    if problems:
        print("Problems encountered:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
    return 1 if args.strict and problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
