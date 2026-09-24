#!/usr/bin/env python3
"""Generate the descriptive RQ3 summaries reported in the thesis."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


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


@dataclass(frozen=True)
class Run:
    node: str
    package: int
    package_device: int
    device: int
    repeat: int
    position: int
    result_path: Path
    result: dict


def read_rows(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dig(data: dict, path: tuple[str, ...]):
    current = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def load_runs(run_dir: Path) -> list[Run]:
    runs = []
    for node_dir in sorted(path for path in (run_dir / "nodes").iterdir() if path.is_dir()):
        for row in read_rows(node_dir / "orders.tsv", "\t"):
            repeat = int(row["repeat"])
            device = int(row["device"])
            result_path = node_dir / f"repeat_{repeat:02d}" / f"device_{device:02d}" / "result.json"
            if not result_path.exists():
                result_path = result_path.parent / "full" / "result.json"
            if not result_path.is_file():
                raise ValueError(f"missing {result_path}")
            runs.append(Run(
                node=node_dir.name,
                package=int(row["package"]),
                package_device=int(row["package_device"]),
                device=device,
                repeat=repeat,
                position=int(row["position"]),
                result_path=result_path,
                result=json.loads(result_path.read_text()),
            ))
    if len(runs) != 432:
        raise ValueError(f"expected 432 runs, found {len(runs)}")
    return runs


def reduce_runs(runs: list[Run]):
    observations = []
    for run in runs:
        for metric in METRICS:
            value = dig(run.result, metric.path)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"missing {metric.name}: {run.result_path}")
            observations.append(dict(
                node=run.node,
                package=run.package,
                package_device=run.package_device,
                logical_device=run.device,
                repeat=run.repeat,
                position=run.position,
                metric=metric.name,
                unit=metric.unit,
                value=float(value),
                normalized_percent=math.nan,
                result=str(run.result_path),
            ))

    sweeps = defaultdict(list)
    for row in observations:
        sweeps[(row["node"], row["package"], row["metric"], row["repeat"])].append(row)
    for key, rows in sweeps.items():
        if len(rows) != 6 or len({row["logical_device"] for row in rows}) != 6:
            raise ValueError(f"incomplete package sweep: {key}")
        center = st.median(row["value"] for row in rows)
        for row in rows:
            row["normalized_percent"] = 100 * (row["value"] / center - 1)

    groups = defaultdict(list)
    for row in observations:
        groups[(row["node"], row["package"], row["metric"])].append(row)

    per_device = []
    package_summary = []
    for (node, package, metric_name), rows in sorted(groups.items()):
        metric = next(metric for metric in METRICS if metric.name == metric_name)
        devices = sorted({row["logical_device"] for row in rows})
        if len(rows) != 72 or len(devices) != 6:
            raise ValueError(f"expected 72 observations for {node}/{package}/{metric_name}")
        raw_means = []
        normalized_means = []
        normalized_sds = []
        for device in devices:
            selected = [row for row in rows if row["logical_device"] == device]
            raw = [row["value"] for row in selected]
            normalized = [row["normalized_percent"] for row in selected]
            if len(raw) != 12:
                raise ValueError(f"expected 12 observations for {node}/{package}/{device}/{metric_name}")
            raw_mean = st.mean(raw)
            normalized_mean = st.mean(normalized)
            raw_means.append(raw_mean)
            normalized_means.append(normalized_mean)
            normalized_sds.append(st.stdev(normalized))
            per_device.append(dict(
                node=node,
                package=package,
                package_device=device % 6,
                logical_device=device,
                metric=metric_name,
                unit=metric.unit,
                n=len(raw),
                raw_mean=raw_mean,
                raw_sd=st.stdev(raw),
                normalized_mean_percent=normalized_mean,
                normalized_sd_percent=st.stdev(normalized),
                raw_min=min(raw),
                raw_max=max(raw),
            ))
        package_summary.append(dict(
            node=node,
            package=package,
            metric=metric_name,
            unit=metric.unit,
            sweeps=12,
            observations=72,
            package_mean=st.mean(row["value"] for row in rows),
            device_mean_min=min(raw_means),
            device_mean_max=max(raw_means),
            device_mean_range=max(raw_means) - min(raw_means),
            device_mean_range_percent=max(normalized_means) - min(normalized_means),
            mean_within_device_sd_percent=st.mean(normalized_sds),
        ))
    return observations, per_device, package_summary


def read_grid(result_path: Path, metric: Metric) -> dict[tuple[int, int, int], float]:
    prefix = {"l3": "L3", "main": "MainMemory"}[metric.path[1]]
    direction = "Read" if "read" in metric.name else "Write"
    files = list((result_path.parent / "results").rglob(f"*__{prefix}_{direction}_BW_Grid.csv"))
    if len(files) != 1:
        raise ValueError(f"expected one grid for {result_path}/{metric.name}")
    grid = {}
    for row in read_rows(files[0]):
        config = int(row["blocks"]), int(row["threads"]), int(row["reps"])
        value = float(row["bandwidth"])
        if config in grid or not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid grid row in {files[0]}")
        grid[config] = value
    return grid


def fixed_launches(observations: list[dict]):
    metrics = {metric.name: metric for metric in METRICS if metric.path[1] in ("l3", "main") and "bandwidth" in metric.name}
    groups = defaultdict(list)
    for row in observations:
        if row["metric"] in metrics:
            groups[(row["node"], row["package"], row["metric"])].append(row)
    all_rows = []
    selected_rows = []
    for (node, package, metric_name), records in sorted(groups.items()):
        records.sort(key=lambda row: (int(row["repeat"]), int(row["logical_device"])))
        if len(records) != 72:
            raise ValueError(f"expected 72 grids for {node}/{package}/{metric_name}")
        metric = metrics[metric_name]
        grids = [read_grid(Path(row["result"]), metric) for row in records]
        common = set.intersection(*(set(grid) for grid in grids))
        if not common:
            raise ValueError(f"no common launch for {node}/{package}/{metric_name}")
        winners = Counter()
        for record in records:
            peak = dig(json.loads(Path(record["result"]).read_text()), metric.path[:-1])
            winners[tuple(int(peak[key]) for key in ("numBlocks", "numThreads", "numReps"))] += 1
        modal = min(common, key=lambda config: (-winners[config], config))
        for config in sorted(common):
            values = [grid[config] for grid in grids]
            matrix = [values[index:index + 6] for index in range(0, 72, 6)]
            raw_means = [st.mean(column) for column in zip(*matrix)]
            normalized = [[100 * (value / st.median(sweep) - 1) for value in sweep] for sweep in matrix]
            effects = [st.mean(column) for column in zip(*normalized)]
            row = dict(
                node=node,
                package=package,
                metric=metric_name,
                blocks=config[0],
                threads=config[1],
                reps=config[2],
                modal=config == modal,
                raw_device_min=min(raw_means),
                raw_device_max=max(raw_means),
                device_range_percent=max(effects) - min(effects),
                common_configurations=len(common),
            )
            all_rows.append(row)
            if config == modal:
                selected_rows.append(row)
    if len(selected_rows) != 24:
        raise ValueError(f"expected 24 fixed-launch summaries, found {len(selected_rows)}")
    return all_rows, selected_rows


def metric_summaries(package_rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in package_rows:
        grouped[row["metric"]].append(row)
    result = []
    for metric in METRICS:
        rows = grouped[metric.name]
        if len(rows) != 6:
            raise ValueError(f"expected six packages for {metric.name}")
        result.append(dict(
            metric=metric.name,
            unit=metric.unit,
            package_mean_min=min(row["package_mean"] for row in rows),
            package_mean_max=max(row["package_mean"] for row in rows),
            device_range_min_percent=min(row["device_mean_range_percent"] for row in rows),
            device_range_max_percent=max(row["device_mean_range_percent"] for row in rows),
            device_range_min=min(row["device_mean_range"] for row in rows),
            device_range_max=max(row["device_mean_range"] for row in rows),
        ))
    return result


def node_averages(observations: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in observations:
        if row["metric"].startswith(("L3", "Main-memory")):
            grouped[(row["node"], int(row["package"]), row["metric"])].append(float(row["value"]))
    output = []
    nodes = sorted({key[0] for key in grouped})
    metric_names = sorted({key[2] for key in grouped})
    for node in nodes:
        for metric_name in metric_names:
            output.append(dict(
                node=node,
                metric=metric_name,
                unit="cycles" if "latency" in metric_name else "GiB/s",
                mean=st.mean(st.mean(grouped[(node, package, metric_name)]) for package in (0, 1)),
            ))
    return output


def span(row: dict, left: str, right: str, digits: int) -> str:
    return f"{float(row[left]):.{digits}f}--{float(row[right]):.{digits}f}"


def markdown(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        + ["| " + " | ".join(row) + " |" for row in rows]
    )


def summary_text(run_dir: Path, metrics: list[dict], fixed: list[dict], nodes: list[dict]) -> str:
    primary = [row for row in metrics if row["metric"].startswith(("L3", "Main-memory"))]
    bandwidth = [
        [row["metric"].replace(" bandwidth", ""), span(row, "package_mean_min", "package_mean_max", 1), span(row, "device_range_min", "device_range_max", 2)]
        for row in primary if "bandwidth" in row["metric"]
    ]
    node_names = sorted({row["node"] for row in nodes})
    bandwidth_names = ["L3 read bandwidth", "L3 write bandwidth", "Main-memory read bandwidth", "Main-memory write bandwidth"]
    node_table = []
    for node in node_names:
        lookup = {row["metric"]: row["mean"] for row in nodes if row["node"] == node}
        node_table.append([node] + [f"{lookup[name]:.2f}" for name in bandwidth_names])
    fixed_table = []
    for row in fixed:
        if (row["node"], str(row["package"])) in (("vipa1020", "0"), ("vipa1099", "1")) and row["metric"].startswith("L3"):
            fixed_table.append([
                f"{row['node']}/{row['package']}",
                row["metric"].split()[1],
                f"{row['blocks']} × {row['threads']}",
                span(row, "raw_device_min", "raw_device_max", 1),
            ])
    text = [
        "# CPX device sweep: L3 and main-memory performance",
        "",
        "**Conclusion:** Main-memory peak bandwidth is nearly equal across devices. Larger, repeatable L3 bandwidth differences occur in two of the six packages, but no device number is consistently preferable across hosts.",
        "",
        "## What was measured?",
        "",
        f"Run `{run_dir.name}` contains three nodes, two packages per node, six CPX devices per package, and twelve sequential sweeps: **3 × 12 × 12 = 432 runs**. CPX, NPS1, `hipMalloc`, and XNACK=1 remain fixed.",
        "",
        "## Average behavior across nodes",
        "",
        markdown(["Node", "L3 read", "L3 write", "Main read", "Main write"], node_table),
        "",
        "GiB/s. Each entry averages per-device peaks and gives both packages equal weight; it is not concurrent node throughput.",
        "",
        "## Peak bandwidth across packages",
        "",
        markdown(["Metric", "Package averages (GiB/s)", "Device difference (GiB/s)"], bandwidth),
        "",
        "Ranges cover six packages. Device difference is the highest minus the lowest twelve-run device average within a package.",
        "",
        "## L3 bandwidth at common launch settings",
        "",
        markdown(["Node/package", "Direction", "Blocks × threads", "Device averages (GiB/s)"], fixed_table),
        "",
        "All listed configurations use a 64 MiB working set and 2,048 repetitions. Holding launch settings fixed checks whether independent peak selection alone explains a device difference.",
        "",
        "## Generated files",
        "",
        "- `details/per_device.csv`: twelve-run device averages and variation.",
        "- `details/package_summary.csv`: within-package device ranges.",
        "- `details/metric_summary.csv`: ranges across the six packages.",
        "- `details/fixed_configurations.csv`: every launch configuration shared by all runs in a package.",
        "- `details/fixed_summary.csv`: the commonly winning shared launch used for the concise comparison.",
        "- `details/node_averages.csv`: package-balanced node averages.",
        "",
    ]
    return "\n".join(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "analysis")
    args = parser.parse_args()
    details = args.output_dir / "details"
    details.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.run)
    observations, per_device, packages = reduce_runs(runs)
    fixed_all, fixed_selected = fixed_launches(observations)
    metrics = metric_summaries(packages)
    nodes = node_averages(observations)

    write_rows(details / "observations.csv", observations)
    write_rows(details / "per_device.csv", per_device)
    write_rows(details / "package_summary.csv", packages)
    write_rows(details / "metric_summary.csv", metrics)
    write_rows(details / "fixed_configurations.csv", fixed_all)
    write_rows(details / "fixed_summary.csv", fixed_selected)
    write_rows(details / "node_averages.csv", nodes)
    (args.output_dir / "SUMMARY.md").write_text(summary_text(args.run, metrics, fixed_selected, nodes))
    print(f"Analyzed 432 runs; results: {args.output_dir / 'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
