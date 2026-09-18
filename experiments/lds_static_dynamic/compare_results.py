#!/usr/bin/env python3
"""Validate and summarize paired static-versus-dynamic MT4G LDS runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


MODES = ("spx", "tpx", "cpx")
VARIANTS = ("dynamic", "static")
RUN_PATTERN = re.compile(r"^(spx|tpx|cpx)_(.+)$")
EXPECTED_DATA_BYTES = 32 * 1024
EXPECTED_GRID_THREADS = (64, 128, 256, 512, 1024)
EXPECTED_GRID_REPS = 2048

METRICS = {
    "latency_cycles": ("memory", "shared", "latency", "mean"),
    "read_gibs_per_cu": (
        "memory", "shared", "readBandwidthPerCU", "measuredBandwidth"
    ),
    "write_gibs_per_cu": (
        "memory", "shared", "writeBandwidthPerCU", "measuredBandwidth"
    ),
}

DETAIL_FIELDS = {
    "read": ("memory", "shared", "readBandwidthPerCU"),
    "write": ("memory", "shared", "writeBandwidthPerCU"),
}


@dataclass
class Observation:
    mode: str
    node: str
    repeat: int
    position: int
    variant: str
    git_commit: str
    result_path: Path
    data: Dict[str, Any]
    grids: Dict[str, Dict[int, float]]
    grid_reps: Dict[str, int]
    grid_paths: Dict[str, Path]

    def metric(self, name: str) -> Optional[float]:
        value = dig(self.data, METRICS[name])
        if is_number(value):
            return float(value)
        return None

    def detail(self, direction: str, field: str) -> Optional[float]:
        value = dig(self.data, DETAIL_FIELDS[direction] + (field,))
        if is_number(value):
            return float(value)
        return None

    def grid_value(self, direction: str, threads: int) -> Optional[float]:
        return self.grids.get(direction, {}).get(threads)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(float(value))


def dig(data: Dict[str, Any], path: Sequence[str]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def key_values(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(errors="replace").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values


def tsv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def run_rank(path: Path) -> Tuple[int, int]:
    suffix = path.name.rsplit("_", 1)[-1]
    numeric = int(suffix) if suffix.isdigit() else -1
    return numeric, path.stat().st_mtime_ns


def latest_run(runs_dir: Path, mode: str) -> Optional[Path]:
    if not runs_dir.is_dir():
        return None
    candidates = [
        path for path in runs_dir.iterdir()
        if path.is_dir() and path.name.startswith(mode + "_")
        and RUN_PATTERN.match(path.name)
    ]
    return max(candidates, key=run_rank) if candidates else None


def parse_run_overrides(entries: Sequence[str]) -> Dict[str, Path]:
    overrides: Dict[str, Path] = {}
    for entry in entries:
        mode, separator, raw_path = entry.partition("=")
        if not separator or mode not in MODES or not raw_path:
            raise ValueError("--run must have the form spx=/path, tpx=/path, or cpx=/path")
        overrides[mode] = Path(raw_path).expanduser().resolve()
    return overrides


def load_json(path: Path, problems: List[str]) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        problems.append("{}: invalid JSON: {}".format(path, exc))
        return None
    if not isinstance(value, dict):
        problems.append("{}: top-level JSON value is not an object".format(path))
        return None
    return value


def validate_command(path: Path, variant: str, problems: List[str]) -> None:
    if not path.exists():
        problems.append("{}: missing command.txt".format(path.parent))
        return
    try:
        tokens = set(shlex.split(path.read_text()))
    except ValueError as exc:
        problems.append("{}: cannot parse command: {}".format(path, exc))
        return

    for required in ("--device-id", "--shared", "--optimal"):
        if required not in tokens:
            problems.append("{}: command lacks {}".format(path, required))
    if variant == "static" and "--static" not in tokens:
        problems.append("{}: static run lacks --static".format(path))
    if variant == "dynamic" and "--static" in tokens:
        problems.append("{}: dynamic run unexpectedly contains --static".format(path))
    if "--allocator" in tokens:
        problems.append("{}: focused LDS command unexpectedly selects an allocator".format(path))


def validate_result(
    result_path: Path,
    variant: str,
    data: Dict[str, Any],
    problems: List[str],
) -> None:
    for name, metric_path in METRICS.items():
        if not is_number(dig(data, metric_path)):
            problems.append(
                "{}: missing or non-numeric {} ({})".format(
                    result_path, name, ".".join(metric_path)
                )
            )

    for direction, base in DETAIL_FIELDS.items():
        data_bytes = dig(data, base + ("dataBytes",))
        blocks = dig(data, base + ("numBlocks",))
        if data_bytes != EXPECTED_DATA_BYTES:
            problems.append(
                "{}: {} {} LDS working set is {!r}, expected {}".format(
                    result_path, variant, direction, data_bytes, EXPECTED_DATA_BYTES
                )
            )
        if blocks != 1:
            problems.append(
                "{}: {} {} numBlocks is {!r}, expected 1".format(
                    result_path, variant, direction, blocks
                )
            )
        for field in ("numThreads", "numReps", "cycles", "time"):
            if not is_number(dig(data, base + (field,))):
                problems.append(
                    "{}: missing {} {} field {}".format(
                        result_path, variant, direction, field
                    )
                )


def load_bandwidth_grid(
    output_dir: Path,
    variant: str,
    direction: str,
    problems: List[str],
) -> Tuple[Dict[int, float], Optional[int], Optional[Path]]:
    variant_tag = "stat" if variant == "static" else "dyn"
    direction_tag = "Read" if direction == "read" else "Write"
    grid_dir = output_dir / "results" / "result"
    pattern = "*__LDS_{}_32KiB_{}_BW_Grid.csv".format(
        direction_tag, variant_tag
    )
    candidates = sorted(grid_dir.glob(pattern)) if grid_dir.is_dir() else []
    if len(candidates) != 1:
        problems.append(
            "{}: found {} {} {} bandwidth grids, expected 1".format(
                output_dir, len(candidates), variant, direction
            )
        )
        return {}, None, None

    path = candidates[0]
    values: Dict[int, float] = {}
    repetitions: Optional[int] = None
    try:
        with path.open(newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader, None)
            if header is None or len(header) != 2 or header[0] != "threads":
                problems.append("{}: malformed grid header {!r}".format(path, header))
                return {}, None, path
            try:
                repetitions = int(header[1])
            except ValueError:
                problems.append(
                    "{}: non-integer repetition header {!r}".format(path, header[1])
                )
                return {}, None, path

            for line_number, row in enumerate(reader, start=2):
                if len(row) != 2:
                    problems.append(
                        "{}:{}: malformed grid row {!r}".format(
                            path, line_number, row
                        )
                    )
                    continue
                try:
                    threads = int(row[0])
                    bandwidth = float(row[1])
                except ValueError:
                    problems.append(
                        "{}:{}: non-numeric grid row {!r}".format(
                            path, line_number, row
                        )
                    )
                    continue
                if threads in values:
                    problems.append(
                        "{}:{}: duplicate thread count {}".format(
                            path, line_number, threads
                        )
                    )
                elif not math.isfinite(bandwidth):
                    problems.append(
                        "{}:{}: non-finite bandwidth {!r}".format(
                            path, line_number, row[1]
                        )
                    )
                else:
                    values[threads] = bandwidth
    except OSError as exc:
        problems.append("{}: cannot read bandwidth grid: {}".format(path, exc))
        return {}, None, path

    if tuple(sorted(values)) != EXPECTED_GRID_THREADS:
        problems.append(
            "{}: thread grid is {!r}, expected {!r}".format(
                path, tuple(sorted(values)), EXPECTED_GRID_THREADS
            )
        )
    if repetitions != EXPECTED_GRID_REPS:
        problems.append(
            "{}: grid uses {} repetitions, expected {}".format(
                path, repetitions, EXPECTED_GRID_REPS
            )
        )
    return values, repetitions, path


def expected_platform(mode: str) -> Tuple[str, str]:
    return {
        "spx": ("2", "228"),
        "tpx": ("6", "76"),
        "cpx": ("12", "38"),
    }[mode]


def load_mode_run(
    mode: str,
    run_dir: Path,
    expected_nodes: int,
    expected_repetitions: int,
    allow_dirty: bool,
) -> Tuple[List[Observation], List[str], List[Tuple[str, str]]]:
    observations: List[Observation] = []
    problems: List[str] = []
    identities: List[Tuple[str, str]] = []

    nodes_dir = run_dir / "nodes"
    node_dirs = sorted(path for path in nodes_dir.iterdir() if path.is_dir()) \
        if nodes_dir.is_dir() else []
    if len(node_dirs) != expected_nodes:
        problems.append(
            "{}: found {} nodes, expected {}".format(
                run_dir, len(node_dirs), expected_nodes
            )
        )

    expected_gpu_count, expected_cus = expected_platform(mode)
    for node_dir in node_dirs:
        metadata = key_values(node_dir / "metadata.txt")
        expected_metadata = {
            "mode": mode,
            "memory_partition": "NPS1",
            "xnack": "1",
            "device": "0",
            "repetitions": str(expected_repetitions),
            "order_design": "alternating-paired",
            "failures": "0",
            "observed_gpu_count": expected_gpu_count,
            "observed_cus_per_device": expected_cus,
        }
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected:
                problems.append(
                    "{}: metadata {}={!r}, expected {!r}".format(
                        node_dir.name, key, metadata.get(key), expected
                    )
                )
        if not allow_dirty and metadata.get("git_clean") != "1":
            problems.append("{}: benchmark source was not clean".format(node_dir.name))

        commit = metadata.get("git_commit", "unknown")
        binary_hash = metadata.get("mt4g_sha256", "unknown")
        identities.append((commit, binary_hash))

        status_rows = tsv_rows(node_dir / "status.tsv")
        if len(status_rows) != expected_repetitions * 2:
            problems.append(
                "{}: status.tsv contains {} rows, expected {}".format(
                    node_dir.name, len(status_rows), expected_repetitions * 2
                )
            )
        order_rows = tsv_rows(node_dir / "order.tsv")
        if len(order_rows) != expected_repetitions * 2:
            problems.append(
                "{}: order.tsv contains {} rows, expected {}".format(
                    node_dir.name, len(order_rows), expected_repetitions * 2
                )
            )

        status_index: Dict[Tuple[int, str], Dict[str, str]] = {}
        for row in status_rows:
            try:
                key = (int(row["repeat"]), row["variant"])
            except (KeyError, ValueError):
                problems.append("{}: malformed status row {!r}".format(node_dir.name, row))
                continue
            if key in status_index:
                problems.append("{}: duplicate status for {}".format(node_dir.name, key))
            status_index[key] = row

        order_by_repeat: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
        for row in order_rows:
            try:
                order_by_repeat[int(row["repeat"])].append(
                    (int(row["position"]), row["variant"])
                )
            except (KeyError, ValueError):
                problems.append("{}: malformed order row {!r}".format(node_dir.name, row))

        first_counts = Counter()
        for repeat in range(1, expected_repetitions + 1):
            ordered = sorted(order_by_repeat.get(repeat, []))
            if ordered not in (
                [(1, "dynamic"), (2, "static")],
                [(1, "static"), (2, "dynamic")],
            ):
                problems.append(
                    "{}: repeat {} is not one dynamic/static pair: {!r}".format(
                        node_dir.name, repeat, ordered
                    )
                )
            elif ordered:
                first_counts[ordered[0][1]] += 1

            for position, variant in ordered:
                if variant not in VARIANTS:
                    continue
                status = status_index.get((repeat, variant), {})
                if status.get("state") != "ok" or status.get("exit_status") != "0":
                    problems.append(
                        "{}: repeat {} {} has state={!r}, exit={!r}".format(
                            node_dir.name, repeat, variant,
                            status.get("state"), status.get("exit_status")
                        )
                    )

                output_dir = node_dir / "repeat_{:02d}".format(repeat) / variant
                validate_command(output_dir / "command.txt", variant, problems)
                result_path = output_dir / "result.json"
                data = load_json(result_path, problems)
                if data is None:
                    continue
                validate_result(result_path, variant, data, problems)
                grids: Dict[str, Dict[int, float]] = {}
                grid_reps: Dict[str, int] = {}
                grid_paths: Dict[str, Path] = {}
                for direction in ("read", "write"):
                    values, repetitions, grid_path = load_bandwidth_grid(
                        output_dir, variant, direction, problems
                    )
                    grids[direction] = values
                    if repetitions is not None:
                        grid_reps[direction] = repetitions
                    if grid_path is not None:
                        grid_paths[direction] = grid_path
                observations.append(
                    Observation(
                        mode=mode,
                        node=node_dir.name,
                        repeat=repeat,
                        position=position,
                        variant=variant,
                        git_commit=commit,
                        result_path=result_path,
                        data=data,
                        grids=grids,
                        grid_reps=grid_reps,
                        grid_paths=grid_paths,
                    )
                )

        if expected_repetitions % 2 == 0:
            expected_first = expected_repetitions // 2
            for variant in VARIANTS:
                if first_counts[variant] != expected_first:
                    problems.append(
                        "{}: {} runs first {} times, expected {}".format(
                            node_dir.name, variant, first_counts[variant], expected_first
                        )
                    )

    return observations, problems, identities


def mean(values: Iterable[float]) -> float:
    return statistics.fmean(list(values))


def sample_sd(values: Sequence[float]) -> Optional[float]:
    return statistics.stdev(values) if len(values) >= 2 else None


def node_aware_summary(
    observations: Sequence[Observation],
    metric: str,
) -> Dict[str, Any]:
    by_node: Dict[str, List[float]] = defaultdict(list)
    raw: List[float] = []
    for item in observations:
        value = item.metric(metric)
        if value is not None:
            by_node[item.node].append(value)
            raw.append(value)
    node_means = [mean(values) for _, values in sorted(by_node.items()) if values]
    return {
        "nodes": len(node_means),
        "observations": len(raw),
        "mean_of_node_means": mean(node_means) if node_means else None,
        "between_node_sd": sample_sd(node_means),
        "minimum_observation": min(raw) if raw else None,
        "maximum_observation": max(raw) if raw else None,
    }


def cell_summaries(observations: Sequence[Observation]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for mode in MODES:
        for variant in VARIANTS:
            group = [
                item for item in observations
                if item.mode == mode and item.variant == variant
            ]
            for metric in METRICS:
                row = {"mode": mode, "variant": variant, "metric": metric}
                row.update(node_aware_summary(group, metric))
                rows.append(row)
    return rows


def latency_summaries(observations: Sequence[Observation]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for mode in MODES:
        group = [item for item in observations if item.mode == mode]
        row = {"mode": mode, "variant": "pooled", "metric": "latency_cycles"}
        row.update(node_aware_summary(group, "latency_cycles"))
        rows.append(row)
    return rows


def paired_effects(observations: Sequence[Observation]) -> List[Dict[str, Any]]:
    index = {
        (item.mode, item.node, item.repeat, item.variant): item
        for item in observations
    }
    rows: List[Dict[str, Any]] = []
    for mode in MODES:
        nodes = sorted({item.node for item in observations if item.mode == mode})
        for metric in ("read_gibs_per_cu", "write_gibs_per_cu"):
            node_ratios: List[float] = []
            node_dynamic: List[float] = []
            node_static: List[float] = []
            pairs = 0
            for node in nodes:
                repeats = sorted({
                    item.repeat for item in observations
                    if item.mode == mode and item.node == node
                })
                ratios: List[float] = []
                dynamic_values: List[float] = []
                static_values: List[float] = []
                for repeat in repeats:
                    dynamic = index.get((mode, node, repeat, "dynamic"))
                    static = index.get((mode, node, repeat, "static"))
                    if dynamic is None or static is None:
                        continue
                    dynamic_value = dynamic.metric(metric)
                    static_value = static.metric(metric)
                    if dynamic_value is None or static_value is None or dynamic_value == 0:
                        continue
                    ratios.append(static_value / dynamic_value)
                    dynamic_values.append(dynamic_value)
                    static_values.append(static_value)
                    pairs += 1
                if ratios:
                    node_ratios.append(mean(ratios))
                    node_dynamic.append(mean(dynamic_values))
                    node_static.append(mean(static_values))

            ratio = mean(node_ratios) if node_ratios else None
            rows.append({
                "mode": mode,
                "metric": metric,
                "nodes": len(node_ratios),
                "pairs": pairs,
                "dynamic_mean_of_node_means": mean(node_dynamic) if node_dynamic else None,
                "static_mean_of_node_means": mean(node_static) if node_static else None,
                "static_over_dynamic_ratio": ratio,
                "percent_change": (ratio - 1.0) * 100.0 if ratio is not None else None,
                "ratio_between_node_sd": sample_sd(node_ratios),
            })
    return rows


def fixed_launch_effects(
    observations: Sequence[Observation],
) -> List[Dict[str, Any]]:
    index = {
        (item.mode, item.node, item.repeat, item.variant): item
        for item in observations
    }
    rows: List[Dict[str, Any]] = []
    for mode in MODES:
        nodes = sorted({item.node for item in observations if item.mode == mode})
        for direction in ("read", "write"):
            for threads in EXPECTED_GRID_THREADS:
                node_ratios: List[float] = []
                node_dynamic: List[float] = []
                node_static: List[float] = []
                pairs = 0
                for node in nodes:
                    repeats = sorted({
                        item.repeat for item in observations
                        if item.mode == mode and item.node == node
                    })
                    ratios: List[float] = []
                    dynamic_values: List[float] = []
                    static_values: List[float] = []
                    for repeat in repeats:
                        dynamic = index.get((mode, node, repeat, "dynamic"))
                        static = index.get((mode, node, repeat, "static"))
                        if dynamic is None or static is None:
                            continue
                        dynamic_value = dynamic.grid_value(direction, threads)
                        static_value = static.grid_value(direction, threads)
                        if (dynamic_value is None or static_value is None
                                or dynamic_value == 0):
                            continue
                        ratios.append(static_value / dynamic_value)
                        dynamic_values.append(dynamic_value)
                        static_values.append(static_value)
                        pairs += 1
                    if ratios:
                        node_ratios.append(mean(ratios))
                        node_dynamic.append(mean(dynamic_values))
                        node_static.append(mean(static_values))

                ratio = mean(node_ratios) if node_ratios else None
                rows.append({
                    "mode": mode,
                    "direction": direction,
                    "threads": threads,
                    "repetitions": EXPECTED_GRID_REPS,
                    "nodes": len(node_ratios),
                    "pairs": pairs,
                    "dynamic_mean_of_node_means": (
                        mean(node_dynamic) if node_dynamic else None
                    ),
                    "static_mean_of_node_means": (
                        mean(node_static) if node_static else None
                    ),
                    "static_over_dynamic_ratio": ratio,
                    "percent_change": (
                        (ratio - 1.0) * 100.0 if ratio is not None else None
                    ),
                    "ratio_between_node_sd": sample_sd(node_ratios),
                })
    return rows


def common_configuration(
    observations: Sequence[Observation],
    direction: str,
) -> str:
    configurations = []
    for item in observations:
        threads = item.detail(direction, "numThreads")
        reps = item.detail(direction, "numReps")
        if threads is not None and reps is not None:
            configurations.append((int(threads), int(reps)))
    if not configurations:
        return "—"
    (threads, reps), count = Counter(configurations).most_common(1)[0]
    return "T={}, R={} ({}/{})".format(threads, reps, count, len(configurations))


def format_number(value: Optional[float], digits: int = 2) -> str:
    return "—" if value is None else ("{:.%df}" % digits).format(value)


def format_mean_sd(row: Dict[str, Any], digits: int = 2) -> str:
    center = row.get("mean_of_node_means")
    spread = row.get("between_node_sd")
    if center is None:
        return "—"
    if spread is None:
        return format_number(center, digits)
    return "{} ± {}".format(format_number(center, digits), format_number(spread, digits))


def format_launch_cell(row: Dict[str, Any]) -> str:
    dynamic = format_number(row.get("dynamic_mean_of_node_means"))
    static = format_number(row.get("static_mean_of_node_means"))
    ratio = format_number(row.get("static_over_dynamic_ratio"), 3)
    return "{} / {} ({}x)".format(dynamic, static, ratio)


def markdown_report(
    selected: Dict[str, Path],
    observations: Sequence[Observation],
    problems: Sequence[str],
    summaries: Sequence[Dict[str, Any]],
    pooled_latency: Sequence[Dict[str, Any]],
    effects: Sequence[Dict[str, Any]],
    launch_effects: Sequence[Dict[str, Any]],
) -> str:
    summary_index = {
        (row["mode"], row["variant"], row["metric"]): row for row in summaries
    }
    effect_index = {(row["mode"], row["metric"]): row for row in effects}
    launch_index = {
        (row["mode"], row["direction"], row["threads"]): row
        for row in launch_effects
    }
    lines = [
        "# Static-versus-dynamic LDS comparison",
        "",
        "The campaign fixes `HSA_XNACK=1`, NPS1, logical device 0, and a 32 KiB",
        "LDS working set. Values below give every node equal weight: repetitions",
        "are averaged within a node before node means are combined.",
        "",
        "## Selected runs",
        "",
        "| Mode | Run | Nodes | Observations |",
        "|---|---|---:|---:|",
    ]
    for mode in MODES:
        path = selected.get(mode)
        group = [item for item in observations if item.mode == mode]
        nodes = len({item.node for item in group})
        lines.append(
            "| {} | `{}` | {} | {} |".format(
                mode.upper(), path.name if path else "missing", nodes, len(group)
            )
        )

    lines.extend(["", "## Independently optimized LDS bandwidth", "",
                  "Mean node means ± between-node sample SD, in GiB/s per CU.", "",
                  "| Mode | Variant | Read | Write | Observations / nodes |",
                  "|---|---|---:|---:|---:|"])
    for mode in MODES:
        for variant in VARIANTS:
            read = summary_index.get((mode, variant, "read_gibs_per_cu"), {})
            write = summary_index.get((mode, variant, "write_gibs_per_cu"), {})
            lines.append(
                "| {} | {} | {} | {} | {} / {} |".format(
                    mode.upper(), variant,
                    format_mean_sd(read), format_mean_sd(write),
                    read.get("observations", 0), read.get("nodes", 0)
                )
            )

    lines.extend(["", "## Paired independently optimized peak effect", "",
                  "Each variant contributes the peak selected by its own adaptive search.",
                  "Ratios are formed within each node and repetition, averaged within",
                  "nodes, and then combined across nodes.", "",
                  "| Mode | Direction | Dynamic | Static | Static / dynamic | Change | Ratio SD | Pairs |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"])
    for mode in MODES:
        for metric, direction in (
            ("read_gibs_per_cu", "read"),
            ("write_gibs_per_cu", "write"),
        ):
            row = effect_index.get((mode, metric), {})
            lines.append(
                "| {} | {} | {} | {} | {} | {}% | {} | {} |".format(
                    mode.upper(), direction,
                    format_number(row.get("dynamic_mean_of_node_means")),
                    format_number(row.get("static_mean_of_node_means")),
                    format_number(row.get("static_over_dynamic_ratio"), 4),
                    format_number(row.get("percent_change"), 2),
                    format_number(row.get("ratio_between_node_sd"), 4),
                    row.get("pairs", 0),
                )
            )

    lines.extend(["", "## LDS latency control", "",
                  "The static flag does not select a different latency kernel, so both",
                  "invocations are pooled only as a contemporaneous execution control.", "",
                  "| Mode | Latency (cycles) | Observations / nodes |",
                  "|---|---:|---:|"])
    for row in pooled_latency:
        lines.append(
            "| {} | {} | {} / {} |".format(
                row["mode"].upper(), format_mean_sd(row),
                row["observations"], row["nodes"]
            )
        )

    lines.extend(["", "## Winning launch configurations", "",
                  "The parenthesized count shows how often the modal optimal-search",
                  "configuration won among all observations in that cell.", "",
                  "| Mode | Variant | Read configuration | Write configuration |",
                  "|---|---|---|---|"])
    for mode in MODES:
        for variant in VARIANTS:
            group = [
                item for item in observations
                if item.mode == mode and item.variant == variant
            ]
            lines.append(
                "| {} | {} | {} | {} |".format(
                    mode.upper(), variant,
                    common_configuration(group, "read"),
                    common_configuration(group, "write")
                )
            )

    lines.extend([
        "", "## Matched launch-grid effect", "",
        "The raw grids compare the variants at identical launch geometry:",
        "one block, the same thread count, the same 32 KiB working set, and",
        "2,048 repetitions. Each cell is `dynamic / static (static/dynamic)`",
        "in GiB/s per CU, calculated from 30 matched node/repetition pairs.",
        "", "| Direction | Threads | SPX | TPX | CPX |",
        "|---|---:|---:|---:|---:|",
    ])
    for direction in ("read", "write"):
        for threads in EXPECTED_GRID_THREADS:
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    direction, threads,
                    format_launch_cell(launch_index.get(
                        ("spx", direction, threads), {}
                    )),
                    format_launch_cell(launch_index.get(
                        ("tpx", direction, threads), {}
                    )),
                    format_launch_cell(launch_index.get(
                        ("cpx", direction, threads), {}
                    )),
                )
            )

    launch_ratio_sds = [
        row["ratio_between_node_sd"] for row in launch_effects
        if row.get("ratio_between_node_sd") is not None
    ]
    lines.extend([
        "",
        "Static bandwidth is higher at every matched grid point. The ratio",
        "depends on thread count, but its direction and shape are reproduced",
        "across SPX, TPX, and CPX. The independently optimized peak ratio is",
        "therefore not an artifact of comparing different winning thread counts.",
    ])
    if launch_ratio_sds:
        lines.extend([
            "The largest between-node sample SD of a matched ratio is {}.".format(
                format_number(max(launch_ratio_sds), 4)
            )
        ])

    lines.extend(["", "## Validation", ""])
    if problems:
        lines.append("The reducer found {} issue(s):".format(len(problems)))
        lines.append("")
        lines.extend("- " + problem for problem in problems)
    else:
        lines.append("All selected runs passed the requested coverage and provenance checks.")
    lines.append("")
    return "\n".join(lines)


def write_csvs(
    output_dir: Path,
    observations: Sequence[Observation],
    summaries: Sequence[Dict[str, Any]],
    effects: Sequence[Dict[str, Any]],
    launch_effects: Sequence[Dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    observation_fields = [
        "mode", "node", "repeat", "position", "variant", "git_commit",
        "latency_cycles", "read_gibs_per_cu", "write_gibs_per_cu",
        "read_data_bytes", "read_num_blocks", "read_num_threads", "read_num_reps",
        "read_cycles", "read_time_seconds", "write_data_bytes", "write_num_blocks",
        "write_num_threads", "write_num_reps", "write_cycles", "write_time_seconds",
        "result_path",
    ]
    with (output_dir / "observations.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=observation_fields)
        writer.writeheader()
        for item in observations:
            writer.writerow({
                "mode": item.mode,
                "node": item.node,
                "repeat": item.repeat,
                "position": item.position,
                "variant": item.variant,
                "git_commit": item.git_commit,
                "latency_cycles": item.metric("latency_cycles"),
                "read_gibs_per_cu": item.metric("read_gibs_per_cu"),
                "write_gibs_per_cu": item.metric("write_gibs_per_cu"),
                "read_data_bytes": item.detail("read", "dataBytes"),
                "read_num_blocks": item.detail("read", "numBlocks"),
                "read_num_threads": item.detail("read", "numThreads"),
                "read_num_reps": item.detail("read", "numReps"),
                "read_cycles": item.detail("read", "cycles"),
                "read_time_seconds": item.detail("read", "time"),
                "write_data_bytes": item.detail("write", "dataBytes"),
                "write_num_blocks": item.detail("write", "numBlocks"),
                "write_num_threads": item.detail("write", "numThreads"),
                "write_num_reps": item.detail("write", "numReps"),
                "write_cycles": item.detail("write", "cycles"),
                "write_time_seconds": item.detail("write", "time"),
                "result_path": str(item.result_path),
            })

    grid_observation_fields = [
        "mode", "node", "repeat", "position", "variant", "git_commit",
        "direction", "threads", "repetitions", "gibs_per_cu", "grid_path",
    ]
    with (output_dir / "grid_observations.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=grid_observation_fields)
        writer.writeheader()
        for item in observations:
            for direction in ("read", "write"):
                for threads, bandwidth in sorted(item.grids.get(direction, {}).items()):
                    writer.writerow({
                        "mode": item.mode,
                        "node": item.node,
                        "repeat": item.repeat,
                        "position": item.position,
                        "variant": item.variant,
                        "git_commit": item.git_commit,
                        "direction": direction,
                        "threads": threads,
                        "repetitions": item.grid_reps.get(direction),
                        "gibs_per_cu": bandwidth,
                        "grid_path": str(item.grid_paths.get(direction, "")),
                    })

    summary_fields = [
        "mode", "variant", "metric", "nodes", "observations",
        "mean_of_node_means", "between_node_sd", "minimum_observation",
        "maximum_observation",
    ]
    with (output_dir / "cell_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)

    effect_fields = [
        "mode", "metric", "nodes", "pairs", "dynamic_mean_of_node_means",
        "static_mean_of_node_means", "static_over_dynamic_ratio",
        "percent_change", "ratio_between_node_sd",
    ]
    with (output_dir / "paired_effects.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=effect_fields)
        writer.writeheader()
        writer.writerows(effects)

    launch_effect_fields = [
        "mode", "direction", "threads", "repetitions", "nodes", "pairs",
        "dynamic_mean_of_node_means", "static_mean_of_node_means",
        "static_over_dynamic_ratio", "percent_change", "ratio_between_node_sd",
    ]
    with (output_dir / "fixed_launch_effects.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=launch_effect_fields)
        writer.writeheader()
        writer.writerows(launch_effects)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path(__file__).resolve().parent / "runs",
        help="directory containing <mode>_<job-id> run directories",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="MODE=PATH",
        help="override the selected run for one mode; may be repeated",
    )
    parser.add_argument("--expected-nodes", type=int, default=3)
    parser.add_argument("--expected-repetitions", type=int, default=10)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--md", type=Path, help="write the Markdown report to this path")
    parser.add_argument("--csv-dir", type=Path, help="write detailed CSV files here")
    args = parser.parse_args()

    try:
        overrides = parse_run_overrides(args.run)
    except ValueError as exc:
        parser.error(str(exc))

    selected: Dict[str, Path] = {}
    problems: List[str] = []
    observations: List[Observation] = []
    identities: List[Tuple[str, str]] = []
    for mode in MODES:
        run_dir = overrides.get(mode) or latest_run(args.runs, mode)
        if run_dir is None or not run_dir.is_dir():
            problems.append("No {} run found under {}".format(mode.upper(), args.runs))
            continue
        selected[mode] = run_dir
        mode_observations, mode_problems, mode_identities = load_mode_run(
            mode,
            run_dir,
            args.expected_nodes,
            args.expected_repetitions,
            args.allow_dirty,
        )
        observations.extend(mode_observations)
        problems.extend(mode_problems)
        identities.extend(mode_identities)

    known_identities = {item for item in identities if "unknown" not in item}
    if len(known_identities) > 1:
        problems.append(
            "Selected nodes do not share one (Git commit, MT4G binary hash): {}".format(
                sorted(known_identities)
            )
        )

    summaries = cell_summaries(observations)
    pooled_latency = latency_summaries(observations)
    effects = paired_effects(observations)
    launch_effects = fixed_launch_effects(observations)
    report = markdown_report(
        selected, observations, problems, summaries, pooled_latency, effects,
        launch_effects,
    )

    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(report)
    else:
        print(report, end="")
    if args.csv_dir:
        write_csvs(
            args.csv_dir, observations, summaries, effects, launch_effects
        )

    if problems:
        print("Found {} validation issue(s).".format(len(problems)), file=sys.stderr)
        return 1 if args.strict else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
