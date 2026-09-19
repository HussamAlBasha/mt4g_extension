#!/usr/bin/env python3
"""Reduce the replicated MT4G allocator/partition/XNACK experiment.

The report keeps two kinds of evidence separate:

* L3 and main-memory measurements are allocator-aware and are summarized for
  every mode/XNACK/allocator cell.
* vector L1, L2, scalar L1, and LDS do not use ``--allocator``. Repeated
  allocator-labelled invocations are therefore pooled as partition controls,
  not interpreted as allocator comparisons.

All headline summaries give every node equal weight. Repetitions are first
averaged within a node; the displayed standard deviation is the sample
standard deviation of those node means.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


MODES = ("spx", "tpx", "cpx")
XNACK_SETTINGS = ("1", "0")
ALLOCATORS = ("hipmalloc", "hipmallocmanaged", "hiphostmalloc", "malloc")
RUN_PATTERN = re.compile(r"^(spx|tpx|cpx)_xnack([01])_(.+)$")
REPEAT_PATTERN = re.compile(r"^repeat_([0-9]+)$")
MISSING = "—"

LEVEL_LABELS = {
    "shared": "LDS",
    "l1": "Vector L1",
    "scalarL1": "Scalar L1",
    "l2": "L2",
    "l3": "L3",
    "main": "Main memory",
}


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    level: str
    kind: str
    statistic: str
    unit: str
    path: tuple[str, ...]
    allocator_aware: bool = False


def latency_metrics(level: str, allocator_aware: bool = False) -> tuple[Metric, ...]:
    prefix = LEVEL_LABELS[level]
    return tuple(
        Metric(
            f"{level}_latency_{statistic}",
            f"{prefix} {label}",
            level,
            "latency",
            statistic,
            "cycles",
            ("memory", level, "latency", statistic),
            allocator_aware,
        )
        for statistic, label in (
            ("mean", "mean"),
            ("p50", "p50"),
            ("p95", "p95"),
            ("stdev", "within-run SD"),
        )
    )


def bandwidth_metrics(
    level: str,
    read_key: str,
    write_key: str,
    allocator_aware: bool = False,
) -> tuple[Metric, ...]:
    prefix = LEVEL_LABELS[level]
    return (
        Metric(
            f"{level}_read_bandwidth",
            f"{prefix} read",
            level,
            "bandwidth",
            "peak",
            "GiB/s",
            ("memory", level, read_key, "measuredBandwidth"),
            allocator_aware,
        ),
        Metric(
            f"{level}_write_bandwidth",
            f"{prefix} write",
            level,
            "bandwidth",
            "peak",
            "GiB/s",
            ("memory", level, write_key, "measuredBandwidth"),
            allocator_aware,
        ),
    )


CONTROL_LATENCY_METRICS = tuple(
    metric
    for level in ("shared", "l1", "scalarL1", "l2")
    for metric in latency_metrics(level)
)
ALLOCATOR_LATENCY_METRICS = tuple(
    metric
    for level in ("l3", "main")
    for metric in latency_metrics(level, allocator_aware=True)
)
CONTROL_BANDWIDTH_METRICS = (
    *bandwidth_metrics("shared", "readBandwidthPerCU", "writeBandwidthPerCU"),
    *bandwidth_metrics("l1", "readBandwidthPerCU", "writeBandwidthPerCU"),
    *bandwidth_metrics("scalarL1", "readBandwidthPerCU", "writeBandwidthPerCU"),
    *bandwidth_metrics("l2", "readBandwidth", "writeBandwidth"),
)
ALLOCATOR_BANDWIDTH_METRICS = (
    *bandwidth_metrics("l3", "readBandwidth", "writeBandwidth", True),
    *bandwidth_metrics("main", "readBandwidth", "writeBandwidth", True),
)
ALL_METRICS = (
    *CONTROL_LATENCY_METRICS,
    *ALLOCATOR_LATENCY_METRICS,
    *CONTROL_BANDWIDTH_METRICS,
    *ALLOCATOR_BANDWIDTH_METRICS,
)
PRIMARY_CONTROL_LATENCY = tuple(
    metric for metric in CONTROL_LATENCY_METRICS if metric.statistic == "mean"
)
PRIMARY_ALLOCATOR_LATENCY = tuple(
    metric
    for metric in ALLOCATOR_LATENCY_METRICS
    if metric.statistic in ("mean", "p50", "p95")
)


@dataclass
class Observation:
    mode: str
    xnack: str
    node: str
    repeat: int
    allocator: str
    path: Path
    status: dict[str, str]
    result: dict[str, Any] | None
    command: str
    metadata: dict[str, str]


@dataclass(frozen=True)
class Summary:
    mean: float
    between_node_sd: float
    minimum: float
    maximum: float
    nodes: int
    observations: int


def dig(data: dict[str, Any] | None, *path: str) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def numeric(data: dict[str, Any] | None, path: tuple[str, ...]) -> float | None:
    value = dig(data, *path)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def table(headers: list[str], rows: list[list[str]], markdown: bool) -> str:
    if markdown:
        return "\n".join(
            [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join("---" for _ in headers) + " |",
            ]
            + ["| " + " | ".join(row) + " |" for row in rows]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    layout = " | ".join(f"{{:<{width}}}" for width in widths)
    separator = "-+-".join("-" * width for width in widths)
    return "\n".join(
        [layout.format(*headers), separator]
        + [layout.format(*row) for row in rows]
    )


def run_rank(path: Path) -> tuple[int, int]:
    suffix = path.name.rsplit("_", 1)[-1]
    job_number = int(suffix) if suffix.isdigit() else -1
    return job_number, path.stat().st_mtime_ns


def latest_job_dirs(runs_dir: Path) -> dict[tuple[str, str], Path]:
    selected: dict[tuple[str, str], Path] = {}
    if not runs_dir.exists():
        return selected
    for path in runs_dir.iterdir():
        if not path.is_dir():
            continue
        match = RUN_PATTERN.match(path.name)
        if not match:
            continue
        key = match.group(1), match.group(2)
        if key not in selected or run_rank(path) > run_rank(selected[key]):
            selected[key] = path
    return selected


def read_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(errors="replace").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values


def read_result(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def read_command(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(errors="replace").strip()


def load_observations(runs_dir: Path) -> tuple[
    list[Observation],
    dict[tuple[str, str], Path],
    dict[tuple[str, str], list[str]],
]:
    jobs = latest_job_dirs(runs_dir)
    observations: list[Observation] = []
    job_nodes: dict[tuple[str, str], list[str]] = {}

    for (mode, xnack), job_dir in jobs.items():
        nodes_dir = job_dir / "nodes"
        if nodes_dir.is_dir():
            node_dirs = sorted(path for path in nodes_dir.iterdir() if path.is_dir())
            job_nodes[(mode, xnack)] = [path.name for path in node_dirs]
            for node_dir in node_dirs:
                metadata = read_key_values(node_dir / "metadata.txt")
                repeat_dirs = sorted(
                    path
                    for path in node_dir.iterdir()
                    if path.is_dir() and REPEAT_PATTERN.match(path.name)
                )
                for repeat_dir in repeat_dirs:
                    match = REPEAT_PATTERN.match(repeat_dir.name)
                    assert match is not None
                    repeat = int(match.group(1))
                    for allocator in ALLOCATORS:
                        allocator_dir = repeat_dir / allocator
                        observations.append(
                            Observation(
                                mode,
                                xnack,
                                node_dir.name,
                                repeat,
                                allocator,
                                allocator_dir,
                                read_key_values(allocator_dir / "status.txt"),
                                read_result(allocator_dir / "result.json"),
                                read_command(allocator_dir / "command.txt"),
                                metadata,
                            )
                        )
            continue

        # Preserve support for the original one-node/one-run layout. Strict
        # mode will identify it as an incomplete replicated campaign.
        metadata = read_key_values(job_dir / "metadata.txt")
        node = metadata.get("hostname", "legacy")
        job_nodes[(mode, xnack)] = [node]
        for allocator in ALLOCATORS:
            allocator_dir = job_dir / allocator
            observations.append(
                Observation(
                    mode,
                    xnack,
                    node,
                    1,
                    allocator,
                    allocator_dir,
                    read_key_values(allocator_dir / "status.txt"),
                    read_result(allocator_dir / "result.json"),
                    read_command(allocator_dir / "command.txt"),
                    metadata,
                )
            )

    return observations, jobs, job_nodes


def matching(
    observations: Iterable[Observation],
    *,
    mode: str | None = None,
    xnack: str | None = None,
    allocator: str | None = None,
    node: str | None = None,
) -> list[Observation]:
    return [
        item
        for item in observations
        if (mode is None or item.mode == mode)
        and (xnack is None or item.xnack == xnack)
        and (allocator is None or item.allocator == allocator)
        and (node is None or item.node == node)
    ]


def successful(observations: Iterable[Observation]) -> list[Observation]:
    return [item for item in observations if item.result is not None]


def values_for(observations: Iterable[Observation], metric: Metric) -> list[float]:
    values = [numeric(item.result, metric.path) for item in successful(observations)]
    return [value for value in values if value is not None]


def node_aware_summary(
    observations: Iterable[Observation], metric: Metric
) -> Summary | None:
    items = list(observations)
    all_values = values_for(items, metric)
    node_means = []
    for node in sorted({item.node for item in items}):
        node_values = values_for(matching(items, node=node), metric)
        if node_values:
            node_means.append(statistics.mean(node_values))
    if not node_means:
        return None
    return Summary(
        mean=statistics.mean(node_means),
        between_node_sd=(statistics.stdev(node_means) if len(node_means) > 1 else 0.0),
        minimum=min(all_values),
        maximum=max(all_values),
        nodes=len(node_means),
        observations=len(all_values),
    )


def paired_ratio_summary(
    target: Iterable[Observation],
    reference: Iterable[Observation],
    metric: Metric,
) -> Summary | None:
    """Summarize target/reference ratios paired by node and repetition."""
    target_values = {
        (item.node, item.repeat): numeric(item.result, metric.path)
        for item in successful(target)
    }
    reference_values = {
        (item.node, item.repeat): numeric(item.result, metric.path)
        for item in successful(reference)
    }
    paired: dict[str, list[float]] = {}
    for key in sorted(target_values.keys() & reference_values.keys()):
        target_value = target_values[key]
        reference_value = reference_values[key]
        if target_value is None or reference_value in (None, 0):
            continue
        paired.setdefault(key[0], []).append(target_value / reference_value)
    ratios = [ratio for node_ratios in paired.values() for ratio in node_ratios]
    if not ratios:
        return None
    node_means = [statistics.mean(node_ratios) for node_ratios in paired.values()]
    return Summary(
        mean=statistics.mean(node_means),
        between_node_sd=(statistics.stdev(node_means) if len(node_means) > 1 else 0.0),
        minimum=min(ratios),
        maximum=max(ratios),
        nodes=len(node_means),
        observations=len(ratios),
    )


def format_summary(summary: Summary | None, digits: int = 1) -> str:
    if summary is None:
        return MISSING
    return (
        f"{summary.mean:.{digits}f} ± {summary.between_node_sd:.{digits}f} "
        f"[{summary.minimum:.{digits}f}, {summary.maximum:.{digits}f}]"
    )


def run_groups(command: str) -> tuple[str, ...]:
    if not command:
        return ()
    try:
        tokens = set(shlex.split(command))
    except ValueError:
        tokens = set(command.split())
    group_flags = {
        "--shared": "shared",
        "--l1": "l1",
        "--scalar": "scalarL1",
        "--l2": "l2",
        "--l3": "l3",
        "--memory": "main",
    }
    selected = tuple(level for flag, level in group_flags.items() if flag in tokens)
    if selected:
        return selected
    # With no group flag, MT4G enables every AMD-compatible group.
    return ("shared", "l1", "scalarL1", "l2", "l3", "main")


def command_profile(group: list[Observation]) -> str:
    commands = [item.command for item in group if item.command]
    if not commands:
        return MISSING
    groups = run_groups(commands[0])
    tokens = set(shlex.split(commands[0]))
    default = not any(
        flag in tokens
        for flag in ("--shared", "--l1", "--scalar", "--l2", "--l3", "--memory")
    )
    label = ", ".join(LEVEL_LABELS[level] for level in groups)
    return f"default AMD set ({label})" if default else f"explicit ({label})"


def rows_in_order() -> Iterable[tuple[str, str, str]]:
    for xnack in XNACK_SETTINGS:
        for mode in MODES:
            for allocator in ALLOCATORS:
                yield mode, xnack, allocator


def metric_by_key(key: str) -> Metric:
    return next(metric for metric in ALL_METRICS if metric.key == key)


def anomaly_candidates(group: list[Observation]) -> list[list[str]]:
    rows: list[list[str]] = []
    mean_latency_metrics = tuple(
        metric
        for metric in (*CONTROL_LATENCY_METRICS, *ALLOCATOR_LATENCY_METRICS)
        if metric.statistic == "mean"
    )
    for metric in mean_latency_metrics:
        samples = [
            (item, numeric(item.result, metric.path))
            for item in successful(group)
        ]
        present = [(item, value) for item, value in samples if value is not None]
        if len(present) < 3:
            continue
        values = [value for _, value in present]
        center = statistics.median(values)
        deviations = [abs(value - center) for value in values]
        mad = statistics.median(deviations)
        for (item, value), deviation in zip(present, deviations):
            if mad == 0:
                # Quantized cache latencies often yield a zero MAD with two
                # adjacent legitimate values. No robust score is defined in
                # that case, so do not manufacture an infinite outlier score.
                continue
            score = 0.6745 * deviation / mad
            anomalous = score > 3.5
            if anomalous:
                rows.append(
                    [
                        item.mode.upper(),
                        item.xnack,
                        item.allocator,
                        item.node,
                        metric.label,
                        str(item.repeat),
                        f"{value:.1f}",
                        f"{center:.1f}",
                        f"{score:.1f}",
                    ]
                )
    return rows


def report(
    observations: list[Observation],
    jobs: dict[tuple[str, str], Path],
    job_nodes: dict[tuple[str, str], list[str]],
    markdown: bool = False,
) -> str:
    heading = (
        "# MT4G allocator/partition/XNACK comparison"
        if markdown
        else "MT4G ALLOCATOR/PARTITION/XNACK COMPARISON"
    )
    parts = [heading, ""]

    run_rows = []
    for mode in MODES:
        for xnack in XNACK_SETTINGS:
            path = jobs.get((mode, xnack))
            group = matching(observations, mode=mode, xnack=xnack)
            nodes = job_nodes.get((mode, xnack), [])
            commits = sorted(
                {
                    item.metadata.get("git_commit", "unknown")[:7]
                    for item in group
                    if item.metadata
                }
            )
            run_rows.append(
                [
                    mode.upper(),
                    xnack,
                    path.name if path else MISSING,
                    str(len(nodes)),
                    ", ".join(nodes) if nodes else MISSING,
                    ", ".join(commits) if commits else MISSING,
                    command_profile(group),
                    "static" if any("--static" in item.command for item in group) else "dynamic",
                ]
            )
    parts += [
        "## Selected jobs and coverage" if markdown else "SELECTED JOBS AND COVERAGE",
        table(
            ["Mode", "XNACK", "Run", "Nodes", "Hostnames", "Commit", "Groups", "LDS"],
            run_rows,
            markdown,
        ),
        "",
    ]

    status_rows = []
    for mode, xnack, allocator in rows_in_order():
        group = matching(observations, mode=mode, xnack=xnack, allocator=allocator)
        states = [
            item.status.get("state", "ok" if item.result else "missing") for item in group
        ]
        status_rows.append(
            [
                mode.upper(),
                xnack,
                allocator,
                str(len({item.node for item in group})),
                str(len(group)),
                str(sum(state == "ok" for state in states)),
                str(sum(state == "expected-failure" for state in states)),
                str(sum(state not in ("ok", "expected-failure") for state in states)),
            ]
        )
    parts += [
        "## Run status" if markdown else "RUN STATUS",
        table(
            [
                "Mode",
                "XNACK",
                "Allocator",
                "Nodes",
                "Attempts",
                "OK",
                "Expected failures",
                "Other failures",
            ],
            status_rows,
            markdown,
        ),
        "",
    ]

    coverage_rows = []
    for mode in MODES:
        for xnack in XNACK_SETTINGS:
            group = successful(matching(observations, mode=mode, xnack=xnack))
            coverage_rows.append(
                [
                    mode.upper(),
                    xnack,
                    str(len(group)),
                    *[
                        str(sum(isinstance(dig(item.result, "memory", level), dict) for item in group))
                        for level in ("shared", "l1", "scalarL1", "l2", "l3", "main")
                    ],
                ]
            )
    parts += [
        "## Successful-result coverage" if markdown else "SUCCESSFUL-RESULT COVERAGE",
        "Counts are result files containing each group. CPX scalar L1 was not selected; it is not a failure.",
        table(
            ["Mode", "XNACK", "Results", "LDS", "Vector L1", "Scalar L1", "L2", "L3", "Main"],
            coverage_rows,
            markdown,
        ),
        "",
    ]

    summary_note = (
        "Cells are the equal-weight mean of node means ± between-node sample SD "
        "[minimum observation, maximum observation]."
    )
    parts += [
        "## Partition controls" if markdown else "PARTITION CONTROLS",
        "Allocator labels are pooled because these benchmark groups do not consume the allocator option. "
        + summary_note,
        "",
    ]

    control_latency_rows = []
    for mode in MODES:
        for xnack in XNACK_SETTINGS:
            group = matching(observations, mode=mode, xnack=xnack)
            summaries = [node_aware_summary(group, metric) for metric in PRIMARY_CONTROL_LATENCY]
            control_latency_rows.append(
                [
                    mode.upper(),
                    xnack,
                    str(max((summary.observations for summary in summaries if summary), default=0)),
                    *[format_summary(summary) for summary in summaries],
                ]
            )
    parts += [
        "### Latency (cycles)" if markdown else "CONTROL LATENCY (CYCLES)",
        table(
            ["Mode", "XNACK", "Max N", *[metric.label for metric in PRIMARY_CONTROL_LATENCY]],
            control_latency_rows,
            markdown,
        ),
        "",
    ]

    control_bandwidth_rows = []
    for mode in MODES:
        for xnack in XNACK_SETTINGS:
            group = matching(observations, mode=mode, xnack=xnack)
            summaries = [node_aware_summary(group, metric) for metric in CONTROL_BANDWIDTH_METRICS]
            control_bandwidth_rows.append(
                [
                    mode.upper(),
                    xnack,
                    str(max((summary.observations for summary in summaries if summary), default=0)),
                    *[format_summary(summary) for summary in summaries],
                ]
            )
    parts += [
        "### Peak bandwidth (GiB/s)" if markdown else "CONTROL PEAK BANDWIDTH (GIB/S)",
        "LDS, vector-L1, and scalar-L1 values are per CU; L2 is aggregate for the selected logical device.",
        table(
            ["Mode", "XNACK", "Max N", *[metric.label for metric in CONTROL_BANDWIDTH_METRICS]],
            control_bandwidth_rows,
            markdown,
        ),
        "",
    ]

    allocator_latency_rows = []
    for mode, xnack, allocator in rows_in_order():
        group = matching(observations, mode=mode, xnack=xnack, allocator=allocator)
        summaries = [node_aware_summary(group, metric) for metric in PRIMARY_ALLOCATOR_LATENCY]
        allocator_latency_rows.append(
            [
                mode.upper(),
                xnack,
                allocator,
                str(max((summary.observations for summary in summaries if summary), default=0)),
                *[format_summary(summary) for summary in summaries],
            ]
        )
    parts += [
        "## Allocator-aware L3 and main-memory latency (cycles)"
        if markdown
        else "ALLOCATOR-AWARE L3 AND MAIN-MEMORY LATENCY (CYCLES)",
        summary_note,
        table(
            ["Mode", "XNACK", "Allocator", "Max N", *[metric.label for metric in PRIMARY_ALLOCATOR_LATENCY]],
            allocator_latency_rows,
            markdown,
        ),
        "",
    ]

    allocator_bandwidth_rows = []
    for mode, xnack, allocator in rows_in_order():
        group = matching(observations, mode=mode, xnack=xnack, allocator=allocator)
        summaries = [node_aware_summary(group, metric) for metric in ALLOCATOR_BANDWIDTH_METRICS]
        allocator_bandwidth_rows.append(
            [
                mode.upper(),
                xnack,
                allocator,
                str(max((summary.observations for summary in summaries if summary), default=0)),
                *[format_summary(summary) for summary in summaries],
            ]
        )
    parts += [
        "## Allocator-aware peak bandwidth (GiB/s)"
        if markdown
        else "ALLOCATOR-AWARE PEAK BANDWIDTH (GIB/S)",
        summary_note,
        table(
            ["Mode", "XNACK", "Allocator", "Max N", *[metric.label for metric in ALLOCATOR_BANDWIDTH_METRICS]],
            allocator_bandwidth_rows,
            markdown,
        ),
        "",
    ]

    ratio_rows = []
    for mode, xnack, allocator in rows_in_order():
        if allocator == "hipmalloc":
            continue
        target = matching(observations, mode=mode, xnack=xnack, allocator=allocator)
        reference = matching(observations, mode=mode, xnack=xnack, allocator="hipmalloc")
        ratios = []
        for metric in (
            metric_by_key("l3_read_bandwidth"),
            metric_by_key("l3_write_bandwidth"),
            metric_by_key("main_read_bandwidth"),
            metric_by_key("main_write_bandwidth"),
        ):
            ratio = paired_ratio_summary(target, reference, metric)
            ratios.append(
                f"{ratio.mean:.3f} ± {ratio.between_node_sd:.3f}×"
                if ratio is not None
                else MISSING
            )
        ratio_rows.append([mode.upper(), xnack, allocator, *ratios])
    parts += [
        "## Bandwidth ratios relative to hipMalloc" if markdown else "BANDWIDTH RATIOS RELATIVE TO HIPMALLOC",
        "Each allocator ratio is paired by node and repetition, then summarized as the equal-weight mean of three node-level ratio means ± between-node sample SD.",
        table(
            ["Mode", "XNACK", "Allocator", "L3 read", "L3 write", "Main read", "Main write"],
            ratio_rows,
            markdown,
        ),
        "",
    ]

    anomaly_rows: list[list[str]] = []
    for mode, xnack, allocator in rows_in_order():
        group = matching(observations, mode=mode, xnack=xnack, allocator=allocator)
        for node in job_nodes.get((mode, xnack), []):
            anomaly_rows += anomaly_candidates(matching(group, node=node))
    parts += [
        "## Within-node latency anomaly candidates" if markdown else "WITHIN-NODE LATENCY ANOMALY CANDIDATES"
    ]
    if anomaly_rows:
        parts += [
            table(
                ["Mode", "XNACK", "Allocator", "Node", "Metric", "Repeat", "Value", "Median", "Modified z"],
                anomaly_rows,
                markdown,
            ),
            "",
        ]
    else:
        parts += ["None detected by the median-absolute-deviation rule.", ""]

    parts += [
        "## Interpretation rules" if markdown else "INTERPRETATION RULES",
        "* `malloc` with XNACK disabled is unsupported and excluded from numeric summaries."
        if markdown
        else "- malloc with XNACK disabled is unsupported and excluded from numeric summaries.",
        "* Scalar L1 is available for SPX and TPX only; no CPX scalar value is imputed."
        if markdown
        else "- Scalar L1 is available for SPX and TPX only; no CPX scalar value is imputed.",
        "* L3/main allocator comparisons are the intended allocator analysis. Lower-level repetitions increase precision but do not create an allocator effect."
        if markdown
        else "- L3/main allocator comparisons are the intended allocator analysis. Lower-level repetitions increase precision but do not create an allocator effect.",
        "* Reported bandwidth is the peak selected by MT4G's adaptive search, not an average of independent timed rounds."
        if markdown
        else "- Reported bandwidth is the peak selected by MT4G's adaptive search, not an average of independent timed rounds.",
        "",
    ]
    return "\n".join(parts)


def expected_groups(item: Observation) -> tuple[str, ...]:
    return run_groups(item.command)


def unexpected_problems(
    observations: list[Observation],
    jobs: dict[tuple[str, str], Path],
    job_nodes: dict[tuple[str, str], list[str]],
    expected_nodes: int,
    expected_repetitions: int,
) -> list[str]:
    problems: list[str] = []
    expected_repeat_set = set(range(1, expected_repetitions + 1))
    for mode in MODES:
        for xnack in XNACK_SETTINGS:
            if (mode, xnack) not in jobs:
                problems.append(f"missing job for {mode}/XNACK={xnack}")
                continue
            nodes = job_nodes.get((mode, xnack), [])
            if len(nodes) != expected_nodes:
                problems.append(
                    f"{mode}/XNACK={xnack}: expected {expected_nodes} nodes, found {len(nodes)}"
                )
            commits = {
                item.metadata.get("git_commit", "unknown")
                for item in matching(observations, mode=mode, xnack=xnack)
            }
            if len(commits) != 1:
                problems.append(
                    f"{mode}/XNACK={xnack}: inconsistent commits={sorted(commits)}"
                )
            for node in nodes:
                node_items = matching(observations, mode=mode, xnack=xnack, node=node)
                repeats = {item.repeat for item in node_items}
                if repeats != expected_repeat_set:
                    problems.append(
                        f"{mode}/XNACK={xnack}/{node}: repeats={sorted(repeats)}, "
                        f"expected={sorted(expected_repeat_set)}"
                    )
                for allocator in ALLOCATORS:
                    items = matching(node_items, allocator=allocator)
                    if len(items) != expected_repetitions:
                        problems.append(
                            f"{mode}/XNACK={xnack}/{node}/{allocator}: expected "
                            f"{expected_repetitions} attempts, found {len(items)}"
                        )
                    expected_failure = allocator == "malloc" and xnack == "0"
                    for item in items:
                        state = item.status.get("state", "missing-status")
                        if item.result is None:
                            if not (expected_failure and state == "expected-failure"):
                                problems.append(
                                    f"{mode}/XNACK={xnack}/{node}/repeat={item.repeat}/"
                                    f"{allocator}: {state}"
                                )
                            continue
                        if state != "ok":
                            problems.append(
                                f"{mode}/XNACK={xnack}/{node}/repeat={item.repeat}/"
                                f"{allocator}: result exists but state={state}"
                            )

                        declared_allocators = (
                            dig(item.result, "memory", "l3", "latency", "allocator"),
                            dig(item.result, "memory", "main", "latency", "allocator"),
                            dig(item.result, "memory", "l3", "bandwidthAllocator"),
                            dig(item.result, "memory", "main", "bandwidthAllocator"),
                        )
                        if any(value != allocator for value in declared_allocators):
                            problems.append(
                                f"{mode}/XNACK={xnack}/{node}/repeat={item.repeat}/"
                                f"{allocator}: allocator metadata mismatch"
                            )

                        groups = expected_groups(item)
                        for level in groups:
                            if not isinstance(dig(item.result, "memory", level), dict):
                                problems.append(
                                    f"{mode}/XNACK={xnack}/{node}/repeat={item.repeat}/"
                                    f"{allocator}: missing selected group {LEVEL_LABELS[level]}"
                                )
                        for metric in ALL_METRICS:
                            if metric.level not in groups:
                                continue
                            if numeric(item.result, metric.path) is None:
                                problems.append(
                                    f"{mode}/XNACK={xnack}/{node}/repeat={item.repeat}/"
                                    f"{allocator}: missing {metric.label} ({metric.statistic})"
                                )
    return problems


def bandwidth_configuration(result: dict[str, Any] | None, metric: Metric) -> dict[str, Any]:
    bandwidth_object = dig(result, *metric.path[:-1])
    if not isinstance(bandwidth_object, dict):
        return {}
    return {
        key: bandwidth_object.get(key)
        for key in ("numBlocks", "numThreads", "numReps", "dataBytes", "time", "cycles")
    }


def write_csv_outputs(output_dir: Path, observations: list[Observation]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    measurement_fields = [
        "mode", "xnack", "node", "repeat", "allocator", "git_commit",
        "metric", "label", "level", "kind", "statistic", "unit",
        "allocator_aware", "value", "num_blocks", "num_threads", "num_reps",
        "data_bytes", "time_seconds", "cycles", "result_path",
    ]
    with (output_dir / "measurements.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=measurement_fields)
        writer.writeheader()
        for item in successful(observations):
            for metric in ALL_METRICS:
                value = numeric(item.result, metric.path)
                if value is None:
                    continue
                configuration = bandwidth_configuration(item.result, metric)
                writer.writerow(
                    {
                        "mode": item.mode,
                        "xnack": item.xnack,
                        "node": item.node,
                        "repeat": item.repeat,
                        "allocator": item.allocator,
                        "git_commit": item.metadata.get("git_commit", ""),
                        "metric": metric.key,
                        "label": metric.label,
                        "level": metric.level,
                        "kind": metric.kind,
                        "statistic": metric.statistic,
                        "unit": metric.unit,
                        "allocator_aware": str(metric.allocator_aware).lower(),
                        "value": value,
                        "num_blocks": configuration.get("numBlocks", ""),
                        "num_threads": configuration.get("numThreads", ""),
                        "num_reps": configuration.get("numReps", ""),
                        "data_bytes": configuration.get("dataBytes", ""),
                        "time_seconds": configuration.get("time", ""),
                        "cycles": configuration.get("cycles", ""),
                        "result_path": item.path / "result.json",
                    }
                )

    summary_fields = [
        "mode", "xnack", "allocator", "metric", "label", "level", "kind",
        "statistic", "unit", "nodes", "observations", "mean_of_node_means",
        "between_node_sd", "minimum_observation", "maximum_observation",
    ]
    with (output_dir / "cell_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for mode, xnack, allocator in rows_in_order():
            group = matching(observations, mode=mode, xnack=xnack, allocator=allocator)
            for metric in ALL_METRICS:
                summary = node_aware_summary(group, metric)
                if summary is None:
                    continue
                writer.writerow(
                    {
                        "mode": mode,
                        "xnack": xnack,
                        "allocator": allocator,
                        "metric": metric.key,
                        "label": metric.label,
                        "level": metric.level,
                        "kind": metric.kind,
                        "statistic": metric.statistic,
                        "unit": metric.unit,
                        "nodes": summary.nodes,
                        "observations": summary.observations,
                        "mean_of_node_means": summary.mean,
                        "between_node_sd": summary.between_node_sd,
                        "minimum_observation": summary.minimum,
                        "maximum_observation": summary.maximum,
                    }
                )

    control_fields = [field for field in summary_fields if field != "allocator"]
    with (output_dir / "partition_controls.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=control_fields)
        writer.writeheader()
        for mode in MODES:
            for xnack in XNACK_SETTINGS:
                group = matching(observations, mode=mode, xnack=xnack)
                for metric in (*CONTROL_LATENCY_METRICS, *CONTROL_BANDWIDTH_METRICS):
                    summary = node_aware_summary(group, metric)
                    if summary is None:
                        continue
                    writer.writerow(
                        {
                            "mode": mode,
                            "xnack": xnack,
                            "metric": metric.key,
                            "label": metric.label,
                            "level": metric.level,
                            "kind": metric.kind,
                            "statistic": metric.statistic,
                            "unit": metric.unit,
                            "nodes": summary.nodes,
                            "observations": summary.observations,
                            "mean_of_node_means": summary.mean,
                            "between_node_sd": summary.between_node_sd,
                            "minimum_observation": summary.minimum,
                            "maximum_observation": summary.maximum,
                        }
                    )

    effect_fields = [
        "comparison", "mode", "xnack", "allocator", "metric", "target",
        "reference", "ratio", "percent_change", "ratio_between_node_sd",
        "paired_observations", "target_nodes", "reference_nodes",
    ]
    with (output_dir / "effects.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=effect_fields)
        writer.writeheader()

        def emit_effect(
            comparison: str,
            mode: str,
            xnack: str,
            allocator: str,
            metric: Metric,
            target: Summary | None,
            reference: Summary | None,
            paired_ratio: Summary | None = None,
        ) -> None:
            if target is None or reference is None or reference.mean == 0:
                return
            ratio = paired_ratio.mean if paired_ratio else target.mean / reference.mean
            writer.writerow(
                {
                    "comparison": comparison,
                    "mode": mode,
                    "xnack": xnack,
                    "allocator": allocator,
                    "metric": metric.key,
                    "target": target.mean,
                    "reference": reference.mean,
                    "ratio": ratio,
                    "percent_change": (ratio - 1.0) * 100.0,
                    "ratio_between_node_sd": (
                        paired_ratio.between_node_sd if paired_ratio else ""
                    ),
                    "paired_observations": (
                        paired_ratio.observations if paired_ratio else ""
                    ),
                    "target_nodes": target.nodes,
                    "reference_nodes": reference.nodes,
                }
            )

        allocator_metrics = (*PRIMARY_ALLOCATOR_LATENCY, *ALLOCATOR_BANDWIDTH_METRICS)
        for mode in MODES:
            for xnack in XNACK_SETTINGS:
                reference_group = matching(
                    observations, mode=mode, xnack=xnack, allocator="hipmalloc"
                )
                for allocator in ALLOCATORS[1:]:
                    target_group = matching(
                        observations, mode=mode, xnack=xnack, allocator=allocator
                    )
                    for metric in allocator_metrics:
                        emit_effect(
                            "allocator_vs_hipmalloc", mode, xnack, allocator, metric,
                            node_aware_summary(target_group, metric),
                            node_aware_summary(reference_group, metric),
                            paired_ratio_summary(target_group, reference_group, metric),
                        )

        for xnack in XNACK_SETTINGS:
            for allocator in ALLOCATORS:
                reference_group = matching(
                    observations, mode="cpx", xnack=xnack, allocator=allocator
                )
                for mode in ("tpx", "spx"):
                    target_group = matching(
                        observations, mode=mode, xnack=xnack, allocator=allocator
                    )
                    for metric in allocator_metrics:
                        emit_effect(
                            "partition_vs_cpx", mode, xnack, allocator, metric,
                            node_aware_summary(target_group, metric),
                            node_aware_summary(reference_group, metric),
                        )

        for mode in MODES:
            for allocator in ALLOCATORS[:-1]:
                target_group = matching(
                    observations, mode=mode, xnack="1", allocator=allocator
                )
                reference_group = matching(
                    observations, mode=mode, xnack="0", allocator=allocator
                )
                for metric in allocator_metrics:
                    emit_effect(
                        "xnack1_vs_xnack0", mode, "1/0", allocator, metric,
                        node_aware_summary(target_group, metric),
                        node_aware_summary(reference_group, metric),
                    )

        control_metrics = (*PRIMARY_CONTROL_LATENCY, *CONTROL_BANDWIDTH_METRICS)
        for xnack in XNACK_SETTINGS:
            reference_group = matching(observations, mode="cpx", xnack=xnack)
            for mode in ("tpx", "spx"):
                target_group = matching(observations, mode=mode, xnack=xnack)
                for metric in control_metrics:
                    emit_effect(
                        "control_partition_vs_cpx", mode, xnack, "pooled", metric,
                        node_aware_summary(target_group, metric),
                        node_aware_summary(reference_group, metric),
                    )

        for mode in MODES:
            target_group = matching(observations, mode=mode, xnack="1")
            reference_group = matching(observations, mode=mode, xnack="0")
            for metric in control_metrics:
                emit_effect(
                    "control_xnack1_vs_xnack0", mode, "1/0", "pooled", metric,
                    node_aware_summary(target_group, metric),
                    node_aware_summary(reference_group, metric),
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", type=Path,
        default=Path(__file__).resolve().parent / "runs",
    )
    parser.add_argument("--md", type=Path, help="also write a Markdown report")
    parser.add_argument(
        "--csv-dir", type=Path,
        help="write measurements, cell summaries, pooled controls, and effects as CSV",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="fail unless every command-selected metric in the replicated matrix is present",
    )
    parser.add_argument("--expected-nodes", type=int, default=3)
    parser.add_argument("--expected-repetitions", type=int, default=5)
    args = parser.parse_args()

    if args.expected_nodes < 1 or args.expected_repetitions < 1:
        parser.error("expected nodes and repetitions must be positive")

    try:
        observations, jobs, job_nodes = load_observations(args.runs)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error reading results: {exc}", file=sys.stderr)
        return 1

    if not jobs:
        print(f"no matrix runs found under {args.runs}", file=sys.stderr)
        return 1

    print(report(observations, jobs, job_nodes))
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(report(observations, jobs, job_nodes, markdown=True) + "\n")
        print(f"Markdown report written to {args.md}")
    if args.csv_dir:
        write_csv_outputs(args.csv_dir, observations)
        print(f"CSV tables written to {args.csv_dir}")

    problems = unexpected_problems(
        observations, jobs, job_nodes,
        args.expected_nodes, args.expected_repetitions,
    )
    if problems:
        print("Incomplete or inconsistent replicated matrix:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
    return 1 if args.strict and problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
