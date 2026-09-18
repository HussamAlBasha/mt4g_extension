#!/usr/bin/env python3
"""Strictly validate a completed randomized CPX sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


RUN_PATTERN = re.compile(r"^cpx_(.+)$")
EXPECTED_DEVICES = set(range(12))
EXPECTED_POSITIONS = set(range(12))
EXPECTED_CUS_PER_DEVICE = 38
EXPECTED_SIMDS_PER_CU = 4
EXPECTED_SIMDS_PER_DEVICE = EXPECTED_CUS_PER_DEVICE * EXPECTED_SIMDS_PER_CU

REQUIRED_FULL_METRICS = (
    ("vector L1 latency", ("memory", "l1", "latency", "mean")),
    ("vector L1 read bandwidth", ("memory", "l1", "readBandwidthPerCU", "measuredBandwidth")),
    ("vector L1 write bandwidth", ("memory", "l1", "writeBandwidthPerCU", "measuredBandwidth")),
    ("L2 latency", ("memory", "l2", "latency", "mean")),
    ("L2 read bandwidth", ("memory", "l2", "readBandwidth", "measuredBandwidth")),
    ("L2 write bandwidth", ("memory", "l2", "writeBandwidth", "measuredBandwidth")),
    ("L3 latency", ("memory", "l3", "latency", "mean")),
    ("L3 read bandwidth", ("memory", "l3", "readBandwidth", "measuredBandwidth")),
    ("L3 write bandwidth", ("memory", "l3", "writeBandwidth", "measuredBandwidth")),
    ("LDS latency", ("memory", "shared", "latency", "mean")),
    ("LDS read bandwidth", ("memory", "shared", "readBandwidthPerCU", "measuredBandwidth")),
    ("LDS write bandwidth", ("memory", "shared", "writeBandwidthPerCU", "measuredBandwidth")),
    ("main-memory latency", ("memory", "main", "latency", "mean")),
    ("main-memory read bandwidth", ("memory", "main", "readBandwidth", "measuredBandwidth")),
    ("main-memory write bandwidth", ("memory", "main", "writeBandwidth", "measuredBandwidth")),
)

def run_rank(path: Path) -> tuple[int, int]:
    suffix = path.name.rsplit("_", 1)[-1]
    return (int(suffix) if suffix.isdigit() else -1, path.stat().st_mtime_ns)


def latest_run(runs_dir: Path) -> Path | None:
    candidates = [
        path for path in runs_dir.iterdir()
        if path.is_dir() and RUN_PATTERN.match(path.name)
    ] if runs_dir.is_dir() else []
    return max(candidates, key=run_rank) if candidates else None


def key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(errors="replace").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values


def tsv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def dig(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def rocminfo_gpu_agents(path: Path, problems: list[str]) -> list[dict[str, str]]:
    agents: list[dict[str, str]] = []
    current: dict[str, str] | None = None

    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError as exc:
        problems.append(f"{path}: cannot read rocminfo snapshot: {exc}")
        return agents

    def retain_current() -> None:
        if current is not None and current.get("device_type") == "GPU":
            agents.append(current.copy())

    field_names = {
        "Uuid": "uuid",
        "Device Type": "device_type",
        "BDFID": "bdfid",
        "Compute Unit": "compute_units",
        "SIMDs per CU": "simds_per_cu",
    }
    for raw_line in lines:
        line = raw_line.strip()
        if re.fullmatch(r"Agent\s+\d+", line):
            retain_current()
            current = {}
            continue
        if current is None:
            continue
        label, separator, value = line.partition(":")
        key = field_names.get(label)
        if separator and key:
            current[key] = value.strip().split()[0]
    retain_current()
    return agents


def validate_rocminfo_device_map(
    path: Path,
    devices: list[dict[str, str]],
    problems: list[str],
) -> None:
    agents = rocminfo_gpu_agents(path, problems)
    if len(agents) != len(EXPECTED_DEVICES):
        problems.append(
            f"{path}: found {len(agents)} GPU agents, expected {len(EXPECTED_DEVICES)}"
        )
        return
    if len(devices) != len(EXPECTED_DEVICES):
        return

    try:
        cu_values = {int(agent["compute_units"]) for agent in agents}
        simds_per_cu_values = {int(agent["simds_per_cu"]) for agent in agents}
    except (KeyError, ValueError) as exc:
        problems.append(f"{path}: malformed GPU-agent properties: {exc}")
        return

    if cu_values != {EXPECTED_CUS_PER_DEVICE}:
        problems.append(
            f"{path}: unexpected ROCr CU counts {sorted(cu_values)}, "
            f"expected [{EXPECTED_CUS_PER_DEVICE}]"
        )
    if simds_per_cu_values != {EXPECTED_SIMDS_PER_CU}:
        problems.append(
            f"{path}: unexpected ROCr SIMDs-per-CU values "
            f"{sorted(simds_per_cu_values)}, expected [{EXPECTED_SIMDS_PER_CU}]"
        )

    try:
        ordered_devices = sorted(devices, key=lambda row: int(row["logical_device"]))
        for ordinal, (row, agent) in enumerate(zip(ordered_devices, agents)):
            bdfid = int(agent["bdfid"])
            location_id = int(row["location_id"])
            unique_id = int(row["unique_id"])
            simd_count = int(row["simd_count"])
            compute_units = int(agent["compute_units"])
            simds_per_cu = int(agent["simds_per_cu"])
            expected_uuid = f"GPU-{unique_id:016x}"

            if location_id != bdfid:
                problems.append(
                    f"{path}: device {ordinal} KFD location_id {location_id} "
                    f"does not match ROCr BDFID {bdfid}"
                )
            if agent.get("uuid", "").lower() != expected_uuid.lower():
                problems.append(
                    f"{path}: device {ordinal} KFD unique_id {unique_id} does "
                    f"not match ROCr UUID {agent.get('uuid')!r}"
                )
            if simd_count != compute_units * simds_per_cu:
                problems.append(
                    f"{path}: device {ordinal} KFD simd_count {simd_count} does "
                    f"not equal {compute_units} CUs x {simds_per_cu} SIMDs/CU"
                )
    except (KeyError, ValueError) as exc:
        problems.append(f"{path}: cannot compare KFD and ROCr device maps: {exc}")


def validate_json(path: Path, metrics: tuple[tuple[str, tuple[str, ...]], ...], problems: list[str]) -> None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        problems.append(f"{path}: invalid JSON: {exc}")
        return
    for name, metric_path in metrics:
        value = dig(data, metric_path)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            problems.append(f"{path}: missing or non-numeric {name} ({'.'.join(metric_path)})")


def validate_node(
    node_dir: Path,
    min_sweeps: int,
) -> tuple[list[str], str | None, str | None, tuple[tuple[int, ...], ...]]:
    problems: list[str] = []
    metadata = key_values(node_dir / "metadata.txt")

    for key, expected in (
        ("mode", "cpx"),
        ("memory_partition", "NPS1"),
        ("xnack", "1"),
        ("allocator", "hipmalloc"),
        ("devices", "12"),
        ("packages", "2"),
        ("order_design", "williams"),
        ("failures", "0"),
    ):
        if metadata.get(key) != expected:
            problems.append(
                f"{node_dir.name}: metadata {key}={metadata.get(key)!r}, expected {expected!r}"
            )
    try:
        sweeps = int(metadata.get("sweeps", "0"))
    except ValueError:
        sweeps = 0
    if sweeps < min_sweeps:
        problems.append(f"{node_dir.name}: only {sweeps} sweeps; require at least {min_sweeps}")

    for phase in ("provenance_before", "provenance_after"):
        path = node_dir / phase / "amd_smi_partition.txt"
        snapshot = path.read_text() if path.exists() else ""
        compute = set(re.findall(r"(?:ACCELERATOR|COMPUTE)_PARTITION:\s*(\S+)", snapshot)) - {"N/A"}
        memory = set(re.findall(r"MEMORY_PARTITION:\s*(\S+)", snapshot)) - {"N/A"}
        if compute != {"CPX"} or memory != {"NPS1"}:
            problems.append(f"{path}: partition snapshot mismatch: {compute}, {memory}")

    before_map = node_dir / "devices.tsv"
    after_map = node_dir / "devices_after.tsv"
    if not before_map.exists() or not after_map.exists():
        problems.append(f"{node_dir.name}: missing before/after device map")
    elif before_map.read_bytes() != after_map.read_bytes():
        problems.append(f"{node_dir.name}: device map changed during execution")

    devices = tsv_rows(before_map)
    after_devices = tsv_rows(after_map)
    if len(devices) != 12:
        problems.append(f"{node_dir.name}: device map has {len(devices)} rows, expected 12")
    else:
        try:
            ordinals = {int(row["logical_device"]) for row in devices}
            package_counts = Counter(int(row["package"]) for row in devices)
            simd_values = {int(row["simd_count"]) for row in devices}
            xcc_values = {int(row["num_xcc"]) for row in devices}
            if ordinals != EXPECTED_DEVICES:
                problems.append(f"{node_dir.name}: device ordinals are incomplete")
            if package_counts != Counter({0: 6, 1: 6}):
                problems.append(f"{node_dir.name}: invalid package grouping {dict(package_counts)}")
            if simd_values != {EXPECTED_SIMDS_PER_DEVICE}:
                problems.append(
                    f"{node_dir.name}: unexpected KFD SIMD counts "
                    f"{sorted(simd_values)}, expected [{EXPECTED_SIMDS_PER_DEVICE}]"
                )
            if xcc_values != {1}:
                problems.append(f"{node_dir.name}: unexpected XCC counts {sorted(xcc_values)}")
        except (KeyError, ValueError) as exc:
            problems.append(f"{node_dir.name}: malformed device map: {exc}")

    validate_rocminfo_device_map(
        node_dir / "provenance_before" / "rocminfo.txt", devices, problems
    )
    validate_rocminfo_device_map(
        node_dir / "provenance_after" / "rocminfo.txt", after_devices, problems
    )

    orders = tsv_rows(node_dir / "orders.tsv")
    by_repeat: dict[int, list[dict[str, str]]] = defaultdict(list)
    try:
        for row in orders:
            by_repeat[int(row["repeat"])].append(row)
    except (KeyError, ValueError) as exc:
        problems.append(f"{node_dir.name}: malformed orders.tsv: {exc}")
        by_repeat.clear()

    expected_repeats = set(range(1, sweeps + 1))
    if set(by_repeat) != expected_repeats:
        problems.append(
            f"{node_dir.name}: repeats in orders.tsv do not match 1..{sweeps}"
        )

    position_counts: Counter[tuple[int, int]] = Counter()
    carryover_counts: Counter[tuple[int, int]] = Counter()
    order_signature: list[tuple[int, ...]] = []
    for repeat in sorted(by_repeat):
        rows = by_repeat[repeat]
        try:
            positions = {int(row["position"]) for row in rows}
            devices_in_repeat = {int(row["device"]) for row in rows}
            ordered = tuple(
                int(row["device"])
                for row in sorted(rows, key=lambda item: int(item["position"]))
            )
        except (KeyError, ValueError) as exc:
            problems.append(f"{node_dir.name}: repeat {repeat} has malformed order: {exc}")
            continue
        if len(rows) != 12 or positions != EXPECTED_POSITIONS or devices_in_repeat != EXPECTED_DEVICES:
            problems.append(f"{node_dir.name}: repeat {repeat} is not a permutation of devices 0..11")
            continue
        order_signature.append(ordered)
        for position, device in enumerate(ordered):
            position_counts[(device, position)] += 1
        for previous, current in zip(ordered, ordered[1:]):
            carryover_counts[(previous, current)] += 1

    if sweeps > 0 and sweeps % 12 == 0 and len(order_signature) == sweeps:
        expected_count = sweeps // 12
        if any(position_counts[(device, position)] != expected_count
               for device in EXPECTED_DEVICES for position in EXPECTED_POSITIONS):
            problems.append(f"{node_dir.name}: device/order positions are not exactly balanced")
        if any(carryover_counts[(previous, current)] != expected_count
               for previous in EXPECTED_DEVICES for current in EXPECTED_DEVICES
               if previous != current):
            problems.append(f"{node_dir.name}: immediate predecessor pairs are not balanced")

    status_rows = tsv_rows(node_dir / "status.tsv")
    # New runs contain one status row per device. Older runs may contain an
    # additional static-LDS row; retain compatibility by validating only their
    # full rows.
    if status_rows and "kind" in status_rows[0]:
        status_rows = [row for row in status_rows if row.get("kind") == "full"]
    expected_status_rows = sweeps * 12
    if len(status_rows) != expected_status_rows:
        problems.append(
            f"{node_dir.name}: status.tsv has {len(status_rows)} rows, expected {expected_status_rows}"
        )

    for repeat, rows in by_repeat.items():
        for row in rows:
            try:
                device = int(row["device"])
            except (KeyError, ValueError):
                continue
            base = node_dir / f"repeat_{repeat:02d}" / f"device_{device:02d}"
            # The simplified layout writes directly into device_XX. Fall back
            # to the original device_XX/full layout for already collected runs.
            result_dir = base if (base / "result.json").exists() else base / "full"
            full_status = key_values(result_dir / "status.txt")
            for key in ("repeat", "position", "device", "package", "package_device"):
                if full_status.get(key) != row[key]:
                    problems.append(f"{result_dir}: status/order mismatch in {key}")
            if full_status.get("exit_status") != "0":
                problems.append(f"{result_dir}: nonzero exit status")
            if full_status.get("state") != "ok":
                problems.append(f"{result_dir}: incomplete run")
            elif not (result_dir / "result.json").exists():
                problems.append(f"{result_dir}: missing result.json")
            else:
                validate_json(result_dir / "result.json", REQUIRED_FULL_METRICS, problems)

    return (
        problems,
        metadata.get("git_commit"),
        metadata.get("mt4g_sha256"),
        tuple(order_signature),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path(__file__).resolve().parent / "runs",
        help="directory containing cpx_<job-id> runs",
    )
    parser.add_argument("--run", type=Path, help="specific cpx_<job-id> directory")
    parser.add_argument("--expected-nodes", type=int, default=3)
    parser.add_argument("--min-sweeps", type=int, default=12)
    args = parser.parse_args()

    run = args.run or latest_run(args.runs)
    if run is None or not run.is_dir():
        print("No CPX sweep run found", file=sys.stderr)
        return 1

    nodes_dir = run / "nodes"
    node_dirs = sorted(path for path in nodes_dir.iterdir() if path.is_dir()) \
        if nodes_dir.is_dir() else []
    problems: list[str] = []
    if len(node_dirs) != args.expected_nodes:
        problems.append(
            f"{run.name}: found {len(node_dirs)} node directories, expected {args.expected_nodes}"
        )

    commits: set[str] = set()
    checksums: set[str] = set()
    signatures: list[tuple[tuple[int, ...], ...]] = []
    for node_dir in node_dirs:
        node_problems, commit, checksum, signature = validate_node(node_dir, args.min_sweeps)
        problems.extend(node_problems)
        if commit:
            commits.add(commit)
        if checksum:
            checksums.add(checksum)
        signatures.append(signature)

    if len(commits) != 1:
        problems.append(f"workers do not share one Git commit: {sorted(commits)}")
    if len(checksums) != 1:
        problems.append(f"workers do not share one MT4G binary checksum: {sorted(checksums)}")
    if len(signatures) > 1 and len(set(signatures)) != len(signatures):
        problems.append("two nodes used identical device-order sequences")

    if problems:
        print(f"Validation failed for {run}", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print(
        f"Validation passed: {run} ({len(node_dirs)} nodes, "
        f">={args.min_sweeps} sweeps per node, commit {next(iter(commits))})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
