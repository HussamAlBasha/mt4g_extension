#!/usr/bin/env python3
"""Generate deterministic, position-balanced CPX device orders."""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from pathlib import Path


def seed_number(seed: str, node: str, block: int) -> int:
    material = f"{seed}\0{node}\0{block}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def williams_rows(size: int) -> list[list[int]]:
    """Return an even-order Williams design with balanced first-order carryover."""
    if size < 2 or size % 2:
        raise ValueError("the Williams design requires a positive even device count")
    first = [0]
    for position in range(1, size):
        if position % 2:
            first.append((position + 1) // 2)
        else:
            first.append(size - position // 2)
    return [[(value + shift) % size for value in first] for shift in range(size)]


def generate(repetitions: int, devices: int, seed: str, node: str) -> list[dict[str, int | str]]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    if devices < 2 or devices % 2:
        raise ValueError("devices must be a positive even number")

    design = williams_rows(devices)
    records: list[dict[str, int | str]] = []
    repeat = 1
    block = 0
    while repeat <= repetitions:
        numeric_seed = seed_number(seed, node, block)
        rng = random.Random(numeric_seed)

        # Randomly relabel device symbols and randomize the order in which the
        # balanced rows are executed. A complete block retains exact position
        # and first-order carryover balance.
        labels = list(range(devices))
        rng.shuffle(labels)
        rows = [row[:] for row in design]
        rng.shuffle(rows)

        for row in rows:
            if repeat > repetitions:
                break
            order = [labels[symbol] for symbol in row]
            for position, device in enumerate(order):
                records.append(
                    {
                        "repeat": repeat,
                        "position": position,
                        "device": device,
                        "package": device // 6,
                        "package_device": device % 6,
                        "block": block,
                        "design_seed": numeric_seed,
                    }
                )
            repeat += 1
        block += 1
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=12)
    parser.add_argument("--devices", type=int, default=12)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--node", required=True)
    args = parser.parse_args()

    records = generate(args.repetitions, args.devices, args.seed, args.node)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "repeat",
                "position",
                "device",
                "package",
                "package_device",
                "block",
                "design_seed",
            ),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
