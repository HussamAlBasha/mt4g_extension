#!/usr/bin/env python3
"""Validate one RQ3 run and regenerate its descriptive summaries."""
import argparse
import subprocess
import sys
from pathlib import Path


def latest_run(runs_dir):
    candidates = [path for path in runs_dir.glob("cpx_*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no cpx_<job-id> run under {runs_dir}")

    def rank(path):
        suffix = path.name.rsplit("_", 1)[-1]
        return int(suffix) if suffix.isdigit() else -1, path.stat().st_mtime_ns

    return max(candidates, key=rank)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        help="run directory to analyze; defaults to the newest runs/cpx_<job-id>",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    run = args.run.resolve() if args.run else latest_run(base / "runs")

    def execute(name, *arguments):
        print(f"Running {name}", flush=True)
        subprocess.run(
            [sys.executable, str(base / name), *map(str, arguments)],
            check=True,
        )

    execute("validate_results.py", "--run", run)
    execute("analyze.py", "--run", run, "--output-dir", base / "analysis")


if __name__ == "__main__":
    main()
