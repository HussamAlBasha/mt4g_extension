#!/usr/bin/env python3
"""Validate the retained RQ3 run and regenerate its descriptive summaries."""
import subprocess
import sys
from pathlib import Path


def main():
    base = Path(__file__).resolve().parent
    run = base / "runs" / "cpx_11360825"

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
