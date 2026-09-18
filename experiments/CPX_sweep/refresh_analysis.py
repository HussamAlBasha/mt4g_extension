#!/usr/bin/env python3
"""Reproduce retained RQ3 analysis only; never submit jobs or execute MT4G."""
import subprocess
import sys
from pathlib import Path

def main():
    base=Path(__file__).resolve().parent
    run=base/'runs'/'cpx_11360825'
    details=base/'analysis'/'details'
    def execute(name,*arguments,quiet=False):
        print(f'Running {name}',flush=True)
        subprocess.run([sys.executable,str(base/name),*map(str,arguments)],check=True,stdout=subprocess.DEVNULL if quiet else None)
    execute('validate_results.py','--run',run)
    subprocess.run([sys.executable,'-m','unittest','test_analysis.py'],cwd=base,check=True)
    execute('compare_results.py','--run',run,'--output-dir',details,'--strict',quiet=True)
    execute('summarize_results.py','--analysis',details)
    print(f'Results: {base / "analysis" / "SUMMARY.md"}')

if __name__=='__main__':main()
