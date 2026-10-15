"""Regenerate the five-seed controlled SAE results without changing references.

By default writes to a user-selected output directory (not the committed
reference arrays); the scripts are the original experimental implementation.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent


def execute(args):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True,help='Non-reference output directory')
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3,4])
    p.add_argument('--repetitions',type=int,default=10)
    p.add_argument('--audit',action='store_true',help='Run original audit (expects seeds 0..4)')
    o=p.parse_args(args)
    out=o.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,str(HERE/'pilot.py'),'--seeds',*map(str,o.seeds),'--out',str(out)],check=True)
    if o.audit:
        if sorted(o.seeds)!=[0,1,2,3,4]:
            p.error('--audit requires all 5 seeds, as in original audit.py')
        subprocess.run([sys.executable,str(HERE/'audit.py'),'--results',str(out),'--repetitions',str(o.repetitions)],check=True)
        subprocess.run([sys.executable,str(HERE/'code_order_bound.py'),'--results',str(out)],check=True)
        subprocess.run([sys.executable,str(HERE/'plot_paper_figures.py'), '--results', str(out),
                        '--figures', str(out/'figures')], check=True)


if __name__=='__main__':
    execute(sys.argv[1:])
