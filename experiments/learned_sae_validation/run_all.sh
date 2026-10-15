#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python pilot.py --seeds 0 1 2 3 4 --out results
python audit.py --results results --repetitions 10
python code_order_bound.py --results results
python plot_paper_figures.py
