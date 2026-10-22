# Learned SAE validation

This experiment trains five overcomplete SAEs and measures input dependence while keeping their representation and activation samples fixed across input maps.

## Setup

| Component | Setting |
|---|---|
| Controlled input | 8 binary factors, 256 states |
| Seeds | 0–4 |
| Representation network | `8 → 24 → 8` ReLU network |
| Train / held-out | 192 / 64 states |
| SAE | `24 → 48 → 24` ReLU, normalized decoder, L1 coefficient 0.01 |
| Input maps | Identity, cyclic XOR arity 3 and 5 |
| Scope optimization | All 256 candidate scopes per feature, exact dynamic programming |

## Results

| Map | Mean budget for 90% explained activation variance |
|---|---:|
| Identity | 136.0 ± 22.6 |
| XOR arity 3 | 192.4 ± 11.6 |
| XOR arity 5 | 202.8 ± 8.6 |

The paired estimator approaches the exact curves as the number of pairs per candidate scope grows:

| Pairs per scope | 32 | 128 | 512 | 2,048 |
|---|---:|---:|---:|---:|
| Mean maximum normalized curve error | 0.123 | 0.0472 | 0.0216 | 0.0109 |

At the same omission tolerance, the spectral lower bounds are 4.2, 12.6, and 20.6, and the feature-order lower bounds are 71.2, 145.0, and 150.2.

## Reproduce

```bash
python reproduce.py --out ./outputs/retrained --audit
```

Individual steps:

```bash
python pilot.py --seeds 0 1 2 3 4 --out ./outputs/retrained
python audit.py --results ./outputs/retrained --repetitions 10
python code_order_bound.py --results ./outputs/retrained
python plot_paper_figures.py --results ./outputs/retrained \
    --figures ./outputs/retrained/figures
```

The `results/` folder contains five model checkpoints (`seed_*_weights.pt`), frozen activations, exact curves, sampled estimates, and result tables. See [the reproduction guide](../../docs/REPRODUCIBILITY.md) for the full workflow.

## Dependencies

From the repository root, install the shared dependencies with `python -m pip install -r requirements.txt`. The same file supports the released demo and the learned SAE validation scripts.
