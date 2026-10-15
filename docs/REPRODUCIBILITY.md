# Reproducing Localize

The repository includes code, saved results, and trained weights for all five controlled learned-SAE experiments.

## Install

```bash
pip install -e ".[dev]"
pip install -e ".[experiments]"  # PyTorch, pandas, matplotlib
```

## Run the core analyses

```bash
python -m pytest -q
python experiments/exact_construction/verify_family.py
python demo/demo.py --no-plot
```

`verify_family.py` enumerates the $7^4=2,401$ states of the exact finite-field construction. The demo recomputes scope curves for each of five learned SAEs under the identity and two XOR input maps.

## Reproduce the learned-SAE experiment

```bash
python experiments/learned_sae_validation/reproduce.py \
    --out ./outputs/retrained --audit
```

The script trains five independent `8 → 24 → 8` representation networks and `24 → 48 → 24` overcomplete SAEs on 192 of the 256 Boolean states, then computes all full-grid scope curves. It also runs the paired sampling audit and regenerates the experiment figures. Training uses the original protocol and five seeds.

To evaluate the saved models and activation arrays directly:

```bash
python experiments/learned_sae_validation/audit.py \
    --results experiments/learned_sae_validation/results \
    --repetitions 10
python experiments/learned_sae_validation/code_order_bound.py \
    --results experiments/learned_sae_validation/results
```

These scripts write results to the specified directory. Use a copy of `results/` if you want to preserve the committed reference files.

## Results and reproduction entrypoints

| Analysis | Code | Reference results |
|---|---|---|
| Exact finite-field curves | `experiments/exact_construction/verify_family.py` | `verification_results.json` |
| Five SAE scope curves | `experiments/learned_sae_validation/pilot.py` | `seed_*.npz`, `seed_*.json`, `models.csv` |
| Paired intervention recovery | `experiments/learned_sae_validation/audit.py` | `sampling.csv`, `sampling_examples.npz` |
| Spectral and compatibility bounds | `experiments/learned_sae_validation/audit.py` | `certificates.csv`, `seed_*.json` |
| Feature-order lower bounds | `experiments/learned_sae_validation/code_order_bound.py` | `code_order_bounds.csv`, `code_order_bounds.npz` |
| Learned-SAE plots | `experiments/learned_sae_validation/plot_paper_figures.py` | `assets/figures/fig3.png`, `fig4.png` |
| Exact curve plot | `experiments/exact_construction/make_frozen_curves.py` | `assets/figures/fig2.png` |

The public API, setup conventions, and input array shapes are described in [API.md](API.md).
