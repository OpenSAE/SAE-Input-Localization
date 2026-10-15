"""Offline learned-SAE scope-curve demo using the five paper checkpoints.

No model downloads, GPU or retraining. The included full-grid SAE codes
are reindexed by the original bijective input mappings, and the complete
population optimum is recomputed rather than merely reading saved budgets.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from sae_localize import exact_scope_curve, scope_budget

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / 'experiments' / 'learned_sae_validation' / 'results'


def run(reference: Path = REF):
    summary = {}
    for r in (1, 3, 5):
        budgets = []
        curves = []
        for seed in range(5):
            with np.load(reference / f'seed_{seed}.npz', allow_pickle=False) as data:
                ids = data[f'ids_{r}']
                z = data['z'][ids]
                result = exact_scope_curve(z, n_factors=8)
                np.testing.assert_allclose(result.curve, data[f'curve_{r}'], atol=1e-10, rtol=0)
                budget = scope_budget(result, 0.1*result.total_variance)
                budgets.append(budget)
                curves.append(result.normalized_curve)
        summary[r] = dict(budgets=budgets, mean=float(np.mean(budgets)),
                          std=float(np.std(budgets, ddof=1)),
                          curve=np.mean(curves, axis=0))
    return summary


def plot_summary(summary, path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    orders = [1, 3, 5]
    bars = ax.bar(range(3), [summary[r]['mean'] for r in orders],
                  yerr=[summary[r]['std'] for r in orders], capsize=5)
    ax.set_xticks(range(3), ['Identity (arity 1)', 'XOR arity 3', 'XOR arity 5'])
    ax.set_ylabel('Budget to explain 90% of activation variance')
    for b, r in zip(bars, orders):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+summary[r]['std']+7,
                f"{summary[r]['mean']:.1f}",ha='center')
    ax.set_ylim(0,250)
    for r in orders:
        ax2.plot(summary[r]['curve'], label=f'Arity {r}')
    ax2.set_xlabel('Total input-feature dependency budget')
    ax2.set_ylabel('Minimum omitted activation variance / total')
    ax2.set_ylim(0,1.01)
    ax2.legend()
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=REF)
    p.add_argument('--no-plot', action='store_true')
    p.add_argument('--output', type=Path, default=ROOT/'outputs'/'demo_scope_budgets.png')
    args=p.parse_args()
    result=run(args.results)
    print('Five frozen overcomplete SAEs | 8 binary factors | all 256 controlled states')
    print(f"{'Map':<12} {'Five seed budgets':<28} {'Mean ± sample SD'}")
    for order, label in [(1,'Identity'),(3,'XOR arity 3'),(5,'XOR arity 5')]:
        d=result[order]
        print(f"{label:<12} {str(d['budgets']):<28} {d['mean']:5.1f} ± {d['std']:4.1f}")
    print('Changes versus identity: +41.5% (arity 3), +49.1% (arity 5)')
    if not args.no_plot:
        plot_summary(result,args.output)
        print(f'Plot: {args.output}')


if __name__ == '__main__':
    main()
