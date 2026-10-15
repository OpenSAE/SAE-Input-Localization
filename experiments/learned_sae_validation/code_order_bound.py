"""An optional, newly derived lower bound; NOT a result in the supplied paper.

For a frozen code and its ANOVA energies w[j,T], define
    a_j(k) = sum_{|T|>k} w[j,T].
Every scope A_j with |A_j|=k omits every component of order greater than k.
Consequently,
    min_{sum k_j <= B} sum_j a_j(k_j) <= L_z(B).
This relaxation discards support identities but retains feature-coordinate
information. Unlike the paper's h-based spectral bound, it needs ANOVA energies
of the learned code z. It does not replace or tighten that theorem using the
same information, and is not a full characterization of the scope curve.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
from pilot import fourier, optimize_scopes


def run(root: Path) -> None:
    rows, arrays = [], {}
    for seed in range(5):
        data = np.load(root / f'seed_{seed}.npz')
        W = data['z'].var(axis=0).sum()
        n = int(np.log2(len(data['z'])))
        card = np.array([i.bit_count() for i in range(1 << n)])
        for width in [1, 3, 5]:
            coefficients = fourier(data['z'][data[f'ids_{width}']])
            costs = np.array([(coefficients[card > k]**2).sum(axis=0) for k in range(n + 1)])
            lower = optimize_scopes(costs[card])[0]
            truth = data[f'curve_{width}']
            if not np.all(lower <= truth + 1e-10):
                raise AssertionError('Order relaxation exceeded the exact curve.')
            true_b = int(np.flatnonzero(truth <= .1 * W)[0])
            lower_b = int(np.flatnonzero(lower <= .1 * W)[0])
            rows.append(dict(seed=seed, map_arity=width, true_B10=true_b,
                             code_order_lower_B10=lower_b, ratio=lower_b/true_b))
            arrays[f'curve_seed{seed}_width{width}'] = lower
    with (root / 'code_order_bounds.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(root / 'code_order_bounds.npz', **arrays)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path(__file__).parent/'results')
    run(parser.parse_args().results)
