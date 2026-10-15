"""Audit the exact finite population; simulate finite paired interventions.

N means independent pairs PER CANDIDATE SCOPE, not total model calls.
The same N base pairs are reused across scopes. Each scope needs its own
hybrid input. The finite 256-state activation cache avoids repeated forwards;
this does not demonstrate query efficiency in a large input space.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
from pilot import fourier, optimize_scopes, recover_assignment


def direct_scope_error(z: np.ndarray, mask: int) -> np.ndarray:
    cells = np.arange(len(z)) & mask
    residual = np.empty_like(z)
    for cell in np.unique(cells):
        inside = cells == cell
        residual[inside] = z[inside] - z[inside].mean(axis=0)
    return (residual ** 2).mean(axis=0)


def certificate(h: np.ndarray, z: np.ndarray, D: np.ndarray, bias: np.ndarray,
                assignment: np.ndarray) -> dict:
    hc, zc = fourier(h), fourier(z)
    mask = np.arange(len(h))
    keep = (mask[:, None] & assignment[None, :]) == mask[:, None]
    omitted = zc * (~keep)
    gamma = float((omitted**2).sum())
    Lambda = float(((omitted @ D.T)**2).sum())
    R = float(((h - z @ D.T - bias) ** 2).sum(axis=1).mean())
    Q = 0.0
    for t in range(1, len(h)):
        columns = D[:, keep[t]]
        effect = hc[t]
        if columns.shape[1]:
            U, singular, _ = np.linalg.svd(columns, full_matrices=False)
            # Orthogonal projection on the numerical span of eligible decoder columns.
            rank = int((singular > singular[0] * 1e-10).sum())
            residual = effect - U[:, :rank] @ (U[:, :rank].T @ effect)
        else:
            residual = effect
        Q += float(residual @ residual)
    lower_gamma = max(np.sqrt(Q) - np.sqrt(R), 0) ** 2 / np.linalg.norm(D, 2)**2
    slack = np.sqrt(R) + np.sqrt(Lambda) - np.sqrt(Q)
    if slack < -1e-8 or lower_gamma > gamma + 1e-8:
        raise AssertionError('Compatibility inequality audit failed.')
    return dict(R=R, Gamma=gamma, Lambda=Lambda, Q=Q,
                compatibility_lower_Gamma=lower_gamma, compatibility_slack=slack)


def finite_errors(z: np.ndarray, N: int, rng: np.random.Generator) -> np.ndarray:
    count, m = z.shape
    x = rng.integers(count, size=N)
    xp = rng.integers(count, size=N)
    zx = z[x]
    result = np.empty((count, m))
    for A in range(count):
        y = (x & A) | (xp & (count - 1 - A))
        result[A] = (0.5 * (zx - z[y])**2).mean(axis=0)
    return result


def run(directory: Path, repetitions: int = 30) -> None:
    start = time.perf_counter()
    model_rows, certificate_rows, checks = [], [], []
    for seed in range(5):
        data = np.load(directory / f'seed_{seed}.npz')
        meta = json.loads((directory / f'seed_{seed}.json').read_text())
        Rbase, Kbase, Wbase = meta['R'], meta['K'], meta['W']
        for r in [1, 3, 5]:
            ids = data[f'ids_{r}']
            h, z, hhat = data['h'][ids], data['z'][ids], data['hhat'][ids]
            errors, curve = data[f'errors_{r}'], data[f'curve_{r}']
            direct_difference = max(float(np.max(np.abs(direct_scope_error(z, A) - errors[A])))
                                    for A in range(len(z)))
            checked_R = ((h - hhat)**2).sum(axis=1).mean()
            checked_K = (z > 0).sum(axis=1).mean()
            checked_W = z.var(axis=0).sum()
            permutation_ok = np.array_equal(np.sort(ids), np.arange(len(ids)))
            if direct_difference > 1e-10 or not permutation_ok:
                raise AssertionError('Exact population audit failed.')
            checks.append(dict(seed=seed, map_arity=r, direct_error_max=direct_difference,
                               delta_R=float(checked_R - Rbase), delta_K=float(checked_K - Kbase),
                               delta_W=float(checked_W - Wbase), permutation_verified=bool(permutation_ok)))
            for tolerance in [.01, .05, .1, .2]:
                budget = meta['maps'][str(r)]['B'][str(tolerance)]
                assignment = recover_assignment(budget, data[f'choices_{r}'], data[f'best_mask_{r}'])
                actual_error = errors[assignment, np.arange(len(assignment))].sum()
                actual_cost = sum(int(A).bit_count() for A in assignment)
                if actual_error > curve[budget] + 1e-10 or actual_cost > budget:
                    raise AssertionError('Dynamic-programming backtracking audit failed.')
                c = certificate(h, z, data['D'], data['bias'], assignment)
                certificate_rows.append(dict(seed=seed, map_arity=r, tolerance=tolerance, budget=budget,
                                             W=Wbase, **c))
            model_rows.append(dict(seed=seed, map_arity=r, R_over_Hvar=meta['R_over_Hvar'],
                                   heldout_R_over_Hvar=meta['heldout_R_over_Hvar'], K=meta['K'], W=meta['W'],
                                   B10=meta['maps'][str(r)]['B']['0.1'],
                                   B20=meta['maps'][str(r)]['B']['0.2'],
                                   spectral_lower_B10=meta['maps'][str(r)]['spectral_lower_B']['0.1']))

    confidence = .95
    sample_sizes = [32, 128, 512, 2048]
    records, example_arrays = [], {}
    for sampling_seed in range(5):
        data = np.load(directory / f'seed_{sampling_seed}.npz')
        W = float(data['z'].var(axis=0).sum())
        m = data['z'].shape[1]
        ranges = np.ptp(data['z'], axis=0)
        candidate_count = m * len(data['z'])
        for N in sample_sizes:
            epsilon = (ranges**2 / 2) * np.sqrt(np.log(2*candidate_count/(1-confidence)) / (2*N))
            radius = float(epsilon.sum())
            for rep in range(repetitions):
                for r in [1, 3, 5]:
                    rng = np.random.default_rng(230000 + sampling_seed*10000000 + N*1000 + rep*10 + r)
                    z = data['z'][data[f'ids_{r}']]
                    estimated = finite_errors(z, N, rng)
                    estimated_curve, choices, masks = optimize_scopes(estimated)
                    truth = data[f'curve_{r}']
                    true_errors = data[f'errors_{r}']
                    low_errors = np.maximum(estimated - epsilon, 0)
                    high_errors = np.minimum(estimated + epsilon, ranges**2 / 2)
                    low_errors[-1] = high_errors[-1] = 0
                    low = optimize_scopes(low_errors)[0]
                    high = optimize_scopes(high_errors)[0]
                    covered = bool(np.all(low <= truth + 1e-10) and np.all(high >= truth - 1e-10))
                    row = dict(seed=sampling_seed, map_arity=r, pairs_per_scope=N, repetition=rep,
                               max_curve_error_over_W=float(np.max(np.abs(estimated_curve - truth))/W),
                               mean_curve_error_over_W=float(np.mean(np.abs(estimated_curve - truth))/W),
                               simultaneous_band_covered=covered,
                               simple_uniform_radius_over_W=radius/W)
                    for tolerance in [.05, .1, .2]:
                        key = int(round(100*tolerance))
                        Bhat = int(np.flatnonzero(estimated_curve <= tolerance*W)[0])
                        Btrue = int(np.flatnonzero(truth <= tolerance*W)[0])
                        chosen = recover_assignment(Bhat, choices, masks)
                        selected_truth = float(true_errors[chosen, np.arange(m)].sum())
                        row[f'estimated_B{key}'] = Bhat
                        row[f'true_B{key}'] = Btrue
                        row[f'B{key}_abs_error'] = abs(Bhat-Btrue)
                        row[f'selected_regret_B{key}_over_W'] = float((selected_truth - truth[Bhat])/W)
                        row[f'band_width_at_true_B{key}_over_W'] = float((high[Btrue]-low[Btrue])/W)
                    records.append(row)
                    if rep == 0 and sampling_seed == 0:
                        example_arrays.update({f'curve_r{r}_N{N}': estimated_curve,
                                               f'lower_r{r}_N{N}': low, f'upper_r{r}_N{N}': high})
            print(f'Completed sampling seed={sampling_seed} N={N}', flush=True)

    for name, rows in [('models',model_rows),('certificates',certificate_rows),('sampling',records),('checks',checks)]:
        with (directory / f'{name}.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(directory / 'sampling_examples.npz', **example_arrays)
    summary = {'sampling_model_seeds': list(range(5)), 'repetitions_per_seed_N_and_map': repetitions,
               'confidence': confidence, 'sample_sizes_per_candidate_scope': sample_sizes,
               'elapsed_seconds': time.perf_counter() - start,
               'max_conditional_projection_discrepancy': max(c['direct_error_max'] for c in checks)}
    for r in [1,3,5]:
        rows = [x for x in model_rows if x['map_arity'] == r]
        summary[f'map_arity_{r}'] = {key: {'mean': float(np.mean([x[key] for x in rows])),
                                     'sd': float(np.std([x[key] for x in rows], ddof=1))}
                                for key in ['B10','B20','R_over_Hvar','heldout_R_over_Hvar','K','spectral_lower_B10']}
    summary['sampling'] = {}
    for N in sample_sizes:
        rows = [x for x in records if x['pairs_per_scope'] == N]
        summary['sampling'][str(N)] = {
            'max_curve_error_mean': float(np.mean([x['max_curve_error_over_W'] for x in rows])),
            'max_curve_error_p95': float(np.quantile([x['max_curve_error_over_W'] for x in rows], .95)),
            'B5_abs_error_mean': float(np.mean([x['B5_abs_error'] for x in rows])),
            'B10_abs_error_mean': float(np.mean([x['B10_abs_error'] for x in rows])),
            'B20_abs_error_mean': float(np.mean([x['B20_abs_error'] for x in rows])),
            'band_coverage_empirical': float(np.mean([x['simultaneous_band_covered'] for x in rows])),
            'band_width_at_true_B10_mean': float(np.mean([x['band_width_at_true_B10_over_W'] for x in rows])),
            'selected_regret_B10_mean': float(np.mean([x['selected_regret_B10_over_W'] for x in rows])),
        }
    (directory / 'audit_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path(__file__).parent / 'results')
    parser.add_argument('--repetitions', type=int, default=30)
    args = parser.parse_args()
    run(args.results, args.repetitions)
