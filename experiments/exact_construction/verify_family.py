"""Independently re-evaluate the finite-field exact scope-curve family.

Uses the four explicitly declared matrices in the reference data and computes
conditional variances by grouping all p^n states (rather than using
closed-form curves). Checks identical code distributions and the exact
r-times-scope-budget formula. Does not regenerate every other auxiliary
theorem diagnostic in verification_results.json.
"""
from __future__ import annotations
import itertools
import json
from pathlib import Path
import numpy as np
from sae_localize import optimize_scopes

ROOT=Path(__file__).resolve().parent


def compute(data):
    n=int(data['n']);p=int(data['p'])
    x=np.asarray(list(itertools.product(range(p), repeat=n)), dtype=np.int64)
    reference=None
    report={}
    for rec in data['records']:
        r=rec['order'];M=np.asarray(rec['matrix'],dtype=np.int64)
        transformed=(x@M.T)%p
        if len({tuple(row) for row in transformed})!=p**n:
            raise AssertionError(f'Finite-field map r={r} is not bijective.')
        q=(transformed==0).astype(float)
        # The population distribution of codes is unchanged.
        signatures=np.asarray(sorted(map(tuple,q)))
        if reference is None:reference=signatures
        elif not np.array_equal(signatures,reference):
            raise AssertionError('Representation distribution changed.')
        errors=np.zeros((1<<n,n),dtype=float)
        for mask in range(1<<n):
            cols=[i for i in range(n) if mask & (1<<i)]
            if cols:
                _, groups=np.unique(x[:,cols],axis=0,return_inverse=True)
                sizes=np.bincount(groups)
                average=np.stack([np.bincount(groups,weights=q[:,j])/sizes for j in range(n)],axis=1)
                residue=q-average[groups]
            else:
                residue=q-q.mean(axis=0)
            errors[mask]=np.mean(residue**2,axis=0)
        curve=optimize_scopes(errors)[0]
        v=(1/p)*(1-1/p)
        expected=np.array([v*(n-min(n,b//r)) for b in range(n*n+1)])
        discrepancy=float(np.max(np.abs(curve-expected)))
        if discrepancy>1e-10 or np.max(np.abs(curve-np.asarray(rec['curve'])))>1e-10:
            raise AssertionError(f'Failed exact family scope curve for r={r}')
        if int(np.flatnonzero(curve<=1e-12)[0])!=r*n:
            raise AssertionError(f'Exact full-localization cost should equal {r*n}')
        report[r]=discrepancy
    return report


def main():
    data=json.loads((ROOT/'verification_results.json').read_text())
    results=compute(data['family'])
    print('Finite-field verification (complete grid, conditional means):')
    for order,error in results.items():
        print(f'  order {order}: maximum closed-form discrepancy {error:.3g}')
    print('All four distributions matched; all scope curves verified.')


if __name__=='__main__':
    main()
