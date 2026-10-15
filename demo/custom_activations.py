"""Try Localize on your own complete, uniformly ordered Boolean SAE codes.

Input: .npy of shape (2**n_factors, n_features), or .npz with a ``z`` key.
Row i corresponds to binary factor values equal to the bits of i. For the
actual measured input map, enumerate *every* combination in that order.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from sae_localize import exact_scope_curve


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('activations',type=Path)
    p.add_argument('--n-factors',type=int,required=True)
    p.add_argument('--relative-omission',type=float,default=.1)
    args=p.parse_args()
    if not 0<=args.relative_omission<=1:
        p.error('--relative-omission must be between 0 and 1')
    loaded = np.load(args.activations, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        with loaded as src:
            z = src['z'].copy()
    else:
        z = loaded
    result=exact_scope_curve(z,n_factors=args.n_factors)
    target=args.relative_omission*result.total_variance
    b=result.budget(target)
    assignment=result.assignment(b)
    print(f'features={result.n_features} factors={result.n_factors}')
    print(f'total activation variance W_z = {result.total_variance:.8g}')
    print(f'budget at omitted fraction <= {args.relative_omission:g}: {b}')
    print('feature scopes (bitmasks):',assignment.tolist())
    print('actual scope cost:',sum(int(s).bit_count() for s in assignment))


if __name__ == '__main__':
    main()
