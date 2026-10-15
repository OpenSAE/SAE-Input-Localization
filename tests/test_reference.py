from pathlib import Path
import json
import numpy as np
import pytest
from sae_localize import (exact_scope_curve, spectral_budget_lower_bound,
                          feature_order_lower_curve, compatibility_certificate)

REF=Path(__file__).resolve().parents[1]/'experiments'/'learned_sae_validation'/'results'

@pytest.mark.parametrize('arity,budget,spectral',[(1,128,4),(3,206,12),(5,215,20)])
def test_reference_seed0(arity,budget,spectral):
    with np.load(REF/'seed_0.npz',allow_pickle=False) as d:
        z=d['z'][d[f'ids_{arity}']]
        result=exact_scope_curve(z,n_factors=8)
        np.testing.assert_allclose(result.curve,d[f'curve_{arity}'],atol=1e-10,rtol=0)
        np.testing.assert_allclose(result.errors,d[f'errors_{arity}'],atol=1e-10,rtol=0)
        assert result.budget(.1*result.total_variance)==budget
        assert spectral_budget_lower_bound(d['h'][d[f'ids_{arity}']],d['D'],
                 float(((d['h']-d['hhat'])**2).sum(axis=1).mean()),.1*result.total_variance)==spectral
        if arity==3:
            assigned=result.assignment(budget)
            bounds=compatibility_certificate(d['h'][d[f'ids_{arity}']],z,d['D'],d['bias'],assigned)
            assert bounds['compatibility_slack']>=-1e-9
            assert bounds['Gamma']+1e-9>=bounds['compatibility_lower_Gamma']
            relaxed=feature_order_lower_curve(z)
            assert np.all(relaxed <= result.curve+1e-9)


def test_reference_all_five_seed_budgets_and_invariance():
    found={1:[],3:[],5:[]}
    for seed in range(5):
        with np.load(REF/f'seed_{seed}.npz',allow_pickle=False) as d:
            meta=json.loads((REF/f'seed_{seed}.json').read_text())
            for r in (1,3,5):
                ids=d[f'ids_{r}']
                np.testing.assert_array_equal(np.sort(ids),np.arange(256))
                W=float(d['z'].var(axis=0).sum())
                curve=d[f'curve_{r}']
                b=int(np.flatnonzero(curve<=.1*W+1e-12)[0])
                assert b==meta['maps'][str(r)]['B']['0.1']
                found[r].append(b)
    for r,target in [(1,136.0),(3,192.4),(5,202.8)]:
        assert np.isclose(np.mean(found[r]),target)
    assert all(a<b<c for a,b,c in zip(found[1],found[3],found[5]))
