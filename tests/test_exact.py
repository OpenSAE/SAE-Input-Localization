import itertools
import numpy as np
import pytest
from sae_localize import (boolean_fourier, direct_omission_error, exact_scope_curve,
                          omission_errors, optimize_scopes, paired_scope_errors,
                          scope_assignment, scope_budget)


def test_omit_matches_conditional_mean_and_parseval():
    rng = np.random.default_rng(9)
    z = rng.standard_normal((8, 4))
    result = exact_scope_curve(z)
    assert np.isclose(result.curve[0], result.total_variance, atol=1e-12)
    assert np.isclose(result.curve[-1], 0, atol=1e-12)
    assert np.all(np.diff(result.curve) <= 1e-12)
    for mask in range(8):
        np.testing.assert_allclose(result.errors[mask], direct_omission_error(z,mask),atol=1e-12)
    np.testing.assert_allclose(np.sum(result.coefficients[1:]**2),result.total_variance,atol=1e-12)
    for B in (0, 1, 4, 6, 12):
        assign=scope_assignment(result,B)
        assert sum(int(x).bit_count() for x in assign)<=B
        np.testing.assert_allclose(result.errors[assign,np.arange(4)].sum(),result.curve[B],atol=1e-12)


def test_dynamic_program_matches_exhaustive_on_two_features():
    rng=np.random.default_rng(42)
    z=rng.standard_normal((4,2))
    result=exact_scope_curve(z)
    for budget in range(5):
        values=[result.errors[i,0]+result.errors[j,1]
                for i,j in itertools.product(range(4),repeat=2)
                if i.bit_count()+j.bit_count()<=budget]
        np.testing.assert_allclose(result.curve[budget],min(values),atol=1e-12)


def test_budget_and_errors():
    x=np.arange(8)
    z=((x&1)>0).reshape(-1,1).astype(float)
    result=exact_scope_curve(z,3)
    assert scope_budget(result,0)==1
    assert scope_budget(result,.25)==0
    assert result.assignment(1).tolist()==[1]
    assert np.isclose(result.curve[0],.25)
    assert np.isclose(result.curve[-1],0)
    with pytest.raises(ValueError):scope_budget(result,-.1)
    with pytest.raises(ValueError):scope_assignment(result,4)


def test_bad_input_shapes_and_order_contract():
    with pytest.raises(ValueError):exact_scope_curve(np.zeros((6,2)))
    with pytest.raises(ValueError):exact_scope_curve(np.zeros((8,2)),n_factors=4)
    with pytest.raises(ValueError):exact_scope_curve(np.zeros((8,2))+np.nan)
    with pytest.raises(ValueError):optimize_scopes(np.array([[-1.],[0.]]))


def test_paired_scope_estimator_matches_population_and_zero_full_scope():
    x=np.arange(8)
    z=np.stack([(x&1).astype(float),((x>>1)&1).astype(float)],axis=1)
    errors,masks=paired_scope_errors(z,30000,seed=12)
    np.testing.assert_allclose(errors[0],[.25,.25],atol=.01)
    np.testing.assert_allclose(errors[1,0],0,atol=0)
    np.testing.assert_allclose(errors[2,1],0,atol=0)
    np.testing.assert_allclose(errors[-1],0,atol=0)
    small,selected=paired_scope_errors(z,100,seed=12,candidate_masks=[3,7])
    assert small.shape==(2,2) and selected.tolist()==[3,7]
    with pytest.raises(ValueError):paired_scope_errors(z,0)
    with pytest.raises(ValueError):paired_scope_errors(z,10,candidate_masks=[1,1])
