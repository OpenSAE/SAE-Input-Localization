# Localize Python API

Localize measures feature activation dependence for a **frozen SAE** on a controlled input design. The core package needs NumPy only.

## Input arrays

The exact implementation takes `z` with shape `(2**n_factors, n_features)`. Row `i` corresponds to the binary input configuration represented by integer `i`, with factor 0 in its least-significant bit. All configurations have equal weight.

```python
from sae_localize import exact_scope_curve
result = exact_scope_curve(z, n_factors=8)
```

## Optimal scope curves

`exact_scope_curve` returns a `ScopeCurveResult`:

| Attribute or method | Meaning |
|---|---|
| `curve[B]` | Minimum omitted activation variance at dependency budget `B` |
| `normalized_curve` | `curve / total_variance` |
| `errors[mask, feature]` | Omitted variance for one candidate feature scope |
| `total_variance` | Sum of SAE activation variances, $W_z$ |
| `budget(tolerance)` | Smallest budget with omitted variance at most `tolerance` |
| `assignment(B)` | Optimal input scopes as integer bitmasks |

`budget` takes an **absolute variance tolerance**. To explain 90% of the activation variance, use `result.budget(0.1 * result.total_variance)`.

The implementation uses Boolean Fourier energies to calculate errors for every scope, followed by dynamic programming over total input-feature cost. Computational cost increases exponentially with the number of controlled input factors.

## Paired resampling

```python
from sae_localize import paired_scope_errors, estimated_scope_curve
errors, masks = paired_scope_errors(z, pairs_per_scope=512, seed=0)
curve = estimated_scope_curve(z, pairs_per_scope=512, seed=0)
```

`paired_scope_errors` estimates $\mathbb E[(z_j(x)-z_j(x_A,x'_{-A}))^2]/2$. Its sample size is the number of **pairs per candidate scope**; a shared set of base pairs is used across candidate scopes. `estimated_scope_curve` optimizes these estimates.

## Theoretical bounds

```python
from sae_localize import (
    spectral_budget_lower_bound,
    feature_order_lower_curve,
    compatibility_certificate,
)
```

- `spectral_budget_lower_bound(h, decoder, reconstruction_error, omission_tolerance)` computes the representation-level bound, using absolute squared reconstruction and omission errors.
- `feature_order_lower_curve(z)` computes the activation-level, feature-order relaxation reported in the appendix.
- `compatibility_certificate(h, z, decoder, bias, assignment)` checks the compatibility inequality for a specified scope assignment.

See `tests/test_exact.py` and `tests/test_reference.py` for runnable examples and numerical checks.
