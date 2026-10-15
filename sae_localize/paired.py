"""Finite paired-intervention estimators on uniform binary product designs.

Each requested scope is estimated from N *pairs per candidate scope*.
Using shared base pairs across masks saves random sampling but still needs
hybrid evaluations for each candidate; the reference implementation uses
an already-cached finite population, not new neural-network forward calls.
"""
from __future__ import annotations
import numpy as np
from .exact import validate_activation_grid, optimize_scopes


def paired_scope_errors(
    z: np.ndarray,
    pairs_per_scope: int,
    *,
    seed: int = 0,
    candidate_masks: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate E[(z(X)-z(X_A,X'_{-A}))^2]/2 for candidate scopes.

    Returns ``(estimated_errors, candidate_masks)``. Rows correspond in order
    to returned masks, and columns to features. Both independent base draws
    are uniformly distributed on the full controlled Boolean grid.
    """
    z, n = validate_activation_grid(z)
    if isinstance(pairs_per_scope, bool) or not isinstance(pairs_per_scope, (int,np.integer)) or pairs_per_scope < 1:
        raise ValueError("pairs_per_scope must be a positive integer")
    count, features = z.shape
    if candidate_masks is None:
        masks = np.arange(count, dtype=np.int64)
    else:
        masks = np.asarray(candidate_masks)
        if masks.ndim != 1 or masks.size == 0 or not np.issubdtype(masks.dtype, np.integer):
            raise ValueError("candidate_masks must be a nonempty integer vector")
        if np.any(masks < 0) or np.any(masks >= count) or len(np.unique(masks)) != len(masks):
            raise ValueError("candidate_masks must be unique, valid Boolean factor masks")
        masks = masks.astype(np.int64)
    rng = np.random.default_rng(seed)
    x = rng.integers(count, size=pairs_per_scope)
    xp = rng.integers(count, size=pairs_per_scope)
    zx = z[x]
    result = np.empty((len(masks), features), dtype=np.float64)
    for idx, mask in enumerate(masks):
        hybrid = (x & int(mask)) | (xp & (count-1-int(mask)))
        result[idx] = (0.5*(zx-z[hybrid])**2).mean(axis=0)
    return result, masks


def estimated_scope_curve(z: np.ndarray, pairs_per_scope: int, *, seed: int = 0) -> np.ndarray:
    """Optimize the *plug-in estimated* curve on all scopes; not a certificate.

    Use independent validation / simultaneous bounds to justify
    post-selection statements. The curve may lie below the true curve.
    """
    estimated, _ = paired_scope_errors(z, pairs_per_scope, seed=seed)
    return optimize_scopes(estimated)[0]
