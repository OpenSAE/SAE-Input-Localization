"""Bounds and diagnostics from the Localize paper and appendix.

All guarantees are conditional on the declared full product input design,
fixed decoder and reconstruction residual, and exact finite population
quantities. Numerical rank choices are diagnostic, not a substitute for
specifying an appropriate tolerance when D is ill conditioned.
"""
from __future__ import annotations
import numpy as np
from .exact import boolean_fourier, validate_activation_grid, optimize_scopes


def spectral_budget_lower_bound(h: np.ndarray, decoder: np.ndarray, reconstruction_error: float,
                                 omission_tolerance: float) -> int:
    """The paper's representation-level lower bound on B_z(tau).

    ``h`` has shape [2**n, d], ``decoder`` shape [d,m]; R and tau are
    squared L2 population quantities. This is *not* an estimator and does
    not supply a tight curve in the learned experiment.
    """
    h, n = validate_activation_grid(h)
    D = np.asarray(decoder, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != h.shape[1] or D.shape[1] < 1 or not np.isfinite(D).all():
        raise ValueError("decoder shape must be [representation_dim, n_features]")
    if not np.isfinite(reconstruction_error) or reconstruction_error < 0 or not np.isfinite(omission_tolerance) or omission_tolerance < 0:
        raise ValueError("R and omission tolerance must be nonnegative finite values")
    coeff = boolean_fourier(h)
    card = np.array([i.bit_count() for i in range(1<<n)])
    threshold = (np.sqrt(reconstruction_error) + np.linalg.norm(D, 2)*np.sqrt(omission_tolerance))**2
    lower = 0
    for r in range(1,n+1):
        mat = coeff[card >= r]
        eig = np.linalg.eigvalsh(mat.T @ mat)[::-1].clip(min=0)
        tail = np.maximum(np.sum(eig) - np.concatenate(([0.0],np.cumsum(eig))),0)
        possible = np.flatnonzero(tail <= threshold + 1e-12)
        lower += int(possible[0]) if len(possible) else len(eig)
    return lower


def feature_order_lower_curve(z: np.ndarray) -> np.ndarray:
    """Appendix-only feature-order relaxation using SAE activation ANOVA energies.

    This is a *different*, more information-rich lower bound, not a
    replacement for the representation-level spectral theorem.
    """
    z, n = validate_activation_grid(z)
    coeff = boolean_fourier(z)
    card = np.array([i.bit_count() for i in range(1<<n)])
    costs = np.array([(coeff[card > k]**2).sum(axis=0) for k in range(n+1)])
    return optimize_scopes(costs[card])[0]


def compatibility_certificate(h: np.ndarray, z: np.ndarray, decoder: np.ndarray,
                              bias: np.ndarray, assignment: np.ndarray) -> dict[str,float]:
    """Evaluate the compatible-direction lower bound for a declared feature scope.

    Retains the numerical SVD rank rule of the paper's original audit code.
    Returns reconstruction error R, feature omission Gamma, decoded omission
    Lambda, compatibility obstruction Q, and lower bound/slack diagnostics.
    """
    h, n = validate_activation_grid(h)
    z, zn = validate_activation_grid(z, n)
    D = np.asarray(decoder, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    A = np.asarray(assignment)
    if D.shape != (h.shape[1],z.shape[1]) or bias.shape != (h.shape[1],):
        raise ValueError("decoder / bias incompatible with h and z")
    if A.shape != (z.shape[1],) or not np.issubdtype(A.dtype,np.integer) or np.any(A<0) or np.any(A>=len(z)):
        raise ValueError("assignment must be valid integer factor masks per feature")
    hc, zc = boolean_fourier(h), boolean_fourier(z)
    masks = np.arange(len(h))
    keep = (masks[:,None] & A[None,:]) == masks[:,None]
    omit = zc * (~keep)
    gamma = float(np.sum(omit**2))
    decoded = float(np.sum((omit @ D.T)**2))
    R = float(np.mean(np.sum((h-z @ D.T - bias)**2,axis=1)))
    Q = 0.0
    for t in range(1,len(h)):
        columns = D[:,keep[t]]
        effect = hc[t]
        if columns.shape[1]:
            U,s,_ = np.linalg.svd(columns,full_matrices=False)
            rank = int((s>s[0]*1e-10).sum())
            diff = effect-U[:,:rank] @ (U[:,:rank].T@effect)
        else:
            diff=effect
        Q+=float(diff@diff)
    opnorm=float(np.linalg.norm(D,2))
    lower=(max(np.sqrt(Q)-np.sqrt(R),0.0)/opnorm)**2 if opnorm>0 else 0.0
    slack=float(np.sqrt(R)+np.sqrt(decoded)-np.sqrt(Q))
    return {'R':R,'Gamma':gamma,'Lambda':decoded,'Q':Q,
            'compatibility_lower_Gamma':lower,'compatibility_slack':slack}
