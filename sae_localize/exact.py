"""Exact optimal input-scope curves on complete uniform Boolean factorial grids.

The rows of ``z`` are ordered by the integer whose binary digits encode
controlled factors (factor zero is the least significant bit). Each row is
one equally weighted configuration of independently controlled binary inputs.
The feature system is frozen; the procedure changes only declared scopes.

The core algorithms are factored from the original paper's ``pilot.py``.
They retain its subset Fourier transform and exact dynamic program.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ScopeCurveResult:
    """Exact scope-curve analysis of a frozen activation table.

    ``curve[b]`` is the minimum omitted activation variance among all
    feature scope assignments with total input-feature cost at most ``b``.
    ``errors[mask, j]`` is omitted variance for feature ``j`` assigned
    precisely the input variables whose bits occur in ``mask``.
    """

    curve: np.ndarray
    errors: np.ndarray
    coefficients: np.ndarray
    choices: np.ndarray
    best_masks: np.ndarray
    total_variance: float
    n_factors: int
    n_features: int

    @property
    def normalized_curve(self) -> np.ndarray:
        """Variance-normalized curve (undefined for constant feature tables)."""
        if self.total_variance <= 0:
            raise ValueError("Cannot normalize when all SAE features are constant.")
        return self.curve / self.total_variance

    def budget(self, tolerance: float) -> int:
        """Smallest cost with omitted variance at most an *absolute* tolerance."""
        return scope_budget(self, tolerance)

    def assignment(self, budget: int) -> np.ndarray:
        """Return one optimal scope per feature as integer bit masks."""
        return scope_assignment(self, budget)


def validate_activation_grid(z: np.ndarray, n_factors: int | None = None) -> tuple[np.ndarray, int]:
    """Validate [2**n_factors, features] observations in Boolean bit-mask order."""
    arr = np.asarray(z, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 1 or arr.shape[0] < 2:
        raise ValueError("Expected nonempty 2-D activation table [2**n_factors, n_features].")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Activation table contains a nonfinite number.")
    states = arr.shape[0]
    if states & (states - 1):
        raise ValueError("Uniform Boolean grid must have exactly 2**n_factors rows.")
    actual = states.bit_length() - 1
    if n_factors is not None and (isinstance(n_factors, bool) or n_factors != actual):
        raise ValueError(f"Expected {2**n_factors} grid rows for n_factors={n_factors}; got {states}.")
    return arr, actual


def boolean_fourier(z: np.ndarray, n_factors: int | None = None) -> np.ndarray:
    """Walsh/Fourier coefficients indexed by Boolean input subset masks.

    For f(x) with x_i in {0,1}, character mask T is (-1)**sum(x_i for i in T).
    The coefficient of mask zero is the population mean.
    """
    out, _ = validate_activation_grid(z, n_factors)
    out = out.copy()
    count = len(out)
    size = 1
    while size < count:
        view = out.reshape(-1, 2 * size, out.shape[1])
        a = view[:, :size].copy()
        b = view[:, size:].copy()
        view[:, :size] = a + b
        view[:, size:] = a - b
        size *= 2
    return out / count


def omission_errors(z: np.ndarray, n_factors: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Exact minimum L2 omission errors for every feature and Boolean input scope.

    Uses ANOVA/Fourier energy: sum the energies of every nonconstant term
    whose support is *not* contained in the candidate scope.
    """
    coeff = boolean_fourier(z, n_factors)
    energy = coeff**2
    captured = energy.copy()
    count, _ = coeff.shape
    n = count.bit_length() - 1
    for bit in range(n):
        for mask in range(count):
            if mask & (1 << bit):
                captured[mask] += captured[mask ^ (1 << bit)]
    err = np.maximum(energy.sum(axis=0)[None, :] - captured, 0.0)
    err[-1] = 0.0
    return err, coeff


def optimize_scopes(errors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact dynamic program over all feature-scope choices (at most B cost).

    ``errors`` has shape [2**n_factors, n_features]. The returned
    ``(curve, choices, best_masks)`` support optimal scope backtracking.
    This algorithm scales exponentially in input factor count; for large n,
    predeclare a restricted candidate scope family rather than claiming an
    unrestricted global optimum.
    """
    err = np.asarray(errors, dtype=np.float64)
    if err.ndim != 2 or err.shape[1] < 1 or err.shape[0] < 2:
        raise ValueError("errors must have shape [2**n_factors, n_features].")
    count, features = err.shape
    if count & (count - 1) or not np.all(np.isfinite(err)) or np.any(err < -1e-12):
        raise ValueError("errors must be finite, nonnegative and have power-of-two scope rows.")
    n = count.bit_length() - 1
    card = np.array([int(i).bit_count() for i in range(count)])
    best = np.empty((features, n + 1), dtype=np.float64)
    best_mask = np.empty((features, n + 1), dtype=np.int64)
    for k in range(n + 1):
        masks = np.flatnonzero(card <= k)
        arg = np.argmin(err[masks], axis=0)
        best_mask[:, k] = masks[arg]
        best[:, k] = err[masks[arg], np.arange(features)]
    max_b = features*n
    dp = np.zeros(max_b + 1, dtype=np.float64)
    choices = np.zeros((features, max_b + 1), dtype=np.int64)
    for j in range(features):
        candidates = np.full((n+1, max_b+1), np.inf)
        for k in range(n+1):
            candidates[k, k:] = dp[:max_b+1-k]+best[j,k]
        choices[j] = np.argmin(candidates, axis=0)
        dp = np.min(candidates, axis=0)
    return dp, choices, best_mask


def exact_scope_curve(z: np.ndarray, n_factors: int | None = None) -> ScopeCurveResult:
    """Compute the global optimal curve from *all* controlled input configurations.

    The intervention distribution must be the uniform product distribution;
    array rows must be complete and ordered by Boolean bitmask. Accepts
    frozen SAE codes or any fixed multivariate scalar feature table.
    """
    arr, n = validate_activation_grid(z, n_factors)
    errors, coeff = omission_errors(arr, n)
    curve, choices, best_masks = optimize_scopes(errors)
    total = float(np.sum(np.var(arr, axis=0)))
    return ScopeCurveResult(curve, errors, coeff, choices, best_masks, total, n, arr.shape[1])


def scope_budget(result: ScopeCurveResult, tolerance: float) -> int:
    """Minimum total input-feature incidence budget B_z(tolerance).

    ``tolerance`` is *absolute omitted variance*. To permit 10% omitted
    activation variance, pass ``0.1 * result.total_variance``.
    """
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be nonnegative and finite")
    choices = np.flatnonzero(result.curve <= tolerance + 1e-12)
    if choices.size == 0:
        raise ValueError("Tolerance is lower than the numerically attainable curve minimum.")
    return int(choices[0])


def recover_assignment(budget: int, choices: np.ndarray, best_mask: np.ndarray) -> np.ndarray:
    """Backtrack one optimal assignment (integer bitmask per SAE feature)."""
    if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)):
        raise ValueError("budget must be an integer")
    if choices.ndim != 2 or best_mask.ndim != 2 or len(choices) != len(best_mask):
        raise ValueError("choices and best_mask must have compatible shapes")
    if budget < 0 or budget >= choices.shape[1]:
        raise ValueError("budget exceeds the total number of input-feature incidences")
    scopes = np.zeros(len(choices), dtype=np.int64)
    for j in range(len(choices)-1, -1, -1):
        k = int(choices[j, budget])
        scopes[j] = best_mask[j, k]
        budget -= k
    return scopes


def scope_assignment(result: ScopeCurveResult, budget: int) -> np.ndarray:
    """Return scope bitmask for each feature under a specified total budget."""
    return recover_assignment(budget, result.choices, result.best_masks)


def direct_omission_error(z: np.ndarray, mask: int) -> np.ndarray:
    """Independent conditional-mean reference for one scope on a complete grid."""
    arr, n = validate_activation_grid(z)
    if isinstance(mask, bool) or not isinstance(mask, (int,np.integer)) or mask < 0 or mask >= (1<<n):
        raise ValueError("mask must be a valid Boolean factor bitmask")
    cells = np.arange(len(arr)) & mask
    residual = np.empty_like(arr)
    for cell in np.unique(cells):
        inside = cells == cell
        residual[inside] = arr[inside] - arr[inside].mean(axis=0)
    return (residual**2).mean(axis=0)
