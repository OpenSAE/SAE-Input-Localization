"""Input localization for frozen sparse autoencoder feature activations."""
from .exact import (ScopeCurveResult, boolean_fourier, direct_omission_error,
                    exact_scope_curve, omission_errors, optimize_scopes,
                    recover_assignment, scope_assignment, scope_budget,
                    validate_activation_grid)
from .paired import estimated_scope_curve, paired_scope_errors
from .certificates import (compatibility_certificate, feature_order_lower_curve,
                           spectral_budget_lower_bound)

__version__ = '0.1.0'
__all__ = [
    'ScopeCurveResult', 'boolean_fourier', 'direct_omission_error',
    'exact_scope_curve', 'omission_errors', 'optimize_scopes',
    'recover_assignment', 'scope_assignment', 'scope_budget',
    'validate_activation_grid', 'paired_scope_errors', 'estimated_scope_curve',
    'compatibility_certificate','feature_order_lower_curve',
    'spectral_budget_lower_bound',
]
