"""
Change-of-Numéraire Reweighting — core library.

Modules:
  discrete_law     – DiscreteLaw: bivariate discrete probability law on [0,∞)×R
  numeraire_change – Γ operator, primitive-law identities
  wasserstein      – Wasserstein distances and ratio-tail diagnostics
"""
from .discrete_law import DiscreteLaw
from .numeraire_change import (
    construct_gamma_discrete,
    construct_boundary_limit_gamma,
    evaluate_gamma_expectation,
    evaluate_primitive_formula,
    call_price_gamma,
    call_price_primitive,
    abs_moment_gamma,
    abs_moment_primitive,
    expected_A,
)
from .wasserstein import (
    wasserstein_p,
    wasserstein_p_dirac,
    compute_mixed_moment,
    compute_ratio_tail,
    compute_tail_envelope,
)

__all__ = [
    "DiscreteLaw",
    "construct_gamma_discrete",
    "construct_boundary_limit_gamma",
    "evaluate_gamma_expectation",
    "evaluate_primitive_formula",
    "call_price_gamma",
    "call_price_primitive",
    "abs_moment_gamma",
    "abs_moment_primitive",
    "expected_A",
    "wasserstein_p",
    "wasserstein_p_dirac",
    "compute_mixed_moment",
    "compute_ratio_tail",
    "compute_tail_envelope",
]
