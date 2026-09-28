"""
Change-of-numéraire reweighting operator Γ(Q).

Given a bivariate discrete law Q on [0,∞) × R, the operator Γ is defined as
the law of Z = B/A under the weighted measure (A / E[A]) dQ.

For states with A = 0, boundary zero-weighting applies: those states
contribute mass 0 after multiplication by A, so they are excluded from the
transformed support.

Reference: paper's foundational identity
    ∫ f dΓ = E[A f(B/A)] / E[A]   for non-negative measurable f.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable

from .discrete_law import DiscreteLaw


# ------------------------------------------------------------------
# Core expectation helpers
# ------------------------------------------------------------------


def expected_A(law: DiscreteLaw) -> float:
    """Return E[A] for a bivariate discrete law."""
    return law.expected_A()


def construct_gamma_discrete(law: DiscreteLaw) -> DiscreteLaw:
    """
    Construct the weighted ratio law Γ(Q) from a discrete bivariate law Q.

    Algorithm
    ---------
    1. Filter to atoms with a_i > 0 (boundary zero-weighting).
    2. Compute ratio z_i = b_i / a_i.
    3. Assign weighted mass q_i = p_i * a_i / m  where m = E[A].
    4. Aggregate masses at identical z values.
    5. Return as a 1-D DiscreteLaw with A=z and B=1 (z stored in A column).

    Returns a 1-D DiscreteLaw on (0,∞) with support stored in the 'A' column
    (representing z values) and 'B' = 1.0.  The 'A' column here carries the
    support of Γ, and the user should interpret it as such.
    """
    m = law.expected_A()
    if m <= 0:
        raise ValueError("E[A] must be strictly positive to construct Γ.")

    atoms = law.atoms
    mask = atoms["A"] > 0
    sub = atoms[mask].copy()

    if sub.empty:
        raise ValueError("All atoms have A=0; Γ is not well-defined.")

    z = sub["B"].values / sub["A"].values
    q = sub["prob"].values * sub["A"].values / m

    df = pd.DataFrame({"z": z, "q": q})
    # Aggregate at identical z values
    df = df.groupby("z", sort=True)["q"].sum().reset_index()
    df.columns = ["z", "q"]

    # Verify mass conservation
    total = df["q"].sum()
    assert np.isclose(total, 1.0, atol=1e-10), f"Γ masses sum to {total}, not 1."

    result_df = pd.DataFrame({"prob": df["q"], "A": df["z"], "B": 1.0})
    return DiscreteLaw(result_df)


def construct_boundary_limit_gamma(law: DiscreteLaw) -> DiscreteLaw:
    """
    Construct Γ for a law that may charge {A=0}, using boundary zero-weighting.

    States with A=0 are excluded: they contribute zero mass to Γ because the
    weighting factor A multiplies out.  Only states with A > 0 are used.

    This implements the boundary-permitting definition:
        ∫ f dΓ = (1/m) ∫_{a>0} a f(b/a) Q(da,db),   m = ∫ a dQ > 0.
    """
    m = law.expected_A()
    if m <= 0:
        raise ValueError("E[A] = ∫ a dQ must be strictly positive.")

    atoms = law.atoms
    mask = atoms["A"] > 0
    sub = atoms[mask].copy()

    if sub.empty:
        # All mass is at A=0; m=0 would have been caught above, so this
        # shouldn't happen after the guard, but handle gracefully.
        raise ValueError("No atoms with A>0; Γ is degenerate.")

    z = sub["B"].values / sub["A"].values
    q = sub["prob"].values * sub["A"].values / m

    df = pd.DataFrame({"z": z, "q": q})
    df = df.groupby("z", sort=True)["q"].sum().reset_index()
    df.columns = ["z", "q"]

    total = df["q"].sum()
    assert np.isclose(total, 1.0, atol=1e-10), f"Boundary Γ masses sum to {total}, not 1."

    result_df = pd.DataFrame({"prob": df["q"], "A": df["z"], "B": 1.0})
    return DiscreteLaw(result_df)


# ------------------------------------------------------------------
# Expectation evaluators
# ------------------------------------------------------------------


def evaluate_gamma_expectation(gamma: DiscreteLaw, f: Callable[[np.ndarray], np.ndarray]) -> float:
    """
    Compute ∫ f dΓ directly from the transformed law Γ.

    Parameters
    ----------
    gamma : DiscreteLaw  – the transformed law (support in 'A' column)
    f     : callable     – a non-negative measurable function f(z)

    Returns
    -------
    float : Σ_j q_j f(z_j)
    """
    z = gamma.atoms["A"].values
    q = gamma.atoms["prob"].values
    fz = np.asarray(f(z), dtype=float)
    return float(np.dot(q, fz))


def evaluate_primitive_formula(law: DiscreteLaw, f: Callable[[np.ndarray], np.ndarray]) -> float:
    """
    Compute E[A f(B/A)] / E[A] directly from the primitive law Q.

    States with A = 0 are excluded (boundary zero-weighting).

    Parameters
    ----------
    law : DiscreteLaw – primitive bivariate law Q
    f   : callable    – a non-negative measurable function f(z)

    Returns
    -------
    float : (Σ_{a_i>0} p_i a_i f(b_i/a_i)) / m
    """
    m = law.expected_A()
    if m <= 0:
        raise ValueError("E[A] must be strictly positive.")

    atoms = law.atoms
    mask = atoms["A"] > 0
    sub = atoms[mask]

    if sub.empty:
        return 0.0

    a = sub["A"].values
    b = sub["B"].values
    p = sub["prob"].values
    z = b / a
    fz = np.asarray(f(z), dtype=float)
    return float(np.dot(p * a, fz) / m)


# ------------------------------------------------------------------
# Special-case identity evaluators
# ------------------------------------------------------------------


def call_price_gamma(gamma: DiscreteLaw, K: float) -> float:
    """
    Compute ∫ (z - K)^+ dΓ from the transformed law.

    Identity: ∫ (z-K)^+ dΓ = E[(B - K A)^+] / E[A]
    """
    return evaluate_gamma_expectation(gamma, lambda z: np.maximum(z - K, 0.0))


def call_price_primitive(law: DiscreteLaw, K: float) -> float:
    """
    Compute E[(B - K A)^+] / E[A] from the primitive law Q.

    Identity: = ∫ (z-K)^+ dΓ
    """
    m = law.expected_A()
    atoms = law.atoms
    payoff = np.maximum(atoms["B"].values - K * atoms["A"].values, 0.0)
    return float(np.dot(atoms["prob"].values, payoff) / m)


def abs_moment_gamma(gamma: DiscreteLaw, s: float) -> float:
    """
    Compute ∫ |z|^s dΓ from the transformed law.

    Identity: = E[|B|^s A^{1-s}] / E[A]
    """
    return evaluate_gamma_expectation(gamma, lambda z: np.abs(z) ** s)


def abs_moment_primitive(law: DiscreteLaw, s: float) -> float:
    """
    Compute E[|B|^s A^{1-s}] / E[A] from the primitive law Q.

    Identity: = ∫ |z|^s dΓ
    """
    return evaluate_primitive_formula(law, lambda z: np.abs(z) ** s)
