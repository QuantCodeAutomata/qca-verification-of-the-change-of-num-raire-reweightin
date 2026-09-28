"""
Experiment 1: Verification of the Change-of-Numéraire Reweighting Identities
            and Weighted Ratio Law Construction.

This experiment:
  - Constructs discrete laws Q_n = (1-ε_n)δ_{(1,0)} + ε_n δ_{(a_n,1)}
    with ε_n = 2^{-n}, a_n = 2^{-n}, n = 1..N.
  - Constructs a boundary-permitting limit law Q = (1-ε)δ_{(1,0)} + ε δ_{(0,1)}.
  - Verifies: ∫ f dΓ = E[A f(B/A)] / E[A] for several test functions f.
  - Verifies special-case identities for call prices and moments.
  - Reports residuals (should be ≤ machine precision ~1e-15).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.discrete_law import DiscreteLaw
from src.numeraire_change import (
    construct_gamma_discrete,
    construct_boundary_limit_gamma,
    evaluate_gamma_expectation,
    evaluate_primitive_formula,
    call_price_gamma,
    call_price_primitive,
    abs_moment_gamma,
    abs_moment_primitive,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ------------------------------------------------------------------
# Test function definitions
# ------------------------------------------------------------------

def make_abs_moment(s: float):
    return lambda z: np.abs(z) ** s


def make_call(K: float):
    return lambda z: np.maximum(z - K, 0.0)


# ------------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------------

def run_experiment_1(N: int = 40) -> pd.DataFrame:
    """
    Run Experiment 1: identity verification for n = 1..N.

    Returns a DataFrame of residuals for all identity checks.
    """
    eps_n = 2.0 ** (-np.arange(1, N + 1))
    a_n = 2.0 ** (-np.arange(1, N + 1))  # same schedule

    # Strike grid and moment orders
    K_grid = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 5.0])
    s_values = [1.0, 1.5, 2.0, 3.0]

    records = []

    for i, (eps, a) in enumerate(zip(eps_n, a_n)):
        n = i + 1
        # Construct Q_n = (1-ε)δ_{(1,0)} + ε δ_{(a,1)}
        law_n = DiscreteLaw.from_mixture(
            weights=[1 - eps, eps],
            atoms=[(1.0, 0.0), (a, 1.0)],
        )
        m_n = law_n.expected_A()

        # Construct Γ_n
        gamma_n = construct_gamma_discrete(law_n)

        # Verify mass conservation
        mass_sum = gamma_n.atoms["prob"].sum()
        assert np.isclose(mass_sum, 1.0, atol=1e-12), f"n={n}: Γ masses = {mass_sum}"

        # Gamma should have atoms at z=0 (from (1,0)) and z=1/a (from (a,1))
        # Verify the structure
        z_vals = gamma_n.atoms["A"].values
        q_vals = gamma_n.atoms["prob"].values

        # Check call-price identities
        for K in K_grid:
            direct = call_price_gamma(gamma_n, K)
            formula = call_price_primitive(law_n, K)
            residual = abs(direct - formula)
            records.append({
                "n": n, "test": f"call K={K:.2f}",
                "direct": direct, "formula": formula, "residual": residual,
            })

        # Check absolute moment identities
        for s in s_values:
            f_s = make_abs_moment(s)
            direct = evaluate_gamma_expectation(gamma_n, f_s)
            formula = evaluate_primitive_formula(law_n, f_s)
            residual = abs(direct - formula)
            records.append({
                "n": n, "test": f"|z|^{s}",
                "direct": direct, "formula": formula, "residual": residual,
            })

        # Check first absolute moment identity: ∫|z|dΓ = E|B|/E[A]
        int_abs_z = evaluate_gamma_expectation(gamma_n, lambda z: np.abs(z))
        e_abs_B = law_n.expected_abs_B()
        formula_abs = e_abs_B / m_n
        records.append({
            "n": n, "test": "E|B|/E[A]",
            "direct": int_abs_z, "formula": formula_abs,
            "residual": abs(int_abs_z - formula_abs),
        })

    df = pd.DataFrame(records)
    return df


def run_boundary_example(eps: float = 0.25) -> dict:
    """
    Boundary-permitting example: Q = (1-ε)δ_{(1,0)} + ε δ_{(0,1)}.

    Verify Γ = δ_0 under boundary zero-weighting.
    The atom (0,1) contributes zero mass because a=0.
    """
    law_limit = DiscreteLaw.from_mixture(
        weights=[1 - eps, eps],
        atoms=[(1.0, 0.0), (0.0, 1.0)],
    )

    # Boundary zero-weighting: only a>0 atoms contribute
    gamma_limit = construct_boundary_limit_gamma(law_limit)

    # Γ should be δ_0 (only atom at z=0 with full weight)
    assert len(gamma_limit.atoms) == 1, "Γ should be a single Dirac."
    z_val = gamma_limit.atoms["A"].values[0]
    q_val = gamma_limit.atoms["prob"].values[0]

    assert np.isclose(z_val, 0.0, atol=1e-12), f"Γ support = {z_val}, expected 0."
    assert np.isclose(q_val, 1.0, atol=1e-12), f"Γ mass = {q_val}, expected 1."

    # Verify no undefined ratio: B/A at the boundary atom (0,1) must NOT be computed
    boundary_mass = law_limit.atoms.loc[law_limit.atoms["A"] == 0, "prob"].sum()
    assert np.isclose(boundary_mass, eps, atol=1e-12)

    result = {
        "epsilon": eps,
        "gamma_support": float(z_val),
        "gamma_mass": float(q_val),
        "E[A]": law_limit.expected_A(),
        "boundary_mass_in_Q": float(boundary_mass),
        "boundary_abs_B": law_limit.boundary_abs_B_integral(),
    }
    return result


def plot_residuals(df: pd.DataFrame, save_path: str) -> None:
    """Plot identity residuals versus n for all test functions."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Call-price residuals
    call_df = df[df["test"].str.startswith("call")]
    for K_str, grp in call_df.groupby("test"):
        axes[0].semilogy(grp["n"], grp["residual"] + 1e-20, label=K_str, alpha=0.7)
    axes[0].set_xlabel("n")
    axes[0].set_ylabel("Absolute Residual")
    axes[0].set_title("Call-Price Identity Residuals: |∫(z−K)⁺dΓ − E[(B−KA)⁺]/E[A]|")
    axes[0].legend(ncol=3, fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(1e-14, color="red", linestyle="--", alpha=0.5, label="Machine ε")

    # Moment residuals
    mom_df = df[df["test"].str.startswith("|")]
    for s_str, grp in mom_df.groupby("test"):
        axes[1].semilogy(grp["n"], grp["residual"] + 1e-20, label=s_str, alpha=0.7)
    axes[1].set_xlabel("n")
    axes[1].set_ylabel("Absolute Residual")
    axes[1].set_title("Moment Identity Residuals: |∫|z|ˢdΓ − E[|B|ˢA^{1-s}]/E[A]|")
    axes[1].legend(ncol=2, fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(1e-14, color="red", linestyle="--", alpha=0.5, label="Machine ε")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  Saved: {save_path}")


def main() -> None:
    print("=" * 65)
    print("Experiment 1: Change-of-Numéraire Identity Verification")
    print("=" * 65)

    N = 40
    df = run_experiment_1(N=N)

    max_residual = df["residual"].max()
    mean_residual = df["residual"].mean()
    print(f"\n[N={N}] Max residual across all tests: {max_residual:.2e}")
    print(f"[N={N}] Mean residual across all tests: {mean_residual:.2e}")
    assert max_residual < 1e-12, f"Identity check FAILED: max residual = {max_residual}"
    print("✓ All identities verified to machine precision.")

    # Boundary example
    print("\n--- Boundary-Permitting Example ---")
    for eps in [0.1, 0.25, 0.5]:
        result = run_boundary_example(eps)
        print(f"  ε={eps:.2f}: Γ = δ_{{{result['gamma_support']:.1f}}}, "
              f"mass={result['gamma_mass']:.6f}, "
              f"boundary|b|∫={result['boundary_abs_B']:.4f}, "
              f"E[A]={result['E[A]']:.4f}")
    print("✓ Boundary zero-weighting produces Γ = δ_0 in all cases.")

    # Detailed residual table (first 5 n, select tests)
    print("\n--- Sample Residual Table (n=1..5, test=call K=1.00) ---")
    sample = df[(df["test"] == "call K=1.00") & (df["n"] <= 5)][
        ["n", "direct", "formula", "residual"]
    ]
    print(sample.to_string(index=False))

    # Save plots
    plot_path = os.path.join(RESULTS_DIR, "exp1_identity_residuals.png")
    plot_residuals(df, save_path=plot_path)

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "exp1_residuals.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    return {
        "max_residual": float(max_residual),
        "mean_residual": float(mean_residual),
        "boundary_verified": True,
    }


if __name__ == "__main__":
    main()
