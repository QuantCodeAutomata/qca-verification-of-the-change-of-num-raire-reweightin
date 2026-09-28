"""
Experiment 3: First-Order Wasserstein Stability and Boundary Obstruction at A=0.

Q_n = (1-ε)δ_{(1,0)} + ε δ_{(α_n,1)},  α_n = 2^{-n},  n = 1..N.
Weak limit: Q = (1-ε)δ_{(1,0)} + ε δ_{(0,1)}.

Key results:
  - Γ_n ⇒ δ_0 weakly (weak convergence holds).
  - W_1(Γ_n, δ_0) → ε/(1-ε) ≠ 0  (W_1-stability fails).
  - Obstruction: ∫_{a=0}|b| dQ = ε.
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
)
from src.wasserstein import wasserstein_p_dirac

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_experiment_3(eps: float = 0.25, N: int = 60) -> pd.DataFrame:
    """
    Run Experiment 3 for a given ε ∈ (0,1).

    Returns a DataFrame recording atom weights, first moments, and W_1 values.
    """
    alpha_n = 2.0 ** (-np.arange(1, N + 1))
    n_arr = np.arange(1, N + 1)

    records = []

    for n, alpha in zip(n_arr, alpha_n):
        # Construct Q_n = (1-ε)δ_{(1,0)} + ε δ_{(α_n,1)}
        law_n = DiscreteLaw.from_mixture(
            weights=[1 - eps, eps],
            atoms=[(1.0, 0.0), (alpha, 1.0)],
        )
        m_n = law_n.expected_A()  # = 1 - ε + ε α_n

        # Construct Γ_n
        gamma_n = construct_gamma_discrete(law_n)

        # Masses in Γ_n:
        #  mass at z=0: (1-ε)/m_n
        #  mass at z=1/α_n: ε α_n / m_n
        mass_at_0 = (1 - eps) / m_n
        mass_at_inf_atom = eps * alpha / m_n

        # First moment of Γ_n (exact formula):
        # ∫|z| dΓ_n = (ε α_n / m_n) * (1/α_n) = ε / m_n
        first_moment_exact = eps / m_n
        first_moment_direct = float(np.dot(
            gamma_n.atoms["prob"].values,
            np.abs(gamma_n.atoms["A"].values),
        ))

        # W_1(Γ_n, δ_0) = ∫|z-0| dΓ_n = first moment (since target is δ_0)
        W1_direct = wasserstein_p_dirac(gamma_n, c=0.0, p=1.0)

        assert np.isclose(first_moment_exact, first_moment_direct, atol=1e-12), \
            f"n={n}: moment mismatch {first_moment_exact} vs {first_moment_direct}"
        assert np.isclose(W1_direct, first_moment_direct, atol=1e-12), \
            f"n={n}: W1 mismatch"

        records.append({
            "n": n,
            "alpha_n": alpha,
            "m_n": m_n,
            "mass_at_0": mass_at_0,
            "mass_at_1_over_alpha": mass_at_inf_atom,
            "ratio_1_over_alpha": 1.0 / alpha,
            "first_moment": first_moment_direct,
            "W1_Gamma_n_delta0": W1_direct,
        })

    return pd.DataFrame(records)


def compute_limit_values(eps: float) -> dict:
    """
    Compute analytical limit quantities for ε given.

    Limit: W_1(Γ_n, δ_0) → ε / (1-ε)   (does NOT vanish)
    Boundary residue: ∫_{a=0}|b| dQ = ε
    Correct limit law: Γ = δ_0 (boundary zero-weighting)
    """
    W1_limit = eps / (1 - eps)

    # Construct the limit law Q = (1-ε)δ_{(1,0)} + ε δ_{(0,1)}
    law_limit = DiscreteLaw.from_mixture(
        weights=[1 - eps, eps],
        atoms=[(1.0, 0.0), (0.0, 1.0)],
    )
    boundary_residue = law_limit.boundary_abs_B_integral()
    E_A_limit = law_limit.expected_A()  # = 1 - ε

    # Construct Γ using boundary zero-weighting
    gamma_limit = construct_boundary_limit_gamma(law_limit)

    assert len(gamma_limit.atoms) == 1, "Limit Γ should be δ_0."
    assert np.isclose(gamma_limit.atoms["A"].values[0], 0.0, atol=1e-12)

    return {
        "epsilon": eps,
        "W1_limit": W1_limit,
        "boundary_residue": boundary_residue,
        "E_A_limit": E_A_limit,
        "gamma_is_delta_0": True,
    }


def plot_experiment_3(df: pd.DataFrame, eps: float, save_path: str) -> None:
    """Plot convergence of Γ_n and W_1 trajectory."""
    W1_limit = eps / (1 - eps)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Atom masses vs n
    axes[0].semilogy(df["n"], df["mass_at_0"], label="mass at z=0", color="steelblue")
    axes[0].semilogy(df["n"], df["mass_at_1_over_alpha"], label="mass at z=1/α_n", color="orange")
    axes[0].set_xlabel("n")
    axes[0].set_ylabel("Mass")
    axes[0].set_title(f"Atom masses of Γ_n  (ε={eps})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(1.0, color="steelblue", linestyle="--", alpha=0.3)
    axes[0].axhline(0.0, color="gray", linestyle="--", alpha=0.3)

    # Panel 2: First moment of Γ_n vs n
    axes[1].plot(df["n"], df["first_moment"], label="∫|z|dΓ_n", color="purple", linewidth=2)
    axes[1].axhline(W1_limit, color="red", linestyle="--",
                    label=f"Limit ε/(1-ε) = {W1_limit:.4f}")
    axes[1].set_xlabel("n")
    axes[1].set_ylabel("∫|z| dΓ_n")
    axes[1].set_title(f"First Moment of Γ_n → ε/(1-ε) = {W1_limit:.4f}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel 3: W_1(Γ_n, δ_0) vs n
    axes[2].plot(df["n"], df["W1_Gamma_n_delta0"], label="W₁(Γₙ, δ₀)", color="darkred", linewidth=2)
    axes[2].axhline(W1_limit, color="red", linestyle="--",
                    label=f"Limit = {W1_limit:.4f} ≠ 0")
    axes[2].axhline(0.0, color="gray", linestyle=":", alpha=0.5)
    axes[2].set_xlabel("n")
    axes[2].set_ylabel("W₁(Γₙ, δ₀)")
    axes[2].set_title(f"W₁-Divergence (ε={eps}): fails despite weak convergence")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(
        f"Exp 3: First-Order Wasserstein Obstruction (ε={eps}), "
        f"Boundary Residue = ε = {eps}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def main() -> dict:
    print("=" * 65)
    print("Experiment 3: First-Order W₁ Stability and Boundary Obstruction")
    print("=" * 65)

    eps_values = [0.1, 0.25, 0.5]
    all_results = {}
    summary_records = []

    for eps in eps_values:
        print(f"\n--- ε = {eps} ---")
        df = run_experiment_3(eps=eps, N=60)
        limits = compute_limit_values(eps)

        # Convergence diagnostics
        W1_final = df["W1_Gamma_n_delta0"].iloc[-1]
        W1_theory = limits["W1_limit"]
        discrepancy = abs(W1_final - W1_theory)

        print(f"  W₁(Γ_n, δ_0) → {W1_final:.6f}  (theory: {W1_theory:.6f})  "
              f"discrepancy: {discrepancy:.2e}")
        print(f"  Boundary residue ∫_{{a=0}}|b|dQ = {limits['boundary_residue']:.6f} = ε")
        print(f"  Weak convergence: ✓  W₁-convergence: ✗  (W₁ → {W1_theory:.4f} ≠ 0)")

        # Verify W₁ → ε/(1-ε) numerically
        assert discrepancy < 1e-6, f"W₁ limit discrepancy too large: {discrepancy}"
        assert np.isclose(limits["boundary_residue"], eps, atol=1e-12)

        all_results[eps] = {"df": df, "limits": limits}

        # Plot
        save_path = os.path.join(RESULTS_DIR, f"exp3_eps{eps:.2f}.png")
        plot_experiment_3(df, eps=eps, save_path=save_path)

        summary_records.append({
            "epsilon": eps,
            "W1_limit_numerical": float(W1_final),
            "W1_limit_theory": float(W1_theory),
            "boundary_residue": float(limits["boundary_residue"]),
            "weak_convergence": True,
            "W1_convergence": False,
            "obstruction": f"boundary |b| mass = {eps}",
        })

    df_summary = pd.DataFrame(summary_records)
    csv_path = os.path.join(RESULTS_DIR, "exp3_summary.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    print("\n--- Experiment 3 Summary ---")
    print(df_summary.to_string(index=False))

    return all_results


if __name__ == "__main__":
    main()
