"""
Experiment 4: Necessity of Uniform Integrability of the Numéraire for Weak Stability.

Q_n = (1-ε_n)δ_{(1,0)} + ε_n δ_{(1/ε_n, 1/ε_n)},  ε_n = 2^{-n}.
Weak limit Q = δ_{(1,0)}.

Key results:
  - Q_n ⇒ Q weakly.
  - Γ_n → (1/2)δ_0 + (1/2)δ_1  ≠  δ_0 = Γ(Q).
  - Failure traced to non-UI of {A_n}: E[A_n 1_{A_n>R}] ≥ 1 for all R.
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
    construct_gamma_discrete as gamma_op,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def build_non_ui_sequence(N: int = 60):
    """
    Build the non-UI sequence Q_n = (1-ε_n)δ_{(1,0)} + ε_n δ_{(1/ε_n, 1/ε_n)}.

    Returns: (n_arr, eps_arr, laws_n, gamma_arr, m_arr).
    """
    n_arr = np.arange(1, N + 1)
    eps_arr = 2.0 ** (-n_arr.astype(float))

    laws_n = []
    gamma_arr = []
    m_arr = np.zeros(N)

    for eps in eps_arr:
        A_rare = 1.0 / eps
        B_rare = 1.0 / eps
        law = DiscreteLaw.from_mixture(
            weights=[1 - eps, eps],
            atoms=[(1.0, 0.0), (A_rare, B_rare)],
        )
        laws_n.append(law)
        m_arr[int(np.round(np.log2(1 / eps))) - 1] = law.expected_A()

        gamma = construct_gamma_discrete(law)
        gamma_arr.append(gamma)

    # Recompute m_arr correctly
    m_arr = np.array([law.expected_A() for law in laws_n])

    return n_arr, eps_arr, laws_n, gamma_arr, m_arr


def extract_gamma_masses(gamma: DiscreteLaw, target_z: float = 1.0):
    """
    Extract mass at z=0 and z=1 from a two-atom Γ.

    The rare atom maps to ratio B/A = (1/ε)/(1/ε) = 1, so Γ has support {0, 1}.
    """
    atoms = gamma.atoms
    z_vals = atoms["A"].values
    q_vals = atoms["prob"].values

    mass_0 = float(q_vals[np.isclose(z_vals, 0.0, atol=1e-12)].sum())
    mass_1 = float(q_vals[np.isclose(z_vals, target_z, atol=1e-12)].sum())
    return mass_0, mass_1


def compute_ui_tail_expectations(laws_n: list, R_values: np.ndarray) -> np.ndarray:
    """
    Compute sup_n E[A_n 1_{A_n > R}] for each R.

    For Q_n, E[A_n 1_{A_n > R}] = ε_n * (1/ε_n) * 1_{1/ε_n > R} = 1_{1/ε_n > R}.
    So for any fixed R, there exists n large enough that 1/ε_n > R, yielding value 1.
    Hence sup_n E[A_n 1_{A_n > R}] = 1 for all R > 0.
    """
    results = np.zeros(len(R_values))
    for j, R in enumerate(R_values):
        tail_vals = []
        for law in laws_n:
            atoms = law.atoms
            mask = atoms["A"] > R
            tail_contrib = float(np.dot(
                atoms.loc[mask, "prob"],
                atoms.loc[mask, "A"],
            ))
            tail_vals.append(tail_contrib)
        results[j] = max(tail_vals)
    return results


def run_experiment_4(N: int = 60) -> dict:
    """
    Run Experiment 4: UI failure counterexample.

    Returns dict with sequences, masses, UI tail expectations.
    """
    n_arr, eps_arr, laws_n, gamma_arr, m_arr = build_non_ui_sequence(N)

    # Extract masses at z=0 and z=1 for each n
    records = []
    for n, eps, gamma, m_n in zip(n_arr, eps_arr, gamma_arr, m_arr):
        mass_0, mass_1 = extract_gamma_masses(gamma, target_z=1.0)

        # Verify exact formulas:
        # mass at 0 = (1-ε_n)/m_n, mass at 1 = 1/m_n
        expected_mass_0 = (1 - eps) / m_n
        expected_mass_1 = 1.0 / m_n

        assert np.isclose(mass_0, expected_mass_0, atol=1e-10), \
            f"n={n}: mass_0={mass_0} vs expected={expected_mass_0}"
        assert np.isclose(mass_1, expected_mass_1, atol=1e-10), \
            f"n={n}: mass_1={mass_1} vs expected={expected_mass_1}"

        records.append({
            "n": n,
            "eps_n": eps,
            "m_n": m_n,
            "A_rare": 1.0 / eps,
            "mass_at_0": mass_0,
            "mass_at_1": mass_1,
        })

    df = pd.DataFrame(records)

    # Limiting behavior: m_n → 2, mass_at_0 → 1/2, mass_at_1 → 1/2
    lim_mass_0 = df["mass_at_0"].iloc[-1]
    lim_mass_1 = df["mass_at_1"].iloc[-1]
    print(f"  Limiting Γ_n: mass at 0 → {lim_mass_0:.6f} (expect 0.5)")
    print(f"  Limiting Γ_n: mass at 1 → {lim_mass_1:.6f} (expect 0.5)")
    print(f"  m_n → {df['m_n'].iloc[-1]:.6f} (expect 2.0)")

    assert abs(lim_mass_0 - 0.5) < 1e-4, f"Expected mass_0 → 0.5, got {lim_mass_0}"
    assert abs(lim_mass_1 - 0.5) < 1e-4, f"Expected mass_1 → 0.5, got {lim_mass_1}"

    # Compute the limit of Γ independently: Γ(Q) for Q = δ_{(1,0)} is δ_0
    Q_limit = DiscreteLaw.from_dirac(a=1.0, b=0.0)
    from src.numeraire_change import construct_gamma_discrete
    gamma_limit = construct_gamma_discrete(Q_limit)
    assert len(gamma_limit.atoms) == 1
    assert np.isclose(gamma_limit.atoms["A"].values[0], 0.0, atol=1e-12)
    print(f"  Correct limit Γ(Q) = δ_0: support = {gamma_limit.atoms['A'].values[0]}")

    # UI tail expectations
    R_values = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])
    ui_tails = compute_ui_tail_expectations(laws_n, R_values)
    print(f"\n  UI Tail E[A_n 1_{{A_n>R}}] suprema over n:")
    for R, T in zip(R_values, ui_tails):
        print(f"    R={R:.1f}: sup_n E[A_n 1_{{A_n>R}}] = {T:.4f}")

    # For all R, sup_n ≥ 1 (since there exists n with 1/ε_n > R)
    assert np.all(ui_tails >= 1.0 - 1e-6), \
        "UI should fail: sup_n E[A_n 1_{A_n>R}] must be ≥ 1 for all R."
    print("  ✓ UI failure confirmed: sup_n E[A_n 1_{A_n>R}] = 1 for all R.")

    return {
        "n_arr": n_arr,
        "eps_arr": eps_arr,
        "df": df,
        "R_values": R_values,
        "ui_tails": ui_tails,
        "lim_mass_0": lim_mass_0,
        "lim_mass_1": lim_mass_1,
    }


def plot_experiment_4(results: dict, save_path: str) -> None:
    """Plot Γ_n mass trajectories and UI tail expectations."""
    df = results["df"]
    R_values = results["R_values"]
    ui_tails = results["ui_tails"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: Γ_n atom masses vs n
    axes[0].plot(df["n"], df["mass_at_0"], label="mass at z=0", color="steelblue", linewidth=2)
    axes[0].plot(df["n"], df["mass_at_1"], label="mass at z=1", color="darkorange", linewidth=2)
    axes[0].axhline(0.5, color="gray", linestyle="--", alpha=0.6, label="limit 0.5")
    axes[0].set_xlabel("n")
    axes[0].set_ylabel("Mass")
    axes[0].set_title("Γ_n masses at z=0 and z=1\n(Limit: ½δ₀ + ½δ₁ ≠ δ₀ = Γ(Q))")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.05, 1.05)

    # Panel 2: UI tail expectations
    axes[1].plot(R_values, ui_tails, "o-", color="red", linewidth=2, markersize=6)
    axes[1].axhline(1.0, color="darkred", linestyle="--", alpha=0.7, label="floor = 1")
    axes[1].axhline(0.0, color="gray", linestyle=":", alpha=0.5, label="UI target = 0")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("R (tail threshold)")
    axes[1].set_ylabel("sup_n E[A_n 1_{A_n > R}]")
    axes[1].set_title("UI Diagnostic: Tail Expectation does NOT vanish\n"
                      "(sup_n E[A_n 1_{A_n > R}] = 1 for all R)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.1, 1.5)

    plt.suptitle(
        "Exp 4: Weak Stability Fails when A_n is NOT Uniformly Integrable\n"
        "Q_n → δ_{(1,0)} weakly, but Γ_n → ½δ₀+½δ₁ ≠ Γ(Q) = δ_0",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def main() -> dict:
    print("=" * 65)
    print("Experiment 4: UI Necessity for Weak Stability")
    print("=" * 65)

    results = run_experiment_4(N=60)

    save_path = os.path.join(RESULTS_DIR, "exp4_ui_failure.png")
    plot_experiment_4(results, save_path=save_path)

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "exp4_gamma_masses.csv")
    results["df"].to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Summary
    print("\n--- Experiment 4 Verdict ---")
    print(f"  Q_n ⇒ Q = δ_{{(1,0)}}  (weak convergence): ✓")
    print(f"  UI of A_n fails (sup_n E[A_n 1_{{A_n>R}}] = 1 ∀R):  ✓ confirmed")
    print(f"  Γ_n → ½δ₀ + ½δ₁ ≠ δ₀ = Γ(Q)  (weak stability fails): ✓")
    print(f"  Γ_n limit mass at z=0: {results['lim_mass_0']:.6f} (theory: 0.5)")
    print(f"  Γ_n limit mass at z=1: {results['lim_mass_1']:.6f} (theory: 0.5)")

    return results


if __name__ == "__main__":
    main()
