"""
Experiment 5: Numéraire Inversion Application and Inverse-Moment Stability.

Specialization (A, B) = (X, 1):
  S(μ) = law of 1/X under (X / E[X]) dμ.
  ∫ y^s dS(μ) = E[X^{1-s}] / E[X].

μ_n = (1-ε_n)δ_1 + ε_n δ_{a_n},  ε_n = 2^{-n}.

For a_n = ε_n (threshold s=2 illustration):
  E[X_n] = 1 - ε_n + ε_n a_n = 1 - ε_n + ε_n^2.
  E[X_n^{1-s}] = (1-ε_n) + ε_n a_n^{1-s}.
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
    evaluate_gamma_expectation,
    abs_moment_gamma,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def build_mu_sequence(
    eps_seq: np.ndarray,
    a_seq: np.ndarray,
) -> tuple[list[DiscreteLaw], list[DiscreteLaw], np.ndarray]:
    """
    Build μ_n = (1-ε_n)δ_1 + ε_n δ_{a_n} and the transformed S(μ_n).

    In the (A,B) notation: A = X, B = 1, so S(μ_n) is Γ(Q_n) with
    Q_n = (1-ε_n)δ_{(1,1)} + ε_n δ_{(a_n,1)}.

    Returns: (mu_list, S_mu_list, m_arr).
    """
    mu_list = []
    S_mu_list = []
    m_arr = np.zeros(len(eps_seq))

    for i, (eps, a) in enumerate(zip(eps_seq, a_seq)):
        # Q_n with (A,B) = (X,1): atoms at (1,1) and (a,1)
        law = DiscreteLaw.from_mixture(
            weights=[1 - eps, eps],
            atoms=[(1.0, 1.0), (a, 1.0)],
        )
        mu_list.append(law)
        m_arr[i] = law.expected_A()

        # S(μ_n) = Γ(Q_n): law of B/A = 1/X under X-weighting
        S_mu = construct_gamma_discrete(law)
        S_mu_list.append(S_mu)

    return mu_list, S_mu_list, m_arr


def compute_inverse_moment_formula(law: DiscreteLaw, s: float) -> float:
    """
    Compute E[X^{1-s}] / E[X] using the primitive formula with B=1.

    Since B=1, |B|^s A^{1-s} = A^{1-s}, so:
    E[|B|^s A^{1-s}] / E[A] = E[X^{1-s}] / E[X].
    """
    p = law.atoms["prob"].values
    x = law.atoms["A"].values  # X values
    E_X = law.expected_A()
    E_X_1ms = float(np.dot(p, x ** (1 - s)))
    return E_X_1ms / E_X


def compute_s_mu_moment(S_mu: DiscreteLaw, s: float) -> float:
    """Compute ∫ y^s dS(μ) directly from the transformed law."""
    return abs_moment_gamma(S_mu, s)


def run_experiment_5(N: int = 60) -> dict:
    """
    Run Experiment 5: numéraire inversion with two a_n schedules.

    Schedule 1: a_n = ε_n (threshold at s=2 illustration).
    Schedule 2: a_n = ε_n^{0.5} (alternative exponent c=0.5).
    """
    n_arr = np.arange(1, N + 1)
    eps_seq = 2.0 ** (-n_arr.astype(float))

    s_values = [1.0, 1.5, 2.0, 2.5, 3.0]
    schedules = {
        "a_n = eps_n (c=1)": eps_seq,
        "a_n = eps_n^0.5 (c=0.5)": eps_seq ** 0.5,
        "a_n = eps_n^2 (c=2)": eps_seq ** 2.0,
    }

    results = {}

    for schedule_name, a_seq in schedules.items():
        print(f"\n  Schedule: {schedule_name}")
        mu_list, S_mu_list, m_arr = build_mu_sequence(eps_seq, a_seq)

        # Verify identity for each s value
        identity_records = []
        for s in s_values:
            for n_idx in range(N):
                n = n_idx + 1
                # Direct: ∫ y^s dS(μ_n)
                direct = compute_s_mu_moment(S_mu_list[n_idx], s)
                # Formula: E[X^{1-s}] / E[X]
                formula = compute_inverse_moment_formula(mu_list[n_idx], s)
                residual = abs(direct - formula)
                identity_records.append({
                    "n": n, "s": s, "direct": direct,
                    "formula": formula, "residual": residual,
                })

        id_df = pd.DataFrame(identity_records)
        # Relative tolerance: residual / |formula| ≤ machine-epsilon level
        # (Absolute residuals can be O(ULP) when formula values are O(2^53))
        id_df["rel_residual"] = id_df["residual"] / (id_df["formula"].abs() + 1e-300)
        max_rel = id_df["rel_residual"].max()
        max_abs = id_df["residual"].max()
        print(f"    Max abs residual: {max_abs:.2e}  |  Max rel residual: {max_rel:.2e}")
        assert max_rel < 1e-10, f"Identity check FAILED (rel): {max_rel:.2e}"
        print(f"    ✓ Identity ∫y^s dS(μ) = E[X^{{1-s}}]/E[X] verified.")

        # First-moment automatic stability (s=1)
        s1_df = id_df[id_df["s"] == 1.0].copy()
        # E[X^0] / E[X] = 1 / E[X] → 1/1 = 1 as n → ∞
        first_moments = s1_df.groupby("n")["formula"].first().values
        limit_first_moment = first_moments[-1]
        print(f"    s=1 first moment → {limit_first_moment:.6f} (expect 1.0)")
        assert abs(limit_first_moment - 1.0) < 1e-3, \
            f"First-order stability: expected ~1.0, got {limit_first_moment}"

        # Extract moment sequences for selected s
        moment_seqs = {}
        for s in s_values:
            s_df = id_df[id_df["s"] == s].copy()
            moment_seqs[s] = s_df["formula"].values

        results[schedule_name] = {
            "n_arr": n_arr,
            "eps_seq": eps_seq,
            "a_seq": a_seq,
            "m_arr": m_arr,
            "moment_seqs": moment_seqs,
            "id_df": id_df,
        }

    return results


def plot_results(results: dict, save_path: str) -> None:
    """Plot moment sequences for each schedule and s value."""
    n_schedules = len(results)
    n_s = 5  # number of s values

    fig, axes = plt.subplots(n_schedules, 1, figsize=(11, 4 * n_schedules))
    if n_schedules == 1:
        axes = [axes]

    colors = plt.cm.viridis(np.linspace(0, 1, n_s))
    s_values = [1.0, 1.5, 2.0, 2.5, 3.0]

    for ax, (sched_name, sched_data) in zip(axes, results.items()):
        n_arr = sched_data["n_arr"]
        moment_seqs = sched_data["moment_seqs"]

        for s, color in zip(s_values, colors):
            vals = moment_seqs[s]
            ax.semilogy(n_arr, vals + 1e-300, label=f"s={s}", color=color, linewidth=1.5)

        ax.set_xlabel("n")
        ax.set_ylabel("∫y^s dS(μ_n) = E[X^{1-s}]/E[X]")
        ax.set_title(f"Inverse-Moment Stability: {sched_name}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.3)

    plt.suptitle("Exp 5: Numéraire Inversion – ∫y^s dS(μ_n) = E[X^{1-s}]/E[X]", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_s1_stability(results: dict, save_path: str) -> None:
    """
    Show first-order automatic stability: ∫y dS(μ_n) = 1/E[X_n] → 1.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for sched_name, sched_data in results.items():
        n_arr = sched_data["n_arr"]
        vals = sched_data["moment_seqs"][1.0]
        ax.plot(n_arr, vals, label=sched_name, linewidth=2)

    ax.axhline(1.0, color="red", linestyle="--", alpha=0.7, label="Limit = 1.0")
    ax.set_xlabel("n")
    ax.set_ylabel("∫y dS(μ_n) = 1/E[X_n]")
    ax.set_title("First-Order Automatic Stability: ∫y dS(μ_n) → 1")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.9, 1.1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  Saved: {save_path}")


def main() -> dict:
    print("=" * 65)
    print("Experiment 5: Numéraire Inversion and Inverse-Moment Stability")
    print("=" * 65)

    results = run_experiment_5(N=60)

    # Save plots
    plot_path = os.path.join(RESULTS_DIR, "exp5_inverse_moments.png")
    plot_results(results, save_path=plot_path)

    s1_path = os.path.join(RESULTS_DIR, "exp5_first_order_stability.png")
    plot_s1_stability(results, save_path=s1_path)

    # Save a summary CSV (schedule "a_n = eps_n")
    key = "a_n = eps_n (c=1)"
    id_df = results[key]["id_df"]
    csv_path = os.path.join(RESULTS_DIR, "exp5_identity_residuals.csv")
    id_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Build moment summary table (n=1,5,10,20,40,60; all s; schedule c=1)
    print("\n--- Inverse-Moment Identity (schedule a_n=ε_n, select n) ---")
    sel_n = [1, 5, 10, 20, 40, 60]
    sub = id_df[id_df["n"].isin(sel_n)][["n", "s", "formula", "residual"]]
    print(sub.to_string(index=False))

    print("\n--- Experiment 5 Summary ---")
    for sched_name, sched_data in results.items():
        max_res = sched_data["id_df"]["residual"].max()
        lim_s1 = sched_data["moment_seqs"][1.0][-1]
        print(f"  {sched_name}: max residual={max_res:.2e}, s=1 limit={lim_s1:.6f}")

    return results


if __name__ == "__main__":
    main()
