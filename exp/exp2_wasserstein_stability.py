"""
Experiment 2: Exact Wasserstein Stability Boundary via Theorem 3.3
             and Sharp Threshold Example.

Two-atom sequence Q_n = (1-ε_n)δ_{(1,0)} + ε_n δ_{(a_n,1)}, with
  ε_n = 2^{-n},  a_n = ε_n^{1/(s*-1)},   for n = 1..N.

Threshold theorem:
  M_n(σ) = ∫|z|^σ dΓ_n = ε_n^{(s*-σ)/(s*-1)} / m_n.

  σ < s*  ⟹  M_n(σ) → 0  (W_σ-stable)
  σ = s*  ⟹  M_n(σ) → 1  (marginal failure)
  σ > s*  ⟹  M_n(σ) → ∞  (unstable)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.discrete_law import DiscreteLaw
from src.numeraire_change import construct_gamma_discrete
from src.wasserstein import wasserstein_p_dirac, compute_ratio_tail

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def build_sequence(s_star: float, N: int = 60):
    """
    Build the two-atom sequence Q_n for the given threshold s* > 1.

    Returns lists: eps_arr, a_arr, laws_n, m_arr, and gamma_arr.
    """
    n_arr = np.arange(1, N + 1)
    eps_arr = 2.0 ** (-n_arr.astype(float))
    a_arr = eps_arr ** (1.0 / (s_star - 1.0))

    laws_n = []
    m_arr = np.zeros(N)
    gamma_arr = []

    for i, (eps, a) in enumerate(zip(eps_arr, a_arr)):
        law = DiscreteLaw.from_mixture(
            weights=[1 - eps, eps],
            atoms=[(1.0, 0.0), (a, 1.0)],
        )
        laws_n.append(law)
        m_arr[i] = law.expected_A()

        gamma = construct_gamma_discrete(law)
        gamma_arr.append(gamma)

    return eps_arr, a_arr, laws_n, m_arr, gamma_arr


def compute_moments_exact(eps_arr: np.ndarray, a_arr: np.ndarray, m_arr: np.ndarray,
                          sigma: float, s_star: float) -> np.ndarray:
    """
    Compute M_n(σ) = ε_n^{(s*-σ)/(s*-1)} / m_n using exact closed-form formula.

    Avoids numerical overflow from large 1/a_n values.
    """
    exponent = (s_star - sigma) / (s_star - 1.0)
    log_Mn = exponent * np.log(eps_arr) - np.log(m_arr)
    # Return in linear scale (may be very large for σ > s*)
    return np.exp(log_Mn)


def compute_moments_direct(gamma_arr: list, sigma: float) -> np.ndarray:
    """
    Compute M_n(σ) = ∫|z|^σ dΓ_n directly from the transformed laws.
    Uses log-space for the rare atom term to avoid overflow.
    """
    results = []
    for gamma in gamma_arr:
        z = gamma.atoms["A"].values
        q = gamma.atoms["prob"].values
        # Evaluate |z|^sigma for each atom
        # z[0] = 0 (from atom (1,0)), z[1] = 1/a_n (from atom (a_n,1))
        # |0|^sigma = 0, |1/a_n|^sigma = (1/a_n)^sigma  -- may overflow
        # Use log-space: log(q[1] * (1/a_n)^sigma) = log(q[1]) + sigma*log(1/a_n)
        val = 0.0
        for zi, qi in zip(z, q):
            if zi == 0.0:
                continue
            log_term = np.log(qi) + sigma * np.log(abs(zi))
            val += np.exp(log_term)
        results.append(val)
    return np.array(results)


def compute_ratio_tails_sequence(laws_n: list, sigma: float, R_grid: np.ndarray) -> np.ndarray:
    """
    Compute T_σ(R) = sup_n T_{n,σ}(R) for each R in R_grid.

    For the two-atom law:
    T_{n,σ}(R) = ε_n a_n^{1-σ} * 1_{1 > R a_n}  (the rare atom has |B|=1, A=a_n)
    """
    T_env = np.zeros(len(R_grid))
    for j, R in enumerate(R_grid):
        tails = [compute_ratio_tail(law, sigma, R) for law in laws_n]
        T_env[j] = np.max(tails)
    return T_env


def call_price_curve(gamma: DiscreteLaw, K_grid: np.ndarray) -> np.ndarray:
    """Compute call-price curve C(K) = ∫(z-K)^+ dΓ for a grid of strikes."""
    z = gamma.atoms["A"].values
    q = gamma.atoms["prob"].values
    result = np.zeros(len(K_grid))
    for j, K in enumerate(K_grid):
        result[j] = float(np.dot(q, np.maximum(z - K, 0.0)))
    return result


def run_experiment_2(s_star: float = 2.0, N: int = 60) -> dict:
    """
    Run Experiment 2 for a given threshold s*.

    Returns dict with moment arrays, regime classification, and tail envelopes.
    """
    print(f"\n--- s* = {s_star} ---")

    eps_arr, a_arr, laws_n, m_arr, gamma_arr = build_sequence(s_star, N)
    n_arr = np.arange(1, N + 1)

    # σ grid covering three regimes
    sigma_below = [1.0, max(1.05, s_star * 0.5), s_star * 0.75, s_star * 0.9]
    sigma_at = [s_star]
    sigma_above = [s_star * 1.1, s_star * 1.5, s_star * 2.0]
    sigma_grid = sorted(set(sigma_below + sigma_at + sigma_above))

    moment_data = {}
    for sigma in sigma_grid:
        Mn_exact = compute_moments_exact(eps_arr, a_arr, m_arr, sigma, s_star)
        Mn_direct = compute_moments_direct(gamma_arr, sigma)

        # Max relative discrepancy between exact and direct
        denom = np.maximum(Mn_exact, 1e-300)
        max_disc = np.max(np.abs(Mn_exact - Mn_direct) / denom)

        if max_disc > 1e-8:
            print(f"  WARNING: σ={sigma:.3f} max discrepancy = {max_disc:.2e}")
        else:
            print(f"  σ={sigma:.4f}: M_n formula validated (max disc={max_disc:.2e})")

        # Regime classification based on limit of M_n
        lim_Mn = Mn_exact[-1]
        exponent_val = (s_star - sigma) / (s_star - 1.0)
        regime = "stable (M→0)" if exponent_val > 0 else ("marginal (M→1)" if exponent_val == 0 else "unstable (M→∞)")

        moment_data[sigma] = {
            "Mn_exact": Mn_exact,
            "Mn_direct": Mn_direct,
            "limit": float(lim_Mn),
            "exponent": float(exponent_val),
            "regime": regime,
        }
        print(f"    → regime={regime}, M_n[N]={lim_Mn:.4g}")

    # Ratio-tail envelopes for selected σ
    R_grid = 2.0 ** np.arange(0, 15)  # R = 1, 2, 4, ..., 2^14
    tail_data = {}
    for sigma in [s_star * 0.75, s_star, s_star * 1.5]:
        T_env = compute_ratio_tails_sequence(laws_n, sigma, R_grid)
        tail_data[sigma] = T_env

    # Verify m_n → 1 (since ε_n a_n = ε_n^{s*/(s*-1)} → 0)
    assert np.isclose(m_arr[-1], 1.0, atol=1e-3), f"m_n should → 1, got {m_arr[-1]}"

    return {
        "s_star": s_star,
        "n_arr": n_arr,
        "eps_arr": eps_arr,
        "a_arr": a_arr,
        "m_arr": m_arr,
        "moment_data": moment_data,
        "sigma_grid": sigma_grid,
        "R_grid": R_grid,
        "tail_data": tail_data,
        "gamma_arr": gamma_arr,
        "laws_n": laws_n,
    }


def plot_results(results: dict, save_prefix: str) -> None:
    """Generate all plots for Experiment 2."""
    s_star = results["s_star"]
    n_arr = results["n_arr"]
    moment_data = results["moment_data"]
    sigma_grid = results["sigma_grid"]
    R_grid = results["R_grid"]
    tail_data = results["tail_data"]

    # --- Plot 1: M_n(σ) vs n for three regimes ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    regimes = ["stable", "marginal", "unstable"]
    colors = ["blue", "orange", "red"]

    # Classify σ values
    stable_sigmas = [σ for σ in sigma_grid if (s_star - σ) / (s_star - 1.0) > 1e-8]
    marginal_sigmas = [σ for σ in sigma_grid if abs((s_star - σ) / (s_star - 1.0)) < 1e-8]
    unstable_sigmas = [σ for σ in sigma_grid if (s_star - σ) / (s_star - 1.0) < -1e-8]

    for ax, sigs, color, label in zip(
        axes, [stable_sigmas, [s_star], unstable_sigmas],
        colors, regimes
    ):
        for σ in sigs[:3]:
            if σ in moment_data:
                Mn = moment_data[σ]["Mn_exact"]
                ax.semilogy(n_arr, Mn + 1e-300, label=f"σ={σ:.3g}", color=color, alpha=0.7)
        ax.set_xlabel("n")
        ax.set_ylabel("M_n(σ)")
        ax.set_title(f"s* = {s_star:.2g}: {label} regime")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if label == "stable":
            ax.set_title(f"s* = {s_star:.2g}: σ < s* (W_σ → 0)")
        elif label == "marginal":
            ax.set_title(f"s* = {s_star:.2g}: σ = s* (W_σ fails)")
            ax.set_yscale("linear")
        else:
            ax.set_title(f"s* = {s_star:.2g}: σ > s* (W_σ → ∞)")

    plt.suptitle(f"Wasserstein Stability: M_n(σ) = ε_n^{{(s*-σ)/(s*-1)}} / m_n, s* = {s_star}", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_moments.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_prefix}_moments.png")

    # --- Plot 2: Ratio-tail envelopes T_σ(R) vs R ---
    fig, ax = plt.subplots(figsize=(9, 5))
    colors_t = ["blue", "orange", "red"]
    labels_t = [f"σ={σ:.3g} ({'<' if σ < s_star else ('=' if σ == s_star else '>')} s*)"
                for σ in tail_data]
    for (sigma, T_env), color, lbl in zip(tail_data.items(), colors_t, labels_t):
        ax.semilogx(R_grid, T_env, marker="o", markersize=4, label=lbl, color=color)
    ax.set_xlabel("R (tail threshold)")
    ax.set_ylabel("T_σ(R) = sup_n T_{n,σ}(R)")
    ax.set_title(f"Ratio-Tail Envelope T_σ(R) for s* = {s_star}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_tails.png", dpi=120)
    plt.close()
    print(f"  Saved: {save_prefix}_tails.png")

    # --- Plot 3: W_σ(Γ_n, δ_0) = M_n(σ)^{1/σ} versus n ---
    fig, ax = plt.subplots(figsize=(9, 5))
    for σ, color, ls in zip(
        [s for s in sigma_grid if abs(s - s_star) < 0.01 or
         (s < s_star and s > 1.0) or (s > s_star and s < s_star * 2.5)],
        plt.cm.viridis(np.linspace(0, 1, len(sigma_grid))),
        ["-"] * len(sigma_grid),
    ):
        if σ not in moment_data:
            continue
        Mn = moment_data[σ]["Mn_exact"]
        Wsigma = np.where(Mn > 0, Mn ** (1.0 / σ), 0)
        label = f"σ={σ:.3g}"
        ax.semilogy(n_arr, Wsigma + 1e-300, label=label, alpha=0.75)
    ax.set_xlabel("n")
    ax.set_ylabel("W_σ(Γ_n, δ_0)")
    ax.set_title(f"Wasserstein Distance W_σ(Γ_n, δ_0), s* = {s_star}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_wasserstein.png", dpi=120)
    plt.close()
    print(f"  Saved: {save_prefix}_wasserstein.png")


def main() -> dict:
    print("=" * 65)
    print("Experiment 2: Exact Wasserstein Stability Boundary")
    print("=" * 65)

    all_results = {}
    summary_records = []

    for s_star in [1.5, 2.0, 3.0]:
        results = run_experiment_2(s_star=s_star, N=60)
        all_results[s_star] = results

        prefix = os.path.join(RESULTS_DIR, f"exp2_sstar{s_star:.1f}")
        plot_results(results, save_prefix=prefix)

        # Build summary table
        for sigma, data in results["moment_data"].items():
            summary_records.append({
                "s_star": s_star,
                "sigma": sigma,
                "exponent": data["exponent"],
                "M_n_final": data["limit"],
                "regime": data["regime"],
            })

    df_summary = pd.DataFrame(summary_records)
    csv_path = os.path.join(RESULTS_DIR, "exp2_threshold_summary.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"\nSaved summary: {csv_path}")

    print("\n--- Threshold Summary ---")
    print(df_summary.to_string(index=False))

    return all_results


if __name__ == "__main__":
    main()
