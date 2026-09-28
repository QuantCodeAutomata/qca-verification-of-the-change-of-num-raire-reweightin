"""
Wasserstein distance and ratio-tail diagnostics for discrete 1-D laws.

For 1-D discrete measures μ and ν, the p-Wasserstein distance is computed via
the quantile formula:
    W_p(μ, ν)^p = ∫₀¹ |F_μ^{-1}(t) - F_ν^{-1}(t)|^p dt.

When ν = δ_c (a Dirac at c), this simplifies to:
    W_p(μ, δ_c)^p = ∫ |z - c|^p dμ(z).

# Using scipy.stats.wasserstein_distance — Context7 confirmed for 1-D OT
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance

from .discrete_law import DiscreteLaw


def wasserstein_p(
    mu: DiscreteLaw,
    nu: DiscreteLaw,
    p: float = 1.0,
) -> float:
    """
    Compute W_p(μ, ν) between two discrete 1-D laws.

    The support of each law is taken from the 'A' column (z-values of Γ).

    Parameters
    ----------
    mu, nu : DiscreteLaw – 1-D transformed laws with support in 'A' column
    p      : float       – Wasserstein order (p ≥ 1)

    Returns
    -------
    float : W_p(μ, ν)  (the distance itself, not the p-th power)
    """
    z_mu = mu.atoms["A"].values
    w_mu = mu.atoms["prob"].values
    z_nu = nu.atoms["A"].values
    w_nu = nu.atoms["prob"].values

    if p == 1.0:
        # scipy.stats.wasserstein_distance computes W_1
        return float(wasserstein_distance(z_mu, z_nu, u_weights=w_mu, v_weights=w_nu))

    # For p > 1: use exact formula via quantile-function quadrature on combined grid
    # Combine supports and sort
    all_z = np.sort(np.unique(np.concatenate([z_mu, z_nu])))

    def cdf(z_arr: np.ndarray, support: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """CDF of a discrete measure evaluated at each point in z_arr."""
        result = np.zeros(len(z_arr))
        for i, z in enumerate(z_arr):
            result[i] = weights[support <= z].sum()
        return result

    # Gauss-Legendre quadrature on [0,1] for quantile integral
    n_quad = 10000
    t = np.linspace(0, 1, n_quad + 1)[1:]  # avoid t=0

    def quantile(t_arr: np.ndarray, support: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Quantile function of a discrete measure."""
        idx = support.argsort()
        s_sorted = support[idx]
        w_sorted = weights[idx]
        cdf_vals = np.cumsum(w_sorted)
        # Q(t) = min{z : F(z) >= t}
        result = np.zeros(len(t_arr))
        for i, ti in enumerate(t_arr):
            pos = np.searchsorted(cdf_vals, ti, side="left")
            pos = min(pos, len(s_sorted) - 1)
            result[i] = s_sorted[pos]
        return result

    q_mu = quantile(t, z_mu, w_mu)
    q_nu = quantile(t, z_nu, w_nu)
    integrand = np.abs(q_mu - q_nu) ** p
    wp_p = integrand.mean()  # approximation of ∫₀¹ |F^{-1}(t)-G^{-1}(t)|^p dt
    return float(wp_p ** (1.0 / p))


def wasserstein_p_dirac(gamma: DiscreteLaw, c: float, p: float) -> float:
    """
    Compute W_p(Γ, δ_c) using the exact formula:
        W_p(Γ, δ_c)^p = ∫ |z - c|^p dΓ = Σ_j q_j |z_j - c|^p.

    Parameters
    ----------
    gamma : DiscreteLaw – 1-D transformed law with support in 'A' column
    c     : float       – position of the Dirac target
    p     : float       – Wasserstein order

    Returns
    -------
    float : W_p(Γ, δ_c)
    """
    z = gamma.atoms["A"].values
    q = gamma.atoms["prob"].values
    wp_p = float(np.dot(q, np.abs(z - c) ** p))
    return wp_p ** (1.0 / p)


def compute_mixed_moment(law: DiscreteLaw, sigma: float) -> float:
    """
    Compute E[|B_n|^σ A_n^{1-σ}] from the primitive bivariate law Q_n.

    This is the key quantity governing W_σ stability in Theorem 3.3.
    States with A = 0 contribute 0.

    Parameters
    ----------
    law   : DiscreteLaw – primitive bivariate law Q_n
    sigma : float       – Wasserstein order σ ≥ 1

    Returns
    -------
    float : E[|B|^σ A^{1-σ}]
    """
    return law.expected_mixed_moment(sigma)


def compute_ratio_tail(law: DiscreteLaw, sigma: float, R: float) -> float:
    """
    Compute T_{n,σ}(R) = E[|B|^σ A^{1-σ} 1_{|B| > R A}].

    This is the ratio-tail diagnostic for Wasserstein stability.
    States with A = 0 contribute 0.

    Parameters
    ----------
    law   : DiscreteLaw – primitive bivariate law Q_n
    sigma : float       – Wasserstein order
    R     : float       – tail threshold (R > 0)

    Returns
    -------
    float : T_{n,σ}(R)
    """
    atoms = law.atoms
    mask = atoms["A"] > 0
    sub = atoms[mask]

    if sub.empty:
        return 0.0

    a = sub["A"].values
    b = np.abs(sub["B"].values)
    p = sub["prob"].values

    # Ratio event: |B| > R A
    tail_mask = b > R * a
    contributions = np.where(
        tail_mask,
        p * (b ** sigma) * (a ** (1 - sigma)),
        0.0
    )
    return float(contributions.sum())


def compute_tail_envelope(
    laws: list[DiscreteLaw],
    sigma: float,
    R: float,
) -> float:
    """
    Compute T_σ(R) = sup_{1 ≤ n ≤ N} T_{n,σ}(R) over a sequence of laws.

    Parameters
    ----------
    laws  : list of DiscreteLaw – sequence Q_1, ..., Q_N
    sigma : float               – Wasserstein order
    R     : float               – tail threshold

    Returns
    -------
    float : supremum of T_{n,σ}(R) over the sequence
    """
    tails = [compute_ratio_tail(law, sigma, R) for law in laws]
    return float(np.max(tails))
