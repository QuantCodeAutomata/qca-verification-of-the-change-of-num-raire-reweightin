"""
Tests for src/wasserstein.py — Wasserstein distances and ratio-tail diagnostics.
"""

import numpy as np
import pytest

from src.discrete_law import DiscreteLaw
from src.wasserstein import (
    wasserstein_p_dirac,
    compute_mixed_moment,
    compute_ratio_tail,
    compute_tail_envelope,
)


class TestWassersteinPDirac:
    def test_dirac_vs_itself_is_zero(self):
        """W_p(δ_c, δ_c) = 0."""
        gamma = DiscreteLaw.from_dirac(a=2.0, b=1.0)
        # delta_c = delta at c=2.0
        w = wasserstein_p_dirac(gamma, c=2.0, p=1.0)
        assert np.isclose(w, 0.0, atol=1e-12)

    def test_w1_dirac_vs_dirac(self):
        """W_1(δ_x, δ_0) = |x|."""
        x = 3.5
        gamma = DiscreteLaw.from_dirac(a=x, b=1.0)
        w = wasserstein_p_dirac(gamma, c=0.0, p=1.0)
        assert np.isclose(w, x, atol=1e-12)

    def test_w1_two_atom_vs_delta_0(self):
        """W_1(q₀δ_0 + q₁δ_z, δ_0) = q₁ * z."""
        q0, q1, z = 0.7, 0.3, 4.0
        gamma = DiscreteLaw.from_mixture(weights=[q0, q1], atoms=[(0.0, 1.0), (z, 1.0)])
        w = wasserstein_p_dirac(gamma, c=0.0, p=1.0)
        assert np.isclose(w, q1 * z, atol=1e-12)

    def test_w2_dirac_vs_dirac(self):
        """W_2(δ_x, δ_0) = |x|."""
        x = 5.0
        gamma = DiscreteLaw.from_dirac(a=x, b=1.0)
        w = wasserstein_p_dirac(gamma, c=0.0, p=2.0)
        assert np.isclose(w, x, atol=1e-12)

    def test_wasserstein_non_negative(self):
        """W_p is always non-negative."""
        gamma = DiscreteLaw.from_mixture(
            weights=[0.5, 0.5], atoms=[(1.0, 1.0), (3.0, 1.0)]
        )
        for p in [1.0, 1.5, 2.0, 3.0]:
            assert wasserstein_p_dirac(gamma, c=0.0, p=p) >= 0.0


class TestComputeMixedMoment:
    def test_s1_equals_E_abs_B(self):
        """At s=1: E[|B|^1 A^0] = E[|B|]."""
        law = DiscreteLaw.from_mixture(
            weights=[0.5, 0.5],
            atoms=[(1.0, 3.0), (2.0, 4.0)],
        )
        assert np.isclose(compute_mixed_moment(law, sigma=1.0), law.expected_abs_B())

    def test_s2_two_atom(self):
        """At s=2: E[|B|² A^{-1}] = Σ p_i b_i² / a_i."""
        law = DiscreteLaw.from_mixture(
            weights=[0.3, 0.7],
            atoms=[(2.0, 4.0), (1.0, 3.0)],
        )
        expected = 0.3 * (16.0 / 2.0) + 0.7 * (9.0 / 1.0)
        assert np.isclose(compute_mixed_moment(law, sigma=2.0), expected, atol=1e-12)

    def test_boundary_atoms_contribute_zero(self):
        """A=0 atoms must not contribute (0^{1-s}=0 for s≥1)."""
        law = DiscreteLaw.from_mixture(
            weights=[0.5, 0.5],
            atoms=[(0.0, 1.0), (1.0, 2.0)],
        )
        # Only atom (1,2): E = 0.5 * |2|^2 * 1^{-1} = 0.5*4 = 2.0
        assert np.isclose(compute_mixed_moment(law, sigma=2.0), 2.0, atol=1e-12)


class TestComputeRatioTail:
    def test_no_ratio_exceeds_R(self):
        """If max ratio is below R, tail should be 0."""
        # Both atoms: B/A = 0 and B/A = 1, set R=2 → tail = 0
        law = DiscreteLaw.from_mixture(
            weights=[0.5, 0.5],
            atoms=[(1.0, 0.0), (1.0, 1.0)],
        )
        assert np.isclose(compute_ratio_tail(law, sigma=2.0, R=2.0), 0.0)

    def test_rare_atom_satisfies_tail(self):
        """When rare atom has |B|/A > R, tail should be positive."""
        eps, a, R = 0.1, 0.01, 0.5
        # Atom (a, 1): ratio = 1/a = 100 > R = 0.5
        law = DiscreteLaw.from_mixture(
            weights=[1 - eps, eps],
            atoms=[(1.0, 0.0), (a, 1.0)],
        )
        tail = compute_ratio_tail(law, sigma=1.0, R=R)
        # Contribution from rare atom: eps * |1|^1 * a^{1-1} = eps * 1 * 1 = eps
        assert np.isclose(tail, eps, atol=1e-12)

    def test_tail_vanishes_above_ratio(self):
        """Tail should vanish when R ≥ max ratio."""
        eps, a = 0.1, 0.5
        # max ratio = 1/a = 2
        law = DiscreteLaw.from_mixture(
            weights=[1 - eps, eps],
            atoms=[(1.0, 0.0), (a, 1.0)],
        )
        # R = 3 > 1/a = 2: no atom satisfies |B| > 3*A
        tail = compute_ratio_tail(law, sigma=2.0, R=3.0)
        assert np.isclose(tail, 0.0, atol=1e-12)

    def test_tail_envelope_supremum(self):
        """Envelope should be ≥ individual values."""
        laws = [
            DiscreteLaw.from_mixture([0.9, 0.1], [(1.0, 0.0), (0.1, 1.0)]),
            DiscreteLaw.from_mixture([0.8, 0.2], [(1.0, 0.0), (0.5, 1.0)]),
        ]
        R = 0.5
        sigma = 1.0
        env = compute_tail_envelope(laws, sigma, R)
        for law in laws:
            assert env >= compute_ratio_tail(law, sigma, R) - 1e-12


class TestWassersteinThresholdFormula:
    """
    Verify the closed-form Wasserstein threshold from Experiment 2.
    M_n(σ) = ε_n^{(s*-σ)/(s*-1)} / m_n  should equal  Σ q_j |z_j|^σ.
    """

    def test_threshold_formula_s1_5(self):
        """s* = 1.5: exponent = (1.5-σ)/0.5."""
        s_star = 1.5
        eps = 2.0 ** (-5)  # n=5
        a = eps ** (1.0 / (s_star - 1.0))
        law = DiscreteLaw.from_mixture(
            weights=[1 - eps, eps],
            atoms=[(1.0, 0.0), (a, 1.0)],
        )
        from src.numeraire_change import construct_gamma_discrete
        gamma = construct_gamma_discrete(law)
        m = law.expected_A()

        for sigma in [1.0, 1.5, 2.0]:
            Mn_formula = eps ** ((s_star - sigma) / (s_star - 1.0)) / m
            Mn_direct = float(np.dot(
                gamma.atoms["prob"].values,
                np.abs(gamma.atoms["A"].values) ** sigma,
            ))
            assert np.isclose(Mn_formula, Mn_direct, rtol=1e-6), \
                f"σ={sigma}: formula={Mn_formula}, direct={Mn_direct}"

    def test_threshold_formula_s2(self):
        """s* = 2.0."""
        s_star = 2.0
        n = 8
        eps = 2.0 ** (-n)
        a = eps ** (1.0 / (s_star - 1.0))
        law = DiscreteLaw.from_mixture(
            weights=[1 - eps, eps],
            atoms=[(1.0, 0.0), (a, 1.0)],
        )
        from src.numeraire_change import construct_gamma_discrete
        gamma = construct_gamma_discrete(law)
        m = law.expected_A()

        for sigma in [1.0, 2.0, 3.0]:
            Mn_formula = eps ** ((s_star - sigma) / (s_star - 1.0)) / m
            Mn_direct = float(np.dot(
                gamma.atoms["prob"].values,
                np.abs(gamma.atoms["A"].values) ** sigma,
            ))
            assert np.isclose(Mn_formula, Mn_direct, rtol=1e-6), \
                f"σ={sigma}: formula={Mn_formula}, direct={Mn_direct}"
