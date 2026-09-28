"""
Tests for src/numeraire_change.py — Γ operator and identity verification.
"""

import numpy as np
import pytest

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
    expected_A,
)


class TestConstructGammaDiscrete:
    def test_dirac_at_non_zero_A(self):
        """Γ(δ_{(2,6)}) = δ_{3}, since z = B/A = 6/2 = 3."""
        law = DiscreteLaw.from_dirac(a=2.0, b=6.0)
        gamma = construct_gamma_discrete(law)
        assert len(gamma.atoms) == 1
        assert np.isclose(gamma.atoms["A"].values[0], 3.0)
        assert np.isclose(gamma.atoms["prob"].values[0], 1.0)

    def test_two_atom_law_ratios(self):
        """Verify atom z-values and weighted masses for a two-atom law."""
        eps = 0.25
        a = 0.5
        law = DiscreteLaw.from_mixture(
            weights=[1 - eps, eps],
            atoms=[(1.0, 0.0), (a, 1.0)],
        )
        m = law.expected_A()  # = 1 - eps + eps*a
        gamma = construct_gamma_discrete(law)

        # z-values: 0 (from (1,0)) and 1/a (from (a,1))
        z_vals = sorted(gamma.atoms["A"].values)
        q_vals = gamma.atoms.sort_values("A")["prob"].values

        assert np.isclose(z_vals[0], 0.0, atol=1e-12)
        assert np.isclose(z_vals[1], 1.0 / a, atol=1e-12)
        assert np.isclose(q_vals[0], (1 - eps) / m, atol=1e-12)
        assert np.isclose(q_vals[1], eps * a / m, atol=1e-12)

    def test_mass_conservation(self):
        """Γ masses must sum to 1."""
        law = DiscreteLaw.from_mixture(
            weights=[0.2, 0.5, 0.3],
            atoms=[(1.0, 2.0), (3.0, 6.0), (0.5, 1.0)],
        )
        gamma = construct_gamma_discrete(law)
        assert np.isclose(gamma.atoms["prob"].sum(), 1.0, atol=1e-12)

    def test_all_zero_A_raises(self):
        """If all A=0 then Γ cannot be constructed."""
        law = DiscreteLaw(
            __import__("pandas").DataFrame({
                "prob": [1.0], "A": [0.0], "B": [1.0]
            })
        )
        with pytest.raises((ValueError, ZeroDivisionError)):
            construct_gamma_discrete(law)


class TestBoundaryLimitGamma:
    def test_boundary_atom_excluded(self):
        """Γ from Q=(1-ε)δ_{(1,0)}+εδ_{(0,1)} should be δ_0."""
        eps = 0.4
        law = DiscreteLaw.from_mixture(
            weights=[1 - eps, eps],
            atoms=[(1.0, 0.0), (0.0, 1.0)],
        )
        gamma = construct_boundary_limit_gamma(law)
        assert len(gamma.atoms) == 1
        assert np.isclose(gamma.atoms["A"].values[0], 0.0, atol=1e-12)
        assert np.isclose(gamma.atoms["prob"].values[0], 1.0, atol=1e-12)

    def test_boundary_limit_mass_conservation(self):
        """Boundary Γ must also sum to 1."""
        law = DiscreteLaw.from_mixture(
            weights=[0.6, 0.4],
            atoms=[(2.0, 4.0), (0.0, 1.0)],
        )
        gamma = construct_boundary_limit_gamma(law)
        assert np.isclose(gamma.atoms["prob"].sum(), 1.0, atol=1e-12)


class TestIdentityVerification:
    """
    Test the core identity: ∫ f dΓ = E[A f(B/A)] / E[A].
    """

    def _make_law(self, eps=0.25, a=0.5):
        return DiscreteLaw.from_mixture(
            weights=[1 - eps, eps],
            atoms=[(1.0, 0.0), (a, 1.0)],
        )

    def test_perspective_cancellation_abs(self):
        """∫|z| dΓ = E[A|B/A|]/E[A] = E[|B|]/E[A]."""
        law = self._make_law()
        gamma = construct_gamma_discrete(law)
        direct = evaluate_gamma_expectation(gamma, lambda z: np.abs(z))
        formula = evaluate_primitive_formula(law, lambda z: np.abs(z))
        assert np.isclose(direct, formula, atol=1e-12)

    def test_call_price_identity(self):
        """∫(z-K)^+ dΓ = E[(B-KA)^+] / E[A]."""
        law = self._make_law(eps=0.3, a=0.4)
        gamma = construct_gamma_discrete(law)
        for K in [0.0, 0.5, 1.0, 2.0, 5.0]:
            direct = call_price_gamma(gamma, K)
            formula = call_price_primitive(law, K)
            assert np.isclose(direct, formula, atol=1e-12), \
                f"K={K}: direct={direct}, formula={formula}"

    def test_abs_moment_identity_s1(self):
        """∫|z| dΓ = E[|B|] / E[A]."""
        law = self._make_law(eps=0.2, a=0.3)
        gamma = construct_gamma_discrete(law)
        direct = abs_moment_gamma(gamma, s=1.0)
        formula = abs_moment_primitive(law, s=1.0)
        assert np.isclose(direct, formula, atol=1e-12)

    def test_abs_moment_identity_s2(self):
        """∫|z|² dΓ = E[|B|²/A] / E[A]."""
        law = self._make_law(eps=0.15, a=0.5)
        gamma = construct_gamma_discrete(law)
        direct = abs_moment_gamma(gamma, s=2.0)
        formula = abs_moment_primitive(law, s=2.0)
        assert np.isclose(direct, formula, atol=1e-12)

    def test_abs_moment_identity_s3(self):
        """∫|z|³ dΓ = E[|B|³ A^{-2}] / E[A]."""
        law = self._make_law(eps=0.1, a=0.5)
        gamma = construct_gamma_discrete(law)
        direct = abs_moment_gamma(gamma, s=3.0)
        formula = abs_moment_primitive(law, s=3.0)
        assert np.isclose(direct, formula, atol=1e-12)

    def test_identity_with_three_atoms(self):
        """Identity should hold for a three-atom law."""
        law = DiscreteLaw.from_mixture(
            weights=[0.3, 0.4, 0.3],
            atoms=[(1.0, 2.0), (2.0, 4.0), (0.5, -1.0)],
        )
        gamma = construct_gamma_discrete(law)
        f = lambda z: z ** 2 + np.abs(z)
        direct = evaluate_gamma_expectation(gamma, f)
        formula = evaluate_primitive_formula(law, f)
        assert np.isclose(direct, formula, atol=1e-12)

    def test_identity_exact_first_moment_formula(self):
        """∫|z| dΓ = E|B| / E[A] (special case of the general identity)."""
        eps = 0.2
        a = 0.25
        law = DiscreteLaw.from_mixture(
            weights=[1 - eps, eps],
            atoms=[(1.0, 0.0), (a, 1.0)],
        )
        gamma = construct_gamma_discrete(law)
        m = law.expected_A()
        E_abs_B = law.expected_abs_B()

        int_abs_z = evaluate_gamma_expectation(gamma, lambda z: np.abs(z))
        assert np.isclose(int_abs_z, E_abs_B / m, atol=1e-12)


class TestExpectedA:
    def test_wrapper(self):
        law = DiscreteLaw.from_dirac(a=3.0, b=1.0)
        assert np.isclose(expected_A(law), 3.0)
