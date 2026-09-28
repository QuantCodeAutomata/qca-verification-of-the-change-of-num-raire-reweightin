"""
Integration tests for experiments — verify paper methodology implementation.

These tests validate that each experiment's key hypotheses are correctly
implemented: exact formulas, convergence regimes, and obstruction mechanisms.
"""

import numpy as np
import pytest

from src.discrete_law import DiscreteLaw
from src.numeraire_change import (
    construct_gamma_discrete,
    construct_boundary_limit_gamma,
    call_price_gamma,
    call_price_primitive,
    abs_moment_gamma,
    abs_moment_primitive,
)
from src.wasserstein import wasserstein_p_dirac, compute_ratio_tail


# =============================================================================
# Exp 1: Identity verification tests
# =============================================================================

class TestExp1Identities:
    """Paper identity: ∫ f dΓ = E[A f(B/A)] / E[A]."""

    @pytest.mark.parametrize("n", [1, 5, 10, 20])
    def test_call_price_identity_for_sequence(self, n: int):
        eps = 2.0 ** (-n)
        a = 2.0 ** (-n)
        law = DiscreteLaw.from_mixture(
            weights=[1 - eps, eps], atoms=[(1.0, 0.0), (a, 1.0)]
        )
        gamma = construct_gamma_discrete(law)
        for K in [0.0, 0.5, 1.0, 2.0]:
            d = call_price_gamma(gamma, K)
            f = call_price_primitive(law, K)
            assert np.isclose(d, f, atol=1e-12), f"n={n}, K={K}: {d} vs {f}"

    @pytest.mark.parametrize("n", [1, 10, 30])
    def test_moment_identity_for_sequence(self, n: int):
        eps = 2.0 ** (-n)
        a = 2.0 ** (-n)
        law = DiscreteLaw.from_mixture(
            weights=[1 - eps, eps], atoms=[(1.0, 0.0), (a, 1.0)]
        )
        gamma = construct_gamma_discrete(law)
        for s in [1.0, 1.5, 2.0, 3.0]:
            d = abs_moment_gamma(gamma, s)
            f = abs_moment_primitive(law, s)
            assert np.isclose(d, f, atol=1e-12), f"n={n}, s={s}: {d} vs {f}"

    def test_boundary_example_gives_delta_0(self):
        """Boundary Q = (1-ε)δ_{(1,0)} + εδ_{(0,1)} → Γ = δ_0."""
        for eps in [0.1, 0.3, 0.5, 0.9]:
            law = DiscreteLaw.from_mixture(
                weights=[1 - eps, eps], atoms=[(1.0, 0.0), (0.0, 1.0)]
            )
            gamma = construct_boundary_limit_gamma(law)
            assert len(gamma.atoms) == 1
            assert np.isclose(gamma.atoms["A"].values[0], 0.0, atol=1e-12)

    def test_mass_conservation_always_holds(self):
        """Γ masses must sum to 1 for any valid input."""
        for eps in [0.01, 0.1, 0.5, 0.99]:
            a = eps ** 0.5
            law = DiscreteLaw.from_mixture(
                weights=[1 - eps, eps], atoms=[(1.0, 0.0), (a, 1.0)]
            )
            gamma = construct_gamma_discrete(law)
            assert np.isclose(gamma.atoms["prob"].sum(), 1.0, atol=1e-12)


# =============================================================================
# Exp 2: Wasserstein threshold tests
# =============================================================================

class TestExp2WassersteinThreshold:
    """
    Paper Theorem 3.3: M_n(σ) = ε_n^{(s*-σ)/(s*-1)} / m_n.
    Three regimes determined by sign of (s*-σ).
    """

    @pytest.mark.parametrize("s_star", [1.5, 2.0, 3.0])
    def test_moment_formula_matches_direct(self, s_star: float):
        """Closed-form M_n(σ) matches direct computation from Γ_n."""
        n = 10
        eps = 2.0 ** (-n)
        a = eps ** (1.0 / (s_star - 1.0))
        law = DiscreteLaw.from_mixture(
            weights=[1 - eps, eps], atoms=[(1.0, 0.0), (a, 1.0)]
        )
        gamma = construct_gamma_discrete(law)
        m = law.expected_A()

        for sigma in [1.0, s_star * 0.9, s_star, s_star * 1.1]:
            Mn_formula = eps ** ((s_star - sigma) / (s_star - 1.0)) / m
            Mn_direct = float(np.dot(
                gamma.atoms["prob"].values,
                np.abs(gamma.atoms["A"].values) ** sigma,
            ))
            assert np.isclose(Mn_formula, Mn_direct, rtol=1e-6), \
                f"s*={s_star}, σ={sigma}: {Mn_formula} vs {Mn_direct}"

    @pytest.mark.parametrize("s_star", [1.5, 2.0, 3.0])
    def test_stable_regime_Mn_decreasing(self, s_star: float):
        """For σ < s*, M_n(σ) should decrease toward 0."""
        sigma = s_star * 0.5
        N = 30
        Mn_vals = []
        for n in range(1, N + 1):
            eps = 2.0 ** (-n)
            a = eps ** (1.0 / (s_star - 1.0))
            m = 1 - eps + eps * a
            Mn_vals.append(eps ** ((s_star - sigma) / (s_star - 1.0)) / m)
        Mn_vals = np.array(Mn_vals)
        assert Mn_vals[-1] < Mn_vals[0], "M_n should decrease in stable regime"
        assert Mn_vals[-1] < 1e-5, f"M_n should → 0, got {Mn_vals[-1]}"

    @pytest.mark.parametrize("s_star", [1.5, 2.0, 3.0])
    def test_marginal_regime_Mn_approaches_one(self, s_star: float):
        """At σ = s*, M_n(σ) → 1."""
        sigma = s_star  # exponent = 0
        N = 50
        eps = 2.0 ** (-N)
        a = eps ** (1.0 / (s_star - 1.0))
        m = 1 - eps + eps * a
        Mn_final = eps ** (0.0) / m  # = 1 / m → 1
        assert abs(Mn_final - 1.0) < 1e-3, f"At s*={s_star}: M_n→{Mn_final}, not 1"

    @pytest.mark.parametrize("s_star", [1.5, 2.0, 3.0])
    def test_unstable_regime_Mn_increasing(self, s_star: float):
        """For σ > s*, M_n(σ) should grow unboundedly."""
        sigma = s_star * 1.5
        N = 30
        Mn_vals = []
        for n in range(1, N + 1):
            eps = 2.0 ** (-n)
            a = eps ** (1.0 / (s_star - 1.0))
            m = 1 - eps + eps * a
            exp_val = (s_star - sigma) / (s_star - 1.0)  # negative
            Mn_vals.append(eps ** exp_val / m)
        Mn_vals = np.array(Mn_vals)
        assert Mn_vals[-1] > Mn_vals[0], "M_n should grow in unstable regime"
        assert Mn_vals[-1] > 1e4, f"M_n should → ∞, got {Mn_vals[-1]}"


# =============================================================================
# Exp 3: First-order W₁ obstruction tests
# =============================================================================

class TestExp3BoundaryObstruction:
    """
    Corollary 3.4(a): W_1 fails iff ∫_{a=0}|b| dQ > 0.
    """

    @pytest.mark.parametrize("eps", [0.1, 0.25, 0.5])
    def test_W1_limit_equals_eps_over_one_minus_eps(self, eps: float):
        """W_1(Γ_n, δ_0) → ε/(1-ε) as n → ∞."""
        n = 50
        alpha = 2.0 ** (-n)
        law = DiscreteLaw.from_mixture(
            weights=[1 - eps, eps], atoms=[(1.0, 0.0), (alpha, 1.0)]
        )
        gamma = construct_gamma_discrete(law)
        W1 = wasserstein_p_dirac(gamma, c=0.0, p=1.0)
        # Exact formula: W_1 = ε / m_n
        m_n = law.expected_A()
        assert np.isclose(W1, eps / m_n, atol=1e-12)
        # For large n, m_n ≈ 1-ε, so W_1 ≈ ε/(1-ε)
        assert abs(W1 - eps / (1 - eps)) < 1e-3

    @pytest.mark.parametrize("eps", [0.1, 0.25, 0.5])
    def test_weak_convergence_holds_but_W1_fails(self, eps: float):
        """Γ_n ⇒ δ_0 (mass at big point → 0) but W_1 ≠ 0."""
        n = 40
        alpha = 2.0 ** (-n)
        law = DiscreteLaw.from_mixture(
            weights=[1 - eps, eps], atoms=[(1.0, 0.0), (alpha, 1.0)]
        )
        gamma = construct_gamma_discrete(law)
        m_n = law.expected_A()

        # Weak convergence: mass at the big atom → 0
        big_mass = eps * alpha / m_n
        assert big_mass < 1e-9, f"Big-atom mass should → 0, got {big_mass}"

        # W_1 does not go to 0
        W1 = wasserstein_p_dirac(gamma, c=0.0, p=1.0)
        assert W1 > 0.5 * eps / (1 - eps), "W_1 should stay away from 0"

    @pytest.mark.parametrize("eps", [0.1, 0.25, 0.5])
    def test_boundary_residue_equals_eps(self, eps: float):
        """∫_{a=0}|b| dQ = ε for the limit law Q."""
        law_limit = DiscreteLaw.from_mixture(
            weights=[1 - eps, eps], atoms=[(1.0, 0.0), (0.0, 1.0)]
        )
        residue = law_limit.boundary_abs_B_integral()
        assert np.isclose(residue, eps, atol=1e-12)


# =============================================================================
# Exp 4: UI necessity tests
# =============================================================================

class TestExp4UINecessity:
    """UI failure counterexample: Q_n ⇒ Q but Γ_n ↛ Γ(Q)."""

    @pytest.mark.parametrize("n", [5, 10, 30, 50])
    def test_gamma_n_masses_converge_to_half(self, n: int):
        """Γ_n masses at 0 and 1 both → 1/2 as n → ∞."""
        eps = 2.0 ** (-n)
        A_rare = 1.0 / eps
        B_rare = 1.0 / eps
        law = DiscreteLaw.from_mixture(
            weights=[1 - eps, eps], atoms=[(1.0, 0.0), (A_rare, B_rare)]
        )
        gamma = construct_gamma_discrete(law)
        m = law.expected_A()  # = 2 - eps

        # z=0 from (1,0), z=1 from (1/ε, 1/ε)
        atoms = gamma.atoms.sort_values("A").reset_index(drop=True)
        mass_0 = atoms.loc[np.isclose(atoms["A"], 0.0, atol=1e-12), "prob"].sum()
        mass_1 = atoms.loc[np.isclose(atoms["A"], 1.0, atol=1e-12), "prob"].sum()

        expected_mass_0 = (1 - eps) / m
        expected_mass_1 = 1.0 / m

        assert np.isclose(mass_0, expected_mass_0, atol=1e-12)
        assert np.isclose(mass_1, expected_mass_1, atol=1e-12)

        # For large n, both approach 0.5
        if n >= 30:
            assert abs(mass_0 - 0.5) < 1e-3
            assert abs(mass_1 - 0.5) < 1e-3

    def test_gamma_of_limit_is_delta_0(self):
        """Γ(Q) = Γ(δ_{(1,0)}) = δ_0."""
        Q = DiscreteLaw.from_dirac(a=1.0, b=0.0)
        gamma = construct_gamma_discrete(Q)
        assert len(gamma.atoms) == 1
        assert np.isclose(gamma.atoms["A"].values[0], 0.0, atol=1e-12)

    def test_ui_tail_bounded_away_from_zero(self):
        """sup_n E[A_n 1_{A_n > R}] ≥ 1 for all R > 0."""
        N = 30
        R = 100.0  # large threshold

        # For each n, find if there's a law with 1/ε_n > R
        max_tail = 0.0
        for n in range(1, N + 1):
            eps = 2.0 ** (-n)
            A_rare = 1.0 / eps
            if A_rare > R:
                # E[A_n 1_{A_n > R}] = eps * A_rare = 1
                tail = eps * A_rare
                max_tail = max(max_tail, tail)

        assert max_tail >= 1.0 - 1e-12, \
            f"UI should fail: sup_n E[A_n 1_{{A_n>R}}] ≥ 1, got {max_tail}"


# =============================================================================
# Exp 5: Numéraire inversion tests
# =============================================================================

class TestExp5NumeraireInversion:
    """S(μ) specialization: (A,B) = (X,1); ∫y^s dS(μ) = E[X^{1-s}]/E[X]."""

    @pytest.mark.parametrize("n", [1, 5, 10, 20])
    def test_inverse_moment_identity(self, n: int):
        """∫y^s dS(μ_n) = E[X_n^{1-s}] / E[X_n] for several s values."""
        eps = 2.0 ** (-n)
        a = eps  # schedule a_n = eps_n
        # (A, B) = (X, 1): atoms at (1,1) and (a,1)
        law = DiscreteLaw.from_mixture(
            weights=[1 - eps, eps], atoms=[(1.0, 1.0), (a, 1.0)]
        )
        S_mu = construct_gamma_discrete(law)
        m = law.expected_A()

        for s in [1.0, 1.5, 2.0, 3.0]:
            # Direct: ∫y^s dS(μ)
            direct = abs_moment_gamma(S_mu, s)
            # Formula: E[X^{1-s}] / E[X]
            p = law.atoms["prob"].values
            x = law.atoms["A"].values
            E_X_1ms = float(np.dot(p, x ** (1 - s)))
            formula = E_X_1ms / m
            assert np.isclose(direct, formula, atol=1e-10), \
                f"n={n}, s={s}: {direct} vs {formula}"

    def test_first_order_automatic_stability(self):
        """At s=1: ∫y dS(μ_n) = 1/E[X_n] → 1 as ε_n → 0."""
        for n in [10, 30, 50]:
            eps = 2.0 ** (-n)
            a = eps
            law = DiscreteLaw.from_mixture(
                weights=[1 - eps, eps], atoms=[(1.0, 1.0), (a, 1.0)]
            )
            S_mu = construct_gamma_discrete(law)
            m = law.expected_A()
            # ∫y dS(μ_n) = 1/m
            direct = abs_moment_gamma(S_mu, s=1.0)
            assert np.isclose(direct, 1.0 / m, atol=1e-12)
            # As n→∞, 1/m → 1
            if n >= 30:
                assert abs(direct - 1.0) < 1e-3

    def test_s_mu_atoms(self):
        """S(μ_n) has support at 1 (from X=1) and 1/a (from X=a)."""
        eps, a = 0.25, 0.5
        law = DiscreteLaw.from_mixture(
            weights=[1 - eps, eps], atoms=[(1.0, 1.0), (a, 1.0)]
        )
        S_mu = construct_gamma_discrete(law)
        z_vals = sorted(S_mu.atoms["A"].values)
        assert np.isclose(z_vals[0], 1.0, atol=1e-12)   # 1/1
        assert np.isclose(z_vals[1], 1.0 / a, atol=1e-12)  # 1/a

    def test_inverse_moment_values_in_valid_range(self):
        """∫y^s dS(μ) must be > 0 for all valid s and μ with X > 0."""
        eps, a = 0.1, 0.5
        law = DiscreteLaw.from_mixture(
            weights=[1 - eps, eps], atoms=[(1.0, 1.0), (a, 1.0)]
        )
        S_mu = construct_gamma_discrete(law)
        for s in [1.0, 2.0, 3.0]:
            val = abs_moment_gamma(S_mu, s)
            assert val > 0.0, f"Moment should be > 0 for s={s}"
