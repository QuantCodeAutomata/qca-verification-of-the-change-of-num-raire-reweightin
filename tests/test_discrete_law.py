"""
Tests for src/discrete_law.py — DiscreteLaw class.
"""

import numpy as np
import pandas as pd
import pytest

from src.discrete_law import DiscreteLaw


class TestDiscreteLawConstruction:
    def test_from_mixture_two_atoms(self):
        law = DiscreteLaw.from_mixture(
            weights=[0.7, 0.3],
            atoms=[(1.0, 0.0), (2.0, 1.0)],
        )
        assert len(law.atoms) == 2
        assert np.isclose(law.atoms["prob"].sum(), 1.0)

    def test_from_dirac(self):
        law = DiscreteLaw.from_dirac(a=3.0, b=7.0)
        assert len(law.atoms) == 1
        assert np.isclose(law.atoms["A"].values[0], 3.0)
        assert np.isclose(law.atoms["B"].values[0], 7.0)

    def test_probabilities_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1"):
            DiscreteLaw.from_mixture(
                weights=[0.5, 0.3],
                atoms=[(1.0, 0.0), (2.0, 1.0)],
            )

    def test_negative_probability_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            DiscreteLaw(pd.DataFrame({
                "prob": [-0.1, 1.1],
                "A": [1.0, 2.0],
                "B": [0.0, 1.0],
            }))

    def test_negative_A_allowed_for_gamma_support(self):
        """DiscreteLaw allows negative A to support Γ whose 1D support lives on ℝ."""
        law = DiscreteLaw(pd.DataFrame({
            "prob": [0.5, 0.5],
            "A": [-2.0, 3.0],
            "B": [0.0, 1.0],
        }))
        assert len(law.atoms) == 2

    def test_missing_column_raises(self):
        with pytest.raises(ValueError, match="columns"):
            DiscreteLaw(pd.DataFrame({"prob": [1.0], "A": [1.0]}))


class TestDiscreteLawMoments:
    def test_expected_A_dirac(self):
        law = DiscreteLaw.from_dirac(a=5.0, b=2.0)
        assert np.isclose(law.expected_A(), 5.0)

    def test_expected_B_dirac(self):
        law = DiscreteLaw.from_dirac(a=5.0, b=2.0)
        assert np.isclose(law.expected_B(), 2.0)

    def test_expected_abs_B(self):
        law = DiscreteLaw.from_mixture(
            weights=[0.5, 0.5],
            atoms=[(1.0, -3.0), (1.0, 5.0)],
        )
        assert np.isclose(law.expected_abs_B(), 4.0)

    def test_expected_mixed_moment_s1(self):
        """At s=1, E[|B|^1 A^0] = E[|B|]."""
        law = DiscreteLaw.from_mixture(
            weights=[0.5, 0.5],
            atoms=[(2.0, 4.0), (3.0, 9.0)],
        )
        mixed = law.expected_mixed_moment(s=1.0)
        E_abs_B = law.expected_abs_B()
        assert np.isclose(mixed, E_abs_B)

    def test_expected_mixed_moment_boundary_zero(self):
        """States with A=0 contribute 0 to the mixed moment for s≥1."""
        law = DiscreteLaw.from_mixture(
            weights=[0.5, 0.5],
            atoms=[(0.0, 1.0), (1.0, 0.0)],
        )
        # Only atom (1,0): |0|^s * 1^{1-s} = 0
        assert np.isclose(law.expected_mixed_moment(s=2.0), 0.0)

    def test_boundary_abs_B_integral(self):
        """∫_{a=0}|b| dQ should pick up only A=0 atoms."""
        eps = 0.3
        law = DiscreteLaw.from_mixture(
            weights=[1 - eps, eps],
            atoms=[(1.0, 0.0), (0.0, 1.0)],
        )
        assert np.isclose(law.boundary_abs_B_integral(), eps)

    def test_boundary_abs_B_zero_when_no_boundary(self):
        """No A=0 atoms ⟹ boundary integral = 0."""
        law = DiscreteLaw.from_mixture(
            weights=[0.6, 0.4],
            atoms=[(1.0, 0.0), (2.0, 1.0)],
        )
        assert np.isclose(law.boundary_abs_B_integral(), 0.0)


class TestEdgeCases:
    def test_single_atom(self):
        law = DiscreteLaw.from_dirac(a=1.0, b=0.0)
        assert np.isclose(law.expected_A(), 1.0)
        assert np.isclose(law.expected_B(), 0.0)

    def test_three_atom_mass_conservation(self):
        law = DiscreteLaw.from_mixture(
            weights=[0.2, 0.3, 0.5],
            atoms=[(1.0, 0.0), (2.0, 3.0), (0.5, -1.0)],
        )
        assert np.isclose(law.atoms["prob"].sum(), 1.0)
        assert np.isclose(law.expected_A(), 0.2 * 1 + 0.3 * 2 + 0.5 * 0.5)
