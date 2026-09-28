"""
Discrete probability law representation for (A, B) pairs on [0,∞) × R.

Each law is stored as a pandas DataFrame with columns:
  prob : float  – probability weight (must sum to 1)
  A    : float  – first coordinate (A ≥ 0, the numéraire)
  B    : float  – second coordinate (B ∈ R, the payoff)

A 1-D law on (0,∞) can be stored with B set to a constant (e.g. 1.0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class DiscreteLaw:
    """Discrete bivariate law Q on [0,∞) × R stored as weighted atoms."""

    def __init__(self, atoms: pd.DataFrame) -> None:
        """
        Parameters
        ----------
        atoms : pd.DataFrame
            Must contain columns 'prob', 'A', 'B' with float dtype.
            Probabilities must be non-negative and sum to 1.
        """
        required = {"prob", "A", "B"}
        if not required.issubset(atoms.columns):
            raise ValueError(f"atoms DataFrame must contain columns {required}")

        atoms = atoms[["prob", "A", "B"]].copy().astype(float)
        if np.any(atoms["prob"] < 0):
            raise ValueError("All probabilities must be non-negative.")

        total = atoms["prob"].sum()
        if not np.isclose(total, 1.0, atol=1e-10):
            raise ValueError(f"Probabilities must sum to 1, got {total}")

        self.atoms: pd.DataFrame = atoms.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_dirac(cls, a: float, b: float) -> "DiscreteLaw":
        """Create a Dirac mass δ_{(a,b)}."""
        return cls(pd.DataFrame({"prob": [1.0], "A": [a], "B": [b]}))

    @classmethod
    def from_mixture(cls, weights: list[float], atoms: list[tuple[float, float]]) -> "DiscreteLaw":
        """
        Create a mixture from explicit weights and (A, B) pairs.

        Parameters
        ----------
        weights : list of float – probabilities (must sum to 1)
        atoms   : list of (a, b) tuples
        """
        probs = np.array(weights, dtype=float)
        As = np.array([x[0] for x in atoms], dtype=float)
        Bs = np.array([x[1] for x in atoms], dtype=float)
        df = pd.DataFrame({"prob": probs, "A": As, "B": Bs})
        return cls(df)

    # ------------------------------------------------------------------
    # Moment utilities
    # ------------------------------------------------------------------

    def expected_A(self) -> float:
        """Compute E[A] = Σ p_i a_i."""
        return float(np.dot(self.atoms["prob"], self.atoms["A"]))

    def expected_B(self) -> float:
        """Compute E[B] = Σ p_i b_i."""
        return float(np.dot(self.atoms["prob"], self.atoms["B"]))

    def expected_abs_B(self) -> float:
        """Compute E[|B|] = Σ p_i |b_i|."""
        return float(np.dot(self.atoms["prob"], np.abs(self.atoms["B"])))

    def expected_mixed_moment(self, s: float) -> float:
        """
        Compute E[|B|^s A^{1-s}] = Σ p_i |b_i|^s a_i^{1-s}.

        States with A=0 contribute 0 (boundary zero-weighting when s < 1 would
        require care; here only s ≥ 1 is relevant and 0^{1-s} = 0 for a_i=0).
        """
        p = self.atoms["prob"].values
        a = self.atoms["A"].values
        b = np.abs(self.atoms["B"].values)
        mask = a > 0
        result = np.zeros(len(p))
        result[mask] = p[mask] * (b[mask] ** s) * (a[mask] ** (1 - s))
        # boundary states (a=0): contribution is 0 (b^s * 0^{1-s} = 0 for s≥1)
        return float(result.sum())

    def boundary_abs_B_integral(self) -> float:
        """
        Compute ∫_{a=0} |b| dQ = Σ_{a_i=0} p_i |b_i|.
        This is the first-order boundary obstruction for W_1-stability.
        """
        mask = self.atoms["A"] == 0
        sub = self.atoms[mask]
        return float(np.dot(sub["prob"], np.abs(sub["B"])))

    def to_table_str(self) -> str:
        """Return a formatted string of the atoms table."""
        return self.atoms.to_string(index=False)

    def __repr__(self) -> str:
        return f"DiscreteLaw(n_atoms={len(self.atoms)}, E[A]={self.expected_A():.6g})"
