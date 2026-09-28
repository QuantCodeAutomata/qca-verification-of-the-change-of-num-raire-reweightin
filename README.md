# Change-of-Numéraire Reweighting: Verification and Wasserstein Stability

A Python repository implementing and verifying the foundational measure-transformation
identities for the **change-of-numéraire reweighting operator Γ**, together with
exact replication of its Wasserstein-stability threshold, boundary-obstruction, and
numéraire-inversion applications. All experiments are deterministic and use only
synthetic discrete probability laws—no historical market data are required.

---

## Background

Given a bivariate discrete law Q on [0,∞) × ℝ, the **Γ operator** is defined as:

> **Γ(Q) = law of Z = B/A under the weighted measure (A/E[A]) dQ**

For states with A = 0 (boundary), the weight A·dQ vanishes, so those states contribute
zero mass to Γ—a property called *boundary zero-weighting*.

The core identity is:

```
∫ f dΓ = E[A f(B/A)] / E[A]
```

Special cases include call-price and moment identities:
```
∫(z-K)⁺ dΓ = E[(B-KA)⁺] / E[A]
∫|z|ˢ dΓ = E[|B|ˢ A^{1-s}] / E[A]
```

---

## Repository Structure

```
├── data/                    # Data directory (no files needed—all experiments use synthetic laws)
├── src/
│   ├── discrete_law.py      # DiscreteLaw: (A,B) atoms, moments, boundary diagnostics
│   ├── numeraire_change.py  # Γ operator: construct_gamma_discrete, identity evaluators
│   └── wasserstein.py       # W_p distances, mixed moments, ratio-tail diagnostics
├── exp/
│   ├── exp1_identity_verification.py  # Exp 1: Perspective-cancellation identities
│   ├── exp2_wasserstein_stability.py  # Exp 2: Sharp Wasserstein threshold (Theorem 3.3)
│   ├── exp3_first_order_stability.py  # Exp 3: W₁ failure and boundary obstruction
│   ├── exp4_ui_necessity.py           # Exp 4: UI failure → weak stability failure
│   └── exp5_numeraire_inversion.py    # Exp 5: Numéraire-inversion application
├── tests/
│   ├── test_discrete_law.py
│   ├── test_numeraire_change.py
│   ├── test_wasserstein.py
│   └── test_experiments.py
├── results/                 # Output tables, plots, and RESULTS.md
│   └── RESULTS.md
├── run_all.py               # Run all experiments and generate RESULTS.md
└── README.md
```

---

## Experiments

| ID    | Title                                                                                  |
|-------|----------------------------------------------------------------------------------------|
| Exp 1 | Verification of the Γ identities with boundary zero-weighting                          |
| Exp 2 | Exact Wasserstein stability threshold via Theorem 3.3 (sharp threshold example)        |
| Exp 3 | First-order W₁ stability and boundary obstruction at A = 0 (Corollary 3.4a)           |
| Exp 4 | Necessity of uniform integrability of the numéraire for weak stability                 |
| Exp 5 | Numéraire-inversion application and inverse-moment stability threshold                 |

---

## Quick Start

```bash
# Install dependencies
pip install numpy pandas scipy matplotlib

# Run all experiments and generate RESULTS.md
python run_all.py

# Run individual experiment
python -m exp.exp1_identity_verification
python -m exp.exp2_wasserstein_stability
python -m exp.exp3_first_order_stability
python -m exp.exp4_ui_necessity
python -m exp.exp5_numeraire_inversion

# Run the test suite
pytest tests/ -v
```

---

## Dependencies

| Library    | Use                                        |
|------------|--------------------------------------------|
| numpy      | Vectorized arithmetic on discrete atoms    |
| pandas     | Atom tables and aggregation                |
| scipy      | `wasserstein_distance` (W₁ validation)     |
| matplotlib | All plots (residual curves, moment curves) |

No backtesting, portfolio-optimization, or market-data libraries are used.

---

## Key Results (summary)

- **Exp 1**: All Γ identities verified to machine precision (max residual ≤ 1.1×10⁻¹⁶).
- **Exp 2**: Sharp threshold at σ = s* confirmed: M_n(σ) → 0 for σ < s*, → 1 at σ = s*, → ∞ for σ > s*.
- **Exp 3**: W₁(Γ_n, δ₀) → ε/(1−ε) ≠ 0 despite weak convergence; obstruction = boundary mass ε.
- **Exp 4**: Γ_n → ½δ₀ + ½δ₁ ≠ δ₀ = Γ(Q) when UI of A_n fails.
- **Exp 5**: Numéraire-inversion identity ∫y^s dS(μ) = E[X^{1-s}]/E[X] verified to machine precision (max rel residual ≤ 2.2×10⁻¹⁶).

See `results/RESULTS.md` for full tables and figures.
