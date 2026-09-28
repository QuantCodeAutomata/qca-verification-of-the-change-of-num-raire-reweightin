# Experiment Results: Change-of-Numéraire Reweighting Verification

## Run Status
| Experiment | Status |
|------------|--------|
| Exp 1 – Γ Identity Verification | ✓ PASSED |
| Exp 2 – Wasserstein Stability Threshold | ✓ PASSED |
| Exp 3 – First-Order W₁ Obstruction | ✓ PASSED |
| Exp 4 – UI Necessity for Weak Stability | ✓ PASSED |
| Exp 5 – Numéraire Inversion | ✓ PASSED |

---

## Experiment 1 – Γ Identity Verification

**Objective**: Verify ∫ f dΓ = E[A f(B/A)] / E[A] for call-price and moment test functions.

**Sequence**: Q_n = (1−ε_n)δ_{(1,0)} + ε_n δ_{(a_n,1)}, ε_n = 2^{-n}, a_n = 2^{-n}, N=40.

**Result**: All identities verified to machine precision. Max absolute residual ≤ 1.11×10⁻¹⁶.

**Boundary example**: Q = (1−ε)δ_{(1,0)} + ε δ_{(0,1)} → Γ = δ₀ (boundary atom contributes zero mass after A-weighting).

**Figures**: `results/exp1_identity_residuals.png`

**Table**: `results/exp1_residuals.csv`

---

## Experiment 2 – Exact Wasserstein Stability Threshold (Theorem 3.3)

**Objective**: Reproduce the sharp threshold at σ = s* for the sequence Q_n = (1−ε_n)δ_{(1,0)} + ε_n δ_{(a_n,1)} with ε_n = 2^{-n}, a_n = ε_n^{1/(s*−1)}.

**Moment formula**: M_n(σ) = ∫|z|^σ dΓ_n = ε_n^{(s*−σ)/(s*−1)} / m_n.

**Three-regime classification**:
- σ < s*: exponent > 0 → M_n(σ) → 0 → W_σ(Γ_n, δ₀) → 0 (stable)
- σ = s*: exponent = 0 → M_n(σ) → 1 → W_σ does NOT converge to 0 (marginal)
- σ > s*: exponent < 0 → M_n(σ) → ∞ (unstable)

**Results for s* ∈ {1.5, 2.0, 3.0}**: threshold confirmed exactly in all cases.

**Figures**: `results/exp2_sstar{s*}_moments.png`, `exp2_sstar{s*}_tails.png`, `exp2_sstar{s*}_wasserstein.png`

**Table**: `results/exp2_threshold_summary.csv`

---

## Experiment 3 – First-Order W₁ Obstruction (Corollary 3.4a)

**Objective**: Show Γ_n ⇒ δ₀ weakly but W₁(Γ_n, δ₀) → ε/(1−ε) ≠ 0.

**Sequence**: Q_n = (1−ε)δ_{(1,0)} + ε δ_{(α_n,1)}, α_n = 2^{-n}, limit Q = (1−ε)δ_{(1,0)} + ε δ_{(0,1)}.

**Key formula**: ∫|z| dΓ_n = ε/m_n → ε/(1−ε).

**Results for ε ∈ {0.1, 0.25, 0.5}**:
| ε    | W₁ limit (numerical) | W₁ limit (theory) | Boundary residue | Weak ✓ | W₁ ✗ |
|------|----------------------|-------------------|-----------------|--------|------|
| 0.10 | 0.111111             | 0.111111          | 0.10            | ✓      | ✗    |
| 0.25 | 0.333333             | 0.333333          | 0.25            | ✓      | ✗    |
| 0.50 | 1.000000             | 1.000000          | 0.50            | ✓      | ✗    |

**Figures**: `results/exp3_eps{ε}.png`  |  **Table**: `results/exp3_summary.csv`

---

## Experiment 4 – UI Necessity for Weak Stability

**Objective**: Show Q_n ⇒ δ_{(1,0)} but Γ_n → ½δ₀ + ½δ₁ ≠ δ₀ = Γ(Q) when UI of A_n fails.

**Sequence**: Q_n = (1−ε_n)δ_{(1,0)} + ε_n δ_{(1/ε_n, 1/ε_n)}, ε_n = 2^{-n}.

**Mechanism**: E[A_n] = 2−ε_n → 2; rare atom has ratio z = 1 with persistent weighted mass 1/m_n → 1/2.

**UI failure**: sup_n E[A_n 1_{A_n>R}] = 1 for all R > 0.

**Result**: Γ_n → ½δ₀ + ½δ₁ (confirmed); UI failure confirmed.

**Figure**: `results/exp4_ui_failure.png`  |  **Table**: `results/exp4_gamma_masses.csv`

---

## Experiment 5 – Numéraire Inversion (S(μ) specialization)

**Objective**: Verify ∫y^s dS(μ) = E[X^{1-s}]/E[X] for (A,B) = (X,1), S(μ) = law of 1/X under X-weighting.

**Schedules tested**: a_n = ε_n (c=1), a_n = ε_n^{0.5} (c=0.5), a_n = ε_n² (c=2).

**Identity residuals**: max relative residual ≤ 2.22×10⁻¹⁶ (machine precision) for all schedules.

**First-order stability**: ∫y dS(μ_n) = 1/m_n → 1/1 = 1 as n → ∞.

**Figures**: `results/exp5_inverse_moments.png`, `results/exp5_first_order_stability.png`

**Table**: `results/exp5_identity_residuals.csv`

---

## Summary

All five experiments confirm the theoretical predictions exactly:
- **Exp 1**: Γ identities hold to machine precision; boundary zero-weighting is correct.
- **Exp 2**: The Wasserstein stability threshold at σ = s* is sharp and necessary.
- **Exp 3**: At s = 1 the numéraire cancels exactly; only boundary payoff mass obstructs W₁.
- **Exp 4**: UI of the numéraire is not a technical artifact—its failure breaks weak stability.
- **Exp 5**: The numéraire-inversion identity holds to machine precision across all schedules.
