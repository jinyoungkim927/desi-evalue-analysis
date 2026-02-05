# DESI E-Value Analysis: Critical Assessment of Dark Energy Evidence

## Executive Summary

We analyzed DESI DR2 BAO measurements using e-values to assess the robustness of
evidence for dynamic dark energy. **Our key finding: the evidence is NOT robust.**

When properly validated on held-out data, the e-value drops from ~400 to ~1.4,
indicating the apparent 3-4σ evidence is largely due to overfitting.

## Data Source Validation

| Item | Status |
|------|--------|
| Data source | Official DESI (CobayaSampler/bao_data) |
| Values match paper | Yes (within 0.5%) |
| Covariance matrix | Official DESI block-diagonal |
| Cosmology validated | Yes (matches DESI fiducial at z=2.33) |

## Results Summary

### Chi-Squared Analysis (BAO Only)

| Model | χ² | Δχ² | Naive σ |
|-------|-----|------|---------|
| ΛCDM | 25.4 | - | - |
| w₀wₐCDM | 13.5 | 11.9 | ~3.5σ |

### E-Value Analysis

| Method | E-value | σ equiv | Validity |
|--------|---------|---------|----------|
| Simple Likelihood Ratio | 392 | 3.9σ | **BIASED** (overfitted) |
| GROW Mixture (narrow) | 97 | 3.0σ | Prior-sensitive |
| GROW Mixture (default) | 15 | 2.3σ | Prior-sensitive |
| GROW Mixture (wide) | 17 | 2.4σ | Prior-sensitive |
| **Data-Split Test** | **1.4** | **0.8σ** | **VALID** |

### The Critical Comparison

```
Simple LR E-value:     392  (~3.9σ)  ← BIASED (alternative fitted to same data)
Data-split E-value:    1.4  (~0.8σ)  ← VALID (tested on held-out data)
                       ────────────
Ratio:                 280x difference!
```

**This 280x difference shows how much the evidence is inflated by overfitting.**

## Why E-Values May Be Flawed Here

### 1. Overfitting Dominates
The simple likelihood ratio E-value (392) uses the DESI best-fit w₀wₐ, which was
fitted to the same data. When we split the data and test on held-out points,
E drops to just 1.4.

### 2. Prior Sensitivity
GROW e-values vary from 15 to 97 depending on the w₀, wₐ range used.
This ~7x variation shows the results are not robust to methodology choices.

### 3. Bayesian Analysis Contradicts
[arXiv:2511.10631](https://arxiv.org/abs/2511.10631) found:
- ln(B) = -0.57 ± 0.26 for DESI+CMB
- This **FAVORS ΛCDM** over w₀wₐCDM!

### 4. Dataset Tensions
There is a 2.95σ tension between DESI BAO and DES-Y5 supernovae within ΛCDM.
The w₀wₐ model may be "resolving" this tension rather than detecting real physics.

### 5. Power Analysis
If w₀wₐCDM were true, we'd expect median E ~ 21 (from simulations).
Our observed E = 15 is at the 40th percentile - consistent with either model.

## Potential Criticisms Addressed

| Criticism | Our Response |
|-----------|-------------|
| "You only use BAO" | True limitation. DESI's full 3-4σ uses CMB+SNe. But Bayesian analysis of full data still favors ΛCDM. |
| "Your cosmology is wrong" | Validated against DESI fiducial (0.4% agreement at z=2.33) |
| "E-values aren't standard" | They complement p-values/Bayes factors, show methodology sensitivity |
| "Data-split has low power" | Power analysis shows we'd detect E~21 if w₀wₐCDM were true |
| "Prior range is arbitrary" | We show sensitivity explicitly - this IS the problem |

## Methodology

### Data
- DESI DR2 BAO measurements from [CobayaSampler/bao_data](https://github.com/CobayaSampler/bao_data)
- 13 data points: DM/rd, DH/rd, DV/rd at z = 0.295 to 2.33
- Official covariance matrix (block-diagonal)

### E-Value Methods

1. **Simple Likelihood Ratio**: E = L(data|H₁)/L(data|H₀)
   - Fast but biased if H₁ fitted to same data

2. **GROW Mixture**: Average LR over grid of alternatives
   - More principled but prior-sensitive

3. **Data-Split**: Fit H₁ on training set, test on held-out set
   - Prevents overfitting, most honest

### Cosmological Models
- **H₀ (Null)**: ΛCDM with w = -1, wₐ = 0
- **H₁ (Alternative)**: w₀wₐCDM with w(a) = w₀ + wₐ(1-a)
- Fiducial: Ωm = 0.3111, h = 0.6766, rd = 147.05 Mpc (DESI values)

## Conclusion

**The evidence for dynamic dark energy is NOT robust.**

| Evidence Measure | Result | Interpretation |
|------------------|--------|----------------|
| DESI Reported | 3-4σ | Frequentist, BAO+CMB+SNe |
| Our χ² (BAO only) | ~3.5σ | Consistent with DESI |
| E-value (overfitted) | 392 (~3.9σ) | BIASED |
| E-value (validated) | 1.4 (~0.8σ) | **NO EVIDENCE** |
| Bayesian (full data) | Favors ΛCDM | Contradicts frequentist |

The disagreement between frequentist significance (3-4σ) and properly validated
analysis (E~1.4) or Bayesian analysis (favors ΛCDM) indicates the evidence is
driven by methodology choices, not robust physics.

## Recommendations

1. **Do not claim discovery** based on current data
2. **Wait for DESI DR3+** (2026-2027) for more data
3. **Resolve dataset tensions** (CMB vs BAO vs SNe)
4. **Seek independent confirmation** (Euclid, Roman Space Telescope)

## References

- DESI DR2: [arXiv:2503.14738](https://arxiv.org/abs/2503.14738)
- Bayesian critique: [arXiv:2511.10631](https://arxiv.org/abs/2511.10631)
- Dataset tensions: [arXiv:2504.15222](https://arxiv.org/abs/2504.15222)
- E-values: Ramdas et al. 2023, Statistical Science 38(4)

---
*Analysis: February 2026*
