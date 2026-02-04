# DESI E-Value Analysis: Key Findings

## Executive Summary

This analysis examines the DESI DR2 BAO measurements through the lens of e-values
to assess the evidence for dynamic dark energy (w0waCDM vs LCDM).

**Main Finding**: The evidence for dynamic dark energy is **not robust** when
analyzed with proper statistical methods that prevent overfitting.

## Results

### 1. Traditional Chi-Squared Analysis

| Model | Chi-squared | Parameters |
|-------|-------------|------------|
| LCDM | 24.75 | 0 extra |
| w0waCDM | 14.92 | 2 extra (w0, wa) |
| **Delta chi2** | **9.82** | |

Naive interpretation: ~3.1 sigma preference for w0waCDM (consistent with DESI claims)

### 2. E-Value Analysis

| Method | E-value | log(E) | Sigma equiv. | Notes |
|--------|---------|--------|--------------|-------|
| Simple LR | 135.95 | 4.91 | 2.6 | BIASED (fitted to same data) |
| GROW mixture | 14.56 | 2.68 | 1.8 | Prior-sensitive |
| **Data-split test** | **1.36** | **0.31** | **0.3** | **VALID - no overfitting** |

### 3. Key Observations

1. **Data-split analysis shows weak evidence (E=1.36)**
   - Training on z<1 and testing on z>=1 gives essentially no signal
   - The apparent 3-sigma evidence disappears with proper validation

2. **Prior sensitivity is severe**
   - Narrow prior: E = 97
   - Default prior: E = 15
   - Wide prior: E = 17
   - E-values vary by factor of ~7x depending on prior choice

3. **DR1 vs DR2 comparison**
   - DR1 E-value: 0.81 (no evidence)
   - DR2 E-value: 14.56
   - The jump may indicate systematic issues rather than real signal

## Why E-Values May Be Flawed Here

### 1. Prior/Mixture Sensitivity
E-values depend critically on the choice of alternative hypothesis distribution.
There is no principled way to choose this for dark energy models.

### 2. Bayesian Contradiction
Bayesian analysis of the same data (arxiv:2511.10631) finds:
- **ln B = -0.57 ± 0.26 for DESI+CMB**
- This **FAVORS LCDM**, contradicting the 3.1σ frequentist result

### 3. Dataset Tensions
- 2.95σ tension between DESI BAO and DES-Y5 supernovae within LCDM
- w0waCDM may be "resolving" dataset inconsistencies, not detecting physics
- The extra parameters absorb systematic differences between surveys

### 4. No Complexity Penalty
Unlike Bayesian evidence (which includes Occam's razor), e-values don't
inherently penalize the 2 extra parameters in w0waCDM.

### 5. Overfitting Risk
The DESI best-fit (w0=-0.75, wa=-1.05) was derived from the same data used
to compute significance. This inflates the apparent evidence.

## Comparison: Frequentist vs Bayesian vs E-Values

| Approach | Result | Interpretation |
|----------|--------|----------------|
| Frequentist (DESI) | 3-4σ | Strong evidence for DDE |
| Bayesian evidence | ln B = -0.57 | Weak preference for LCDM |
| E-value (GROW) | E = 14.6 (~1.8σ) | Moderate evidence |
| E-value (split) | E = 1.36 (~0.3σ) | No evidence |

The disagreement between methods indicates the evidence is **not robust**.

## Recommendations

1. **Do not claim discovery** based on current data
2. **Wait for more data**: DESI DR3+ (2026-2027) will double the sample
3. **Resolve dataset tensions** before combining CMB + BAO + SNe
4. **Seek independent confirmation**: Euclid, Roman Space Telescope

## Data Sources

- DESI DR2 BAO: [arXiv:2503.14738](https://arxiv.org/abs/2503.14738)
- Data files: [CobayaSampler/bao_data](https://github.com/CobayaSampler/bao_data)
- Bayesian critique: [arXiv:2511.10631](https://arxiv.org/abs/2511.10631)
- Dataset tension analysis: [arXiv:2504.15222](https://arxiv.org/abs/2504.15222)

## Project Files

```
desi-evalue-analysis/
├── code/
│   ├── cosmology.py       # Cosmological distance calculations
│   ├── data_loader.py     # DESI data loading utilities
│   ├── evalue_analysis.py # E-value computation methods
│   └── run_analysis.py    # Main analysis script
├── data/
│   ├── dr1/               # DESI DR1 BAO measurements
│   └── dr2/               # DESI DR2 BAO measurements
├── notebooks/
│   └── desi_evalue_analysis.ipynb
└── FINDINGS.md            # This document
```

---

*Analysis performed: February 2026*
*Using DESI DR2 public data release (March 2025)*
