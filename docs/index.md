# DESI E-Value Analysis

## Assessment of Dark Energy Evidence

This project applies e-value methodology to the DESI DR2 BAO measurements to assess the evidence for dynamical dark energy (w₀wₐCDM) over the cosmological constant (ΛCDM).

---

## Key Findings

1. **Cross-dataset tension is the strongest finding.** DES-Y5's best-fit w₀wₐCDM parameters predict DESI data *worse* than ΛCDM (E = 0.19), while Pantheon+'s predict it well (E = 2049). This ~10,000× asymmetry indicates the supernova catalogs disagree on what dark energy dynamics look like.

2. **Valid e-values converge on moderate evidence.** Two independent valid methods — the uniform mixture (E ≈ 15) and the LOO average (E ≈ 10) — both indicate moderate evidence for w₀wₐCDM, roughly 2.2–2.3σ.

3. **The data-split test is inconclusive (not negative).** E = 1.4, but power calibration shows the median E under H₁ is only ~2.7. The test is underpowered for wₐ.

---

## Results at a Glance

| Method | E-value | ~σ | Status |
|--------|---------|-----|--------|
| Maximized Likelihood Ratio | 392 | — | **NOT AN E-VALUE** (E[E\|H₀] = ∞) |
| LOO product | 1062 | — | **NOT VALID** (dependent folds) |
| Uniform Mixture (default) | **15** | **2.3** | **Valid e-value** |
| **LOO Average** | **10** | **2.2** | **Valid e-value** |
| Data-Split | 1.4 | 0.8 | Valid, but underpowered |
| Cross-dataset (DES-Y5 → DESI) | 0.19 | — | Evidence of tension |

---

## Quick Links

### Main Results
- **[Full Mathematical Analysis](full_analysis.md)** — Complete derivations and calculations
- **[Analysis Report (HTML)](analysis_report.html)** — Visual overview

### Background Materials
- **[E-Values Explained](background_evalues.md)** — Mathematical theory including LOO average validity
- **[Cosmology Background](background_cosmology.md)** — BAO and dark energy physics
- **[DESI Data Documentation](background_desi_data.md)** — Raw data and validation

### Papers
- **[Main Paper (PDF)](../paper/main.pdf)** — Full research paper
- **[Walkthrough (PDF)](../paper/walkthrough.pdf)** — Step-by-step walkthrough with worked examples

### Resources
- **[GitHub Repository](https://github.com/jinyoungkim927/desi-evalue-analysis)** — Full code and data

---

## Data Sources

- DESI DR2 BAO: [arXiv:2503.14738](https://arxiv.org/abs/2503.14738)
- Official data: [CobayaSampler/bao_data](https://github.com/CobayaSampler/bao_data)
- Bayesian critique: [arXiv:2511.10631](https://arxiv.org/abs/2511.10631)
- Dataset tensions: [arXiv:2504.15222](https://arxiv.org/abs/2504.15222)

---

*Analysis: February 2026*
