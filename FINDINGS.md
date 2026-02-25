# DESI E-Value Analysis: Assessment of Dark Energy Evidence

## Executive Summary

We analyzed DESI DR2 BAO measurements using e-values to assess the evidence for
dynamical dark energy. The key findings:

- **Cross-dataset tension is the strongest finding.** DES-Y5's best-fit w0waCDM
  parameters predict DESI data *worse* than LCDM (E = 0.19), while Pantheon+'s
  parameters predict it well (E = 2049). This ~10,000x asymmetry indicates the
  supernova catalogs disagree on what dark energy dynamics look like, raising the
  possibility that w0waCDM is absorbing inter-dataset tension rather than detecting
  new physics.

- **Valid e-values converge on moderate evidence.** Two independent valid methods
  give consistent results: the uniform mixture e-value (E ~ 15, ln E ~ 2.7) and
  the LOO average (E ~ 10, ln E ~ 2.3) both indicate moderate evidence for
  w0waCDM over LCDM in DESI BAO data alone.

- **The data-split test is inconclusive (not negative).** E = 1.4 from the
  redshift-based data split, but power calibration shows the median E_split under
  H1 is only ~2.7. The test has limited power for w_a because the low-redshift
  training set (z < 1) has only 23-48% leverage on the time-evolution parameter.
  The observed E = 1.4 is consistent with both hypotheses.

## Data Source Validation

| Item | Status |
|------|--------|
| Data source | Official DESI (CobayaSampler/bao_data) |
| Values match paper | Yes (within 0.5%) |
| Covariance matrix | Official DESI block-diagonal |
| Cosmology validated | Yes (matches DESI fiducial at z=2.33) |

## Results Summary

### Chi-Squared Analysis (BAO Only)

| Model | chi-sq | Delta chi-sq | Correct sigma (k=2) |
|-------|--------|--------------|---------------------|
| LCDM | 25.4 | - | - |
| w0waCDM | 13.5 | 11.9 | 2.8 sigma |

Note: The sometimes-quoted 3.5 sigma uses sqrt(Delta chi-sq), which is only valid
for k=1 extra parameter. For k=2, the correct conversion uses the chi-sq(2)
distribution: p = exp(-11.9/2) = 0.0026, corresponding to 2.8 sigma (two-sided).

### E-Value Analysis

| Method | E-value | ln(E) | ~sigma | Status |
|--------|---------|-------|--------|--------|
| Maximized Likelihood Ratio | 392 | 5.97 | --- | **NOT AN E-VALUE** (E[E\|H0] = infinity) |
| LOO product | 1062 | 6.97 | --- | **NOT VALID** (dependent folds) |
| Uniform Mixture (narrow) | 97 | 4.57 | 3.0 | Valid e-value, prior-sensitive |
| **Uniform Mixture (default)** | **15** | **2.71** | **2.3** | **Valid e-value, moderate evidence** |
| Uniform Mixture (wide) | 17 | 2.83 | 2.4 | Valid e-value, prior-sensitive |
| **LOO average** | **10.2** | **2.32** | **2.2** | **Valid e-value, moderate evidence** |
| Data-Split Test | 1.4 | 0.34 | 0.8 | Valid e-value, but underpowered |

### Valid Methods Converge

The two independent valid methods give consistent results:

```
Uniform mixture (default):  E ~ 15   (ln E ~ 2.7)
LOO average:                E ~ 10   (ln E ~ 2.3)
                            ────────────────────
Convergence:                E ~ 10-15 (moderate evidence)
```

The data-split E = 1.4 is consistent with these once power loss is accounted for
(median E_split under H1 ~ 2.7).

### Invalid Statistics (for reference only)

The maximized likelihood ratio (392) is NOT a valid e-value -- its expected value
under H0 is infinite for k >= 2 extra parameters. The LOO product (1062) is NOT
valid because the leave-one-out folds have overlapping training sets, violating the
independence required for product combination. Neither should be interpreted as
calibrated evidence.

### Cross-Dataset E-Values (Strongest Finding)

| Training Dataset | Test Dataset | (w0, wa) | E-value |
|------------------|-------------|----------|---------|
| DESI (fitted) | Pantheon+ | (-0.86, -0.43) | 1.5 |
| DESI (fitted) | DES-Y5 | (-0.86, -0.43) | 86 |
| Pantheon+ | DESI | (-0.90, -0.20) | 2049 |
| **DES-Y5** | **DESI** | **(-0.65, -1.20)** | **0.19** |

The DES-Y5 -> DESI result (E = 0.19) is the analysis's most informative finding:
if dark energy dynamics were real and consistent across datasets, DES-Y5's
parameters should predict DESI data at least as well as LCDM.

### Information Criteria

| Method | Value | Favors | Penalty |
|--------|-------|--------|---------|
| Frequentist p (k=2) | p = 0.0026 (2.8 sigma) | w0waCDM | None |
| Delta AIC | 7.9 | w0waCDM | 2k = 4 |
| Delta BIC (n=13) | 6.8 | w0waCDM | k ln n ~ 5.1 |
| Uniform mixture E-value | E ~ 15 | w0waCDM | Grid averaging |
| LOO average E-value | E ~ 10 | w0waCDM | Out-of-sample (avg) |
| Bayes factor (Ong et al.) | ln B = -0.57 | LCDM | Full prior volume |
| Data-split E-value | E = 1.4 | Inconclusive | Out-of-sample |
| Cross-dataset (DES-Y5) | E = 0.19 | Tension | Cross-prediction |

## Discussion of Key Issues

### 1. Invalid Statistics Must Be Clearly Labeled
The maximized likelihood ratio (392) has E[E|H0] = infinity for k=2, so it is not
a valid e-value. The LOO product (1062) uses overlapping training sets, so the
product has unknown expectation under H0. Both are included as descriptive
statistics only.

### 2. LOO Average vs. LOO Product
Each individual LOO e-value E_k is valid (conditional on training data, the
held-out bin tests a pre-specified alternative). The AVERAGE of e-values is valid
by linearity of expectation: E[avg] = (1/K) sum E[E_k] <= 1. The PRODUCT requires
independence, which fails because LOO training sets overlap. E_1 uses theta_{-1}
(fitted on bins 2-7) and E_2 uses theta_{-2} (fitted on bins 1,3-7): both share
bin 3, creating dependence.

### 3. Prior Sensitivity as Occam Razor
Uniform mixture e-values vary from 15 to 97 depending on the w0, wa range.
This variation reflects the Occam razor operating as expected: a narrower prior
concentrating mass near the MLE produces larger e-values because it wastes less
probability on distant parameter values. This is a standard feature of
mixture-based tests, not a deficiency.

### 4. Bayesian Analysis
Ong et al. (arXiv:2511.10631) found ln(B) = -0.57 +/- 0.26 for DESI+CMB,
modestly favoring LCDM. However, the Bayes factor is itself prior-dependent, and
this result uses a different dataset combination (DESI+CMB rather than DESI BAO
alone). It should be considered alongside, not in place of, the e-value results.

### 5. Dataset Tensions
There is a 2.95 sigma tension between DESI BAO and DES-Y5 supernovae within LCDM.
The w0wa model may be "resolving" this tension (acting as a tension absorber)
rather than detecting real physics. Our cross-dataset e-values make this tension
quantitative: E = 0.19 means DES-Y5's preferred parameters make DESI data less
probable than LCDM.

### 6. Data-Split Power Limitation
The data-split E = 1.4 is inconclusive, not negative. Power calibration (500 Monte
Carlo simulations) shows that even when w0waCDM is the true model, the data-split
test gives median E ~ 2.7. The observed E = 1.4 cannot distinguish the hypotheses.

## Potential Criticisms Addressed

| Criticism | Response |
|-----------|----------|
| "You only use BAO" | True limitation. DESI's full 3-4 sigma uses CMB+SNe. |
| "Your cosmology is wrong" | Validated against DESI fiducial (0.4% agreement at z=2.33) |
| "E-values aren't standard" | They complement p-values/Bayes factors, showing methodology sensitivity |
| "Data-split has low power" | Confirmed by power calibration. This is why we also report LOO average and mixture methods. |
| "Prior range is arbitrary" | The variation (15-97) reflects normal Occam razor behavior |
| "LOO product gives E=1062" | The LOO product is NOT valid (overlapping training sets). The LOO average E~10 is valid. |
| "The sigma conversion is wrong" | For k=2 d.o.f., the correct conversion gives 2.8 sigma, not 3.5 sigma. |

## Methodology

### Data
- DESI DR2 BAO measurements from [CobayaSampler/bao_data](https://github.com/CobayaSampler/bao_data)
- 13 data points: DM/rd, DH/rd, DV/rd at z = 0.295 to 2.33
- Official covariance matrix (block-diagonal)

### E-Value Methods

1. **Maximized Likelihood Ratio**: LR = L(data|H1)/L(data|H0) with MLE parameters
   - NOT a valid e-value when H1 is fitted to the same data (E[E|H0] = infinity for k >= 2)
   - Included for reference only

2. **Uniform Mixture E-Value**: Average LR over uniform grid of alternatives
   - Valid e-value; prior-sensitive (as expected for any mixture/Bayesian method)
   - Default range yields E ~ 15, moderate evidence

3. **Data-Split**: Fit H1 on training set, test on held-out set
   - Valid e-value; but limited power for w_a with redshift-based splits
   - Power calibration: median E under H1 ~ 2.7

4. **LOO Average**: Average of leave-one-out e-values across 7 redshift bins
   - Valid e-value (by linearity of expectation)
   - E ~ 10, moderate evidence
   - Note: LOO PRODUCT (1062) is NOT valid (overlapping training sets)

### Cosmological Models
- **H0 (Null)**: LCDM with w = -1, wa = 0
- **H1 (Alternative)**: w0waCDM with w(a) = w0 + wa(1-a)
- Fiducial: Omega_m = 0.3111, h = 0.6766, rd = 147.05 Mpc (DESI values)

## Conclusion

The evidence for dynamical dark energy from DESI BAO data is moderate but
undermined by cross-dataset inconsistency.

| Evidence Measure | Result | Interpretation |
|------------------|--------|----------------|
| DESI Reported | 3-4 sigma | Frequentist, BAO+CMB+SNe |
| Our chi-sq (BAO only, k=2) | 2.8 sigma | Correct conversion |
| Maximized LR | 392 | NOT AN E-VALUE |
| LOO product | 1062 | NOT VALID (dependent folds) |
| Uniform Mixture E-value | 15 (ln E ~ 2.7) | Moderate evidence |
| LOO Average E-value | 10.2 (ln E ~ 2.3) | Moderate evidence |
| Data-Split E-value | 1.4 (~0.8 sigma) | Inconclusive (underpowered) |
| Cross-dataset (DES-Y5 -> DESI) | E = 0.19 | Evidence of tension |
| Bayesian (DESI+CMB) | ln B = -0.57 | Modestly favors LCDM |

Valid e-values converge on E ~ 10-15 (moderate evidence for w0waCDM). This is
real evidence that should be taken seriously, but it does not constitute a
discovery. The most informative finding is the cross-dataset tension: DES-Y5 and
DESI disagree on what dark energy dynamics should look like (E = 0.19), which is
a more fundamental problem than the strength of evidence in any single dataset.

## Recommendations

1. **Resolve dataset tensions** between DES-Y5, Pantheon+, and DESI -- this is the priority
2. **Wait for DESI DR3+** (2026-2027) for more data and independent confirmation
3. **Seek cross-survey consistency** (Euclid, Roman Space Telescope)
4. **Interpret with appropriate nuance** -- the evidence is moderate but not compelling

## References

- DESI DR2: [arXiv:2503.14738](https://arxiv.org/abs/2503.14738)
- Bayesian analysis: [arXiv:2511.10631](https://arxiv.org/abs/2511.10631)
- Dataset tensions: [arXiv:2504.15222](https://arxiv.org/abs/2504.15222)
- E-values: Ramdas et al. 2023, Statistical Science 38(4)

---
*Analysis: February 2026*
