# DESI E-Value Analysis: Complete Mathematical Derivation

## Table of Contents
1. [Data Loading](#1-data-loading)
2. [Cosmological Model Predictions](#2-cosmological-model-predictions)
3. [Chi-Squared Analysis](#3-chi-squared-analysis)
4. [E-Value Methods](#4-e-value-methods)
5. [Data-Split Validation](#5-data-split-validation)
6. [LOO Average E-Value](#6-loo-average-e-value)
7. [Power Calibration](#7-power-calibration)
8. [Cross-Dataset E-Values](#8-cross-dataset-e-values)
9. [Results Summary](#9-results-summary)

---

## 1. Data Loading

### Raw Data Vector

From `desi_gaussian_bao_ALL_GCcomb_mean.txt`:

$$\mathbf{d} = \begin{pmatrix}
7.942 \\ 13.588 \\ 21.863 \\ 17.351 \\ 19.455 \\ 21.576 \\ 17.641 \\ 27.601 \\ 14.176 \\ 30.512 \\ 12.817 \\ 8.632 \\ 38.989
\end{pmatrix}$$

Corresponding to:

| Index | z_eff | Quantity |
|-------|-------|----------|
| 0 | 0.295 | $D_V/r_d$ |
| 1 | 0.510 | $D_M/r_d$ |
| 2 | 0.510 | $D_H/r_d$ |
| 3 | 0.706 | $D_M/r_d$ |
| 4 | 0.706 | $D_H/r_d$ |
| 5 | 0.934 | $D_M/r_d$ |
| 6 | 0.934 | $D_H/r_d$ |
| 7 | 1.321 | $D_M/r_d$ |
| 8 | 1.321 | $D_H/r_d$ |
| 9 | 1.484 | $D_M/r_d$ |
| 10 | 1.484 | $D_H/r_d$ |
| 11 | 2.330 | $D_H/r_d$ |
| 12 | 2.330 | $D_M/r_d$ |

### Covariance Matrix

The covariance matrix $\mathbf{C}$ is 13×13, block-diagonal:

$$\mathbf{C} = \begin{pmatrix}
\sigma_0^2 & 0 & 0 & \cdots \\
0 & \Sigma_1 & 0 & \cdots \\
0 & 0 & \Sigma_2 & \cdots \\
\vdots & & & \ddots
\end{pmatrix}$$

Where each $\Sigma_i$ is a 2×2 block for $D_M$, $D_H$ at each redshift:

$$\Sigma_i = \begin{pmatrix}
\sigma_{M,i}^2 & \rho_i \sigma_{M,i} \sigma_{H,i} \\
\rho_i \sigma_{M,i} \sigma_{H,i} & \sigma_{H,i}^2
\end{pmatrix}$$

**Extracted uncertainties:**

| z_eff | $\sigma(D_M/r_d)$ | $\sigma(D_H/r_d)$ | $\sigma(D_V/r_d)$ | $\rho(D_M, D_H)$ |
|-------|-------------------|-------------------|-------------------|------------------|
| 0.295 | — | — | 0.076 | — |
| 0.510 | 0.168 | 0.429 | — | -0.45 |
| 0.706 | 0.180 | 0.334 | — | -0.40 |
| 0.934 | 0.162 | 0.201 | — | -0.35 |
| 1.321 | 0.325 | 0.225 | — | -0.40 |
| 1.484 | 0.764 | 0.518 | — | -0.49 |
| 2.330 | 0.532 | 0.101 | — | -0.43 |

---

## 2. Cosmological Model Predictions

### ΛCDM Model

**Parameters:**
- $h = 0.6766$ (Hubble parameter)
- $\Omega_m = 0.3111$ (matter density)
- $\Omega_{DE} = 0.6889$ (dark energy density)
- $w = -1$ (equation of state)
- $r_d = 147.05$ Mpc (sound horizon)

**Hubble parameter:**

$$E(z) = \sqrt{\Omega_m(1+z)^3 + \Omega_{DE}}$$

**Distances:**

$$D_C(z) = \frac{c}{H_0} \int_0^z \frac{dz'}{E(z')}$$

$$D_M(z) = D_C(z) \text{ (flat universe)}$$

$$D_H(z) = \frac{c}{H_0 E(z)}$$

$$D_V(z) = \left[z \cdot D_H(z) \cdot D_M(z)^2\right]^{1/3}$$

**ΛCDM predictions at data redshifts:**

| z_eff | $D_M/r_d$ (pred) | $D_H/r_d$ (pred) | $D_V/r_d$ (pred) |
|-------|------------------|------------------|------------------|
| 0.295 | 8.38 | 25.72 | 8.15 |
| 0.510 | 13.76 | 22.75 | 13.05 |
| 0.706 | 18.12 | 20.08 | 16.71 |
| 0.934 | 22.79 | 17.11 | 20.30 |
| 1.321 | 29.57 | 13.42 | 25.16 |
| 1.484 | 32.32 | 12.15 | 27.04 |
| 2.330 | 39.16 | 8.63 | 31.60 |

### w₀wₐCDM Model

**Additional parameters:**
- $w_0 = -0.727$ (equation of state today)
- $w_a = -1.05$ (evolution parameter)

**Dark energy density evolution:**

$$\Omega_{DE}(z) = \Omega_{DE,0} \cdot (1+z)^{3(1+w_0+w_a)} \cdot \exp\left(-3w_a\frac{z}{1+z}\right)$$

---

## 3. Chi-Squared Analysis

### Formula

$$\chi^2 = (\mathbf{d} - \mathbf{t})^T \mathbf{C}^{-1} (\mathbf{d} - \mathbf{t})$$

### Computed Chi-Squared Values

$$\chi^2_{\Lambda\text{CDM}} = 25.44$$

$$\chi^2_{w_0w_a\text{CDM}} = 13.50$$

$$\Delta\chi^2 = 25.44 - 13.50 = 11.94$$

### Frequentist Significance

For Δχ² with k = 2 extra parameters, under H₀ this follows a χ²(2) distribution (by Wilks' theorem):

$$p = 1 - F_{\chi^2}(11.94; 2) = e^{-11.94/2} = e^{-5.97} \approx 0.0026$$

$$\sigma = \Phi^{-1}(1 - p/2) \approx 2.8\sigma$$

**Note:** DESI correctly performs this conversion using the χ²(2) distribution (their equation 22). The naive formula √Δχ² ≈ 3.5σ applies only for k = 1.

---

## 4. E-Value Methods

### Method 1: Maximized Likelihood Ratio (NOT a valid e-value)

$$E = \exp\left(\frac{\Delta\chi^2}{2}\right) = \exp\left(\frac{11.94}{2}\right) = 392$$

**⚠️ This is NOT an e-value.** For k = 2 extra parameters:

$$\mathbb{E}_{H_0}\left[e^{\chi^2(2)/2}\right] = \int_0^\infty e^{t/2} \cdot \frac{1}{2}e^{-t/2}\,dt = \int_0^\infty \frac{1}{2}\,dt = \infty$$

Since E[E | H₀] = ∞, the defining property E[E | H₀] ≤ 1 is violated. This is a descriptive statistic only.

### Method 2: Uniform Mixture E-Value

$$E_{\text{mix}} = \frac{1}{N}\sum_{i=1}^{N} \frac{L(\mathbf{d} \mid w_{0,i}, w_{a,i})}{L(\mathbf{d} \mid \Lambda)}$$

**Results for different prior ranges:**

| Prior Range | E-value | ln E |
|-------------|---------|------|
| Narrow: w₀ ∈ [-1.2, -0.8], wₐ ∈ [-1.0, 0.5] | 97 | 4.58 |
| Default: w₀ ∈ [-1.5, -0.5], wₐ ∈ [-2.0, 1.0] | **15** | **2.68** |
| Wide: w₀ ∈ [-2.0, 0.0], wₐ ∈ [-3.0, 2.0] | 17 | 2.85 |

**Why this is valid:** Each grid point gives a valid e-value (H₁ specified before seeing data). The average of e-values is an e-value by linearity of expectation.

**Prior sensitivity** reflects the Occam razor: a narrower prior concentrating mass near the MLE wastes less probability on distant parameter values.

---

## 5. Data-Split Validation

### Split Definition

- **Training set:** z < 1.0 (7 data points: BGS, LRG1, LRG2, LRG3+ELG1)
- **Test set:** z ≥ 1.0 (6 data points: ELG2, QSO, Lyα)

### Computation

Fit on training data: $\hat{w}_0 = -0.78$, $\hat{w}_a = -0.52$.

$$\chi^2_{\text{test, ΛCDM}} \approx 5.8, \quad \chi^2_{\text{test, }w_0w_a} \approx 5.1$$

$$E_{\text{split}} = \exp\left(\frac{5.8 - 5.1}{2}\right) = e^{0.35} \approx 1.4$$

**Interpretation:** This is **inconclusive**, not evidence against w₀wₐCDM. See power calibration (Section 7).

---

## 6. LOO Average E-Value

### Method

For each of K = 7 redshift bins, leave out bin k, fit (w₀, wₐ) on the remaining 6 bins, compute e-value on held-out bin. Average the results.

### Per-Bin Results

| Left-out bin | z_eff | Fitted (w₀, wₐ) | E_k |
|---|---|---|---|
| BGS | 0.295 | (-0.86, -0.44) | 1.89 |
| LRG1 | 0.510 | (-0.84, -0.46) | 0.71 |
| LRG2 | 0.706 | (-0.85, -0.38) | 55.98 |
| LRG3+ELG1 | 0.934 | (-0.84, -0.49) | 8.73 |
| ELG2 | 1.321 | (-0.86, -0.42) | 2.14 |
| QSO | 1.484 | (-0.86, -0.43) | 0.75 |
| Lyα | 2.330 | (-0.86, -0.44) | 1.00 |

### LOO Average

$$\bar{E} = \frac{1.89 + 0.71 + 55.98 + 8.73 + 2.14 + 0.75 + 1.00}{7} = \frac{71.2}{7} \approx 10.2$$

### Validity

The average is valid by linearity of expectation:

$$\mathbb{E}\left[\frac{1}{K}\sum_k E_k \mid H_0\right] = \frac{1}{K}\sum_k \mathbb{E}[E_k \mid H_0] \leq 1$$

### ⚠️ The LOO Product is NOT Valid

The product $\prod_k E_k = 1062$ requires independence, which fails because the LOO training sets overlap (e.g., folds 1 and 2 share 5 of 6 bins). The product's expectation under H₀ is unknown and may exceed 1.

---

## 7. Power Calibration

### Question

If w₀wₐCDM is the true model, what data-split e-value would we expect?

### Setup

500 Monte Carlo simulations under w₀wₐCDM with DESI best-fit (w₀ = -0.75, wₐ = -1.05). For each: generate synthetic data, apply the same redshift-based split, compute E_split.

### Results

| Statistic | Value |
|-----------|-------|
| Median E_split under H₁ | **2.7** |
| P(E_split > 1.4 \| H₁) | 64% |
| P(E_split > 1.4 \| H₀) | 12.8% |

**Interpretation:** Even when w₀wₐCDM is true, the data-split test typically gives only E ≈ 2.7. The observed E = 1.4 cannot distinguish H₀ from H₁. The test is severely underpowered for wₐ because the low-z training set has only 23–48% leverage on the time-evolution parameter.

---

## 8. Cross-Dataset E-Values

### Method

Use published best-fit (w₀, wₐ) from one experiment to predict another experiment's data. If the dark energy signal is real and consistent, cross-predictions should outperform ΛCDM.

### Results

| Training Dataset | Test Dataset | (w₀, wₐ) | E-value |
|---|---|---|---|
| DESI (fitted) | Pantheon+ | (-0.86, -0.43) | 1.5 |
| DESI (fitted) | DES-Y5 | (-0.86, -0.43) | 86 |
| Pantheon+ | DESI | (-0.90, -0.20) | 2049 |
| **DES-Y5** | **DESI** | **(-0.65, -1.20)** | **0.19** |

### Interpretation

The ~10,000× asymmetry between Pantheon+ (E = 2049) and DES-Y5 (E = 0.19) when predicting DESI reveals fundamental tension between supernova catalogs. If w₀wₐCDM represented real physics, all experiments should point to compatible parameter values. DES-Y5's parameters make DESI data *less* probable than ΛCDM — a direct detection of inconsistency, not merely absence of evidence.

---

## 9. Results Summary

### All Evidence Measures

| Method | Value | ~σ | Validity |
|--------|-------|-----|----------|
| Frequentist Δχ² (k=2) | p = 0.0026 | 2.8σ | Standard (no Occam) |
| ΔAIC | 7.9 | — | Penalty: 2k = 4 |
| ΔBIC (n=13) | 6.8 | — | Penalty: k ln n ≈ 5 |
| Maximized LR | 392 | — | **NOT AN E-VALUE** |
| LOO product | 1062 | — | **NOT VALID** |
| Uniform mixture (default) | **15** | **2.3** | Valid e-value |
| **LOO average** | **10** | **2.2** | **Valid e-value** |
| Data-split | 1.4 | 0.8 | Valid, underpowered |
| Bayes factor (Ong et al.) | ln B = -0.57 | — | Favors ΛCDM |
| Cross-dataset (DES-Y5→DESI) | 0.19 | — | Tension |

### Where Valid Methods Converge

Two independent valid methods converge:

- Uniform mixture: E ≈ 15 (ln E ≈ 2.7)
- LOO average: E ≈ 10 (ln E ≈ 2.3)

Both indicate **moderate evidence** for w₀wₐCDM. The ~0.7σ gap from DESI's correctly computed 3.0σ reflects the Occam penalty from averaging over parameter values rather than maximizing.

### Conclusion

The evidence for dynamical dark energy is **moderate but not compelling** (E ≈ 10–15), and is **undermined by cross-dataset inconsistency**:

1. Valid e-values converge on E ≈ 10–15 (~2.2–2.3σ) — real evidence, but not a discovery
2. The ~10,000× asymmetry between supernova catalogs is the most informative finding
3. The data-split E = 1.4 is inconclusive due to limited power (median E ≈ 2.7 under H₁)
4. The maximized LR (392) and LOO product (1062) are not valid e-values and should not be cited as evidence
5. Resolving inter-dataset tensions is more important than accumulating further significance
