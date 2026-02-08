# DESI E-Value Analysis: Complete Mathematical Derivation

## Table of Contents
1. [Data Loading](#1-data-loading)
2. [Cosmological Model Predictions](#2-cosmological-model-predictions)
3. [Chi-Squared Analysis](#3-chi-squared-analysis)
4. [E-Value Computation](#4-e-value-computation)
5. [Data-Split Validation](#5-data-split-validation)
6. [Power Analysis](#6-power-analysis)
7. [Results Summary](#7-results-summary)

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

**Equation of state:**

$$w(a) = w_0 + w_a(1-a)$$

**Dark energy density evolution:**

$$\Omega_{DE}(z) = \Omega_{DE,0} \cdot (1+z)^{3(1+w_0+w_a)} \cdot \exp\left(-3w_a\frac{z}{1+z}\right)$$

**w₀wₐCDM predictions:**

| z_eff | $D_M/r_d$ (pred) | $D_H/r_d$ (pred) | Diff from ΛCDM |
|-------|------------------|------------------|----------------|
| 0.295 | 8.27 | 25.15 | -1.3%, -2.2% |
| 0.510 | 13.54 | 22.43 | -1.6%, -1.4% |
| 0.706 | 17.83 | 19.90 | -1.6%, -0.9% |
| 0.934 | 22.44 | 17.10 | -1.5%, -0.1% |
| 1.321 | 29.17 | 13.55 | -1.4%, +1.0% |
| 1.484 | 31.89 | 12.30 | -1.3%, +1.2% |
| 2.330 | 38.70 | 8.73 | -1.2%, +1.2% |

---

## 3. Chi-Squared Analysis

### Formula

$$\chi^2 = (\mathbf{d} - \mathbf{t})^T \mathbf{C}^{-1} (\mathbf{d} - \mathbf{t})$$

where:
- $\mathbf{d}$ = data vector (13 elements)
- $\mathbf{t}$ = theory predictions
- $\mathbf{C}$ = covariance matrix

### Residuals: Data - ΛCDM

| Index | z_eff | Quantity | Data | ΛCDM | Residual | Residual/σ |
|-------|-------|----------|------|------|----------|------------|
| 0 | 0.295 | $D_V/r_d$ | 7.942 | 8.15 | -0.21 | -2.8σ |
| 1 | 0.510 | $D_M/r_d$ | 13.588 | 13.76 | -0.17 | -1.0σ |
| 2 | 0.510 | $D_H/r_d$ | 21.863 | 22.75 | -0.89 | -2.1σ |
| 3 | 0.706 | $D_M/r_d$ | 17.351 | 18.12 | -0.77 | -4.3σ |
| 4 | 0.706 | $D_H/r_d$ | 19.455 | 20.08 | -0.62 | -1.9σ |
| 5 | 0.934 | $D_M/r_d$ | 21.576 | 22.79 | -1.21 | -7.5σ |
| 6 | 0.934 | $D_H/r_d$ | 17.641 | 17.11 | +0.53 | +2.6σ |
| ... | ... | ... | ... | ... | ... | ... |

**Note:** Large individual residuals, but correlations in $\mathbf{C}$ affect overall $\chi^2$.

### Computed Chi-Squared Values

$$\chi^2_{\Lambda\text{CDM}} = 25.44$$

$$\chi^2_{w_0w_a\text{CDM}} = 13.50$$

$$\Delta\chi^2 = 25.44 - 13.50 = 11.94$$

### Frequentist Significance

For $\Delta\chi^2$ with 2 extra parameters (w₀, wₐ):

$$p = P(\chi^2_2 > 11.94) = 0.0026$$

Converting to sigma:

$$\sigma = \Phi^{-1}(1 - p/2) \approx 3.0$$

**Interpretation:** ~3σ preference for w₀wₐCDM over ΛCDM (BAO only)

---

## 4. E-Value Computation

### Method 1: Simple Likelihood Ratio

**Log-likelihood:**

$$\ln L = -\frac{1}{2}\chi^2 - \frac{1}{2}\ln|\mathbf{C}| - \frac{n}{2}\ln(2\pi)$$

**E-value:**

$$E = \frac{L(\mathbf{d} | w_0w_a)}{L(\mathbf{d} | \Lambda\text{CDM})} = \exp\left(\ln L_{w_0w_a} - \ln L_{\Lambda}\right)$$

**Calculation:**

$$\ln L_{\Lambda} = -\frac{1}{2}(25.44) - \frac{1}{2}\ln|\mathbf{C}| - \frac{13}{2}\ln(2\pi)$$

$$\ln L_{w_0w_a} = -\frac{1}{2}(13.50) - \frac{1}{2}\ln|\mathbf{C}| - \frac{13}{2}\ln(2\pi)$$

$$\ln E = \ln L_{w_0w_a} - \ln L_{\Lambda} = -\frac{1}{2}(13.50 - 25.44) = \frac{11.94}{2} = 5.97$$

$$E = e^{5.97} = 392$$

**⚠️ WARNING:** This is BIASED because w₀wₐ was fitted to the same data!

### Method 2: GROW Mixture E-Value

**Grid over alternatives:**

$$w_0 \in [-1.5, -0.5], \quad w_a \in [-2.0, 1.0]$$

Using 15×15 = 225 grid points.

**Mixture e-value:**

$$E_{\text{mix}} = \frac{1}{N}\sum_{i=1}^{N} \frac{L(\mathbf{d} | w_{0,i}, w_{a,i})}{L(\mathbf{d} | \Lambda)}$$

**For numerical stability (log-sum-exp):**

$$\ln E_{\text{mix}} = \max_i(\ln R_i) + \ln\left(\frac{1}{N}\sum_i \exp(\ln R_i - \max_j \ln R_j)\right)$$

where $\ln R_i = \ln L_{w_{0,i},w_{a,i}} - \ln L_\Lambda$

**Results for different prior ranges:**

| Prior Range | $\ln E$ | $E$ |
|-------------|---------|-----|
| Narrow: $w_0 \in [-1.3, -0.7]$, $w_a \in [-1.5, 0.5]$ | 4.58 | 97 |
| Default: $w_0 \in [-1.5, -0.5]$, $w_a \in [-2.0, 1.0]$ | 2.68 | 15 |
| Wide: $w_0 \in [-2.0, 0.0]$, $w_a \in [-3.0, 2.0]$ | 2.85 | 17 |

**Sensitivity:** E varies from 15 to 97 (~7×) depending on prior choice!

---

## 5. Data-Split Validation

### Split Definition

- **Training set:** $z < 1.0$ (7 data points)
- **Test set:** $z \geq 1.0$ (6 data points)

### Step 1: Fit Alternative on Training Data

Minimize negative log-likelihood:

$$-\ln L_{\text{train}}(w_0, w_a) = \frac{1}{2}(\mathbf{d}_{\text{train}} - \mathbf{t}(w_0,w_a))^T \mathbf{C}_{\text{train}}^{-1} (\mathbf{d}_{\text{train}} - \mathbf{t}(w_0,w_a))$$

**Result:** $\hat{w}_0 = -0.872$, $\hat{w}_a = -0.327$

### Step 2: Evaluate E-Value on Test Data

Using the fitted $(\hat{w}_0, \hat{w}_a)$:

$$E_{\text{test}} = \frac{L(\mathbf{d}_{\text{test}} | \hat{w}_0, \hat{w}_a)}{L(\mathbf{d}_{\text{test}} | \Lambda)}$$

**Calculation:**

$$\chi^2_{\Lambda,\text{test}} = 10.23$$

$$\chi^2_{w_0w_a,\text{test}} = 9.51$$

$$\ln E_{\text{test}} = \frac{10.23 - 9.51}{2} = 0.36$$

$$E_{\text{test}} = e^{0.36} = 1.43$$

### Why This Matters

| Method | E-value | Note |
|--------|---------|------|
| Simple LR (same data) | 392 | BIASED |
| Data-split (held-out) | 1.43 | VALID |

**The 280× difference shows overfitting is dominating the simple LR result!**

---

## 6. Power Analysis

### Question

If w₀wₐCDM is true, what E-value would we expect?

### Simulation Setup

1. Assume true cosmology: $w_0 = -0.727$, $w_a = -1.05$
2. Generate synthetic data: $\mathbf{d}_{\text{sim}} = \mathbf{t}_{w_0w_a} + \mathbf{n}$
3. Where $\mathbf{n} \sim \mathcal{N}(0, \mathbf{C})$
4. Compute GROW E-value for each simulation
5. Repeat 500 times

### Results

**E-value distribution if w₀wₐCDM is true:**

| Statistic | Value |
|-----------|-------|
| Median E | 21 |
| Mean E | 35 |
| 5th percentile | 3.2 |
| 95th percentile | 95 |

**Our observed E = 15 is at the 40th percentile**

### Interpretation

- If w₀wₐCDM were true, we'd typically see E ~ 21
- Our E = 15 is consistent with EITHER model
- Not conclusive evidence for or against

---

## 7. Results Summary

### All Evidence Measures

| Method | Value | σ equiv | Validity |
|--------|-------|---------|----------|
| Δχ² frequentist | 11.94 | 3.0σ | Standard |
| E-value (simple LR) | 392 | 3.9σ | **BIASED** |
| E-value (GROW narrow) | 97 | 3.0σ | Prior-sensitive |
| E-value (GROW default) | 15 | 2.3σ | Prior-sensitive |
| E-value (GROW wide) | 17 | 2.4σ | Prior-sensitive |
| E-value (data-split) | **1.43** | **0.8σ** | **VALID** |
| Bayesian evidence | ln B = -0.57 | Favors ΛCDM | External |

### Key Mathematical Insight

The likelihood ratio e-value:

$$E = \exp\left(\frac{\Delta\chi^2}{2}\right) = \exp\left(\frac{11.94}{2}\right) = 392$$

This looks impressive, but it's **biased** because $(\hat{w}_0, \hat{w}_a)$ was chosen to maximize this ratio on the same data.

When we use held-out validation:

$$E_{\text{valid}} = \exp\left(\frac{0.72}{2}\right) = 1.43$$

The evidence essentially vanishes.

### Conclusion

The apparent 3-4σ evidence for dynamic dark energy is **not robust**:

1. **Overfitting:** E drops from 392 → 1.43 with proper validation
2. **Prior sensitivity:** GROW E varies 7× with different choices
3. **Bayesian contradiction:** Full data analysis favors ΛCDM
4. **Power analysis:** E = 15 is consistent with either model

**Recommendation:** Do not claim discovery. Wait for more data and independent confirmation.
