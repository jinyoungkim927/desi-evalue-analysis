# E-Values: Mathematical Background

## What Are E-Values?

E-values are a framework for hypothesis testing developed by Shafer, Vovk, Grünwald, and Ramdas. They provide an alternative to p-values with better mathematical properties for sequential testing and overfitting control.

### Formal Definition

An **e-value** is a non-negative random variable $E$ such that under the null hypothesis $H_0$:

$$\mathbb{E}[E \mid H_0] \leq 1$$

This simple constraint has powerful implications.

### Key Property (Ville's Inequality)

By Markov's inequality: $P(E \geq 1/\alpha \mid H_0) \leq \alpha$ for any $\alpha$.

So if $E = 100$, the probability of seeing $E \geq 100$ under $H_0$ is at most 1/100 = 1%.

## Comparison to P-Values

| Property | P-Value | E-Value |
|----------|---------|---------|
| Definition | $P(\text{data} \geq \text{observed} \mid H_0)$ | Random variable with $\mathbb{E}[E \mid H_0] \leq 1$ |
| Interpretation | Probability of extreme data | Evidence against $H_0$ |
| Combination | Complex (Fisher, Stouffer) | Simple: multiply (if independent) or average |
| Optional stopping | Invalid | Valid |
| Anytime validity | No | Yes |

## The Likelihood Ratio E-Value

For **simple** (fully specified) hypotheses, the likelihood ratio is a valid e-value:

$$E = \frac{L(\text{data} \mid H_1)}{L(\text{data} \mid H_0)}$$

**Proof:** $\mathbb{E}_{H_0}[E] = \int \frac{P(x \mid H_1)}{P(x \mid H_0)} P(x \mid H_0)\, dx = \int P(x \mid H_1)\, dx = 1$

### ⚠️ The Maximized Likelihood Ratio is NOT an E-Value

When $H_1$ involves parameters fitted to the **same data** used for testing, the likelihood ratio is not a valid e-value. For k = 2 extra parameters (the w₀wₐCDM case):

$$\mathbb{E}_{H_0}\left[\exp\left(\frac{\chi^2(2)}{2}\right)\right] = \int_0^\infty e^{t/2} \cdot \frac{1}{2}e^{-t/2}\,dt = \int_0^\infty \frac{1}{2}\,dt = \infty$$

Since $\mathbb{E}[E \mid H_0] = \infty$, the defining property is violated. The maximized LR of 392 in our DESI analysis is a **descriptive statistic**, not calibrated evidence.

## Uniform Mixture E-Values

Average the likelihood ratio over a grid of pre-specified alternatives:

$$E_{\text{mix}} = \frac{1}{N}\sum_{i=1}^{N} \frac{L(\mathbf{d} \mid \theta_i)}{L(\mathbf{d} \mid H_0)}$$

**Why valid:** Each term is a valid e-value (H₁ specified before seeing data). The average of e-values is an e-value by linearity of expectation.

**Trade-off:** The result depends on the prior range (Occam razor). Narrower priors concentrating mass near the true parameters produce larger e-values.

### Relation to GROW

GROW (Growth Rate Optimal in Worst case), from Grünwald, de Heide & Koolen (2024), optimizes the mixture weights for maximum power. Our implementation uses simple uniform weights — a valid but not necessarily optimal choice.

## Data-Split E-Values

1. **Split** data into training set $D_{\text{train}}$ and test set $D_{\text{test}}$
2. **Fit** $H_1$ using only $D_{\text{train}}$
3. **Compute** e-value using only $D_{\text{test}}$

$$E_{\text{split}} = \frac{L(D_{\text{test}} \mid \hat{\theta}(D_{\text{train}}))}{L(D_{\text{test}} \mid H_0)}$$

**Why valid:** Conditional on $D_{\text{train}}$, $\hat{\theta}$ is fixed. From $D_{\text{test}}$'s perspective, $H_1$ is pre-specified.

**Trade-off:** Reduced statistical power from using only part of the data for testing. Power calibration shows the median E_split is only ~2.7 even when H₁ is true for the DESI data.

## LOO Average E-Values

For K redshift bins, leave out bin k, fit on the remaining K-1 bins, compute e-value on held-out bin:

$$\bar{E} = \frac{1}{K}\sum_{k=1}^K E_k$$

**Why the average is valid:** By linearity of expectation:

$$\mathbb{E}\left[\frac{1}{K}\sum_k E_k \mid H_0\right] = \frac{1}{K}\sum_k \mathbb{E}[E_k \mid H_0] \leq \frac{1}{K} \cdot K = 1$$

### ⚠️ The LOO Product is NOT Valid

The product $\prod_k E_k$ requires **independence**, which fails because LOO training sets overlap. For example, folds 1 and 2 share 5 of 6 bins. The product's expectation under H₀ is unknown and may exceed 1.

In our DESI analysis: LOO average = 10.2 (valid), LOO product = 1062 (NOT valid).

## Combining E-Values

### Product Rule (requires independence)

If $E_1, E_2, \ldots, E_n$ are e-values from **independent** data:

$$E_{\text{combined}} = E_1 \times E_2 \times \cdots \times E_n$$

Proof: $\mathbb{E}[E_1 E_2 \mid H_0] = \mathbb{E}[E_1 \mid H_0] \mathbb{E}[E_2 \mid H_0] \leq 1$

### Averaging Rule (works for dependent e-values)

$$E_{\text{avg}} = \frac{1}{n}\sum_{i=1}^n E_i$$

By linearity: $\mathbb{E}[E_{\text{avg}} \mid H_0] = \frac{1}{n}\sum \mathbb{E}[E_i \mid H_0] \leq 1$

This is valid even when the e-values are dependent — which is why the LOO average works but the LOO product does not.

## Interpreting E-Values

| E-Value | Interpretation |
|---------|----------------|
| E < 1 | Evidence **favors** H₀ (null) |
| E = 1 | No evidence either way |
| E ~ 3 | Weak evidence against H₀ |
| E ~ 10 | Moderate evidence against H₀ |
| E ~ 100 | Strong evidence against H₀ |
| E ~ 1000 | Very strong evidence against H₀ |

### Approximate σ Conversion

$$\sigma \approx \sqrt{2 \ln E}$$

| E-Value | Approximate σ |
|---------|---------------|
| 3 | 1.5σ |
| 10 | 2.1σ |
| 15 | 2.3σ |
| 20 | 2.4σ |
| 100 | 3.0σ |

**Caution:** This conversion is approximate. E-values and p-values answer different questions.

## References

1. **Vovk & Wang (2021)**: "E-values: Calibration, combination, and applications" — Annals of Statistics, 49, 1736
2. **Grünwald, de Heide & Koolen (2024)**: "Safe Testing" — Journal of the Royal Statistical Society B
3. **Ramdas et al. (2023)**: "Game-Theoretic Statistics and Safe Anytime-Valid Inference" — Statistical Science, 38, 576
4. **Shafer (2021)**: "Testing by betting: A strategy for statistical and scientific communication" — JRSS-A, 184, 407

## Key Takeaways for DESI Analysis

1. The **maximized likelihood ratio (392)** is NOT an e-value — E[E|H₀] = ∞ for k ≥ 2
2. The **LOO product (1062)** is NOT valid — overlapping training sets violate independence
3. The **uniform mixture (E ≈ 15)** and **LOO average (E ≈ 10)** are both valid and converge on moderate evidence
4. The **data-split (E = 1.4)** is valid but underpowered — power calibration confirms median E ≈ 2.7 under H₁
5. The **convergence** of two independent valid methods on E ≈ 10–15 is the robust finding
