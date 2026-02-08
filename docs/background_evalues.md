# E-Values: Mathematical Background

## What Are E-Values?

E-values (expectation values) are a relatively new framework for hypothesis testing developed by Shafer, Vovk, Grünwald, and Ramdas. They provide an alternative to p-values with better mathematical properties.

### Formal Definition

An **e-value** is a non-negative random variable $E$ such that under the null hypothesis $H_0$:

$$\mathbb{E}[E \mid H_0] \leq 1$$

This simple constraint has powerful implications.

### Why "E" for Expectation?

The name comes from the key property: the **expected value** under the null is bounded by 1. This means:
- If $H_0$ is true, $E$ is unlikely to be large (by Markov's inequality)
- $P(E \geq 1/\alpha \mid H_0) \leq \alpha$ for any $\alpha$

## Comparison to P-Values

| Property | P-Value | E-Value |
|----------|---------|---------|
| Definition | $P(\text{data} \geq \text{observed} \mid H_0)$ | Random variable with $\mathbb{E}[E \mid H_0] \leq 1$ |
| Interpretation | Probability of extreme data | Evidence against $H_0$ |
| Combination | Complex (Fisher, Stouffer) | Simple: multiply or average |
| Optional stopping | Invalid | Valid |
| Anytime validity | No | Yes |

## The Likelihood Ratio E-Value

The simplest e-value is the **likelihood ratio**:

$$E = \frac{L(\text{data} \mid H_1)}{L(\text{data} \mid H_0)} = \frac{P(\text{data} \mid H_1)}{P(\text{data} \mid H_0)}$$

### Why Is This an E-Value?

Under $H_0$, the expected value is:

$$\mathbb{E}\left[\frac{P(\text{data} \mid H_1)}{P(\text{data} \mid H_0)} \mid H_0\right] = \int \frac{P(x \mid H_1)}{P(x \mid H_0)} P(x \mid H_0) dx = \int P(x \mid H_1) dx = 1$$

So the likelihood ratio has expected value exactly 1 under $H_0$.

### The Problem: Overfitting

If $H_1$ is chosen **after** seeing the data (fitted to the data), the likelihood ratio can be arbitrarily large even when $H_0$ is true. This is **overfitting**.

Example:
- True model: $H_0$
- We observe data and fit $H_1$ to maximize likelihood
- $L(H_1)$ will be higher than $L(H_0)$ just by chance
- This inflates the apparent evidence

## GROW E-Values (Growth Rate Optimal)

To address overfitting, we use **mixture e-values**:

$$E_{\text{mix}} = \int \frac{P(\text{data} \mid \theta)}{P(\text{data} \mid H_0)} \pi(\theta) d\theta$$

where $\pi(\theta)$ is a prior/mixture distribution over the alternative hypothesis space.

### Why This Works

1. The alternative is specified **before** seeing data (via the prior)
2. Averaging prevents cherry-picking the best-fitting alternative
3. Still satisfies $\mathbb{E}[E_{\text{mix}} \mid H_0] \leq 1$

### GROW Criterion

GROW stands for **Growth Rate Optimal in the Worst case**. It chooses the mixture $\pi$ to maximize:

$$\inf_{\theta \in \Theta_1} \mathbb{E}_\theta[\log E]$$

This ensures good power against all alternatives in $\Theta_1$.

## Data-Split E-Values

Another approach to prevent overfitting:

1. **Split** data into training set $D_{\text{train}}$ and test set $D_{\text{test}}$
2. **Fit** the alternative $H_1$ using only $D_{\text{train}}$
3. **Compute** e-value using only $D_{\text{test}}$

$$E_{\text{split}} = \frac{P(D_{\text{test}} \mid \hat{H}_1(D_{\text{train}}))}{P(D_{\text{test}} \mid H_0)}$$

### Why This Is Valid

- The alternative $\hat{H}_1$ is determined by $D_{\text{train}}$
- $D_{\text{test}}$ is independent of $D_{\text{train}}$
- So $\hat{H}_1$ is "pre-specified" from $D_{\text{test}}$'s perspective
- No overfitting on the test set

### Trade-off

Data splitting reduces sample size in each half, reducing statistical power. But it provides an **honest** assessment of evidence.

## Interpreting E-Values

| E-Value | Interpretation |
|---------|----------------|
| E < 1 | Evidence **favors** $H_0$ |
| E = 1 | No evidence either way |
| E = 3 | Weak evidence against $H_0$ |
| E = 10 | Moderate evidence against $H_0$ |
| E = 100 | Strong evidence against $H_0$ |
| E = 1000 | Very strong evidence against $H_0$ |

### Approximate σ Conversion

For rough comparison to frequentist significance:

$$\sigma \approx \sqrt{2 \ln E}$$

| E-Value | Approximate σ |
|---------|---------------|
| 3 | 1.5σ |
| 10 | 2.1σ |
| 20 | 2.4σ |
| 100 | 3.0σ |
| 1000 | 3.7σ |

**Caution:** This conversion is approximate. E-values and p-values answer different questions.

## Combining E-Values

### Product Rule (Sequential Testing)

If $E_1, E_2, \ldots, E_n$ are e-values from independent data:

$$E_{\text{combined}} = E_1 \times E_2 \times \cdots \times E_n$$

This is also an e-value! Proof:

$$\mathbb{E}[E_1 E_2 \mid H_0] = \mathbb{E}[E_1 \mid H_0] \mathbb{E}[E_2 \mid H_0] \leq 1 \times 1 = 1$$

### Averaging Rule

For **dependent** e-values, use averaging:

$$E_{\text{avg}} = \frac{1}{n}\sum_{i=1}^n E_i$$

By Jensen's inequality, this is also an e-value:

$$\mathbb{E}\left[\frac{1}{n}\sum E_i \mid H_0\right] = \frac{1}{n}\sum \mathbb{E}[E_i \mid H_0] \leq 1$$

## References

1. **Vovk & Wang (2021)**: "E-values: Calibration, combination, and applications" - Annals of Statistics
2. **Grünwald, de Heide & Koolen (2024)**: "Safe Testing" - Journal of the Royal Statistical Society B
3. **Ramdas et al. (2023)**: "Game-Theoretic Statistics and Safe Anytime-Valid Inference" - Statistical Science 38(4)
4. **Shafer (2021)**: "Testing by betting: A strategy for statistical and scientific communication" - JRSS-A

## Key Takeaways for DESI Analysis

1. **Simple likelihood ratio e-values are BIASED** if the alternative is fitted to the same data
2. **GROW mixture e-values** are more principled but depend on prior choice
3. **Data-split e-values** prevent overfitting and give honest assessment
4. **Large differences between methods** indicate non-robust evidence
