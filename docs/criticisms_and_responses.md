# Potential Criticisms and Pre-Prepared Responses

## 1. Data Criticisms

### 1.1 "You only use BAO, not the full CMB+SNe dataset"
**Validity**: HIGH - This is a real limitation
**Response**:
- DESI's 3-4σ comes from BAO+CMB+SNe combined
- BAO-only gives ~1-2σ even in DESI's own analysis
- However, the Bayesian critique (arXiv:2511.10631) analyzed full data and STILL found ΛCDM favored
- Our BAO-only analysis is informative about the BAO contribution specifically
**Action**: Add CMB compressed likelihood, note limitation clearly

### 1.2 "Your covariance matrix might be incomplete"
**Validity**: MEDIUM
**Response**:
- We use official DESI covariance from CobayaSampler
- Block-diagonal structure matches paper description
- Systematic errors may not be fully included
**Action**: Verify covariance includes systematics, compare errors to paper Table IV

### 1.3 "You're missing correlations between tracers"
**Validity**: LOW
**Response**:
- DESI explicitly states different tracer bins are uncorrelated
- The block-diagonal covariance is intentional, not an oversight
**Action**: Document this in the notebook

## 2. Physics/Cosmology Criticisms

### 2.1 "Your cosmological calculations might be wrong"
**Validity**: HIGH - Must verify
**Response**:
- Should validate against CAMB/CLASS
- Compare our DM/rd, DH/rd to published fiducial values
**Action**: Add validation cell comparing to DESI's fiducial model

### 2.2 "You use wrong fiducial parameters"
**Validity**: MEDIUM
**Response**:
- DESI uses: Ωm=0.3111, h=0.6766, rd=147.05 Mpc
- Check our values match
**Action**: Use exact DESI parameters, document clearly

### 2.3 "The CPL parametrization w(a)=w0+wa(1-a) might be implemented wrong"
**Validity**: MEDIUM
**Response**:
- The dark energy density evolution is: ρ_DE(a) ∝ a^(-3(1+w0+wa)) exp(-3wa(1-a))
- This is standard but easy to get wrong
**Action**: Validate against published w0waCDM predictions

### 2.4 "rd is model-dependent"
**Validity**: LOW for this analysis
**Response**:
- Using rd-normalized quantities (DM/rd, DH/rd) is standard
- This removes most rd-dependence for model comparison
- DESI does the same thing
**Action**: Note this is standard practice

## 3. E-Value Methodology Criticisms

### 3.1 "E-values aren't designed for this problem"
**Validity**: MEDIUM
**Response**:
- E-values are designed for sequential testing with optional stopping
- We're using them for model comparison, which is related but different
- They provide a different perspective than p-values or Bayes factors
- The key advantage: they're calibrated (E[E|H0] ≤ 1)
**Action**: Clearly explain what e-values are and why we use them

### 3.2 "Your uniform mixture prior is arbitrary"
**Validity**: HIGH - This is a fundamental issue
**Response**:
- Yes, different priors give different e-values (we show this varies ~7x)
- This is a limitation of ANY model comparison method
- Bayes factors have the same issue (prior sensitivity)
- We show sensitivity analysis explicitly
**Action**: Emphasize this limitation, show multiple priors

### 3.3 "Data splitting reduces statistical power"
**Validity**: HIGH
**Response**:
- Yes, splitting means each half has fewer data points
- E=1.36 could be due to low power, not absence of signal
- Should compute expected e-value under alternative hypothesis
**Action**: Add power analysis - what E would we expect if w0waCDM is true?

### 3.4 "Your sigma-equivalent conversion is misleading"
**Validity**: MEDIUM
**Response**:
- The conversion E → σ via p ≈ 1/E is approximate
- E-values and p-values have different interpretations
- Should be clearer about this
**Action**: Remove or caveat the sigma conversion, focus on E directly

### 3.5 "Simple likelihood ratio e-values can be infinite"
**Validity**: HIGH
**Response**:
- If alternative is perfectly fitted, E can be arbitrarily large
- This is why we use uniform mixture (bounded by averaging)
- Data splitting also addresses this (alternative fitted to different data)
**Action**: Explain why simple LR is biased, show multiple methods

## 4. Statistical Analysis Criticisms

### 4.1 "Your chi-squared seems too high"
**Validity**: MUST CHECK
**Response**:
- chi2=24.75 for 13 points with ΛCDM
- This is chi2/dof ≈ 1.9, which seems high
- Could indicate: wrong theory, wrong covariance, or real tension
**Action**: Investigate - compare to DESI's reported chi-squared

### 4.2 "You don't marginalize over nuisance parameters"
**Validity**: MEDIUM
**Response**:
- DESI marginalizes over Ωm, h, rd, etc.
- We fix these to fiducial values
- For model COMPARISON (ΛCDM vs w0waCDM), this is approximately valid
  because both models use same fiducial
**Action**: Note this simplification, discuss impact

### 4.3 "You should use DESI's published posteriors"
**Validity**: HIGH - Good suggestion
**Response**:
- DESI released MCMC chains publicly
- Could directly compute e-values from their posteriors
- More accurate than our simplified analysis
**Action**: Download and analyze DESI chains

## 5. Interpretation Criticisms

### 5.1 "Bayesian evidence ≠ e-value, you can't compare them"
**Validity**: MEDIUM
**Response**:
- True, they answer different questions
- Bayes factor: posterior odds given prior odds
- E-value: evidence measure with frequentist guarantee
- But discrepancy between methods suggests non-robust evidence
**Action**: Clarify what each measure means

### 5.2 "The tension might be real physics, not statistics"
**Validity**: MEDIUM
**Response**:
- Dataset tensions COULD indicate new physics
- But they could also indicate systematics
- Standard scientific practice: investigate systematics first
**Action**: Note this possibility fairly

### 5.3 "You're being too skeptical of an important result"
**Validity**: PHILOSOPHICAL
**Response**:
- Extraordinary claims require extraordinary evidence
- 5σ threshold exists for good reason
- Multiple independent confirmations needed
**Action**: Present analysis fairly, let reader judge

## Summary: Key Weaknesses to Address

1. **Validate cosmology code** against CAMB/published values
2. **Power analysis** for data-split e-value
3. **Use DESI chains** for more accurate comparison
4. **Chi-squared investigation** - why is it high?
5. **Clear e-value explanation** - what they mean, limitations
6. **CMB data** - add or clearly note limitation
