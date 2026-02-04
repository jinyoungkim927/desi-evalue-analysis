"""
E-value analysis for DESI BAO cosmological model comparison.

This module implements e-values (expectation values) for comparing
cosmological models using BAO data. E-values provide an alternative
to p-values with better properties for sequential testing and
model comparison.

Key concepts:
- E-value: Non-negative random variable with E[E] <= 1 under null hypothesis
- Can be combined by multiplication (sequential) or averaging
- Related to likelihood ratios but without requiring full Bayesian priors
- GROW (Growth Rate Optimality): Maximizes expected log(E) under alternative

References:
- Shafer (2021): "Testing by betting" - JRSS-A 184, 407-478
- Ramdas et al. (2023): "Game-Theoretic Statistics" - Statistical Science 38(4)
- Vovk & Wang (2021): "E-values: Calibration, combination, and applications"

Criticisms and limitations:
- Prior sensitivity: Results depend on choice of alternative hypothesis
- Computational cost: GROW-optimal e-values require optimization
- Interpretation: Not as intuitive as p-values for many practitioners
- See also: arxiv:2511.10631 (Bayesian critique of DESI evidence)
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.stats import chi2
from typing import Tuple, Dict, Optional, Callable
from dataclasses import dataclass
import warnings

from cosmology import (
    CosmologyParams, LCDM, compute_bao_predictions,
    chi_squared, log_likelihood
)
from data_loader import BAODataset, get_theory_vector


@dataclass
class EValueResult:
    """Results from e-value computation."""
    e_value: float  # The e-value
    log_e: float  # log(e-value) for numerical stability
    chi2_null: float  # Chi-squared under null (LCDM)
    chi2_alt: float  # Chi-squared under alternative
    delta_chi2: float  # chi2_null - chi2_alt
    null_params: CosmologyParams
    alt_params: CosmologyParams
    method: str  # How e-value was computed

    @property
    def sigma_equivalent(self) -> float:
        """Convert to approximate sigma significance."""
        if self.e_value <= 1:
            return 0.0
        # E-value of E corresponds roughly to 1/E p-value
        # Convert to sigma using chi2 with 1 dof
        p_approx = 1.0 / self.e_value
        if p_approx >= 1:
            return 0.0
        return np.sqrt(chi2.ppf(1 - p_approx, df=1))


def likelihood_ratio_evalue(
    data: np.ndarray,
    cov: np.ndarray,
    theory_null: np.ndarray,
    theory_alt: np.ndarray
) -> EValueResult:
    """
    Compute e-value as simple likelihood ratio.

    E = L(data | H1) / L(data | H0)

    This is the most basic e-value, equivalent to a Bayes factor
    with point mass priors on specific parameter values.

    Note: This can give arbitrarily large e-values if the alternative
    is fine-tuned to the data, which is why GROW-optimal approaches
    are preferred.
    """
    log_L_null = log_likelihood(data, theory_null, cov)
    log_L_alt = log_likelihood(data, theory_alt, cov)

    chi2_null = chi_squared(data, theory_null, cov)
    chi2_alt = chi_squared(data, theory_alt, cov)

    log_e = log_L_alt - log_L_null
    e_value = np.exp(log_e)

    return EValueResult(
        e_value=e_value,
        log_e=log_e,
        chi2_null=chi2_null,
        chi2_alt=chi2_alt,
        delta_chi2=chi2_null - chi2_alt,
        null_params=LCDM,
        alt_params=None,  # Point specified
        method='likelihood_ratio'
    )


def split_evalue(
    data: np.ndarray,
    cov: np.ndarray,
    z_values: np.ndarray,
    quantities: list,
    split_z: float = 1.0
) -> Tuple[EValueResult, EValueResult]:
    """
    Compute e-values using data splitting.

    Split data into low-z (z < split_z) and high-z (z >= split_z).
    Use low-z to train the alternative, high-z to test.

    This addresses the issue of overfitting by ensuring the
    alternative hypothesis is not fitted to the same data
    used for testing.

    Returns:
    --------
    (E_train, E_test) : Results from training and test sets
    """
    # Find split indices
    low_z_mask = z_values < split_z
    high_z_mask = ~low_z_mask

    n_low = np.sum(low_z_mask)
    n_high = np.sum(high_z_mask)

    if n_low < 2 or n_high < 2:
        raise ValueError(f"Need at least 2 points in each split. Got {n_low} low-z, {n_high} high-z")

    # Extract subsets
    low_z_idx = np.where(low_z_mask)[0]
    high_z_idx = np.where(high_z_mask)[0]

    data_low = data[low_z_idx]
    data_high = data[high_z_idx]

    cov_low = cov[np.ix_(low_z_idx, low_z_idx)]
    cov_high = cov[np.ix_(high_z_idx, high_z_idx)]

    z_low = z_values[low_z_idx]
    z_high = z_values[high_z_idx]

    # Fit alternative model to low-z data
    def neg_log_likelihood(params):
        w0, wa = params
        cosmo = CosmologyParams(w0=w0, wa=wa)
        pred = compute_bao_predictions(z_low, cosmo)
        theory = _build_theory_vector(pred, z_low, [quantities[i] for i in low_z_idx])
        return -log_likelihood(data_low, theory, cov_low)

    # Optimize over reasonable w0, wa range
    result = minimize(
        neg_log_likelihood,
        x0=[-0.9, -0.5],
        bounds=[(-2.0, 0.0), (-3.0, 2.0)],
        method='L-BFGS-B'
    )
    w0_fit, wa_fit = result.x
    alt_cosmo = CosmologyParams(w0=w0_fit, wa=wa_fit)

    # Compute predictions for both halves
    pred_null_low = compute_bao_predictions(z_low, LCDM)
    pred_null_high = compute_bao_predictions(z_high, LCDM)
    pred_alt_low = compute_bao_predictions(z_low, alt_cosmo)
    pred_alt_high = compute_bao_predictions(z_high, alt_cosmo)

    theory_null_low = _build_theory_vector(pred_null_low, z_low, [quantities[i] for i in low_z_idx])
    theory_null_high = _build_theory_vector(pred_null_high, z_high, [quantities[i] for i in high_z_idx])
    theory_alt_low = _build_theory_vector(pred_alt_low, z_low, [quantities[i] for i in low_z_idx])
    theory_alt_high = _build_theory_vector(pred_alt_high, z_high, [quantities[i] for i in high_z_idx])

    # E-value on training data (for reference - NOT used for inference)
    e_train = likelihood_ratio_evalue(data_low, cov_low, theory_null_low, theory_alt_low)
    e_train.method = 'split_train'
    e_train.alt_params = alt_cosmo

    # E-value on test data (the valid one)
    e_test = likelihood_ratio_evalue(data_high, cov_high, theory_null_high, theory_alt_high)
    e_test.method = 'split_test'
    e_test.alt_params = alt_cosmo

    return e_train, e_test


def grow_evalue(
    data: np.ndarray,
    cov: np.ndarray,
    z_values: np.ndarray,
    quantities: list,
    n_samples: int = 1000
) -> EValueResult:
    """
    Compute GROW (Growth Rate Optimal) e-value.

    GROW e-values maximize the expected log(E) under the alternative
    hypothesis, subject to E[E|H0] <= 1.

    This is done by:
    1. Define a mixture alternative over plausible (w0, wa) values
    2. Compute the e-value as weighted average of likelihood ratios
    3. Optimize the weights to maximize growth rate

    Note: This is computationally expensive and the result depends
    on the choice of prior/mixture distribution over alternatives.
    """
    # Define grid of alternative hypotheses
    w0_grid = np.linspace(-1.5, -0.5, 10)
    wa_grid = np.linspace(-2.0, 1.0, 10)

    # Compute likelihood ratios for each alternative
    log_ratios = []
    theories_null = None

    for w0 in w0_grid:
        for wa in wa_grid:
            cosmo = CosmologyParams(w0=w0, wa=wa)
            pred = compute_bao_predictions(z_values, cosmo)
            theory_alt = _build_theory_vector(pred, z_values, quantities)

            if theories_null is None:
                pred_null = compute_bao_predictions(z_values, LCDM)
                theory_null = _build_theory_vector(pred_null, z_values, quantities)

            log_L_alt = log_likelihood(data, theory_alt, cov)
            log_L_null = log_likelihood(data, theory_null, cov)
            log_ratios.append(log_L_alt - log_L_null)

    log_ratios = np.array(log_ratios)

    # Simple approach: use uniform mixture (proper Bayesian averaging)
    # This gives a valid e-value by Jensen's inequality
    max_log_ratio = np.max(log_ratios)
    log_e = max_log_ratio + np.log(np.mean(np.exp(log_ratios - max_log_ratio)))
    e_value = np.exp(log_e)

    # Also compute best-fit statistics
    best_idx = np.argmax(log_ratios)
    best_w0 = w0_grid[best_idx // len(wa_grid)]
    best_wa = wa_grid[best_idx % len(wa_grid)]
    best_cosmo = CosmologyParams(w0=best_w0, wa=best_wa)

    pred_best = compute_bao_predictions(z_values, best_cosmo)
    theory_best = _build_theory_vector(pred_best, z_values, quantities)

    chi2_null = chi_squared(data, theory_null, cov)
    chi2_alt = chi_squared(data, theory_best, cov)

    return EValueResult(
        e_value=e_value,
        log_e=log_e,
        chi2_null=chi2_null,
        chi2_alt=chi2_alt,
        delta_chi2=chi2_null - chi2_alt,
        null_params=LCDM,
        alt_params=best_cosmo,
        method='grow_mixture'
    )


def sequential_evalue(
    datasets: list,
    covs: list,
    z_arrays: list,
    quantities_arrays: list
) -> EValueResult:
    """
    Compute sequential e-value by multiplying e-values from multiple datasets.

    E_combined = E_1 * E_2 * ... * E_n

    This is valid even with optional stopping - you can stop when
    E exceeds a threshold and still have valid type-I error control.

    Parameters:
    -----------
    datasets : list of np.ndarray
        Data vectors from each dataset (e.g., DR1, DR2)
    covs : list of np.ndarray
        Covariance matrices
    z_arrays : list of np.ndarray
        Redshift arrays
    quantities_arrays : list of list
        Quantity labels for each dataset
    """
    log_e_total = 0.0
    chi2_null_total = 0.0
    chi2_alt_total = 0.0

    for data, cov, z_values, quantities in zip(datasets, covs, z_arrays, quantities_arrays):
        e_result = grow_evalue(data, cov, z_values, quantities)
        log_e_total += e_result.log_e
        chi2_null_total += e_result.chi2_null
        chi2_alt_total += e_result.chi2_alt

    return EValueResult(
        e_value=np.exp(log_e_total),
        log_e=log_e_total,
        chi2_null=chi2_null_total,
        chi2_alt=chi2_alt_total,
        delta_chi2=chi2_null_total - chi2_alt_total,
        null_params=LCDM,
        alt_params=None,
        method='sequential_grow'
    )


def _build_theory_vector(pred: dict, z_values: np.ndarray, quantities: list) -> np.ndarray:
    """Build theory vector matching data structure."""
    theory = np.zeros(len(quantities))
    for i, (z, q) in enumerate(zip(z_values, quantities)):
        z_idx = np.argmin(np.abs(pred['z'] - z))
        if 'DM' in q:
            theory[i] = pred['DM_over_rd'][z_idx]
        elif 'DH' in q:
            theory[i] = pred['DH_over_rd'][z_idx]
        elif 'DV' in q:
            theory[i] = pred['DV_over_rd'][z_idx]
    return theory


# =============================================================================
# CRITICISM AND CAVEATS
# =============================================================================

"""
Why E-values May Be Flawed for DESI Dark Energy Evidence
=========================================================

1. PRIOR/MIXTURE SENSITIVITY
   - GROW e-values depend on the choice of mixture distribution over alternatives
   - Different choices of w0, wa ranges give different e-values
   - No principled way to choose the "right" prior for dark energy

2. POINT-ALTERNATIVE PROBLEM
   - Simple likelihood ratio e-values can be arbitrarily large
   - If alternative is fit to data, E can be huge even under null
   - Data splitting helps but reduces statistical power

3. MODEL COMPLEXITY NOT PENALIZED
   - Unlike Bayesian evidence, e-values don't inherently penalize complexity
   - w0waCDM has 2 extra parameters vs LCDM - should be penalized
   - Bayes factors naturally include Occam's razor

4. DATASET TENSIONS
   - DESI BAO + CMB + SNe show internal tensions (2.95σ per arxiv:2511.10631)
   - E-values from tensioned data may just reflect inconsistency, not physics
   - The w0waCDM model may be "resolving tensions" rather than detecting physics

5. SEQUENTIAL TESTING ASSUMPTIONS
   - Sequential e-values assume data batches are independent
   - DESI DR1 and DR2 share some observations and analysis choices
   - Independence assumption may be violated

6. COMPARISON TO BAYESIAN EVIDENCE
   - Bayesian analysis finds ln B = -0.57 ± 0.26 for DESI+CMB (favors LCDM!)
   - This contradicts 3.1σ frequentist result
   - E-values sit between frequentist and Bayesian - unclear interpretation

7. STOPPING RULE CONCERNS
   - While e-values are valid under optional stopping, interpretation changes
   - Stopping when E > threshold selects for high E values
   - Replication studies may show regression to mean

RECOMMENDATION:
E-values provide ONE perspective on evidence, but should not be the sole
basis for claiming dynamic dark energy. The discrepancy between Bayesian
and frequentist analyses suggests the evidence is not robust. Wait for:
- More data (DESI DR3+)
- Resolution of dataset tensions
- Independent confirmation from other surveys (Euclid, Roman)
"""


def summarize_evidence_caveats(e_result: EValueResult) -> str:
    """Generate summary with appropriate caveats."""
    lines = [
        "E-VALUE ANALYSIS SUMMARY",
        "=" * 50,
        f"E-value: {e_result.e_value:.2f}",
        f"log(E): {e_result.log_e:.2f}",
        f"Approximate sigma: {e_result.sigma_equivalent:.1f}σ",
        f"Delta chi²: {e_result.delta_chi2:.2f}",
        "",
        "INTERPRETATION CAVEATS:",
        "-" * 50,
    ]

    if e_result.e_value > 100:
        lines.append("WARNING: Very large E-value may indicate overfitting")
        lines.append("         or sensitivity to prior/mixture choice.")

    if e_result.sigma_equivalent > 3:
        lines.extend([
            "NOTE: While >3σ seems significant, Bayesian analysis",
            "      of the same data FAVORS LCDM (ln B ≈ -0.57).",
            "      The discrepancy suggests evidence is not robust.",
        ])

    lines.extend([
        "",
        "Key concerns for DESI evidence:",
        "• Dataset tensions between CMB, BAO, and SNe",
        "• w0waCDM may be resolving tensions, not detecting physics",
        "• Prior/mixture choice significantly affects e-values",
        "• Await independent confirmation before strong conclusions",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    # Quick test with mock data
    print("Testing e-value analysis module...")

    # Generate mock data
    np.random.seed(42)
    z_test = np.array([0.3, 0.5, 0.5, 0.7, 0.7, 1.0, 1.0, 1.3, 1.3])
    quantities = ['DV_over_rs', 'DM_over_rs', 'DH_over_rs',
                  'DM_over_rs', 'DH_over_rs', 'DM_over_rs', 'DH_over_rs',
                  'DM_over_rs', 'DH_over_rs']

    # Generate "data" from slightly non-LCDM model
    true_cosmo = CosmologyParams(w0=-0.9, wa=-0.3)
    pred_true = compute_bao_predictions(z_test, true_cosmo)
    theory_true = _build_theory_vector(pred_true, z_test, quantities)

    # Add noise
    errors = theory_true * 0.02  # 2% errors
    cov = np.diag(errors**2)
    data = theory_true + np.random.multivariate_normal(np.zeros(len(z_test)), cov)

    # Compute e-values
    print("\n1. Simple likelihood ratio e-value:")
    pred_lcdm = compute_bao_predictions(z_test, LCDM)
    theory_lcdm = _build_theory_vector(pred_lcdm, z_test, quantities)
    e_simple = likelihood_ratio_evalue(data, cov, theory_lcdm, theory_true)
    print(f"   E = {e_simple.e_value:.2f}, Δχ² = {e_simple.delta_chi2:.2f}")

    print("\n2. GROW mixture e-value:")
    e_grow = grow_evalue(data, cov, z_test, quantities)
    print(f"   E = {e_grow.e_value:.2f}, Δχ² = {e_grow.delta_chi2:.2f}")
    print(f"   Best-fit: w0={e_grow.alt_params.w0:.3f}, wa={e_grow.alt_params.wa:.3f}")

    print("\n" + summarize_evidence_caveats(e_grow))
