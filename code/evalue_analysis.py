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
- Uniform mixture: Bayes factor with discrete uniform prior over a grid of alternatives
- WARNING: The maximized likelihood ratio (plugging in the MLE) is NOT a valid
  e-value. For k>=2 extra parameters, E[exp(chi^2(k)/2)] = infinity under H0.

References:
- Shafer (2021): "Testing by betting" - JRSS-A 184, 407-478
- Ramdas et al. (2023): "Game-Theoretic Statistics" - Statistical Science 38(4)
- Vovk & Wang (2021): "E-values: Calibration, combination, and applications"
- Grünwald, de Heide & Koolen (2024): "Safe Testing" - JRSS-B (GROW-optimal procedure)

Criticisms and limitations:
- Prior sensitivity: Results depend on choice of alternative hypothesis
- Computational cost: Optimized mixture weights (e.g., GROW) require optimization
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
        """Convert to approximate sigma significance.

        Uses sigma = sqrt(2 * ln(E)), which equals sqrt(delta_chi2)
        when E = exp(delta_chi2 / 2). This is the standard conversion
        used in cosmology for comparing nested models.

        Note: This is approximate. E-values and p-values answer
        different questions.
        """
        if self.e_value <= 1:
            return 0.0
        return np.sqrt(2.0 * np.log(self.e_value))


def likelihood_ratio_evalue(
    data: np.ndarray,
    cov: np.ndarray,
    theory_null: np.ndarray,
    theory_alt: np.ndarray
) -> EValueResult:
    """
    Compute likelihood ratio statistic.

    LR = L(data | H1) / L(data | H0)

    This is a valid e-value ONLY when theory_alt is specified independently
    of the data (e.g., from a pre-registered alternative or from a separate
    dataset). When theory_alt comes from MLE fitting on the same data,
    the result is a maximized likelihood ratio, NOT a valid e-value:
    for k>=2 extra parameters, E[exp(chi^2(k)/2) | H0] = infinity.

    This function is used internally by split_evalue (where it IS valid
    because the alternative is fitted on disjoint training data) and for
    cross-dataset validation (also valid, since parameters come from a
    different dataset).
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
) -> EValueResult:
    """
    Compute uniform mixture e-value by averaging over a grid of alternatives.

    Averages the likelihood ratio L(data|theta)/L(data|H0) over a
    uniform grid of (w0, wa) values. This is equivalent to a Bayes
    factor with a discrete uniform prior and produces a valid e-value
    (E[E|H0] = 1 by construction).

    Note: This is NOT the GROW-optimal e-value from Grünwald et al. (2024),
    which would optimize the mixture weights to maximize expected log growth
    rate. We use equal (uniform) weights for simplicity. The result depends
    on the grid range chosen.

    The function name 'grow_evalue' is retained for backwards compatibility;
    use 'uniform_mixture_evalue' as the preferred alias.
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
        method='uniform_mixture'
    )


# Alias for backwards compatibility
uniform_mixture_evalue = grow_evalue


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
        method='sequential_uniform_mixture'
    )


@dataclass
class LOOCVResult:
    """Results from LOOCV e-value computation."""
    e_value: float  # Product of all per-bin E_k values
    log_e: float  # Sum of log(E_k) for numerical stability
    per_bin_e: Dict[float, float]  # E_k for each held-out redshift bin
    per_bin_log_e: Dict[float, float]  # log(E_k) for each bin
    per_bin_w0: Dict[float, float]  # Best-fit w0 for each fold
    per_bin_wa: Dict[float, float]  # Best-fit wa for each fold
    chi2_null_total: float  # Sum of per-bin chi2 under null
    chi2_alt_total: float  # Sum of per-bin chi2 under alt

    @property
    def sigma_equivalent(self) -> float:
        if self.e_value <= 1:
            return 0.0
        return np.sqrt(2.0 * np.log(self.e_value))


def _identify_redshift_bins(z_values: np.ndarray) -> Dict[float, list]:
    """
    Group measurement indices by redshift bin.

    Returns dict mapping each unique redshift to a list of indices
    into the data/z_values arrays.
    """
    bins = {}
    for i, z in enumerate(z_values):
        # Round to avoid floating-point matching issues
        z_key = round(float(z), 4)
        if z_key not in bins:
            bins[z_key] = []
        bins[z_key].append(i)
    return bins


def loocv_evalue(
    data: np.ndarray,
    cov: np.ndarray,
    z_values: np.ndarray,
    quantities: list,
    verbose: bool = True
) -> LOOCVResult:
    """
    Compute Leave-One-Out Cross-Validation e-value over redshift bins.

    For each of the 7 redshift bins:
      1. Hold out that bin's measurements (1 for DV-only, 2 for DM+DH)
      2. Fit (w0, wa) on remaining measurements
      3. Compute likelihood ratio E_k on held-out data
      4. The overall e-value is the product of all E_k

    This is a valid e-value because the covariance matrix is block-diagonal
    (bins are independent), so the product of per-bin likelihood ratios is
    itself a likelihood ratio. Each E_k uses training data disjoint from
    the test point, avoiding overfitting.

    Parameters
    ----------
    data : np.ndarray
        Measurement vector (length 13 for DESI DR2)
    cov : np.ndarray
        Full covariance matrix (13x13, block-diagonal by redshift bin)
    z_values : np.ndarray
        Effective redshifts for each measurement
    quantities : list
        Quantity labels ('DM_over_rs', 'DH_over_rs', 'DV_over_rs')
    verbose : bool
        Print per-fold diagnostics

    Returns
    -------
    LOOCVResult with overall and per-bin e-values
    """
    bins = _identify_redshift_bins(z_values)

    if verbose:
        print(f"LOOCV over {len(bins)} redshift bins:")
        for z_key, idxs in sorted(bins.items()):
            qtys = [quantities[i] for i in idxs]
            print(f"  z={z_key:.3f}: {len(idxs)} point(s) - {qtys}")
        print()

    per_bin_e = {}
    per_bin_log_e = {}
    per_bin_w0 = {}
    per_bin_wa = {}
    chi2_null_total = 0.0
    chi2_alt_total = 0.0

    for z_held_out, held_out_idx in sorted(bins.items()):
        held_out_idx = np.array(held_out_idx)
        # Training indices: everything except held-out bin
        train_idx = np.array([i for i in range(len(data)) if i not in held_out_idx])

        # Extract subsets
        data_train = data[train_idx]
        cov_train = cov[np.ix_(train_idx, train_idx)]
        z_train = z_values[train_idx]
        q_train = [quantities[i] for i in train_idx]

        data_test = data[held_out_idx]
        cov_test = cov[np.ix_(held_out_idx, held_out_idx)]
        z_test = z_values[held_out_idx]
        q_test = [quantities[i] for i in held_out_idx]

        # Fit (w0, wa) on training data
        def neg_log_likelihood_train(params):
            w0, wa = params
            cosmo = CosmologyParams(w0=w0, wa=wa)
            pred = compute_bao_predictions(z_train, cosmo)
            theory = _build_theory_vector(pred, z_train, q_train)
            return -log_likelihood(data_train, theory, cov_train)

        # Use multiple starting points to avoid local minima
        best_result = None
        best_nll = np.inf
        starts = [
            [-0.9, -0.5],
            [-1.0, 0.0],
            [-0.75, -1.0],
            [-0.8, -0.8],
            [-1.1, 0.5],
        ]
        for x0 in starts:
            try:
                result = minimize(
                    neg_log_likelihood_train,
                    x0=x0,
                    bounds=[(-2.0, 0.0), (-4.0, 3.0)],
                    method='L-BFGS-B'
                )
                if result.fun < best_nll:
                    best_nll = result.fun
                    best_result = result
            except Exception:
                continue

        w0_fit, wa_fit = best_result.x
        alt_cosmo = CosmologyParams(w0=w0_fit, wa=wa_fit)

        # Compute E_k on held-out data
        pred_null_test = compute_bao_predictions(z_test, LCDM)
        theory_null_test = _build_theory_vector(pred_null_test, z_test, q_test)
        pred_alt_test = compute_bao_predictions(z_test, alt_cosmo)
        theory_alt_test = _build_theory_vector(pred_alt_test, z_test, q_test)

        log_L_null = log_likelihood(data_test, theory_null_test, cov_test)
        log_L_alt = log_likelihood(data_test, theory_alt_test, cov_test)
        log_ek = log_L_alt - log_L_null
        ek = np.exp(log_ek)

        chi2_null_k = chi_squared(data_test, theory_null_test, cov_test)
        chi2_alt_k = chi_squared(data_test, theory_alt_test, cov_test)

        per_bin_e[z_held_out] = ek
        per_bin_log_e[z_held_out] = log_ek
        per_bin_w0[z_held_out] = w0_fit
        per_bin_wa[z_held_out] = wa_fit
        chi2_null_total += chi2_null_k
        chi2_alt_total += chi2_alt_k

        if verbose:
            print(f"  Fold z={z_held_out:.3f}: "
                  f"E_k={ek:.4f}, log(E_k)={log_ek:+.4f}, "
                  f"dchi2={chi2_null_k - chi2_alt_k:+.3f}, "
                  f"w0={w0_fit:.3f}, wa={wa_fit:.3f}")

    # Overall LOOCV e-value = product of per-bin E_k
    log_e_total = sum(per_bin_log_e.values())
    e_total = np.exp(log_e_total)

    if verbose:
        print(f"\n  LOOCV E-value = {e_total:.4f}")
        print(f"  log(E) = {log_e_total:.4f}")
        if e_total > 1:
            print(f"  sigma equivalent = {np.sqrt(2.0 * np.log(e_total)):.2f}")

    return LOOCVResult(
        e_value=e_total,
        log_e=log_e_total,
        per_bin_e=per_bin_e,
        per_bin_log_e=per_bin_log_e,
        per_bin_w0=per_bin_w0,
        per_bin_wa=per_bin_wa,
        chi2_null_total=chi2_null_total,
        chi2_alt_total=chi2_alt_total,
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
        else:
            raise ValueError(f"Unknown BAO quantity: {q}")
    return theory


# =============================================================================
# CRITICISM AND CAVEATS
# =============================================================================

"""
Why E-values May Be Flawed for DESI Dark Energy Evidence
=========================================================

1. PRIOR/MIXTURE SENSITIVITY
   - Uniform mixture e-values depend on the choice of grid range over alternatives
   - Different choices of w0, wa ranges give different e-values
   - No principled way to choose the "right" prior for dark energy

2. MAXIMIZED LIKELIHOOD RATIO IS NOT AN E-VALUE
   - Plugging in the MLE gives E[exp(chi^2(k)/2)] = infinity for k>=2
   - The maximized LR violates the defining property E[E|H0] <= 1
   - It is a descriptive statistic, not calibrated evidence
   - Data splitting or mixture methods are needed for valid e-values

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
    print("\n1. Likelihood ratio (valid here because alternative is pre-specified):")
    pred_lcdm = compute_bao_predictions(z_test, LCDM)
    theory_lcdm = _build_theory_vector(pred_lcdm, z_test, quantities)
    e_simple = likelihood_ratio_evalue(data, cov, theory_lcdm, theory_true)
    print(f"   E = {e_simple.e_value:.2f}, Δχ² = {e_simple.delta_chi2:.2f}")

    print("\n2. Uniform mixture e-value:")
    e_grow = grow_evalue(data, cov, z_test, quantities)
    print(f"   E = {e_grow.e_value:.2f}, Δχ² = {e_grow.delta_chi2:.2f}")
    print(f"   Best-fit: w0={e_grow.alt_params.w0:.3f}, wa={e_grow.alt_params.wa:.3f}")

    print("\n" + summarize_evidence_caveats(e_grow))
