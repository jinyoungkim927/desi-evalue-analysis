#!/usr/bin/env python3
"""
Uniformly Most Powerful (UMP) E-Value for DESI BAO Analysis.

Implements the UMP e-value construction suggested by Peter Grunwald:

For a simple null H0 (LCDM, w=-1), find a point alternative theta_1 = (w0*, wa*)
such that the likelihood ratio E = L(data|theta_1) / L(data|H0) maximizes
*power* when rejecting at E >= 1/alpha.

The key insight: for a given alpha and sample size n, there exists a theta_1 such
that the rejection region {x : L(x|theta_1)/L(x|H0) >= 1/alpha} is a SUPERSET
of the rejection region for any other theta_1. This theta_1 gives the UMP e-value.

The resulting LR is a valid e-value (simple vs simple, Prop 1 in the paper)
regardless of alpha, but is *optimized* for a particular alpha.

Finding theta_1:
  For each candidate (w0, wa), simulate N datasets under various alternatives,
  compute the rejection rate at threshold 1/alpha, and pick the (w0, wa) that
  maximizes the minimum power across alternatives.

References:
- Grunwald, de Heide & Koolen (2024): "Safe Testing" - JRSS-B
- Author's reply to discussion in the Safe Testing paper
"""

import sys
import time
import numpy as np
from pathlib import Path
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))

from cosmology import (
    CosmologyParams, LCDM, DESI_DR2_BEST_FIT,
    compute_bao_predictions, chi_squared, log_likelihood
)
from data_loader import load_desi_data, BAODataset
from evalue_analysis import _build_theory_vector, EValueResult


# ---------------------------------------------------------------------------
# Core: compute log-likelihood ratio for a point alternative
# ---------------------------------------------------------------------------

def log_lr_at_point(data, cov, z_eff, quantities, w0, wa, theory_null=None):
    """
    Compute log(L(data|w0,wa) / L(data|LCDM)) for a specific (w0, wa).

    Returns log-likelihood ratio and caches/reuses theory_null if provided.
    """
    cosmo = CosmologyParams(w0=w0, wa=wa)
    pred = compute_bao_predictions(z_eff, cosmo)
    theory_alt = _build_theory_vector(pred, z_eff, quantities)

    if theory_null is None:
        pred_null = compute_bao_predictions(z_eff, LCDM)
        theory_null = _build_theory_vector(pred_null, z_eff, quantities)

    log_L_alt = log_likelihood(data, theory_alt, cov)
    log_L_null = log_likelihood(data, theory_null, cov)

    return log_L_alt - log_L_null, theory_null


def generate_simulated_data(z_eff, quantities, cov, true_cosmo):
    """Generate one simulated BAO dataset under a given cosmology."""
    pred = compute_bao_predictions(z_eff, true_cosmo)
    theory = _build_theory_vector(pred, z_eff, quantities)
    noise = np.random.multivariate_normal(np.zeros(len(theory)), cov)
    return theory + noise


# ---------------------------------------------------------------------------
# Find UMP e-value point alternative
# ---------------------------------------------------------------------------

def find_ump_theta(z_eff, quantities, cov, alpha=0.01, n_sims=500,
                   true_alternatives=None, w0_range=(-1.5, -0.5),
                   wa_range=(-2.0, 1.0), grid_resolution=15, verbose=True):
    """
    Find the point alternative theta_1 = (w0*, wa*) that maximizes power.

    For each candidate theta_1 on a grid:
      1. For each true alternative in true_alternatives:
         - Simulate n_sims datasets under the true alternative
         - Compute LR = L(data|theta_1)/L(data|H0) for each
         - Measure power = fraction where LR >= 1/alpha
      2. Score = minimum power across all true alternatives (maximin)

    Pick the theta_1 with the highest score.

    Parameters
    ----------
    z_eff : array, effective redshifts
    quantities : list, BAO quantity types
    cov : array, 13x13 covariance matrix
    alpha : float, significance level for power optimization (default 0.01)
    n_sims : int, simulations per (theta_1, true_alt) pair
    true_alternatives : list of CosmologyParams, true models to test power against
        If None, uses a representative set around DESI best-fit
    w0_range, wa_range : bounds for the theta_1 search grid
    grid_resolution : int, grid points per dimension
    verbose : bool

    Returns
    -------
    dict with keys:
        'w0_star', 'wa_star' : optimal point alternative
        'power_matrix' : power at each (theta_1, true_alt) pair
        'min_power' : minimum power for each theta_1
        'w0_grid', 'wa_grid' : the search grid
    """
    threshold = 1.0 / alpha

    # Default true alternatives: a spread around DESI best-fit
    if true_alternatives is None:
        true_alternatives = [
            CosmologyParams(w0=-0.75, wa=-1.05),   # DESI BAO best-fit
            CosmologyParams(w0=-0.85, wa=-0.50),    # Milder evolution
            CosmologyParams(w0=-0.65, wa=-1.20),    # DES-Y5-like
            CosmologyParams(w0=-0.90, wa=-0.20),    # Pantheon+-like
        ]

    # Precompute null theory vector
    pred_null = compute_bao_predictions(z_eff, LCDM)
    theory_null = _build_theory_vector(pred_null, z_eff, quantities)

    # Grid of candidate theta_1 values
    w0_grid = np.linspace(w0_range[0], w0_range[1], grid_resolution)
    wa_grid = np.linspace(wa_range[0], wa_range[1], grid_resolution)

    n_candidates = len(w0_grid) * len(wa_grid)
    n_alts = len(true_alternatives)

    if verbose:
        print(f"UMP search: {n_candidates} candidates × {n_alts} alternatives × {n_sims} sims")
        print(f"  alpha = {alpha}, threshold = 1/alpha = {threshold}")
        print(f"  w0 range: [{w0_range[0]}, {w0_range[1]}], wa range: [{wa_range[0]}, {wa_range[1]}]")

    # Precompute theory vectors for all candidate theta_1
    if verbose:
        print("  Precomputing theory vectors for candidates...")
    candidate_theories = {}
    for w0 in w0_grid:
        for wa in wa_grid:
            cosmo = CosmologyParams(w0=w0, wa=wa)
            pred = compute_bao_predictions(z_eff, cosmo)
            candidate_theories[(w0, wa)] = _build_theory_vector(pred, z_eff, quantities)

    # For each true alternative, pre-generate simulated datasets
    if verbose:
        print("  Generating simulated datasets...")
    cov_inv = np.linalg.inv(cov)
    sim_datasets = {}
    for j, true_alt in enumerate(true_alternatives):
        np.random.seed(42 + j)  # Reproducible per alternative
        sims = []
        for _ in range(n_sims):
            sims.append(generate_simulated_data(z_eff, quantities, cov, true_alt))
        sim_datasets[j] = sims

    # Compute power for each (candidate, true_alt) pair
    if verbose:
        print("  Computing power matrix...")
    power_matrix = np.zeros((n_candidates, n_alts))
    candidate_list = []
    t0 = time.time()

    for idx, (w0, wa) in enumerate([(w0, wa) for w0 in w0_grid for wa in wa_grid]):
        candidate_list.append((w0, wa))
        theory_alt = candidate_theories[(w0, wa)]

        for j in range(n_alts):
            rejections = 0
            for sim_data in sim_datasets[j]:
                # log LR = -0.5 * (chi2_null - chi2_alt) for Gaussian
                residual_null = sim_data - theory_null
                residual_alt = sim_data - theory_alt
                chi2_null = float(residual_null @ cov_inv @ residual_null)
                chi2_alt = float(residual_alt @ cov_inv @ residual_alt)
                log_lr = 0.5 * (chi2_null - chi2_alt)

                if log_lr >= np.log(threshold):
                    rejections += 1

            power_matrix[idx, j] = rejections / n_sims

        if verbose and (idx + 1) % 25 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            remaining = (n_candidates - idx - 1) / rate
            print(f"    {idx+1}/{n_candidates} ({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    # Maximin: for each candidate, take minimum power across alternatives
    min_power = np.min(power_matrix, axis=1)

    # Also compute average power (alternative criterion)
    avg_power = np.mean(power_matrix, axis=1)

    # Find best candidate
    best_idx_minpower = np.argmax(min_power)
    best_idx_avgpower = np.argmax(avg_power)

    w0_star_min, wa_star_min = candidate_list[best_idx_minpower]
    w0_star_avg, wa_star_avg = candidate_list[best_idx_avgpower]

    if verbose:
        elapsed = time.time() - t0
        print(f"\n  Done in {elapsed:.1f}s")
        print(f"\n  MAXIMIN optimal: w0* = {w0_star_min:.3f}, wa* = {wa_star_min:.3f}")
        print(f"    Min power across alternatives: {min_power[best_idx_minpower]:.3f}")
        print(f"    Power per alternative: {power_matrix[best_idx_minpower]}")
        print(f"\n  AVG-POWER optimal: w0* = {w0_star_avg:.3f}, wa* = {wa_star_avg:.3f}")
        print(f"    Avg power: {avg_power[best_idx_avgpower]:.3f}")
        print(f"    Power per alternative: {power_matrix[best_idx_avgpower]}")

    return {
        'w0_star_minpower': w0_star_min,
        'wa_star_minpower': wa_star_min,
        'w0_star_avgpower': w0_star_avg,
        'wa_star_avgpower': wa_star_avg,
        'power_matrix': power_matrix,
        'min_power': min_power,
        'avg_power': avg_power,
        'candidate_list': candidate_list,
        'w0_grid': w0_grid,
        'wa_grid': wa_grid,
        'true_alternatives': true_alternatives,
        'alpha': alpha,
        'n_sims': n_sims,
    }


def compute_ump_evalue(data, cov, z_eff, quantities, w0_star, wa_star):
    """
    Compute the UMP e-value at the optimal point alternative.

    This is simply the likelihood ratio L(data|theta_1*)/L(data|H0),
    which is a valid e-value because theta_1* is pre-specified (found
    by simulation, not by looking at the actual data).

    Parameters
    ----------
    data : array, observed BAO measurements
    cov : array, covariance matrix
    z_eff : array, effective redshifts
    quantities : list, BAO quantity types
    w0_star, wa_star : optimal point alternative from find_ump_theta

    Returns
    -------
    EValueResult
    """
    cosmo_alt = CosmologyParams(w0=w0_star, wa=wa_star)
    pred_alt = compute_bao_predictions(z_eff, cosmo_alt)
    theory_alt = _build_theory_vector(pred_alt, z_eff, quantities)

    pred_null = compute_bao_predictions(z_eff, LCDM)
    theory_null = _build_theory_vector(pred_null, z_eff, quantities)

    log_L_alt = log_likelihood(data, theory_alt, cov)
    log_L_null = log_likelihood(data, theory_null, cov)

    log_e = log_L_alt - log_L_null
    e_value = np.exp(log_e)

    chi2_null = chi_squared(data, theory_null, cov)
    chi2_alt = chi_squared(data, theory_alt, cov)

    return EValueResult(
        e_value=e_value,
        log_e=log_e,
        chi2_null=chi2_null,
        chi2_alt=chi2_alt,
        delta_chi2=chi2_null - chi2_alt,
        null_params=LCDM,
        alt_params=cosmo_alt,
        method=f'ump_evalue (w0={w0_star:.3f}, wa={wa_star:.3f})'
    )


# ---------------------------------------------------------------------------
# Refinement: optimize continuously around the grid optimum
# ---------------------------------------------------------------------------

def refine_ump_theta(z_eff, quantities, cov, w0_init, wa_init,
                     alpha=0.01, n_sims=1000, true_alternatives=None,
                     verbose=True):
    """
    Refine the UMP theta_1 using continuous optimization (Nelder-Mead)
    starting from the grid search result.

    Uses average power as the objective (smoother than maximin for optimization).
    """
    if true_alternatives is None:
        true_alternatives = [
            CosmologyParams(w0=-0.75, wa=-1.05),
            CosmologyParams(w0=-0.85, wa=-0.50),
            CosmologyParams(w0=-0.65, wa=-1.20),
            CosmologyParams(w0=-0.90, wa=-0.20),
        ]

    threshold = 1.0 / alpha

    # Precompute
    pred_null = compute_bao_predictions(z_eff, LCDM)
    theory_null = _build_theory_vector(pred_null, z_eff, quantities)
    cov_inv = np.linalg.inv(cov)

    # Pre-generate sims (fixed seed for consistency across evaluations)
    sim_datasets = {}
    for j, true_alt in enumerate(true_alternatives):
        np.random.seed(42 + j)
        sims = [generate_simulated_data(z_eff, quantities, cov, true_alt)
                for _ in range(n_sims)]
        sim_datasets[j] = sims

    eval_count = [0]

    def neg_avg_power(params):
        w0, wa = params
        cosmo = CosmologyParams(w0=w0, wa=wa)
        pred = compute_bao_predictions(z_eff, cosmo)
        theory_alt = _build_theory_vector(pred, z_eff, quantities)

        total_power = 0.0
        for j in range(len(true_alternatives)):
            rejections = 0
            for sim_data in sim_datasets[j]:
                residual_null = sim_data - theory_null
                residual_alt = sim_data - theory_alt
                chi2_n = float(residual_null @ cov_inv @ residual_null)
                chi2_a = float(residual_alt @ cov_inv @ residual_alt)
                if 0.5 * (chi2_n - chi2_a) >= np.log(threshold):
                    rejections += 1
            total_power += rejections / n_sims

        avg_power = total_power / len(true_alternatives)
        eval_count[0] += 1

        if verbose and eval_count[0] % 10 == 0:
            print(f"    Eval {eval_count[0]}: w0={w0:.4f}, wa={wa:.4f}, avg_power={avg_power:.4f}")

        return -avg_power  # Minimize negative power

    if verbose:
        print(f"\n  Refining from w0={w0_init:.3f}, wa={wa_init:.3f}...")

    result = minimize(neg_avg_power, x0=[w0_init, wa_init],
                      method='Nelder-Mead',
                      options={'xatol': 0.01, 'fatol': 0.005, 'maxiter': 100})

    w0_refined, wa_refined = result.x

    if verbose:
        print(f"  Refined: w0* = {w0_refined:.4f}, wa* = {wa_refined:.4f}")
        print(f"  Avg power: {-result.fun:.4f}")
        print(f"  Evaluations: {eval_count[0]}")

    return w0_refined, wa_refined, -result.fun


# ---------------------------------------------------------------------------
# Main: run the full UMP analysis
# ---------------------------------------------------------------------------

def run_ump_analysis(verbose=True):
    """
    Run the complete UMP e-value analysis on DESI DR2 data.

    Steps:
    1. Load DESI data
    2. Grid search for optimal theta_1
    3. Refine with continuous optimization
    4. Compute UMP e-value on real data
    5. Compare with other methods
    """
    # Load data
    data_dir = Path(__file__).parent.parent / 'data' / 'dr2'
    dataset = load_desi_data(data_dir)
    data = dataset.data
    cov = dataset.cov
    z_eff = dataset.z_eff
    quantities = dataset.quantities

    if verbose:
        print("=" * 70)
        print("UMP E-VALUE ANALYSIS FOR DESI DR2")
        print("=" * 70)
        print(f"\nData: {len(data)} measurements, 7 redshift bins")
        print(f"Null: LCDM (w0=-1, wa=0)")

    # Step 1: Grid search
    print("\n--- Step 1: Grid search for optimal theta_1 ---")
    grid_result = find_ump_theta(
        z_eff, quantities, cov,
        alpha=0.01,
        n_sims=500,
        grid_resolution=15,
        verbose=verbose
    )

    # Step 2: Refine
    print("\n--- Step 2: Continuous refinement ---")
    w0_refined, wa_refined, refined_power = refine_ump_theta(
        z_eff, quantities, cov,
        w0_init=grid_result['w0_star_avgpower'],
        wa_init=grid_result['wa_star_avgpower'],
        alpha=0.01,
        n_sims=1000,
        verbose=verbose
    )

    # Step 3: Compute UMP e-value on real data
    print("\n--- Step 3: UMP e-value on real DESI data ---")
    ump_result = compute_ump_evalue(data, cov, z_eff, quantities,
                                     w0_refined, wa_refined)

    print(f"\n  UMP point alternative: w0* = {w0_refined:.4f}, wa* = {wa_refined:.4f}")
    print(f"  E_UMP = {ump_result.e_value:.2f}")
    print(f"  log(E_UMP) = {ump_result.log_e:.4f}")
    print(f"  Delta chi2 = {ump_result.delta_chi2:.2f}")
    if ump_result.e_value > 1:
        sigma = np.sqrt(2 * np.log(ump_result.e_value))
        print(f"  sigma equivalent ~ {sigma:.2f}")

    # Step 4: Compare with grid maximin result
    print("\n--- Comparison: grid maximin ---")
    ump_grid = compute_ump_evalue(data, cov, z_eff, quantities,
                                   grid_result['w0_star_minpower'],
                                   grid_result['wa_star_minpower'])
    print(f"  Maximin theta: w0 = {grid_result['w0_star_minpower']:.3f}, wa = {grid_result['wa_star_minpower']:.3f}")
    print(f"  E_UMP (maximin) = {ump_grid.e_value:.2f}")

    # Step 5: Compare with existing methods
    print("\n--- Comparison with existing e-values ---")
    from evalue_analysis import grow_evalue, split_evalue, loocv_evalue

    mix_result = grow_evalue(data, cov, z_eff, quantities)
    _, split_result = split_evalue(data, cov, z_eff, quantities)
    loo_result = loocv_evalue(data, cov, z_eff, quantities)

    print(f"  Uniform mixture: E = {mix_result.e_value:.2f} (log E = {mix_result.log_e:.2f})")
    print(f"  Data-split:      E = {split_result.e_value:.2f} (log E = {split_result.log_e:.2f})")
    print(f"  LOO average:     E = {loo_result.e_value:.2f} (log E = {loo_result.log_e:.2f})")
    print(f"  UMP (refined):   E = {ump_result.e_value:.2f} (log E = {ump_result.log_e:.2f})")

    # Step 6: Validate that UMP e-value is calibrated under H0
    print("\n--- Step 4: Calibration check under H0 ---")
    np.random.seed(12345)
    n_cal = 500
    e_vals_h0 = []
    pred_null = compute_bao_predictions(z_eff, LCDM)
    theory_null = _build_theory_vector(pred_null, z_eff, quantities)
    cosmo_ump = CosmologyParams(w0=w0_refined, wa=wa_refined)
    pred_ump = compute_bao_predictions(z_eff, cosmo_ump)
    theory_ump = _build_theory_vector(pred_ump, z_eff, quantities)
    cov_inv = np.linalg.inv(cov)

    for _ in range(n_cal):
        sim = generate_simulated_data(z_eff, quantities, cov, LCDM)
        r_null = sim - theory_null
        r_alt = sim - theory_ump
        chi2_n = float(r_null @ cov_inv @ r_null)
        chi2_a = float(r_alt @ cov_inv @ r_alt)
        e_vals_h0.append(np.exp(0.5 * (chi2_n - chi2_a)))

    e_vals_h0 = np.array(e_vals_h0)
    print(f"  Mean E under H0: {np.mean(e_vals_h0):.4f} (should be ~1.0)")
    print(f"  Median E under H0: {np.median(e_vals_h0):.4f}")
    print(f"  P(E > 1/alpha) under H0: {np.mean(e_vals_h0 >= 1/0.01):.4f} (should be <= alpha = 0.01)")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  UMP e-value: E = {ump_result.e_value:.2f}")
    print(f"  This is a VALID e-value (simple vs simple LR, theta_1 pre-specified)")
    print(f"  theta_1 was chosen to maximize power at alpha = 0.01")
    print(f"  It uses ALL 13 data points (no splitting) and pays NO Occam penalty")

    return {
        'ump_result': ump_result,
        'grid_result': grid_result,
        'w0_refined': w0_refined,
        'wa_refined': wa_refined,
        'calibration_mean': np.mean(e_vals_h0),
        'calibration_type1': np.mean(e_vals_h0 >= 1/0.01),
    }


if __name__ == '__main__':
    results = run_ump_analysis(verbose=True)
