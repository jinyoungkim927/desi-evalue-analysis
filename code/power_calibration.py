#!/usr/bin/env python3
"""
Monte Carlo Power Calibration for the Data-Split E-Value Procedure.

This script answers the critical question: Is E_split = 1.4 consistent with
the data-split procedure being underpowered, even when the alternative (w0waCDM)
is TRUE?

The paper reports:
  - E_full (uniform mixture) ~ 400 (biased, uses same data to fit and test)
  - E_split = 1.4 (valid, split at z=1, fit on low-z, test on high-z)
  - Ratio: E_full / E_split ~ 280x

We simulate under:
  H0: LCDM is true  -> What does E_split look like? (calibration)
  H1: w0waCDM with DESI best-fit is true -> What does E_split look like? (power)

If median E_split under H1 is ~2-5, the test is underpowered and E=1.4 is
uninformative. If median E_split under H1 is ~50+, then E=1.4 genuinely
suggests no signal in the high-z data.

References:
- Shafer (2021): Testing by betting
- DESI DR2: arXiv:2503.14738
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
from evalue_analysis import _build_theory_vector


# ---------------------------------------------------------------------------
# Core simulation routines
# ---------------------------------------------------------------------------

def generate_simulated_data(z_eff, quantities, cov, true_cosmo):
    """
    Generate one simulated BAO dataset under a given cosmology.

    Steps:
        1. Compute the true theory vector for the given cosmology.
        2. Draw Gaussian noise from the covariance matrix.
        3. Return theory + noise as simulated data.
    """
    pred = compute_bao_predictions(z_eff, true_cosmo)
    theory = _build_theory_vector(pred, z_eff, quantities)
    noise = np.random.multivariate_normal(np.zeros(len(theory)), cov)
    return theory + noise


def run_split_evalue_on_sim(sim_data, z_eff, quantities, cov, split_z=1.0):
    """
    Run the data-split e-value procedure on a simulated dataset.

    Mirrors the split_evalue function in evalue_analysis.py but operates
    on a provided data vector rather than the real data.

    Returns:
        dict with keys: e_split, log_e_split, w0_fit, wa_fit, e_train, chi2_null_test, chi2_alt_test
    """
    # Split into low-z (training) and high-z (test)
    low_z_mask = z_eff < split_z
    high_z_mask = ~low_z_mask

    low_z_idx = np.where(low_z_mask)[0]
    high_z_idx = np.where(high_z_mask)[0]

    n_low = len(low_z_idx)
    n_high = len(high_z_idx)

    if n_low < 2 or n_high < 2:
        return None  # Cannot split

    data_low = sim_data[low_z_idx]
    data_high = sim_data[high_z_idx]
    cov_low = cov[np.ix_(low_z_idx, low_z_idx)]
    cov_high = cov[np.ix_(high_z_idx, high_z_idx)]
    z_low = z_eff[low_z_idx]
    z_high = z_eff[high_z_idx]
    q_low = [quantities[i] for i in low_z_idx]
    q_high = [quantities[i] for i in high_z_idx]

    # Step 1: Fit w0, wa on low-z data
    def neg_log_like_train(params):
        w0, wa = params
        cosmo = CosmologyParams(w0=w0, wa=wa)
        pred = compute_bao_predictions(z_low, cosmo)
        theory = _build_theory_vector(pred, z_low, q_low)
        return -log_likelihood(data_low, theory, cov_low)

    result = minimize(
        neg_log_like_train,
        x0=[-0.9, -0.5],
        bounds=[(-2.0, 0.0), (-3.0, 2.0)],
        method='L-BFGS-B'
    )
    w0_fit, wa_fit = result.x
    alt_cosmo = CosmologyParams(w0=w0_fit, wa=wa_fit)

    # Step 2: Compute e-value on high-z test data
    pred_null_high = compute_bao_predictions(z_high, LCDM)
    pred_alt_high = compute_bao_predictions(z_high, alt_cosmo)
    theory_null_high = _build_theory_vector(pred_null_high, z_high, q_high)
    theory_alt_high = _build_theory_vector(pred_alt_high, z_high, q_high)

    log_L_null = log_likelihood(data_high, theory_null_high, cov_high)
    log_L_alt = log_likelihood(data_high, theory_alt_high, cov_high)
    log_e_test = log_L_alt - log_L_null
    e_test = np.exp(log_e_test)

    chi2_null_test = chi_squared(data_high, theory_null_high, cov_high)
    chi2_alt_test = chi_squared(data_high, theory_alt_high, cov_high)

    # Also compute training e-value for reference
    pred_null_low = compute_bao_predictions(z_low, LCDM)
    pred_alt_low = compute_bao_predictions(z_low, alt_cosmo)
    theory_null_low = _build_theory_vector(pred_null_low, z_low, q_low)
    theory_alt_low = _build_theory_vector(pred_alt_low, z_low, q_low)
    log_L_null_train = log_likelihood(data_low, theory_null_low, cov_low)
    log_L_alt_train = log_likelihood(data_low, theory_alt_low, cov_low)
    log_e_train = log_L_alt_train - log_L_null_train
    e_train = np.exp(log_e_train)

    return {
        'e_split': e_test,
        'log_e_split': log_e_test,
        'w0_fit': w0_fit,
        'wa_fit': wa_fit,
        'e_train': e_train,
        'log_e_train': log_e_train,
        'chi2_null_test': chi2_null_test,
        'chi2_alt_test': chi2_alt_test,
    }


def run_full_evalue_on_sim(sim_data, z_eff, quantities, cov):
    """
    Run the full (non-split) likelihood ratio e-value on simulated data.

    Fits w0, wa to the FULL dataset and computes E = L(alt)/L(null).
    This is the "biased" version (same data for fitting and testing).

    Returns:
        dict with keys: e_full, log_e_full, w0_fit, wa_fit
    """
    def neg_log_like(params):
        w0, wa = params
        cosmo = CosmologyParams(w0=w0, wa=wa)
        pred = compute_bao_predictions(z_eff, cosmo)
        theory = _build_theory_vector(pred, z_eff, quantities)
        return -log_likelihood(sim_data, theory, cov)

    result = minimize(
        neg_log_like,
        x0=[-0.9, -0.5],
        bounds=[(-2.0, 0.0), (-3.0, 2.0)],
        method='L-BFGS-B'
    )
    w0_fit, wa_fit = result.x
    alt_cosmo = CosmologyParams(w0=w0_fit, wa=wa_fit)

    pred_null = compute_bao_predictions(z_eff, LCDM)
    pred_alt = compute_bao_predictions(z_eff, alt_cosmo)
    theory_null = _build_theory_vector(pred_null, z_eff, quantities)
    theory_alt = _build_theory_vector(pred_alt, z_eff, quantities)

    log_L_null = log_likelihood(sim_data, theory_null, cov)
    log_L_alt = log_likelihood(sim_data, theory_alt, cov)
    log_e = log_L_alt - log_L_null
    e_full = np.exp(log_e)

    return {
        'e_full': e_full,
        'log_e_full': log_e,
        'w0_fit': w0_fit,
        'wa_fit': wa_fit,
    }


# ---------------------------------------------------------------------------
# Monte Carlo simulation driver
# ---------------------------------------------------------------------------

def run_monte_carlo(z_eff, quantities, cov, true_cosmo, n_sims=500,
                    split_z=1.0, compute_full=False, label=""):
    """
    Run Monte Carlo simulations of the split e-value procedure.

    Parameters:
        z_eff: redshift array
        quantities: list of BAO quantity labels
        cov: covariance matrix
        true_cosmo: CosmologyParams for generating data
        n_sims: number of simulations
        split_z: redshift split point
        compute_full: also compute the full (unsplit) e-value for ratio analysis
        label: descriptive label for printing

    Returns:
        dict of arrays with simulation results
    """
    e_splits = []
    log_e_splits = []
    w0_fits = []
    wa_fits = []
    e_trains = []
    e_fulls = []
    log_e_fulls = []

    t0 = time.time()
    failed = 0

    for i in range(n_sims):
        sim_data = generate_simulated_data(z_eff, quantities, cov, true_cosmo)

        res = run_split_evalue_on_sim(sim_data, z_eff, quantities, cov, split_z=split_z)
        if res is None:
            failed += 1
            continue

        e_splits.append(res['e_split'])
        log_e_splits.append(res['log_e_split'])
        w0_fits.append(res['w0_fit'])
        wa_fits.append(res['wa_fit'])
        e_trains.append(res['e_train'])

        if compute_full:
            res_full = run_full_evalue_on_sim(sim_data, z_eff, quantities, cov)
            e_fulls.append(res_full['e_full'])
            log_e_fulls.append(res_full['log_e_full'])

        # Progress reporting
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (n_sims - i - 1) / rate
            print(f"    {label} sim {i+1}/{n_sims} "
                  f"({elapsed:.1f}s elapsed, ~{remaining:.0f}s remaining)")

    elapsed = time.time() - t0

    results = {
        'e_split': np.array(e_splits),
        'log_e_split': np.array(log_e_splits),
        'w0_fit': np.array(w0_fits),
        'wa_fit': np.array(wa_fits),
        'e_train': np.array(e_trains),
        'elapsed_s': elapsed,
        'n_success': len(e_splits),
        'n_failed': failed,
    }

    if compute_full:
        results['e_full'] = np.array(e_fulls)
        results['log_e_full'] = np.array(log_e_fulls)

    return results


def print_summary(results, label, observed_e_split=1.4):
    """Print summary statistics for a set of MC results."""
    e = results['e_split']
    n = len(e)

    print(f"\n{'=' * 65}")
    print(f"  {label}")
    print(f"{'=' * 65}")
    print(f"  Simulations: {n} successful, {results['n_failed']} failed")
    print(f"  Time: {results['elapsed_s']:.1f}s ({results['elapsed_s']/max(n,1)*1000:.1f}ms per sim)")
    print(f"")
    print(f"  E_split distribution:")
    print(f"    Median:       {np.median(e):.3f}")
    print(f"    Mean:         {np.mean(e):.3f}")
    print(f"    Geometric mean: {np.exp(np.mean(results['log_e_split'])):.3f}")
    print(f"    5th pctl:     {np.percentile(e, 5):.3f}")
    print(f"    25th pctl:    {np.percentile(e, 25):.3f}")
    print(f"    75th pctl:    {np.percentile(e, 75):.3f}")
    print(f"    95th pctl:    {np.percentile(e, 95):.3f}")
    print(f"    Max:          {np.max(e):.3f}")
    print(f"    Min:          {np.min(e):.6f}")
    print(f"")
    print(f"  log(E_split) distribution:")
    print(f"    Median:       {np.median(results['log_e_split']):.3f}")
    print(f"    Mean:         {np.mean(results['log_e_split']):.3f}")
    print(f"    Std:          {np.std(results['log_e_split']):.3f}")
    print(f"")
    print(f"  Exceedance probabilities:")
    print(f"    P(E > 1)    = {np.mean(e > 1):.3f}  ({np.sum(e > 1)}/{n})")
    print(f"    P(E > 1.4)  = {np.mean(e > 1.4):.3f}  ({np.sum(e > 1.4)}/{n})")
    print(f"    P(E > 3)    = {np.mean(e > 3):.3f}  ({np.sum(e > 3)}/{n})")
    print(f"    P(E > 10)   = {np.mean(e > 10):.3f}  ({np.sum(e > 10)}/{n})")
    print(f"    P(E > 20)   = {np.mean(e > 20):.3f}  ({np.sum(e > 20)}/{n})")
    print(f"    P(E > 100)  = {np.mean(e > 100):.3f}  ({np.sum(e > 100)}/{n})")
    print(f"")
    print(f"  Fitted w0 on low-z training data:")
    print(f"    Mean:  {np.mean(results['w0_fit']):.3f}  Std: {np.std(results['w0_fit']):.3f}")
    print(f"  Fitted wa on low-z training data:")
    print(f"    Mean:  {np.mean(results['wa_fit']):.3f}  Std: {np.std(results['wa_fit']):.3f}")

    if 'e_full' in results:
        ef = results['e_full']
        ratio = ef / np.maximum(e, 1e-10)
        print(f"")
        print(f"  E_full (biased, same-data fit+test):")
        print(f"    Median:       {np.median(ef):.3f}")
        print(f"    Mean:         {np.mean(ef):.3f}")
        print(f"    95th pctl:    {np.percentile(ef, 95):.3f}")
        print(f"")
        print(f"  E_full / E_split ratio:")
        print(f"    Median:       {np.median(ratio):.1f}")
        print(f"    Mean:         {np.mean(ratio):.1f}")
        print(f"    5th pctl:     {np.percentile(ratio, 5):.1f}")
        print(f"    95th pctl:    {np.percentile(ratio, 95):.1f}")
        print(f"    (Paper observed: ~280x)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    DATA_DIR = Path(__file__).parent.parent / 'data'

    print("=" * 70)
    print("MONTE CARLO POWER CALIBRATION FOR DATA-SPLIT E-VALUE")
    print("=" * 70)
    print("")
    print("Question: Is E_split = 1.4 evidence against w0waCDM,")
    print("or is the split procedure simply underpowered?")
    print("")

    # Load DESI DR2 data for covariance and redshift structure
    dr2 = load_desi_data(DATA_DIR / 'dr2', 'DR2')
    print(f"Loaded DESI DR2: {len(dr2.data)} measurements")
    print(f"  Redshifts: {dr2.z_eff}")
    print(f"  Quantities: {dr2.quantities}")
    print(f"  Diagonal errors: {np.sqrt(np.diag(dr2.cov))}")
    print("")

    # Show the split structure
    for sz in [0.8, 1.0, 1.2]:
        n_low = np.sum(dr2.z_eff < sz)
        n_high = np.sum(dr2.z_eff >= sz)
        print(f"  Split at z={sz}: {n_low} low-z points, {n_high} high-z test points")
    print("")

    # Show signal size: how different are LCDM and w0waCDM predictions?
    pred_lcdm = compute_bao_predictions(dr2.z_eff, LCDM)
    pred_h1 = compute_bao_predictions(dr2.z_eff, DESI_DR2_BEST_FIT)
    theory_lcdm = _build_theory_vector(pred_lcdm, dr2.z_eff, dr2.quantities)
    theory_h1 = _build_theory_vector(pred_h1, dr2.z_eff, dr2.quantities)
    errors = np.sqrt(np.diag(dr2.cov))
    signal = theory_h1 - theory_lcdm

    print("Signal vs noise (per data point):")
    print(f"  {'z':>6} {'Quantity':>12} {'Signal':>10} {'Error':>10} {'S/N':>8}")
    print(f"  {'-'*50}")
    for i in range(len(dr2.data)):
        sn = signal[i] / errors[i]
        print(f"  {dr2.z_eff[i]:6.3f} {dr2.quantities[i]:>12} {signal[i]:10.4f} {errors[i]:10.4f} {sn:8.2f}")

    # Full chi2 difference (without splitting)
    full_delta_chi2 = chi_squared(theory_lcdm, theory_h1, dr2.cov)
    # Actually compute expected delta chi2 for signal in noise
    residual = theory_h1 - theory_lcdm
    cov_inv = np.linalg.inv(dr2.cov)
    expected_delta_chi2_full = float(residual @ cov_inv @ residual)
    print(f"\n  Expected full-data delta_chi2 (H1 vs H0): {expected_delta_chi2_full:.2f}")
    print(f"  Expected full-data E = exp(delta_chi2/2) = {np.exp(expected_delta_chi2_full/2):.1f}")

    # Compute expected delta_chi2 for just the test data (high-z)
    # Using the TRUE H1 parameters (best case for the split test)
    split_z = 1.0
    high_z_idx = np.where(dr2.z_eff >= split_z)[0]
    residual_high = residual[high_z_idx]
    cov_high = dr2.cov[np.ix_(high_z_idx, high_z_idx)]
    cov_inv_high = np.linalg.inv(cov_high)
    expected_delta_chi2_high = float(residual_high @ cov_inv_high @ residual_high)
    print(f"\n  Expected test-data delta_chi2 (z>={split_z}, using TRUE H1 params): {expected_delta_chi2_high:.2f}")
    print(f"  Expected test-data E (best case) = exp(delta_chi2/2) = {np.exp(expected_delta_chi2_high/2):.2f}")
    print(f"  NOTE: Actual E_split will be LOWER because w0,wa are ESTIMATED from noisy low-z data")
    print("")

    # Configuration
    N_SIMS = 500  # Increase to 1000 if time permits
    np.random.seed(42)

    # ------------------------------------------------------------------
    # A) Simulations under H0 (LCDM is true)
    # ------------------------------------------------------------------
    print("\n" + "#" * 70)
    print("# PART A: Simulations under H0 (LCDM is true)")
    print("#" * 70)

    results_h0 = run_monte_carlo(
        dr2.z_eff, dr2.quantities, dr2.cov,
        true_cosmo=LCDM,
        n_sims=N_SIMS,
        split_z=1.0,
        compute_full=False,
        label="H0"
    )
    print_summary(results_h0, "H0 (LCDM true), split at z=1.0")

    # ------------------------------------------------------------------
    # B) Simulations under H1 (w0waCDM with DESI best-fit is true)
    # ------------------------------------------------------------------
    print("\n" + "#" * 70)
    print("# PART B: Simulations under H1 (w0waCDM DESI best-fit is true)")
    print("#" * 70)
    print(f"  True parameters: w0={DESI_DR2_BEST_FIT.w0}, wa={DESI_DR2_BEST_FIT.wa}")

    results_h1 = run_monte_carlo(
        dr2.z_eff, dr2.quantities, dr2.cov,
        true_cosmo=DESI_DR2_BEST_FIT,
        n_sims=N_SIMS,
        split_z=1.0,
        compute_full=True,
        label="H1"
    )
    print_summary(results_h1, "H1 (w0waCDM true, w0=-0.75, wa=-1.05), split at z=1.0")

    # ------------------------------------------------------------------
    # C) E_full / E_split ratio analysis under H1
    # ------------------------------------------------------------------
    print("\n" + "#" * 70)
    print("# PART C: E_full / E_split ratio calibration under H1")
    print("#" * 70)

    if 'e_full' in results_h1:
        ef = results_h1['e_full']
        es = results_h1['e_split']
        ratio = ef / np.maximum(es, 1e-10)

        print(f"\n  The paper observes E_full/E_split ~ 400/1.4 ~ 280")
        print(f"")
        print(f"  Under H1 simulations:")
        print(f"    Median E_full/E_split:  {np.median(ratio):.1f}")
        print(f"    Mean E_full/E_split:    {np.mean(ratio):.1f}")
        print(f"    P(ratio > 100):         {np.mean(ratio > 100):.3f}")
        print(f"    P(ratio > 280):         {np.mean(ratio > 280):.3f}")
        print(f"    P(ratio > 500):         {np.mean(ratio > 500):.3f}")
        print(f"")
        if np.mean(ratio > 280) > 0.05:
            print(f"  --> A ratio of 280x is NOT unusual under H1.")
            print(f"      The large gap is expected due to overfitting in E_full.")
        else:
            print(f"  --> A ratio of 280x is somewhat unusual under H1.")
            print(f"      This may indicate the full E-value is additionally inflated")
            print(f"      by overfitting or prior sensitivity.")

    # ------------------------------------------------------------------
    # D) Split-point sensitivity under H1
    # ------------------------------------------------------------------
    print("\n" + "#" * 70)
    print("# PART D: Split-point sensitivity under H1")
    print("#" * 70)

    split_points = [0.8, 1.0, 1.2]
    split_results = {}

    for sz in split_points:
        n_low = np.sum(dr2.z_eff < sz)
        n_high = np.sum(dr2.z_eff >= sz)
        print(f"\n  Split z={sz}: {n_low} training points, {n_high} test points")

        if n_low < 2 or n_high < 2:
            print(f"    SKIPPED: too few points in one split")
            continue

        results_sz = run_monte_carlo(
            dr2.z_eff, dr2.quantities, dr2.cov,
            true_cosmo=DESI_DR2_BEST_FIT,
            n_sims=N_SIMS,
            split_z=sz,
            compute_full=False,
            label=f"H1,z={sz}"
        )
        split_results[sz] = results_sz
        print_summary(results_sz, f"H1 (w0waCDM true), split at z={sz}")

    # Comparison table
    print(f"\n{'=' * 65}")
    print(f"  Split-Point Comparison (H1 true)")
    print(f"{'=' * 65}")
    print(f"  {'z_split':>8} {'N_train':>8} {'N_test':>8} {'Med(E)':>10} {'Mean(E)':>10} {'P(E>1.4)':>10} {'P(E>10)':>10}")
    print(f"  {'-' * 68}")
    for sz in split_points:
        if sz in split_results:
            r = split_results[sz]
            n_low = np.sum(dr2.z_eff < sz)
            n_high = np.sum(dr2.z_eff >= sz)
            e = r['e_split']
            print(f"  {sz:8.1f} {n_low:8d} {n_high:8d} "
                  f"{np.median(e):10.3f} {np.mean(e):10.3f} "
                  f"{np.mean(e > 1.4):10.3f} {np.mean(e > 10):10.3f}")

    # ------------------------------------------------------------------
    # Final interpretation
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL INTERPRETATION")
    print("=" * 70)

    e_h0 = results_h0['e_split']
    e_h1 = results_h1['e_split']

    median_h0 = np.median(e_h0)
    median_h1 = np.median(e_h1)
    p_exceed_h0 = np.mean(e_h0 > 1.4)
    p_exceed_h1 = np.mean(e_h1 > 1.4)

    print(f"""
Observed E_split = 1.4 (from DESI paper, split at z=1)

Under H0 (LCDM true):
  Median E_split = {median_h0:.3f}
  P(E_split > 1.4) = {p_exceed_h0:.3f}
  -> Under the null, E_split > 1.4 happens {p_exceed_h0*100:.1f}% of the time.

Under H1 (w0waCDM true, DESI best-fit):
  Median E_split = {median_h1:.3f}
  P(E_split > 1.4) = {p_exceed_h1:.3f}
  -> Under the alternative, E_split > 1.4 happens {p_exceed_h1*100:.1f}% of the time.
""")

    if median_h1 < 5:
        print("""CONCLUSION: The data-split e-value procedure is SEVERELY UNDERPOWERED.
  Even when w0waCDM is the TRUE model generating the data, the median
  E_split is only ~{:.1f}. This means:

  1. E_split = 1.4 is UNINFORMATIVE -- it cannot distinguish H0 from H1.
  2. The paper's claim that E_split = 1.4 represents "only mild evidence"
     is misleading: the test has almost no power to detect the signal.
  3. The 280x ratio between E_full and E_split is a direct consequence
     of data splitting destroying statistical power, not evidence that
     the signal is absent in high-z data.
  4. The split procedure should NOT be used to calibrate the strength
     of the full-data evidence -- it is too weak to be informative.
""".format(median_h1))
    else:
        print("""CONCLUSION: The data-split e-value procedure HAS meaningful power.
  Under H1, median E_split = {:.1f}, so E_split = 1.4 is genuinely low.
  This suggests the high-z data does NOT support the w0waCDM alternative
  as strongly as expected.
""".format(median_h1))

    # Power analysis summary
    # Compute approximate "power" = P(E_split > threshold | H1)
    # and "size" = P(E_split > threshold | H0)
    print(f"  Power Analysis Table:")
    print(f"  {'Threshold':>12} {'Size (H0)':>12} {'Power (H1)':>12} {'Ratio':>10}")
    print(f"  {'-' * 48}")
    for thresh in [1.0, 1.4, 3.0, 5.0, 10.0, 20.0]:
        size = np.mean(e_h0 > thresh)
        power = np.mean(e_h1 > thresh)
        ratio_str = f"{power/max(size,0.001):.1f}" if size > 0 else "inf"
        print(f"  {thresh:12.1f} {size:12.3f} {power:12.3f} {ratio_str:>10}")

    print("\n" + "=" * 70)
    print("Analysis complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
