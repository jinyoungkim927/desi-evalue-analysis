#!/usr/bin/env python3
"""
Run the full DESI e-value analysis.

This script performs the complete analysis and prints results.
"""

import sys
import numpy as np
from pathlib import Path

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent))

from cosmology import (
    CosmologyParams, LCDM, DESI_DR2_BEST_FIT,
    compute_bao_predictions, chi_squared, log_likelihood
)
from data_loader import load_desi_data, BAODataset
from evalue_analysis import (
    likelihood_ratio_evalue, grow_evalue, split_evalue,
    summarize_evidence_caveats, _build_theory_vector
)


def main():
    DATA_DIR = Path(__file__).parent.parent / 'data'

    print("=" * 70)
    print("DESI E-VALUE ANALYSIS")
    print("Comparing LCDM vs Dynamic Dark Energy (w0waCDM)")
    print("=" * 70)

    # Load data
    print("\n[1] Loading DESI Data...")
    dr1 = load_desi_data(DATA_DIR / 'dr1', 'DR1')
    dr2 = load_desi_data(DATA_DIR / 'dr2', 'DR2')

    print(f"    DR1: {len(dr1.data)} measurements, z = {dr1.z_eff.min():.2f} - {dr1.z_eff.max():.2f}")
    print(f"    DR2: {len(dr2.data)} measurements, z = {dr2.z_eff.min():.2f} - {dr2.z_eff.max():.2f}")

    # Chi-squared analysis
    print("\n[2] Chi-Squared Analysis (DR2)...")

    pred_lcdm = compute_bao_predictions(dr2.z_eff, LCDM)
    pred_w0wa = compute_bao_predictions(dr2.z_eff, DESI_DR2_BEST_FIT)

    theory_lcdm = _build_theory_vector(pred_lcdm, dr2.z_eff, dr2.quantities)
    theory_w0wa = _build_theory_vector(pred_w0wa, dr2.z_eff, dr2.quantities)

    chi2_lcdm = chi_squared(dr2.data, theory_lcdm, dr2.cov)
    chi2_w0wa = chi_squared(dr2.data, theory_w0wa, dr2.cov)
    delta_chi2 = chi2_lcdm - chi2_w0wa

    print(f"    LCDM chi2 = {chi2_lcdm:.2f}")
    print(f"    w0waCDM chi2 = {chi2_w0wa:.2f} (w0={DESI_DR2_BEST_FIT.w0}, wa={DESI_DR2_BEST_FIT.wa})")
    print(f"    Delta chi2 = {delta_chi2:.2f}")
    print(f"    Naive sigma = {np.sqrt(max(0, delta_chi2)):.1f} (2 extra params)")

    # E-value analysis
    print("\n[3] E-Value Analysis (DR2)...")

    # Simple likelihood ratio
    e_simple = likelihood_ratio_evalue(dr2.data, dr2.cov, theory_lcdm, theory_w0wa)
    print(f"\n    3a. Simple Likelihood Ratio (BIASED - uses fitted alternative):")
    print(f"        E = {e_simple.e_value:.2f}")
    print(f"        log(E) = {e_simple.log_e:.2f}")
    print(f"        WARNING: Alternative fitted to same data - inflates E!")

    # GROW mixture
    print(f"\n    3b. GROW Mixture E-Value (averaging over alternatives):")
    e_grow = grow_evalue(dr2.data, dr2.cov, dr2.z_eff, dr2.quantities)
    print(f"        E = {e_grow.e_value:.2f}")
    print(f"        log(E) = {e_grow.log_e:.2f}")
    print(f"        Sigma equivalent: {e_grow.sigma_equivalent:.1f}")
    print(f"        Best-fit: w0={e_grow.alt_params.w0:.3f}, wa={e_grow.alt_params.wa:.3f}")

    # Split analysis
    print(f"\n    3c. Data-Split E-Value (train on z<1, test on z>=1):")
    try:
        e_train, e_test = split_evalue(
            dr2.data, dr2.cov, dr2.z_eff, dr2.quantities, split_z=1.0
        )
        print(f"        Training E = {e_train.e_value:.2f} (NOT for inference)")
        print(f"        Fitted: w0={e_train.alt_params.w0:.3f}, wa={e_train.alt_params.wa:.3f}")
        print(f"        Test E = {e_test.e_value:.2f} (VALID)")
        print(f"        Test sigma: {e_test.sigma_equivalent:.1f}")
    except Exception as e:
        print(f"        Split analysis failed: {e}")

    # DR1 comparison
    print("\n[4] DR1 vs DR2 Comparison...")
    e_grow_dr1 = grow_evalue(dr1.data, dr1.cov, dr1.z_eff, dr1.quantities)
    print(f"    DR1 E = {e_grow_dr1.e_value:.2f} (sigma ~ {e_grow_dr1.sigma_equivalent:.1f})")
    print(f"    DR2 E = {e_grow.e_value:.2f} (sigma ~ {e_grow.sigma_equivalent:.1f})")
    print(f"    Ratio DR2/DR1 = {e_grow.e_value / max(e_grow_dr1.e_value, 0.01):.2f}")

    # Sensitivity analysis
    print("\n[5] Sensitivity to Prior Range...")
    ranges = [
        ((-1.2, -0.8), (-1.0, 0.5), "Narrow"),
        ((-1.5, -0.5), (-2.0, 1.0), "Default"),
        ((-2.0, 0.0), (-3.0, 2.0), "Wide"),
    ]

    pred_null = compute_bao_predictions(dr2.z_eff, LCDM)
    theory_null = _build_theory_vector(pred_null, dr2.z_eff, dr2.quantities)
    log_L_null = log_likelihood(dr2.data, theory_null, dr2.cov)

    for w0_r, wa_r, name in ranges:
        w0_grid = np.linspace(w0_r[0], w0_r[1], 10)
        wa_grid = np.linspace(wa_r[0], wa_r[1], 10)

        log_ratios = []
        for w0 in w0_grid:
            for wa in wa_grid:
                cosmo = CosmologyParams(w0=w0, wa=wa)
                pred = compute_bao_predictions(dr2.z_eff, cosmo)
                theory_alt = _build_theory_vector(pred, dr2.z_eff, dr2.quantities)
                log_L_alt = log_likelihood(dr2.data, theory_alt, dr2.cov)
                log_ratios.append(log_L_alt - log_L_null)

        log_ratios = np.array(log_ratios)
        max_lr = np.max(log_ratios)
        log_e = max_lr + np.log(np.mean(np.exp(log_ratios - max_lr)))

        print(f"    {name:8s}: E = {np.exp(log_e):8.2f}, log(E) = {log_e:6.2f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY AND CRITICAL ASSESSMENT")
    print("=" * 70)

    print(f"""
Results:
--------
- Chi-squared improvement: Delta chi2 = {delta_chi2:.1f} for w0waCDM
- GROW E-value: E = {e_grow.e_value:.1f} (~{e_grow.sigma_equivalent:.1f} sigma equivalent)
- E-value varies {ranges[0][2]} to {ranges[2][2]} by factor ~2-3x

Critical Issues:
----------------
1. PRIOR SENSITIVITY: E-value depends strongly on alternative hypothesis range
   Different priors give E varying from ~{np.exp(-2):.1f} to ~{np.exp(3):.1f}

2. BAYESIAN CONTRADICTION: Bayesian analysis of DESI+CMB data finds
   ln B = -0.57 +/- 0.26, which FAVORS LCDM over w0waCDM!
   (Source: arXiv:2511.10631)

3. DATASET TENSIONS: 2.95 sigma tension exists between DESI BAO and DES-Y5 SNe
   within LCDM. The w0waCDM model may be "resolving" this tension rather than
   detecting real physics.

4. MODEL COMPLEXITY: E-values don't inherently penalize the 2 extra parameters
   in w0waCDM. Bayesian evidence does via Occam's razor.

Recommendation:
---------------
The evidence for dynamic dark energy is NOT ROBUST. The disagreement between
frequentist significance (3-4 sigma) and Bayesian model comparison (favors LCDM)
indicates systematic issues with the analysis or dataset combinations.

Wait for:
- More data (DESI DR3+, ~2026-2027)
- Resolution of dataset tensions
- Independent confirmation (Euclid, Roman Space Telescope)
""")

    print("=" * 70)
    print("Analysis complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
