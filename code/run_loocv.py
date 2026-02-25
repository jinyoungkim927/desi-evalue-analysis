#!/usr/bin/env python3
"""
Run Leave-One-Out Cross-Validation (LOOCV) e-value analysis on DESI DR2 BAO data.

This script:
1. Loads DESI DR2 BAO measurements (13 data points across 7 redshift bins)
2. Performs LOOCV over redshift bins (not individual points)
3. For each fold, fits (w0, wa) on training bins and evaluates on held-out bin
4. Reports the product e-value and per-bin diagnostics
5. Compares with the standard z=1 split result for context
"""

import sys
import numpy as np
from pathlib import Path

# Ensure the code directory is on the path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_desi_data
from cosmology import CosmologyParams, LCDM, compute_bao_predictions
from evalue_analysis import (
    loocv_evalue, split_evalue, grow_evalue,
    _build_theory_vector, likelihood_ratio_evalue,
    summarize_evidence_caveats
)


def main():
    # =========================================================================
    # Load data
    # =========================================================================
    data_dir = Path(__file__).parent.parent / 'data' / 'dr2'
    dataset = load_desi_data(data_dir, 'DR2')

    print("=" * 70)
    print("DESI DR2 BAO: LEAVE-ONE-OUT CROSS-VALIDATION E-VALUE ANALYSIS")
    print("=" * 70)
    print()
    print(dataset.summary())
    print()

    # =========================================================================
    # LOOCV e-value
    # =========================================================================
    print("=" * 70)
    print("LOOCV E-VALUE (leave-one-redshift-bin-out)")
    print("=" * 70)
    print()

    loocv_result = loocv_evalue(
        data=dataset.data,
        cov=dataset.cov,
        z_values=dataset.z_eff,
        quantities=dataset.quantities,
        verbose=True
    )

    print()
    print("-" * 70)
    print("LOOCV SUMMARY")
    print("-" * 70)
    print(f"  Overall E-value (product):  {loocv_result.e_value:.4f}")
    print(f"  log(E):                     {loocv_result.log_e:.4f}")
    if loocv_result.e_value > 1:
        print(f"  Sigma equivalent:           {loocv_result.sigma_equivalent:.2f}sigma")
    else:
        print(f"  Sigma equivalent:           <1 (E <= 1, no evidence against LCDM)")
    print(f"  Sum delta-chi2:             {loocv_result.chi2_null_total - loocv_result.chi2_alt_total:.4f}")
    print()

    # Per-bin breakdown
    print("  Per-bin breakdown:")
    print(f"  {'z':>8}  {'E_k':>10}  {'log(E_k)':>10}  {'w0_fit':>8}  {'wa_fit':>8}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")
    for z_key in sorted(loocv_result.per_bin_e.keys()):
        ek = loocv_result.per_bin_e[z_key]
        lek = loocv_result.per_bin_log_e[z_key]
        w0 = loocv_result.per_bin_w0[z_key]
        wa = loocv_result.per_bin_wa[z_key]
        print(f"  {z_key:8.3f}  {ek:10.4f}  {lek:+10.4f}  {w0:8.3f}  {wa:8.3f}")

    # =========================================================================
    # Compare with z=1 split for reference
    # =========================================================================
    print()
    print("=" * 70)
    print("COMPARISON: STANDARD z=1 SPLIT E-VALUE")
    print("=" * 70)
    print()

    try:
        e_train, e_test = split_evalue(
            data=dataset.data,
            cov=dataset.cov,
            z_values=dataset.z_eff,
            quantities=dataset.quantities,
            split_z=1.0
        )
        print(f"  Split at z=1.0:")
        print(f"    Training (z<1): {np.sum(dataset.z_eff < 1.0)} points")
        print(f"    Testing  (z>=1): {np.sum(dataset.z_eff >= 1.0)} points")
        print(f"    E_test = {e_test.e_value:.4f}")
        print(f"    log(E_test) = {e_test.log_e:.4f}")
        if e_test.e_value > 1:
            print(f"    sigma equivalent = {e_test.sigma_equivalent:.2f}sigma")
        print(f"    Trained alt: w0={e_test.alt_params.w0:.3f}, wa={e_test.alt_params.wa:.3f}")
    except Exception as exc:
        print(f"  Split e-value failed: {exc}")

    # =========================================================================
    # Full-data uniform mixture for reference
    # =========================================================================
    print()
    print("=" * 70)
    print("COMPARISON: UNIFORM MIXTURE E-VALUE (full data, no split)")
    print("=" * 70)
    print()

    e_grow = grow_evalue(
        data=dataset.data,
        cov=dataset.cov,
        z_values=dataset.z_eff,
        quantities=dataset.quantities,
    )
    print(f"  Uniform mixture E-value = {e_grow.e_value:.4f}")
    print(f"  log(E) = {e_grow.log_e:.4f}")
    if e_grow.e_value > 1:
        print(f"  sigma equivalent = {e_grow.sigma_equivalent:.2f}sigma")
    print(f"  Best-fit: w0={e_grow.alt_params.w0:.3f}, wa={e_grow.alt_params.wa:.3f}")

    # =========================================================================
    # Interpretation
    # =========================================================================
    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()
    print("The LOOCV e-value avoids the distribution shift problem of the z=1")
    print("split by including all redshift ranges in training for each fold.")
    print("The block-diagonal covariance guarantees that the product of per-bin")
    print("likelihood ratios is a valid e-value (E[product] <= 1 under H0).")
    print()
    if loocv_result.e_value > 1:
        print(f"LOOCV E = {loocv_result.e_value:.4f} provides evidence against LCDM.")
    else:
        print(f"LOOCV E = {loocv_result.e_value:.4f} does NOT provide evidence against LCDM.")
    print()

    # Identify which bins contribute most
    sorted_bins = sorted(loocv_result.per_bin_log_e.items(), key=lambda x: x[1], reverse=True)
    print("Bins contributing most evidence against LCDM:")
    for z_key, lek in sorted_bins:
        direction = "against LCDM" if lek > 0 else "for LCDM"
        print(f"  z={z_key:.3f}: log(E_k) = {lek:+.4f}  ({direction})")


if __name__ == "__main__":
    main()
