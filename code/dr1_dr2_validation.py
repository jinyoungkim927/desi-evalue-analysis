#!/usr/bin/env python3
"""
DR1 → DR2 Validation Analysis

This script performs the proper temporal validation:
1. Fit w0, wa using DR1 data only
2. Test predictions on DR2 data
3. Compute e-value for generalization

Note: DR2 contains DR1, so they're not fully independent.
But comparing fits and predictions is still informative.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad

# Cosmological parameters (Planck 2018)
H0 = 67.66  # km/s/Mpc
c = 299792.458  # km/s
Om = 0.3111
Or = 9e-5
rd = 147.05  # Mpc, sound horizon

def E_z(z, w0=-1, wa=0):
    """Dimensionless Hubble parameter E(z) = H(z)/H0"""
    Ode_z = (1 - Om - Or) * (1+z)**(3*(1+w0+wa)) * np.exp(-3*wa*z/(1+z))
    return np.sqrt(Om*(1+z)**3 + Or*(1+z)**4 + Ode_z)

def DM_over_rd(z, w0=-1, wa=0):
    """Transverse comoving distance / rd"""
    integrand = lambda zp: 1/E_z(zp, w0, wa)
    result, _ = quad(integrand, 0, z)
    return (c/H0) * result / rd

def DH_over_rd(z, w0=-1, wa=0):
    """Hubble distance / rd"""
    return (c/H0) / E_z(z, w0, wa) / rd

def DV_over_rd(z, w0=-1, wa=0):
    """Volume-averaged distance / rd"""
    dm = DM_over_rd(z, w0, wa)
    dh = DH_over_rd(z, w0, wa)
    return (z * dh * dm**2)**(1/3)

def load_data(filepath_mean, filepath_cov):
    """Load DESI BAO data"""
    data = []
    with open(filepath_mean, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            z = float(parts[0])
            val = float(parts[1])
            qty = parts[2]
            data.append({'z': z, 'value': val, 'type': qty})

    cov = np.loadtxt(filepath_cov)
    return data, cov

def compute_theory(data, w0, wa):
    """Compute theoretical predictions"""
    theory = []
    for d in data:
        z = d['z']
        if d['type'] == 'DV_over_rs':
            theory.append(DV_over_rd(z, w0, wa))
        elif d['type'] == 'DM_over_rs':
            theory.append(DM_over_rd(z, w0, wa))
        elif d['type'] == 'DH_over_rs':
            theory.append(DH_over_rd(z, w0, wa))
    return np.array(theory)

def chi_squared(params, data, cov):
    """Compute chi-squared"""
    w0, wa = params
    obs = np.array([d['value'] for d in data])
    theory = compute_theory(data, w0, wa)
    residuals = obs - theory
    cov_inv = np.linalg.inv(cov)
    return residuals @ cov_inv @ residuals

def fit_model(data, cov, w0_init=-1, wa_init=0, fix_lcdm=False):
    """Fit cosmological model to data"""
    if fix_lcdm:
        chi2 = chi_squared([-1, 0], data, cov)
        return -1, 0, chi2

    result = minimize(
        chi_squared,
        [w0_init, wa_init],
        args=(data, cov),
        method='Nelder-Mead'
    )
    return result.x[0], result.x[1], result.fun

def compute_log_likelihood(data, cov, w0, wa):
    """Compute log-likelihood"""
    chi2 = chi_squared([w0, wa], data, cov)
    n = len(data)
    sign, logdet = np.linalg.slogdet(cov)
    return -0.5 * (chi2 + logdet + n * np.log(2*np.pi))

def main():
    print("="*70)
    print("DR1 → DR2 TEMPORAL VALIDATION ANALYSIS")
    print("="*70)

    # Load data
    dr1_data, dr1_cov = load_data(
        '/Users/jinyoungkim/Desktop/Projects/desi-evalue-analysis/data/dr1/desi_2024_gaussian_bao_ALL_GCcomb_mean.txt',
        '/Users/jinyoungkim/Desktop/Projects/desi-evalue-analysis/data/dr1/desi_2024_gaussian_bao_ALL_GCcomb_cov.txt'
    )

    dr2_data, dr2_cov = load_data(
        '/Users/jinyoungkim/Desktop/Projects/desi-evalue-analysis/data/dr2/desi_gaussian_bao_ALL_GCcomb_mean.txt',
        '/Users/jinyoungkim/Desktop/Projects/desi-evalue-analysis/data/dr2/desi_gaussian_bao_ALL_GCcomb_cov.txt'
    )

    print(f"\nDR1: {len(dr1_data)} measurements")
    print(f"DR2: {len(dr2_data)} measurements")

    # =====================================================
    # PART 1: Compare DR1 vs DR2 measurements
    # =====================================================
    print("\n" + "="*70)
    print("PART 1: DATA COMPARISON (DR1 vs DR2)")
    print("="*70)

    print("\n{:<8} {:<12} {:<12} {:<12} {:<10}".format(
        "z", "DR1", "DR2", "Δ", "% change"))
    print("-"*55)

    # Match by redshift and type
    for d1 in dr1_data:
        for d2 in dr2_data:
            if abs(d1['z'] - d2['z']) < 0.01 and d1['type'] == d2['type']:
                delta = d2['value'] - d1['value']
                pct = 100 * delta / d1['value']
                print(f"{d1['z']:<8.3f} {d1['value']:<12.3f} {d2['value']:<12.3f} {delta:<+12.3f} {pct:<+10.1f}%")

    # =====================================================
    # PART 2: Fit models to each dataset
    # =====================================================
    print("\n" + "="*70)
    print("PART 2: MODEL FITS")
    print("="*70)

    # Fit ΛCDM to both
    _, _, chi2_lcdm_dr1 = fit_model(dr1_data, dr1_cov, fix_lcdm=True)
    _, _, chi2_lcdm_dr2 = fit_model(dr2_data, dr2_cov, fix_lcdm=True)

    # Fit w0waCDM to both
    w0_dr1, wa_dr1, chi2_w0wa_dr1 = fit_model(dr1_data, dr1_cov)
    w0_dr2, wa_dr2, chi2_w0wa_dr2 = fit_model(dr2_data, dr2_cov)

    print("\nΛCDM fits (w0=-1, wa=0 fixed):")
    print(f"  DR1: χ² = {chi2_lcdm_dr1:.2f} (dof = {len(dr1_data)})")
    print(f"  DR2: χ² = {chi2_lcdm_dr2:.2f} (dof = {len(dr2_data)})")

    print("\nw0waCDM fits (free parameters):")
    print(f"  DR1: w0 = {w0_dr1:.3f}, wa = {wa_dr1:.3f}, χ² = {chi2_w0wa_dr1:.2f}")
    print(f"  DR2: w0 = {w0_dr2:.3f}, wa = {wa_dr2:.3f}, χ² = {chi2_w0wa_dr2:.2f}")

    print("\nParameter shifts (DR1 → DR2):")
    print(f"  Δw0 = {w0_dr2 - w0_dr1:+.3f}")
    print(f"  Δwa = {wa_dr2 - wa_dr1:+.3f}")

    # =====================================================
    # PART 3: Cross-validation e-value
    # =====================================================
    print("\n" + "="*70)
    print("PART 3: DR1 → DR2 PREDICTION E-VALUE")
    print("="*70)

    # Key test: How well do DR1-fitted parameters predict DR2?
    # Compare to ΛCDM prediction of DR2

    # Chi-squared of DR1 fit evaluated on DR2 data
    chi2_dr1fit_on_dr2 = chi_squared([w0_dr1, wa_dr1], dr2_data, dr2_cov)

    print("\nPredicting DR2 data:")
    print(f"  ΛCDM prediction:           χ² = {chi2_lcdm_dr2:.2f}")
    print(f"  DR1 w0wa fit prediction:   χ² = {chi2_dr1fit_on_dr2:.2f}")
    print(f"  DR2 w0wa fit (best):       χ² = {chi2_w0wa_dr2:.2f}")

    # E-value: L(DR2 | DR1 fit) / L(DR2 | ΛCDM)
    delta_chi2_dr1_pred = chi2_lcdm_dr2 - chi2_dr1fit_on_dr2
    e_value_dr1_pred = np.exp(delta_chi2_dr1_pred / 2)

    print(f"\nE-value (DR1 fit predicting DR2 vs ΛCDM):")
    print(f"  Δχ² = {delta_chi2_dr1_pred:.2f}")
    print(f"  E = exp(Δχ²/2) = {e_value_dr1_pred:.2f}")

    # For comparison: the naive e-value from DR2 alone
    delta_chi2_naive = chi2_lcdm_dr2 - chi2_w0wa_dr2
    e_value_naive = np.exp(delta_chi2_naive / 2)

    print(f"\nComparison - Naive E-value (DR2 fit on DR2):")
    print(f"  Δχ² = {delta_chi2_naive:.2f}")
    print(f"  E = {e_value_naive:.2f} (BIASED - same data for fit and test)")

    # =====================================================
    # PART 4: Stability analysis
    # =====================================================
    print("\n" + "="*70)
    print("PART 4: STABILITY ANALYSIS")
    print("="*70)

    # How significant is the parameter shift?
    # Rough estimate using Fisher information
    print("\nParameter stability (DR1 → DR2):")
    print(f"  w0: {w0_dr1:.3f} → {w0_dr2:.3f} (shift: {w0_dr2-w0_dr1:+.3f})")
    print(f"  wa: {wa_dr1:.3f} → {wa_dr2:.3f} (shift: {wa_dr2-wa_dr1:+.3f})")

    # If signal is real, DR1 and DR2 fits should be consistent
    # Large shifts suggest overfitting
    shift_magnitude = np.sqrt((w0_dr2-w0_dr1)**2 + (wa_dr2-wa_dr1)**2)
    print(f"\n  Combined shift magnitude: {shift_magnitude:.3f}")

    if shift_magnitude > 0.5:
        print("  → Large shift suggests unstable/overfitted signal")
    elif shift_magnitude > 0.2:
        print("  → Moderate shift - some instability")
    else:
        print("  → Small shift - consistent signal")

    # =====================================================
    # SUMMARY
    # =====================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │  E-VALUE COMPARISON                                             │
    ├─────────────────────────────────────────────────────────────────┤
    │  Naive (DR2 fit on DR2):        E = {:.1f}  ← BIASED           │
    │  Temporal (DR1 fit on DR2):     E = {:.1f}  ← VALID            │
    ├─────────────────────────────────────────────────────────────────┤
    │  Ratio: {:.0f}x reduction when using proper validation         │
    └─────────────────────────────────────────────────────────────────┘
    """.format(e_value_naive, e_value_dr1_pred, e_value_naive/e_value_dr1_pred))

    if e_value_dr1_pred < 3:
        print("CONCLUSION: DR1-fitted model does NOT predict DR2 better than ΛCDM.")
        print("            The apparent evidence for dynamic dark energy is likely overfitting.")
    elif e_value_dr1_pred < 10:
        print("CONCLUSION: Weak evidence that DR1 fit generalizes to DR2.")
    else:
        print("CONCLUSION: Strong evidence - DR1 fit successfully predicts DR2.")

    print("\n" + "="*70)
    print("IMPORTANT CAVEAT")
    print("="*70)
    print("""
    WARNING: DR2 contains DR1! They are NOT independent datasets.

    - DR1: Year 1 observations (~6 million objects)
    - DR2: Years 1-3 observations (~14 million objects, includes DR1)

    The DR2 measurements at each redshift are correlated with DR1
    measurements because they share the same underlying galaxies.

    This "validation" tests STABILITY (do fits stay consistent with more data?)
    NOT true out-of-sample GENERALIZATION.

    Our redshift-split analysis (E = 1.4) is the proper test of whether
    the model generalizes to predict data at different cosmic epochs.
    """)

if __name__ == "__main__":
    main()
