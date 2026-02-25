#!/usr/bin/env python3
"""
Investigation: Non-flat universe inconsistency in cosmology.py

The default CosmologyParams has:
  omega_m  = 0.3111
  omega_de = 0.6889
  omega_r  = 9.0e-5

These sum to 1.00009, giving omega_k = -0.00009 (slightly closed).
The DM() threshold is abs(omega_k) < 1e-10, so the code takes the
curved-universe branch (sin) instead of the flat-universe branch (DM=DC).

This script quantifies the actual impact.
"""

import numpy as np
from scipy.integrate import quad

# ============================================================
# STEP 1: Verify omega_k
# ============================================================
print("=" * 70)
print("STEP 1: VERIFY OMEGA_K FROM DEFAULTS")
print("=" * 70)

omega_m = 0.3111
omega_de = 0.6889
omega_r = 9.0e-5

omega_total = omega_m + omega_de + omega_r
omega_k = 1.0 - omega_total

print(f"  omega_m  = {omega_m}")
print(f"  omega_de = {omega_de}")
print(f"  omega_r  = {omega_r}")
print(f"  Sum      = {omega_total:.10f}")
print(f"  omega_k  = 1 - sum = {omega_k:.10e}")
print(f"  |omega_k| = {abs(omega_k):.10e}")
print(f"  Threshold = 1e-10")
print(f"  |omega_k| > threshold? {abs(omega_k) > 1e-10}")
print(f"  omega_k < 0 => CLOSED universe => sin() branch")

# ============================================================
# STEP 2: Cosmological functions (reproducing cosmology.py)
# ============================================================
C_LIGHT = 299792.458  # km/s
H0 = 67.66  # km/s/Mpc = 100 * 0.6766
h = 0.6766
rd_cosmo_py = 147.09  # rd used in cosmology.py
rd_validation = 147.05  # rd used in dr1_dr2_validation.py

def E_z(z, om=omega_m, ode=omega_de, orad=omega_r, w0=-1.0, wa=0.0):
    """Dimensionless Hubble parameter E(z) = H(z)/H0."""
    ok = 1.0 - om - ode - orad
    a = 1.0 / (1.0 + z)
    de_exponent = -3.0 * (1.0 + w0 + wa) * np.log(a) - 3.0 * wa * (1.0 - a)
    de_term = ode * np.exp(de_exponent)
    matter_term = om * (1.0 + z)**3
    radiation_term = orad * (1.0 + z)**4
    curvature_term = ok * (1.0 + z)**2
    return np.sqrt(matter_term + radiation_term + curvature_term + de_term)

def DC(z, w0=-1.0, wa=0.0):
    """Line-of-sight comoving distance DC(z) in Mpc."""
    integrand = lambda zp: 1.0 / E_z(zp, w0=w0, wa=wa)
    result, _ = quad(integrand, 0, z)
    return C_LIGHT / H0 * result

def DM_flat(z, w0=-1.0, wa=0.0):
    """DM assuming flat universe: DM = DC."""
    return DC(z, w0=w0, wa=wa)

def DM_curved(z, w0=-1.0, wa=0.0, ok=None):
    """DM with curvature correction (what cosmology.py actually does)."""
    if ok is None:
        ok = omega_k
    dc = DC(z, w0=w0, wa=wa)
    DH0 = C_LIGHT / H0  # c/H0
    sqrt_ok = np.sqrt(abs(ok))

    if ok > 0:  # Open
        return DH0 / sqrt_ok * np.sinh(sqrt_ok * dc / DH0)
    else:  # Closed (our case: omega_k = -9e-5)
        return DH0 / sqrt_ok * np.sin(sqrt_ok * dc / DH0)

# ============================================================
# STEP 3: Trace through DM() for each DESI DR2 redshift
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: TRACE DM() CODE PATH")
print("=" * 70)

desi_redshifts = [0.295, 0.51, 0.706, 0.934, 1.321, 1.484, 2.33]

print(f"\n  omega_k = {omega_k:.6e}")
print(f"  abs(omega_k) = {abs(omega_k):.6e}")
print(f"  abs(omega_k) < 1e-10? {abs(omega_k) < 1e-10}")
print(f"  => Code enters CURVED branch (omega_k < 0 => sin() branch)")

# ============================================================
# STEP 4: Compute DM both ways at all 7 redshifts
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: NUMERICAL COMPARISON AT ALL 7 DESI DR2 REDSHIFTS")
print("=" * 70)

print(f"\n  {'z':>8s}  {'DC (Mpc)':>14s}  {'DM_flat (Mpc)':>14s}  {'DM_curved (Mpc)':>16s}  {'Diff (Mpc)':>12s}  {'Frac diff':>14s}")
print("  " + "-" * 90)

dm_flat_values = []
dm_curved_values = []
dc_values = []

for z in desi_redshifts:
    dc = DC(z)
    dm_f = DM_flat(z)
    dm_c = DM_curved(z)
    diff = dm_c - dm_f
    frac = (dm_c - dm_f) / dm_f

    dc_values.append(dc)
    dm_flat_values.append(dm_f)
    dm_curved_values.append(dm_c)

    print(f"  {z:>8.3f}  {dc:>14.6f}  {dm_f:>14.6f}  {dm_c:>16.6f}  {diff:>+12.6e}  {frac:>+14.6e}")

# Verify DC == DM_flat (they should be identical)
print(f"\n  Sanity check: DC == DM_flat? {all(np.isclose(a,b) for a,b in zip(dc_values, dm_flat_values))}")

# ============================================================
# STEP 5: Fractional difference in DM/rd at each redshift
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: FRACTIONAL DIFFERENCE IN DM/rd")
print("=" * 70)

print(f"\n  Using rd = {rd_cosmo_py} Mpc (cosmology.py default)")
print(f"\n  {'z':>8s}  {'DM_flat/rd':>12s}  {'DM_curved/rd':>14s}  {'Frac diff':>14s}")
print("  " + "-" * 60)

for i, z in enumerate(desi_redshifts):
    dm_f_rd = dm_flat_values[i] / rd_cosmo_py
    dm_c_rd = dm_curved_values[i] / rd_cosmo_py
    frac = (dm_c_rd - dm_f_rd) / dm_f_rd
    print(f"  {z:>8.3f}  {dm_f_rd:>12.4f}  {dm_c_rd:>14.4f}  {frac:>+14.6e}")

# ============================================================
# STEP 6: Compare rd mismatch (147.05 vs 147.09)
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: rd MISMATCH (147.05 vs 147.09)")
print("=" * 70)

rd_diff_frac = (rd_cosmo_py - rd_validation) / rd_validation
print(f"\n  cosmology.py rd  = {rd_cosmo_py} Mpc")
print(f"  validation.py rd = {rd_validation} Mpc")
print(f"  Difference       = {rd_cosmo_py - rd_validation:.2f} Mpc")
print(f"  Fractional diff  = {rd_diff_frac:+.6e}")

print(f"\n  Since DM/rd ~ 1/rd, using different rd shifts the ratio by:")
print(f"  (DM/rd_1) / (DM/rd_2) - 1 = rd_2/rd_1 - 1 = {rd_validation/rd_cosmo_py - 1:+.6e}")

print(f"\n  Effect on DM/rd at each redshift:")
print(f"  {'z':>8s}  {'DM/rd (147.09)':>16s}  {'DM/rd (147.05)':>16s}  {'Frac diff':>14s}")
print("  " + "-" * 60)

for i, z in enumerate(desi_redshifts):
    dm_rd_109 = dm_flat_values[i] / rd_cosmo_py
    dm_rd_105 = dm_flat_values[i] / rd_validation
    frac = (dm_rd_105 - dm_rd_109) / dm_rd_109
    print(f"  {z:>8.3f}  {dm_rd_109:>16.4f}  {dm_rd_105:>16.4f}  {frac:>+14.6e}")

# ============================================================
# STEP 7: Mathematical analysis of the sin() correction
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: MATHEMATICAL ANALYSIS OF SIN() CORRECTION")
print("=" * 70)

DH0 = C_LIGHT / H0
sqrt_ok = np.sqrt(abs(omega_k))
print(f"\n  DH0 = c/H0 = {DH0:.4f} Mpc")
print(f"  sqrt(|omega_k|) = {sqrt_ok:.6e}")

print(f"\n  The sin correction: DM = DH0/sqrt_ok * sin(sqrt_ok * DC / DH0)")
print(f"  Let x = sqrt_ok * DC / DH0")
print(f"  Then DM = DH0/sqrt_ok * sin(x)")
print(f"  Taylor: sin(x) = x - x^3/6 + ...")
print(f"  So DM ~ DC * (1 - x^2/6 + ...)")
print(f"  The correction is ~ -x^2/6 = -(omega_k * DC^2 / DH0^2) / 6")

print(f"\n  {'z':>8s}  {'DC (Mpc)':>12s}  {'x=sqrt_ok*DC/DH0':>18s}  {'x^2/6':>14s}  {'Actual frac diff':>18s}")
print("  " + "-" * 80)

for i, z in enumerate(desi_redshifts):
    dc = dc_values[i]
    x = sqrt_ok * dc / DH0
    x2_over_6 = x**2 / 6.0
    actual_frac = (dm_curved_values[i] - dm_flat_values[i]) / dm_flat_values[i]
    print(f"  {z:>8.3f}  {dc:>12.2f}  {x:>18.6e}  {x2_over_6:>14.6e}  {actual_frac:>+18.6e}")

# ============================================================
# STEP 8: Impact assessment on e-value
# ============================================================
print("\n" + "=" * 70)
print("STEP 7: IMPACT ASSESSMENT ON E-VALUE (E=1.4)")
print("=" * 70)

# The e-value depends on chi-squared differences
# chi^2 = sum_i (obs_i - theory_i)^2 / sigma_i^2
# A fractional change delta in theory shifts chi^2 by approximately:
# Delta_chi2 ~ 2 * sum_i (obs_i - theory_i) * delta * theory_i / sigma_i^2

# For context: typical BAO measurement uncertainties
# DESI DR2 DM/rd uncertainties are roughly 0.5-2%
# The curvature effect is ~1e-8 to ~5e-8, which is 6-7 orders of magnitude smaller

print(f"\n  CURVATURE BUG (sin vs flat):")
max_frac_curv = max(abs((dm_curved_values[i] - dm_flat_values[i]) / dm_flat_values[i])
                    for i in range(len(desi_redshifts)))
print(f"    Maximum fractional difference in DM: {max_frac_curv:.2e}")
print(f"    Typical DESI DR2 DM/rd uncertainty:  ~1% = 1e-2")
print(f"    Ratio (effect / uncertainty):         {max_frac_curv / 0.01:.2e}")
print(f"    Effect on chi-squared:                ~ {max_frac_curv**2 / 0.01**2:.2e}")

print(f"\n  rd MISMATCH (147.05 vs 147.09):")
rd_frac = abs(rd_cosmo_py - rd_validation) / rd_cosmo_py
print(f"    Fractional difference in rd:          {rd_frac:.2e}")
print(f"    Typical DESI DR2 DM/rd uncertainty:   ~1% = 1e-2")
print(f"    Ratio (effect / uncertainty):          {rd_frac / 0.01:.2e}")

# More precise: compute the actual shift in DM/rd and compare to typical sigma
print(f"\n  QUANTITATIVE COMPARISON:")
print(f"    DM/rd shift from curvature bug:  {max_frac_curv:.2e} (fractional)")
print(f"    DM/rd shift from rd mismatch:    {rd_frac:.2e} (fractional)")
print(f"    Ratio (rd mismatch / curvature): {rd_frac / max_frac_curv:.0e}")

# E-value sensitivity
# E = exp(Delta_chi2 / 2)
# If E=1.4, then Delta_chi2 = 2*ln(1.4) = 0.673
delta_chi2_baseline = 2 * np.log(1.4)
print(f"\n  E=1.4 corresponds to Delta_chi2 = {delta_chi2_baseline:.4f}")
print(f"\n  To change E meaningfully (say by 0.1), need Delta(Delta_chi2) ~ {2*np.log(1.5) - 2*np.log(1.4):.4f}")

# Estimate chi2 perturbation from curvature bug
# delta(chi2) ~ N * (delta_DM/sigma)^2 where N ~ 7 redshifts, sigma ~ 1%
delta_chi2_curv = 7 * (max_frac_curv / 0.01)**2
print(f"\n  Estimated Delta_chi2 from curvature bug: ~{delta_chi2_curv:.2e}")
print(f"  This is {delta_chi2_curv / delta_chi2_baseline:.2e} of the Delta_chi2 for E=1.4")

delta_chi2_rd = 7 * (rd_frac / 0.01)**2
print(f"\n  Estimated Delta_chi2 from rd mismatch:   ~{delta_chi2_rd:.4f}")
print(f"  This is {delta_chi2_rd / delta_chi2_baseline:.4f} of the Delta_chi2 for E=1.4")

# Now check: does the rd mismatch actually matter?
# The key is: in dr1_dr2_validation.py, rd=147.05 is used for BOTH theory AND data comparison
# so it cancels out! It only matters if cosmology.py and dr1_dr2_validation.py are mixed.
print(f"\n  CRITICAL CONTEXT:")
print(f"  - dr1_dr2_validation.py uses rd=147.05 for ALL calculations (self-consistent)")
print(f"  - cosmology.py uses rd=147.09 for ALL calculations (self-consistent)")
print(f"  - The rd mismatch only matters if you MIX results from both scripts")
print(f"  - Within each script, the rd value cancels in ratios")

# ============================================================
# STEP 9: Also check validation.py's E_z vs cosmology.py's E_z
# ============================================================
print("\n" + "=" * 70)
print("STEP 8: VALIDATE E_z IMPLEMENTATIONS AGREE")
print("=" * 70)

# dr1_dr2_validation.py uses: Ode = (1 - Om - Or) * (1+z)^{3(1+w0+wa)} * exp(-3*wa*z/(1+z))
# cosmology.py uses the equivalent CPL parameterization
# Let's verify they're the same for LCDM

def E_z_validation(z, w0=-1.0, wa=0.0):
    """E(z) from dr1_dr2_validation.py (uses 1-Om-Or for dark energy, no curvature)."""
    Om = 0.3111
    Or = 9e-5
    Ode_z = (1 - Om - Or) * (1+z)**(3*(1+w0+wa)) * np.exp(-3*wa*z/(1+z))
    return np.sqrt(Om*(1+z)**3 + Or*(1+z)**4 + Ode_z)

print(f"\n  Note: validation.py uses Ode = (1-Om-Or) = {1-omega_m-omega_r}")
print(f"  cosmology.py uses omega_de = {omega_de}")
print(f"  Difference: {(1-omega_m-omega_r) - omega_de:.10e}")
print(f"  This means validation.py implicitly assumes FLAT universe (Ode = 1-Om-Or)")
print(f"  while cosmology.py has Ode=0.6889 separately, giving nonzero omega_k")

print(f"\n  {'z':>8s}  {'E(z) cosmology.py':>20s}  {'E(z) validation.py':>20s}  {'Frac diff':>14s}")
print("  " + "-" * 70)

for z in desi_redshifts:
    e_cosmo = E_z(z)
    e_valid = E_z_validation(z)
    frac = (e_cosmo - e_valid) / e_valid
    print(f"  {z:>8.3f}  {e_cosmo:>20.12f}  {e_valid:>20.12f}  {frac:>+14.6e}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"""
  1. CURVATURE BUG CONFIRMED:
     - omega_k = {omega_k:.6e} (should be 0 for intended flat universe)
     - |omega_k| = {abs(omega_k):.1e} >> 1e-10 threshold
     - Code takes sin() branch instead of flat DM=DC branch
     - Maximum fractional error in DM: {max_frac_curv:.2e}
     - This is ~{max_frac_curv/0.01:.0e} of typical measurement uncertainty
     - VERDICT: Completely negligible. Cannot affect e-value.

  2. rd MISMATCH:
     - cosmology.py: rd = {rd_cosmo_py} Mpc
     - dr1_dr2_validation.py: rd = {rd_validation} Mpc
     - Fractional difference: {rd_frac:.2e} = {rd_frac*100:.4f}%
     - VERDICT: Small but {rd_frac/max_frac_curv:.0f}x larger than curvature bug.
       However, each script is internally self-consistent.
       Only matters if mixing results between scripts.

  3. IMPACT ON E-VALUE (E=1.4):
     - E=1.4 requires Delta_chi2 = {delta_chi2_baseline:.3f}
     - Curvature bug contributes Delta_chi2 ~ {delta_chi2_curv:.2e} => ZERO impact
     - rd mismatch contributes Delta_chi2 ~ {delta_chi2_rd:.4f} => {delta_chi2_rd/delta_chi2_baseline*100:.2f}% of signal
     - Neither can change E=1.4 meaningfully.
     - The curvature bug is a code smell but not a numerical error of consequence.
""")
