"""
Savage-Dickey density ratio for LCDM vs w0waCDM using DESI BAO data.

The Savage-Dickey ratio for nested models gives:
    B_01 = p(w0=-1, wa=0 | D, M1) / p(w0=-1, wa=0 | M1)

where:
    - Numerator = posterior density at the null (LCDM) point
    - Denominator = prior density at the null point

We compute the posterior density using two approaches:
  (A) Gaussian approximation: approximate the posterior as a 2D Gaussian
      around the best-fit using the Hessian. Use its normalization but
      evaluate with the EXACT delta-chi2 at the null point.
  (B) Numerical integration: compute the evidence integral over a grid
      and evaluate the posterior density directly.

Also computes AIC and BIC for comparison.

References:
- Dickey (1971): "The Weighted Likelihood Ratio..."
- Verdinelli & Wasserman (1995): "Computing Bayes Factors..."
- Trotta (2008): "Bayes in the sky" - arXiv:0803.4089
"""

import numpy as np
from scipy.optimize import minimize
from pathlib import Path
import sys

# Add the code directory to path
sys.path.insert(0, str(Path(__file__).parent))

from cosmology import (
    CosmologyParams, LCDM, compute_bao_predictions,
    chi_squared
)
from data_loader import load_desi_data, get_theory_vector


def compute_chi2(w0, wa, dataset):
    """Compute chi-squared for given (w0, wa) against BAO data."""
    cosmo = CosmologyParams(w0=w0, wa=wa)
    pred = compute_bao_predictions(dataset.z_eff, cosmo)
    theory = get_theory_vector(dataset, pred)
    return chi_squared(dataset.data, theory, dataset.cov)


def find_best_fit(dataset, w0_init=-0.8, wa_init=-0.5):
    """Find best-fit (w0, wa) by minimizing chi-squared."""

    def objective(params):
        w0, wa = params
        return compute_chi2(w0, wa, dataset)

    result = minimize(
        objective,
        x0=[w0_init, wa_init],
        bounds=[(-2.5, 0.0), (-4.0, 3.0)],
        method='L-BFGS-B',
        options={'ftol': 1e-12, 'gtol': 1e-10}
    )

    # Refine with Nelder-Mead for robustness
    result2 = minimize(
        objective,
        x0=result.x,
        method='Nelder-Mead',
        options={'xatol': 1e-8, 'fatol': 1e-12, 'maxiter': 10000}
    )

    if result2.fun < result.fun:
        return result2.x[0], result2.x[1], result2.fun
    return result.x[0], result.x[1], result.fun


def compute_hessian(dataset, w0_bf, wa_bf, h_w0=1e-4, h_wa=1e-4):
    """
    Compute the Hessian of chi-squared at the best-fit point using
    finite differences.

    Returns the 2x2 Hessian matrix d^2(chi2)/d(theta_i)d(theta_j).
    """
    f00 = compute_chi2(w0_bf, wa_bf, dataset)

    # d^2 chi2 / d(w0)^2
    fp0 = compute_chi2(w0_bf + h_w0, wa_bf, dataset)
    fm0 = compute_chi2(w0_bf - h_w0, wa_bf, dataset)
    d2_dw0dw0 = (fp0 - 2 * f00 + fm0) / h_w0**2

    # d^2 chi2 / d(wa)^2
    f0p = compute_chi2(w0_bf, wa_bf + h_wa, dataset)
    f0m = compute_chi2(w0_bf, wa_bf - h_wa, dataset)
    d2_dwadwa = (f0p - 2 * f00 + f0m) / h_wa**2

    # d^2 chi2 / d(w0)d(wa)
    fpp = compute_chi2(w0_bf + h_w0, wa_bf + h_wa, dataset)
    fpm = compute_chi2(w0_bf + h_w0, wa_bf - h_wa, dataset)
    fmp = compute_chi2(w0_bf - h_w0, wa_bf + h_wa, dataset)
    fmm = compute_chi2(w0_bf - h_w0, wa_bf - h_wa, dataset)
    d2_dw0dwa = (fpp - fpm - fmp + fmm) / (4 * h_w0 * h_wa)

    hessian = np.array([
        [d2_dw0dw0, d2_dw0dwa],
        [d2_dw0dwa, d2_dwadwa]
    ])

    return hessian


def jeffreys_scale(ln_B):
    """Interpret ln(B_10) on the Jeffreys scale."""
    abs_ln = abs(ln_B)
    if abs_ln < 1.0:
        strength = "Not worth more than a bare mention"
    elif abs_ln < 2.5:
        strength = "Substantial"
    elif abs_ln < 5.0:
        strength = "Strong"
    else:
        strength = "Decisive"

    if ln_B > 0:
        direction = "against LCDM (favors w0waCDM)"
    else:
        direction = "for LCDM (against w0waCDM)"

    return f"{strength} evidence {direction}"


def compute_numerical_evidence(dataset, w0_range, wa_range, chi2_min, n_grid=200):
    """
    Compute the marginal likelihood (evidence) by numerical integration
    over a uniform prior on (w0, wa).

    Z = integral exp(-chi2/2) dw0 dwa / V

    where V = (w0_max - w0_min) * (wa_max - wa_min) is the prior volume.

    Returns: log(Z), where Z = (1/V) * integral exp(-chi2/2) dw0 dwa
    """
    w0_lo, w0_hi = w0_range
    wa_lo, wa_hi = wa_range
    volume = (w0_hi - w0_lo) * (wa_hi - wa_lo)

    w0_arr = np.linspace(w0_lo, w0_hi, n_grid)
    wa_arr = np.linspace(wa_lo, wa_hi, n_grid)
    dw0 = w0_arr[1] - w0_arr[0]
    dwa = wa_arr[1] - wa_arr[0]

    # Compute chi2 on grid
    log_likes = np.zeros((n_grid, n_grid))
    for i, w0 in enumerate(w0_arr):
        for j, wa in enumerate(wa_arr):
            chi2_val = compute_chi2(w0, wa, dataset)
            log_likes[i, j] = -0.5 * (chi2_val - chi2_min)  # subtract chi2_min for numerical stability

    # Numerical integration using trapezoidal rule
    # integral exp(-chi2/2) dw0 dwa = exp(-chi2_min/2) * integral exp(log_likes) dw0 dwa
    max_ll = np.max(log_likes)
    log_integral = max_ll + np.log(np.sum(np.exp(log_likes - max_ll)) * dw0 * dwa)

    return log_integral, volume, w0_arr, wa_arr, log_likes


def main():
    # =========================================================================
    # Load DESI DR2 BAO data
    # =========================================================================
    data_dir = Path(__file__).parent.parent / 'data' / 'dr2'
    dataset = load_desi_data(data_dir, 'DR2')
    n_data = len(dataset.data)

    print("=" * 70)
    print("SAVAGE-DICKEY DENSITY RATIO: LCDM vs w0waCDM")
    print("Using DESI DR2 BAO data")
    print("=" * 70)
    print()
    print(dataset.summary())
    print()

    # =========================================================================
    # Compute chi-squared at LCDM point
    # =========================================================================
    chi2_lcdm = compute_chi2(-1.0, 0.0, dataset)
    print(f"Chi-squared at LCDM (w0=-1, wa=0): {chi2_lcdm:.4f}")
    print(f"  chi2/dof = {chi2_lcdm:.4f}/{n_data} = {chi2_lcdm/n_data:.4f}")
    print()

    # =========================================================================
    # Find best-fit w0waCDM
    # =========================================================================
    print("Finding best-fit w0waCDM...")

    # Try multiple starting points to ensure global minimum
    best_fits = []
    for w0_init in [-0.5, -0.8, -1.0, -1.2]:
        for wa_init in [-1.5, -0.5, 0.0, 0.5]:
            try:
                w0_bf, wa_bf, chi2_bf = find_best_fit(
                    dataset, w0_init=w0_init, wa_init=wa_init
                )
                best_fits.append((w0_bf, wa_bf, chi2_bf))
            except Exception:
                pass

    # Select the best
    best_fits.sort(key=lambda x: x[2])
    w0_bf, wa_bf, chi2_bf = best_fits[0]

    delta_chi2 = chi2_lcdm - chi2_bf
    print(f"Best-fit: w0 = {w0_bf:.4f}, wa = {wa_bf:.4f}")
    print(f"Chi-squared at best-fit: {chi2_bf:.4f}")
    print(f"  chi2/dof = {chi2_bf:.4f}/{n_data - 2} = {chi2_bf/(n_data-2):.4f}")
    print(f"Delta chi-squared (LCDM - best-fit): {delta_chi2:.4f}")
    print(f"  Naive sigma = sqrt(Delta chi2) = {np.sqrt(max(delta_chi2, 0)):.2f}")
    print()

    # =========================================================================
    # Compute Hessian and posterior covariance
    # =========================================================================
    print("Computing Hessian of chi-squared at best-fit...")

    hessian = compute_hessian(dataset, w0_bf, wa_bf, h_w0=1e-4, h_wa=1e-4)

    # The posterior covariance is H^{-1} (since chi2 ~ chi2_bf + dtheta^T H dtheta,
    # and the likelihood is exp(-chi2/2), the Gaussian has precision H and cov = H^{-1})
    cov_posterior = np.linalg.inv(hessian)

    sigma_w0 = np.sqrt(cov_posterior[0, 0])
    sigma_wa = np.sqrt(cov_posterior[1, 1])
    rho = cov_posterior[0, 1] / (sigma_w0 * sigma_wa)

    print(f"Hessian of chi2:")
    print(f"  [[{hessian[0,0]:.4f}, {hessian[0,1]:.4f}],")
    print(f"   [{hessian[1,0]:.4f}, {hessian[1,1]:.4f}]]")
    print(f"Posterior (Gaussian approx, using precision = Hessian of chi2):")
    print(f"  sigma(w0) = {sigma_w0:.4f}")
    print(f"  sigma(wa) = {sigma_wa:.4f}")
    print(f"  rho(w0, wa) = {rho:.4f}")
    print()

    # Check Gaussian approximation quality
    dtheta_null = np.array([-1.0 - w0_bf, 0.0 - wa_bf])
    delta_chi2_gauss = float(dtheta_null @ hessian @ dtheta_null)
    print(f"Gaussian approximation quality check at LCDM point:")
    print(f"  Exact Delta chi2 = {delta_chi2:.4f}")
    print(f"  Gaussian Delta chi2 (dtheta^T H dtheta) = {delta_chi2_gauss:.4f}")
    print(f"  Ratio = {delta_chi2_gauss/delta_chi2:.2f}")
    if abs(delta_chi2_gauss - delta_chi2) / delta_chi2 > 0.3:
        print(f"  WARNING: Gaussian approximation is poor at the null point.")
        print(f"           This is expected with strong w0-wa degeneracy (rho={rho:.3f}).")
        print(f"           Using EXACT chi2 for posterior evaluation (Method A).")
        print(f"           Also computing numerical integration (Method B).")
    print()

    # =========================================================================
    # METHOD A: Gaussian-normalized Savage-Dickey with exact delta-chi2
    # =========================================================================
    # The properly normalized Gaussian posterior is:
    #   p(theta|D) = sqrt(det(H)) / (2*pi) * exp(-dtheta^T H dtheta / 2)
    #
    # But when the Gaussian is inaccurate at the null point, we can do better:
    # use the Gaussian normalization but the EXACT likelihood at the null point.
    #
    # p(theta_0 | D) = L(theta_0) / integral L(theta) dtheta
    #                = exp(-chi2_null/2) / integral exp(-chi2/2) dtheta
    #
    # With Gaussian approximation for the integral only:
    #   integral exp(-chi2/2) dtheta ~ exp(-chi2_bf/2) * (2*pi) / sqrt(det(H))
    #
    # So: p(theta_0 | D) = sqrt(det(H)) / (2*pi) * exp(-(chi2_null - chi2_bf)/2)
    #                     = sqrt(det(H)) / (2*pi) * exp(-delta_chi2/2)

    det_H = np.linalg.det(hessian)
    if det_H <= 0:
        print("ERROR: Hessian is not positive definite!")
        return

    # Method A: use exact delta_chi2 with Gaussian normalization
    posterior_at_null_A = np.sqrt(det_H) / (2 * np.pi) * np.exp(-0.5 * delta_chi2)

    print("=" * 70)
    print("SAVAGE-DICKEY DENSITY RATIO")
    print("=" * 70)
    print()
    print("B_01 = p(w0=-1, wa=0 | D, M1) / p(w0=-1, wa=0 | M1)")
    print("B_10 = 1 / B_01  (evidence for w0waCDM over LCDM)")
    print()

    # =========================================================================
    # METHOD B: Numerical integration for each prior range
    # =========================================================================
    prior_configs = {
        'narrow': {'w0_range': (-1.2, -0.8), 'wa_range': (-1.0, 0.5)},
        'default': {'w0_range': (-1.5, -0.5), 'wa_range': (-2.0, 1.0)},
        'wide': {'w0_range': (-2.0, 0.0), 'wa_range': (-3.0, 2.0)},
    }

    print("METHOD A: Gaussian normalization + exact Delta chi2")
    print(f"  Posterior at null = sqrt(det(H))/(2*pi) * exp(-Delta_chi2/2)")
    print(f"  det(H) = {det_H:.4f}")
    print(f"  Delta chi2 = {delta_chi2:.4f}")
    print(f"  p(w0=-1, wa=0 | D) = {posterior_at_null_A:.6e}")
    print()

    print("-" * 100)
    print(f"{'Prior':>10} | {'Volume':>8} | {'pi(null)':>12} | {'B_01':>12} | {'B_10':>12} | {'ln(B_10)':>10} | Interpretation")
    print("-" * 100)

    results_A = {}
    for name, config in prior_configs.items():
        w0_lo, w0_hi = config['w0_range']
        wa_lo, wa_hi = config['wa_range']
        volume = (w0_hi - w0_lo) * (wa_hi - wa_lo)
        prior_density = 1.0 / volume

        B_01 = posterior_at_null_A / prior_density
        B_10 = 1.0 / B_01
        ln_B10 = np.log(B_10)
        interp = jeffreys_scale(ln_B10)

        results_A[name] = {
            'volume': volume,
            'prior_density': prior_density,
            'B_01': B_01,
            'B_10': B_10,
            'ln_B10': ln_B10,
        }

        print(f"{name:>10} | {volume:8.2f} | {prior_density:12.6f} | {B_01:12.6f} | {B_10:12.4f} | {ln_B10:10.4f} | {interp}")

    print()

    # =========================================================================
    # METHOD B: Full numerical integration
    # =========================================================================
    print("METHOD B: Full numerical integration over prior volume")
    print("  (More accurate when posterior is non-Gaussian)")
    print()
    print("  Computing chi2 grid for each prior range (n=200)...")

    print("-" * 100)
    print(f"{'Prior':>10} | {'Volume':>8} | {'pi(null)':>12} | {'B_01':>12} | {'B_10':>12} | {'ln(B_10)':>10} | Interpretation")
    print("-" * 100)

    results_B = {}
    for name, config in prior_configs.items():
        w0_range = config['w0_range']
        wa_range = config['wa_range']
        volume = (w0_range[1] - w0_range[0]) * (wa_range[1] - wa_range[0])
        prior_density = 1.0 / volume

        # Compute evidence by numerical integration
        log_integral, vol, w0_arr, wa_arr, log_likes = compute_numerical_evidence(
            dataset, w0_range, wa_range, chi2_bf, n_grid=200
        )

        # The evidence is: Z = (1/V) * exp(-chi2_min/2) * integral
        # But for Savage-Dickey we need p(theta_0 | D):
        #   p(theta_0 | D) = exp(-chi2_null/2) / [integral exp(-chi2/2) dtheta over prior]
        #
        # The integral in the denominator is computed numerically:
        #   integral exp(-chi2/2) dtheta = exp(-chi2_min/2) * exp(log_integral)
        #
        # So: p(theta_0 | D) = exp(-chi2_null/2) / [exp(-chi2_min/2) * exp(log_integral)]
        #                     = exp(-(chi2_null - chi2_min)/2) / exp(log_integral)
        #                     = exp(-delta_chi2/2 - log_integral)
        log_posterior_at_null = -0.5 * delta_chi2 - log_integral
        posterior_at_null_B = np.exp(log_posterior_at_null)

        B_01 = posterior_at_null_B / prior_density
        B_10 = 1.0 / B_01
        ln_B10 = np.log(B_10)
        interp = jeffreys_scale(ln_B10)

        results_B[name] = {
            'volume': volume,
            'prior_density': prior_density,
            'B_01': B_01,
            'B_10': B_10,
            'ln_B10': ln_B10,
            'posterior_at_null': posterior_at_null_B,
        }

        print(f"{name:>10} | {volume:8.2f} | {prior_density:12.6f} | {B_01:12.6f} | {B_10:12.4f} | {ln_B10:10.4f} | {interp}")

    print()
    print("Prior ranges used:")
    for name, config in prior_configs.items():
        print(f"  {name}: w0 in {config['w0_range']}, wa in {config['wa_range']}")
    print()

    # =========================================================================
    # Information criteria: AIC and BIC
    # =========================================================================
    print("=" * 70)
    print("INFORMATION CRITERIA")
    print("=" * 70)
    print()

    k = 2  # extra parameters in w0waCDM vs LCDM
    n = n_data  # number of data points

    # AIC: lower is better. DAIC = AIC(LCDM) - AIC(w0waCDM)
    # AIC(model) = chi2_model + 2*k_model
    # DAIC = chi2_LCDM - (chi2_best + 2k) = Dchi2 - 2k
    delta_aic = delta_chi2 - 2 * k

    # BIC: lower is better. DBIC = BIC(LCDM) - BIC(w0waCDM)
    # BIC(model) = chi2_model + k_model * ln(n)
    # DBIC = chi2_LCDM - (chi2_best + k*ln(n)) = Dchi2 - k*ln(n)
    delta_bic = delta_chi2 - k * np.log(n)

    # AICc (corrected AIC for small samples)
    # AICc penalty = 2k + 2k(k+1)/(n-k-1)
    aicc_penalty = 2 * k + 2 * k * (k + 1) / (n - k - 1)
    delta_aicc = delta_chi2 - aicc_penalty

    print(f"Number of extra parameters (k): {k}")
    print(f"Number of data points (n): {n}")
    print(f"Delta chi-squared: {delta_chi2:.4f}")
    print()
    print(f"DAIC  = Dchi2 - 2k       = {delta_chi2:.4f} - {2*k} = {delta_aic:.4f}")
    print(f"DAICc = Dchi2 - AICc_pen  = {delta_chi2:.4f} - {aicc_penalty:.4f} = {delta_aicc:.4f}")
    print(f"DBIC  = Dchi2 - k*ln(n)   = {delta_chi2:.4f} - {k}*{np.log(n):.4f} = {delta_bic:.4f}")
    print()

    # Interpretation
    print("Interpretation (positive = favors w0waCDM over LCDM):")
    print(f"  DAIC  = {delta_aic:+.4f}", end="")
    if delta_aic > 0:
        if delta_aic > 10:
            print("  --> Strong preference for w0waCDM")
        elif delta_aic > 4:
            print("  --> Considerable preference for w0waCDM")
        elif delta_aic > 2:
            print("  --> Positive preference for w0waCDM")
        else:
            print("  --> Weak preference for w0waCDM")
    else:
        print("  --> Prefers LCDM (simpler model)")
    print()

    print(f"  DAICc = {delta_aicc:+.4f}", end="")
    if delta_aicc > 0:
        if delta_aicc > 10:
            print("  --> Strong preference for w0waCDM")
        elif delta_aicc > 4:
            print("  --> Considerable preference for w0waCDM")
        elif delta_aicc > 2:
            print("  --> Positive preference for w0waCDM")
        else:
            print("  --> Weak preference for w0waCDM")
    else:
        print("  --> Prefers LCDM (simpler model)")
    print()

    print(f"  DBIC  = {delta_bic:+.4f}", end="")
    if delta_bic > 0:
        if delta_bic > 10:
            print("  --> Very strong preference for w0waCDM (Kass-Raftery)")
        elif delta_bic > 6:
            print("  --> Strong preference for w0waCDM (Kass-Raftery)")
        elif delta_bic > 2:
            print("  --> Positive preference for w0waCDM (Kass-Raftery)")
        else:
            print("  --> Weak preference for w0waCDM")
    else:
        print("  --> Prefers LCDM (simpler model)")
    print()

    # =========================================================================
    # Summary table
    # =========================================================================
    print("=" * 70)
    print("SUMMARY: MODEL COMPARISON RESULTS")
    print("=" * 70)
    print()
    print(f"Data: DESI DR2 BAO ({n_data} measurements)")
    print(f"Null model (M0): LCDM (w0=-1, wa=0)")
    print(f"Alternative (M1): w0waCDM")
    print(f"Best-fit M1: w0={w0_bf:.4f}, wa={wa_bf:.4f}")
    print(f"Delta chi2 = {delta_chi2:.4f}")
    print()

    print("Frequentist:")
    from scipy.stats import chi2 as chi2_dist
    from scipy.special import erfinv
    p_value = 1 - chi2_dist.cdf(delta_chi2, df=2)
    if p_value > 0 and p_value < 1:
        sigma_equiv = np.sqrt(2) * erfinv(1 - p_value)
    else:
        sigma_equiv = float('inf')
    print(f"  p-value (chi2, 2 dof) = {p_value:.6f}")
    print(f"  Equivalent sigma = {sigma_equiv:.2f}")
    print()

    print("Information Criteria (positive = favors w0waCDM):")
    print(f"  DAIC  = {delta_aic:+.4f}")
    print(f"  DAICc = {delta_aicc:+.4f}")
    print(f"  DBIC  = {delta_bic:+.4f}")
    print()

    print("Bayesian -- Savage-Dickey (Method A: Gaussian normalization):")
    for name in ['narrow', 'default', 'wide']:
        r = results_A[name]
        print(f"  [{name:>8} prior] B_01 = {r['B_01']:.6f}, ln(B_10) = {r['ln_B10']:+.4f}  ({jeffreys_scale(r['ln_B10'])})")
    print()

    print("Bayesian -- Savage-Dickey (Method B: numerical integration):")
    for name in ['narrow', 'default', 'wide']:
        r = results_B[name]
        print(f"  [{name:>8} prior] B_01 = {r['B_01']:.6f}, ln(B_10) = {r['ln_B10']:+.4f}  ({jeffreys_scale(r['ln_B10'])})")
    print()

    print("Legend:")
    print("  B_01 > 1 (ln B_10 < 0): favors LCDM over w0waCDM")
    print("  B_01 < 1 (ln B_10 > 0): favors w0waCDM over LCDM")
    print()

    # =========================================================================
    # Comparison notes
    # =========================================================================
    print("=" * 70)
    print("NOTES AND CAVEATS")
    print("=" * 70)
    print()
    print("1. The Savage-Dickey ratio depends on the prior range for (w0, wa).")
    print("   Wider priors penalize the complex model more (Occam's razor),")
    print("   making it harder for w0waCDM to be favored.")
    print()
    print("2. Method A uses the Gaussian approximation for the normalization")
    print("   integral only, but the exact chi2 at the null point. Method B")
    print("   numerically integrates the full likelihood. Method B is more")
    print("   accurate when the posterior is non-Gaussian (as it is here due")
    print(f"   to the strong w0-wa degeneracy with rho = {rho:.3f}).")
    print()
    print("3. The BIC approximation to ln(B_10) is DBIC/2 =", f"{delta_bic/2:.2f}.")
    print("   This can be compared to the Savage-Dickey ln(B_10) values.")
    print()
    print("4. BAO-only constraints are weaker than combined analyses.")
    print("   DESI DR2 + CMB + SNe finds stronger evidence for w0waCDM.")
    print("   However, Bayesian analyses with broader datasets find")
    print("   mixed results due to inter-dataset tensions.")


if __name__ == "__main__":
    main()
