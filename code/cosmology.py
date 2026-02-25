"""
Cosmological distance calculations for BAO analysis.

This module implements the standard cosmological distance measures:
- Comoving distance DM(z)
- Hubble distance DH(z) = c/H(z)
- Volume-averaged distance DV(z)
- Sound horizon rd

References:
- Hogg (1999): Distance measures in cosmology, astro-ph/9905116
- DESI Collaboration (2025): arXiv:2503.14738
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from dataclasses import dataclass
from typing import Optional, Tuple

# Physical constants
C_LIGHT_KM_S = 299792.458  # km/s


@dataclass
class CosmologyParams:
    """Cosmological parameters for distance calculations."""
    h: float = 0.6766  # Hubble parameter H0 = 100*h km/s/Mpc
    omega_m: float = 0.3111  # Matter density today
    omega_de: float = None  # Dark energy density (derived from flatness if None)
    omega_r: float = 9.0e-5  # Radiation density (approximate)
    w0: float = -1.0  # Dark energy equation of state at z=0
    wa: float = 0.0  # Dark energy EoS evolution parameter
    rd: float = 147.05  # Sound horizon in Mpc (DESI fiducial)

    def __post_init__(self):
        if self.omega_de is None:
            self.omega_de = 1.0 - self.omega_m - self.omega_r

    @property
    def omega_k(self) -> float:
        """Curvature density."""
        return 1.0 - self.omega_m - self.omega_de - self.omega_r

    def w(self, a: float) -> float:
        """Dark energy equation of state w(a) = w0 + wa*(1-a)."""
        return self.w0 + self.wa * (1.0 - a)

    def copy(self, **kwargs) -> 'CosmologyParams':
        """Return a copy with updated parameters. omega_de re-derived from flatness unless explicitly set."""
        params = {
            'h': self.h, 'omega_m': self.omega_m,
            'omega_r': self.omega_r, 'w0': self.w0, 'wa': self.wa, 'rd': self.rd
        }
        params.update(kwargs)
        # Only pass omega_de if explicitly requested; otherwise let __post_init__ derive it
        if 'omega_de' not in kwargs:
            params['omega_de'] = None
        return CosmologyParams(**params)


# Standard cosmologies (omega_de derived from flatness constraint)
LCDM = CosmologyParams(w0=-1.0, wa=0.0)
DESI_DR1_BEST_FIT = CosmologyParams(w0=-0.836, wa=-0.807)
DESI_DR2_BEST_FIT = CosmologyParams(w0=-0.75, wa=-1.05)


def E_z(z: float, cosmo: CosmologyParams) -> float:
    """
    Dimensionless Hubble parameter E(z) = H(z)/H0.

    For w0waCDM: includes time-varying dark energy equation of state.
    """
    a = 1.0 / (1.0 + z)

    # Dark energy contribution with CPL parametrization
    # rho_DE(a) = rho_DE(a=1) * a^(-3(1+w0+wa)) * exp(-3*wa*(1-a))
    de_exponent = -3.0 * (1.0 + cosmo.w0 + cosmo.wa) * np.log(a) - 3.0 * cosmo.wa * (1.0 - a)
    de_term = cosmo.omega_de * np.exp(de_exponent)

    matter_term = cosmo.omega_m * (1.0 + z)**3
    radiation_term = cosmo.omega_r * (1.0 + z)**4
    curvature_term = cosmo.omega_k * (1.0 + z)**2

    return np.sqrt(matter_term + radiation_term + curvature_term + de_term)


def H_z(z: float, cosmo: CosmologyParams) -> float:
    """Hubble parameter H(z) in km/s/Mpc."""
    H0 = 100.0 * cosmo.h
    return H0 * E_z(z, cosmo)


def DH(z: float, cosmo: CosmologyParams) -> float:
    """
    Hubble distance DH(z) = c/H(z) in Mpc.

    This is the instantaneous comoving distance scale at redshift z.
    """
    return C_LIGHT_KM_S / H_z(z, cosmo)


def DC(z: float, cosmo: CosmologyParams) -> float:
    """
    Line-of-sight comoving distance DC(z) in Mpc.

    DC(z) = c * integral_0^z dz'/H(z')
    """
    H0 = 100.0 * cosmo.h

    def integrand(zp):
        return 1.0 / E_z(zp, cosmo)

    result, _ = quad(integrand, 0, z)
    return C_LIGHT_KM_S / H0 * result


def DM(z: float, cosmo: CosmologyParams) -> float:
    """
    Transverse comoving distance DM(z) in Mpc.

    For flat universe: DM = DC
    For curved universe: involves sinh or sin correction.
    """
    dc = DC(z, cosmo)

    if abs(cosmo.omega_k) < 1e-6:
        return dc

    DH0 = C_LIGHT_KM_S / (100.0 * cosmo.h)  # c/H0
    sqrt_ok = np.sqrt(abs(cosmo.omega_k))

    if cosmo.omega_k > 0:  # Open universe
        return DH0 / sqrt_ok * np.sinh(sqrt_ok * dc / DH0)
    else:  # Closed universe
        return DH0 / sqrt_ok * np.sin(sqrt_ok * dc / DH0)


def DV(z: float, cosmo: CosmologyParams) -> float:
    """
    Volume-averaged distance DV(z) in Mpc.

    DV(z) = [z * DH(z) * DM(z)^2]^(1/3)

    This is the spherically averaged distance, appropriate for
    angle-averaged BAO measurements.
    """
    dh = DH(z, cosmo)
    dm = DM(z, cosmo)
    return (z * dh * dm**2)**(1.0/3.0)


def compute_bao_predictions(z_values: np.ndarray, cosmo: CosmologyParams) -> dict:
    """
    Compute BAO distance predictions at given redshifts.

    Returns dict with:
    - DM_over_rd: transverse comoving distance / sound horizon
    - DH_over_rd: Hubble distance / sound horizon
    - DV_over_rd: volume-averaged distance / sound horizon
    """
    rd = cosmo.rd

    DM_over_rd = np.array([DM(z, cosmo) / rd for z in z_values])
    DH_over_rd = np.array([DH(z, cosmo) / rd for z in z_values])
    DV_over_rd = np.array([DV(z, cosmo) / rd for z in z_values])

    return {
        'z': z_values,
        'DM_over_rd': DM_over_rd,
        'DH_over_rd': DH_over_rd,
        'DV_over_rd': DV_over_rd
    }


def chi_squared(data: np.ndarray, theory: np.ndarray, cov: np.ndarray) -> float:
    """
    Compute chi-squared statistic.

    chi^2 = (d - t)^T C^-1 (d - t)
    """
    residual = data - theory
    cov_inv = np.linalg.inv(cov)
    return float(residual @ cov_inv @ residual)


def log_likelihood(data: np.ndarray, theory: np.ndarray, cov: np.ndarray) -> float:
    """
    Compute log-likelihood assuming Gaussian errors.

    log L = -0.5 * chi^2 - 0.5 * log|C| - n/2 * log(2*pi)
    """
    n = len(data)
    chi2 = chi_squared(data, theory, cov)
    sign, logdet = np.linalg.slogdet(cov)
    return -0.5 * chi2 - 0.5 * logdet - 0.5 * n * np.log(2 * np.pi)


if __name__ == "__main__":
    # Quick test
    print("Testing cosmology module...")

    z_test = np.array([0.3, 0.5, 0.7, 1.0, 1.5, 2.0])

    print("\nLCDM predictions (DM/rd, DH/rd, DV/rd):")
    pred = compute_bao_predictions(z_test, LCDM)
    for i, z in enumerate(z_test):
        print(f"  z={z:.1f}: {pred['DM_over_rd'][i]:.3f}, {pred['DH_over_rd'][i]:.3f}, {pred['DV_over_rd'][i]:.3f}")

    print("\nw0waCDM (DESI DR2 best-fit) predictions:")
    pred_de = compute_bao_predictions(z_test, DESI_DR2_BEST_FIT)
    for i, z in enumerate(z_test):
        diff_dm = (pred_de['DM_over_rd'][i] - pred['DM_over_rd'][i]) / pred['DM_over_rd'][i] * 100
        diff_dh = (pred_de['DH_over_rd'][i] - pred['DH_over_rd'][i]) / pred['DH_over_rd'][i] * 100
        print(f"  z={z:.1f}: DM diff={diff_dm:+.2f}%, DH diff={diff_dh:+.2f}%")
