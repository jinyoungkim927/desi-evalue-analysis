# Cosmological Background: BAO and Dark Energy

## The Expanding Universe

The universe is expanding, described by the **scale factor** $a(t)$:
- Today: $a_0 = 1$
- Past: $a < 1$
- Redshift: $z = 1/a - 1$

The expansion rate is the **Hubble parameter**:

$$H(t) = \frac{\dot{a}}{a}$$

Today's value: $H_0 \approx 67.7$ km/s/Mpc (from Planck)

## The Friedmann Equation

The expansion is governed by the **Friedmann equation**:

$$H^2(z) = H_0^2 \left[ \Omega_m (1+z)^3 + \Omega_r (1+z)^4 + \Omega_k (1+z)^2 + \Omega_{DE}(z) \right]$$

Where:
- $\Omega_m \approx 0.31$: Matter density (dark matter + baryons)
- $\Omega_r \approx 9 \times 10^{-5}$: Radiation density
- $\Omega_k$: Curvature (≈ 0 for flat universe)
- $\Omega_{DE}$: Dark energy density

## Dark Energy Equation of State

Dark energy is characterized by its **equation of state** $w$:

$$P = w \rho c^2$$

For the **cosmological constant** (ΛCDM): $w = -1$ (constant)

For **dynamic dark energy** (w₀wₐCDM):

$$w(a) = w_0 + w_a(1-a) = w_0 + w_a \frac{z}{1+z}$$

The dark energy density evolves as:

$$\Omega_{DE}(z) = \Omega_{DE,0} \cdot (1+z)^{3(1+w_0+w_a)} \cdot \exp\left(-3w_a \frac{z}{1+z}\right)$$

### Model Comparison

| Model | $w_0$ | $w_a$ | Description |
|-------|-------|-------|-------------|
| ΛCDM | -1 | 0 | Cosmological constant |
| wCDM | free | 0 | Constant but not -1 |
| w₀wₐCDM | free | free | Time-evolving |

## Cosmological Distances

### Comoving Distance

The line-of-sight comoving distance to redshift $z$:

$$D_C(z) = c \int_0^z \frac{dz'}{H(z')} = \frac{c}{H_0} \int_0^z \frac{dz'}{E(z')}$$

where the dimensionless Hubble parameter is:

$$E(z) = H(z)/H_0 = \sqrt{\Omega_m(1+z)^3 + \Omega_r(1+z)^4 + \Omega_{DE}(z)}$$

### Transverse Comoving Distance (DM)

For a flat universe: $D_M = D_C$

For curved universe:
$$D_M = \begin{cases}
\frac{c}{H_0\sqrt{\Omega_k}} \sinh\left(\sqrt{\Omega_k} \frac{H_0 D_C}{c}\right) & \Omega_k > 0 \\
\frac{c}{H_0\sqrt{|\Omega_k|}} \sin\left(\sqrt{|\Omega_k|} \frac{H_0 D_C}{c}\right) & \Omega_k < 0
\end{cases}$$

### Hubble Distance (DH)

The instantaneous expansion scale:

$$D_H(z) = \frac{c}{H(z)} = \frac{c}{H_0 E(z)}$$

### Volume-Averaged Distance (DV)

Spherically averaged distance for isotropic measurements:

$$D_V(z) = \left[ z \cdot D_H(z) \cdot D_M(z)^2 \right]^{1/3}$$

## Baryon Acoustic Oscillations (BAO)

### Physical Origin

In the early universe (before recombination, $z \sim 1100$):
1. Photons, baryons, and electrons form a hot plasma
2. Dark matter clumps gravitationally
3. Pressure waves propagate through the plasma
4. At recombination, waves freeze at the **sound horizon**

$$r_d = \int_{z_{rec}}^\infty \frac{c_s(z)}{H(z)} dz \approx 147 \text{ Mpc}$$

where $c_s \approx c/\sqrt{3}$ is the sound speed.

### BAO as a Standard Ruler

The sound horizon $r_d$ is imprinted on:
- Galaxy clustering (correlation function peak at $r_d$)
- CMB angular power spectrum

Since $r_d$ is known from physics, measuring the **apparent size** of this feature gives distances:

$$\frac{D_M(z)}{r_d}, \quad \frac{D_H(z)}{r_d}, \quad \frac{D_V(z)}{r_d}$$

### What DESI Measures

DESI measures galaxy positions and redshifts. From the 2-point correlation function:

- **Transverse direction**: $D_M(z)/r_d$ from angular separation
- **Radial direction**: $D_H(z)/r_d$ from redshift separation
- **Isotropic**: $D_V(z)/r_d$ from angle-averaged signal

## Why BAO Constrains Dark Energy

Different dark energy models predict different $H(z)$:

$$H(z) = H_0 \sqrt{\Omega_m(1+z)^3 + \Omega_{DE}(z)}$$

This affects both $D_M(z)$ and $D_H(z)$:

$$D_M(z) = c \int_0^z \frac{dz'}{H(z')}$$
$$D_H(z) = \frac{c}{H(z)}$$

If dark energy evolves (w₀wₐCDM), the distances at different redshifts change relative to ΛCDM predictions.

### DESI's Leverage

DESI measures BAO at multiple redshifts:
- $z = 0.3$: BGS (Bright Galaxy Survey)
- $z = 0.5, 0.7$: LRG (Luminous Red Galaxies)
- $z = 0.9$: LRG+ELG combined
- $z = 1.3$: ELG (Emission Line Galaxies)
- $z = 1.5$: QSO (Quasars)
- $z = 2.3$: Lyα forest

This wide redshift range allows distinguishing models.

## DESI DR2 Fiducial Cosmology

DESI uses Planck 2018 bestfit as reference:

| Parameter | Value |
|-----------|-------|
| $h$ | 0.6766 |
| $\Omega_m$ | 0.3111 |
| $\Omega_b$ | 0.0490 |
| $\Omega_{DE}$ | 0.6889 |
| $r_d$ | 147.05 Mpc |
| $w$ | -1 (ΛCDM) |

## References

1. **Hogg (1999)**: "Distance measures in cosmology" - astro-ph/9905116
2. **Eisenstein et al. (2005)**: "Detection of the Baryon Acoustic Peak" - ApJ 633, 560
3. **DESI Collaboration (2024)**: "DESI 2024 VI: Cosmological Constraints" - arXiv:2404.03002
4. **Planck Collaboration (2020)**: "Planck 2018 results. VI. Cosmological parameters"
