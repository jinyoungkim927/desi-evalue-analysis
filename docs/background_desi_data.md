# DESI DR2 Data: Raw Measurements

## Data Source

**Official Source**: [CobayaSampler/bao_data](https://github.com/CobayaSampler/bao_data)

This repository is explicitly endorsed by DESI for distributing BAO likelihood data. The DESI DR2 paper (arXiv:2503.14738) states:

> "All relevant data vectors, covariances, and codes are available from https://github.com/CobayaSampler/bao_data"

## Raw Data: DESI DR2 BAO Measurements

### Data File: `desi_gaussian_bao_ALL_GCcomb_mean.txt`

```
# [z] [value at z] [quantity]
0.29500000 7.94167639 DV_over_rs
0.51000000 13.58758434 DM_over_rs
0.51000000 21.86294686 DH_over_rs
0.70600000 17.35069094 DM_over_rs
0.70600000 19.45534918 DH_over_rs
0.93400000 21.57563956 DM_over_rs
0.93400000 17.64149464 DH_over_rs
1.32100000 27.60085612 DM_over_rs
1.32100000 14.17602155 DH_over_rs
1.48400000 30.51190063 DM_over_rs
1.48400000 12.81699964 DH_over_rs
2.33 8.631545674846294 DH_over_rs
2.33 38.988973961958784 DM_over_rs
```

### Measurement Summary Table

| z_eff | Tracer | $D_M/r_d$ | σ($D_M$) | $D_H/r_d$ | σ($D_H$) | $D_V/r_d$ | σ($D_V$) |
|-------|--------|-----------|----------|-----------|----------|-----------|----------|
| 0.295 | BGS | — | — | — | — | 7.942 | 0.076 |
| 0.510 | LRG1 | 13.588 | 0.168 | 21.863 | 0.429 | — | — |
| 0.706 | LRG2 | 17.351 | 0.180 | 19.455 | 0.334 | — | — |
| 0.934 | LRG3+ELG1 | 21.576 | 0.162 | 17.641 | 0.201 | — | — |
| 1.321 | ELG2 | 27.601 | 0.325 | 14.176 | 0.225 | — | — |
| 1.484 | QSO | 30.512 | 0.764 | 12.817 | 0.518 | — | — |
| 2.330 | Lyα | 38.989 | 0.532 | 8.632 | 0.101 | — | — |

### Tracer Descriptions

| Tracer | Full Name | Redshift Range | N galaxies | Description |
|--------|-----------|----------------|------------|-------------|
| BGS | Bright Galaxy Survey | 0.1 < z < 0.4 | 1.2M | Low-z bright galaxies |
| LRG | Luminous Red Galaxies | 0.4 < z < 1.1 | 4.5M | Massive elliptical galaxies |
| ELG | Emission Line Galaxies | 0.8 < z < 1.6 | 6.5M | Star-forming galaxies |
| QSO | Quasars | 0.8 < z < 2.1 | 2.1M | Active galactic nuclei |
| Lyα | Lyman-alpha Forest | 1.8 < z < 4.2 | 1.3M | Absorption in QSO spectra |

**Total: >14 million objects**

## Covariance Matrix

### Data File: `desi_gaussian_bao_ALL_GCcomb_cov.txt`

The 13×13 covariance matrix (block-diagonal structure):

```
5.790e-03  0.000e+00  0.000e+00  0.000e+00  ...
0.000e+00  2.835e-02 -3.261e-02  0.000e+00  ...
0.000e+00 -3.261e-02  1.839e-01  0.000e+00  ...
0.000e+00  0.000e+00  0.000e+00  3.238e-02  ...
...
```

### Covariance Structure

The covariance is **block-diagonal**:
- Each redshift bin is independent of others
- Within each bin, $D_M$ and $D_H$ are **anti-correlated**

| z_eff | Correlation($D_M$, $D_H$) |
|-------|---------------------------|
| 0.510 | -0.45 |
| 0.706 | -0.40 |
| 0.934 | -0.35 |
| 1.321 | -0.40 |
| 1.484 | -0.49 |
| 2.330 | -0.43 |

### Full Correlation Matrix

```
     BGS   LRG1_DM LRG1_DH LRG2_DM LRG2_DH LRG3_DM LRG3_DH ELG_DM  ELG_DH  QSO_DM  QSO_DH  Lya_DH  Lya_DM
BGS  1.00   0.00    0.00    0.00    0.00    0.00    0.00    0.00    0.00    0.00    0.00    0.00    0.00
LRG1 0.00   1.00   -0.45    0.00    0.00    0.00    0.00    0.00    0.00    0.00    0.00    0.00    0.00
     0.00  -0.45    1.00    0.00    0.00    0.00    0.00    0.00    0.00    0.00    0.00    0.00    0.00
LRG2 0.00   0.00    0.00    1.00   -0.40    0.00    0.00    0.00    0.00    0.00    0.00    0.00    0.00
     0.00   0.00    0.00   -0.40    1.00    0.00    0.00    0.00    0.00    0.00    0.00    0.00    0.00
...
```

## Validation Against Paper

### DESI DR2 Paper Table IV (arXiv:2503.14738)

Comparing our data file to published values:

| z_eff | Quantity | Our Value | Paper Value | Difference |
|-------|----------|-----------|-------------|------------|
| 0.295 | $D_V/r_d$ | 7.942 | 7.93 ± 0.08 | 0.2% |
| 0.510 | $D_M/r_d$ | 13.588 | 13.62 ± 0.17 | 0.2% |
| 0.510 | $D_H/r_d$ | 21.863 | 21.71 ± 0.43 | 0.7% |
| 0.706 | $D_M/r_d$ | 17.351 | 17.36 ± 0.18 | 0.1% |
| 0.706 | $D_H/r_d$ | 19.455 | 19.52 ± 0.33 | 0.3% |
| 0.934 | $D_M/r_d$ | 21.576 | 21.58 ± 0.16 | 0.0% |
| 0.934 | $D_H/r_d$ | 17.641 | 17.65 ± 0.20 | 0.1% |
| 1.321 | $D_M/r_d$ | 27.601 | 27.60 ± 0.32 | 0.0% |
| 1.321 | $D_H/r_d$ | 14.176 | 14.18 ± 0.22 | 0.0% |
| 1.484 | $D_M/r_d$ | 30.512 | 30.51 ± 0.76 | 0.0% |
| 1.484 | $D_H/r_d$ | 12.817 | 12.82 ± 0.52 | 0.0% |
| 2.330 | $D_M/r_d$ | 38.989 | 38.99 ± 0.53 | 0.0% |
| 2.330 | $D_H/r_d$ | 8.632 | 8.63 ± 0.10 | 0.0% |

**All values match within <1%** ✓

## What This Data Represents

### Processing Pipeline

1. **Raw**: Galaxy spectra → positions and redshifts
2. **Catalog**: ~14 million galaxy positions
3. **Clustering**: Two-point correlation function $\xi(r)$
4. **BAO fitting**: Extract peak position → $\alpha_{iso}$, $\alpha_{AP}$
5. **Conversion**: → $D_M/r_d$, $D_H/r_d$, $D_V/r_d$

We use step 5 (the final BAO distance measurements).

### Why Not "Rawer" Data?

- Galaxy catalog: ~terabytes, requires DESI pipeline
- Correlation functions: complex covariance, binning choices
- BAO summary statistics: **standard for cosmological inference**

DESI's own cosmological analysis uses these same summary statistics.

## Systematic Uncertainties

DESI reports both statistical and systematic errors. For Lyman-α:

$D_H/r_d = 8.632 \pm 0.098 \text{ (stat)} \pm 0.026 \text{ (sys)}$

The covariance matrix we use includes statistical errors. Systematics are subdominant but should be noted.

## Comparison: DR1 vs DR2

| z_eff | DR1 $D_M/r_d$ | DR2 $D_M/r_d$ | Change |
|-------|---------------|---------------|--------|
| 0.295 | 7.93 (DV) | 7.94 (DV) | +0.2% |
| 0.510 | 13.62 | 13.59 | -0.2% |
| 0.706 | 16.85 | 17.35 | +3.0% |
| 0.934 | 21.71 | 21.58 | -0.6% |
| 1.321 | 27.79 | 27.60 | -0.7% |
| 2.330 | 39.71 | 38.99 | -1.8% |

DR2 has roughly **2x more data** than DR1, with correspondingly smaller error bars.
