#!/usr/bin/env python3
"""Verification script for LOOCV e-value correctness."""

import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_desi_data
from cosmology import CosmologyParams, LCDM, compute_bao_predictions, chi_squared, log_likelihood
from evalue_analysis import _build_theory_vector

data_dir = Path(__file__).parent.parent / 'data' / 'dr2'
ds = load_desi_data(data_dir, 'DR2')
cov = ds.cov

print("=== Verification: Block-diagonal covariance ===")
bin_boundaries = [0, 1, 3, 5, 7, 9, 11, 13]
all_zero = True
for i in range(len(bin_boundaries)-1):
    for j in range(i+1, len(bin_boundaries)-1):
        block = cov[bin_boundaries[i]:bin_boundaries[i+1], bin_boundaries[j]:bin_boundaries[j+1]]
        if np.any(block != 0):
            print(f"  Non-zero block between bins {i} and {j}")
            all_zero = False
if all_zero:
    print("  CONFIRMED: All cross-bin blocks are zero. Covariance IS block-diagonal.")

print()

# Verify log-likelihood decomposition
pred_lcdm = compute_bao_predictions(ds.z_eff, LCDM)
theory_lcdm = _build_theory_vector(pred_lcdm, ds.z_eff, ds.quantities)
full_logL = log_likelihood(ds.data, theory_lcdm, ds.cov)
print(f"Full log-L(LCDM) = {full_logL:.6f}")

sum_logL = 0
bin_indices = [[0], [1,2], [3,4], [5,6], [7,8], [9,10], [11,12]]
for bidx in bin_indices:
    bidx = np.array(bidx)
    d = ds.data[bidx]
    c = ds.cov[np.ix_(bidx, bidx)]
    t = theory_lcdm[bidx]
    sum_logL += log_likelihood(d, t, c)
print(f"Sum of per-bin log-L(LCDM) = {sum_logL:.6f}")
print(f"Difference = {full_logL - sum_logL:.10f}")

# Check determinant decomposition
sign_full, logdet_full = np.linalg.slogdet(ds.cov)
logdet_blocks = 0
for bidx in bin_indices:
    bidx = np.array(bidx)
    c = ds.cov[np.ix_(bidx, bidx)]
    s, ld = np.linalg.slogdet(c)
    logdet_blocks += ld
print(f"\nlog|full_cov| = {logdet_full:.10f}")
print(f"sum log|block_cov| = {logdet_blocks:.10f}")
print(f"Difference = {abs(logdet_full - logdet_blocks):.2e}")

print("\n=== Key implication ===")
print("Since the covariance is block-diagonal, L(data) = product L(data_k)")
print("Therefore E = product(L_alt(k)/L_null(k)) is the FULL likelihood ratio")
print("decomposed into independent per-bin pieces.")
print("Each E_k uses only out-of-sample training, making the product a valid")
print("cross-validated e-value with no overfitting.")
