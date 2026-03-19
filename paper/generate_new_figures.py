#!/usr/bin/env python3
"""
Generate new figures for the DESI E-Value paper:
  - Figure A: BAO data with error bars + LCDM and w0waCDM predictions
  - Figure B: Correlation matrix heatmap
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))
from cosmology import CosmologyParams, compute_bao_predictions

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 13

# ── Load DESI DR2 data ──────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'dr2')

# Parse mean file
z_eff, values, quantities = [], [], []
with open(os.path.join(DATA_DIR, 'desi_gaussian_bao_ALL_GCcomb_mean.txt')) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        z_eff.append(float(parts[0]))
        values.append(float(parts[1]))
        quantities.append(parts[2])

z_eff = np.array(z_eff)
values = np.array(values)
cov = np.loadtxt(os.path.join(DATA_DIR, 'desi_gaussian_bao_ALL_GCcomb_cov.txt'))
errors = np.sqrt(np.diag(cov))

# Separate by quantity type
dv_mask = np.array(['DV' in q for q in quantities])
dm_mask = np.array(['DM' in q for q in quantities])
dh_mask = np.array(['DH' in q for q in quantities])

# Tracer labels
tracer_labels = {
    0.295: 'BGS', 0.510: 'LRG1', 0.706: 'LRG2', 0.934: 'LRG3+ELG1',
    1.321: 'ELG2', 1.484: 'QSO', 2.330: 'Ly$\\alpha$'
}

# ── Model predictions ───────────────────────────────────────────────
LCDM = CosmologyParams(w0=-1.0, wa=0.0)
W0WA = CosmologyParams(w0=-0.75, wa=-1.05)

z_model = np.linspace(0.1, 2.6, 200)
pred_lcdm = compute_bao_predictions(z_model, LCDM)
pred_w0wa = compute_bao_predictions(z_model, W0WA)


def figure_bao_data():
    """BAO data with error bars and model predictions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── DV/rd panel ─────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(z_model, pred_lcdm['DV_over_rd'], 'k-', lw=1.8, label='$\\Lambda$CDM')
    ax.plot(z_model, pred_w0wa['DV_over_rd'], 'r--', lw=1.8,
            label='$w_0w_a$CDM ($-0.75, -1.05$)')
    idx = np.where(dv_mask)[0]
    ax.errorbar(z_eff[idx], values[idx], yerr=errors[idx],
                fmt='s', color='#1f77b4', capsize=4, markersize=7, lw=1.5,
                label='DESI DR2', zorder=5)
    for i in idx:
        zr = round(z_eff[i], 3)
        lbl = tracer_labels.get(zr, '')
        ax.annotate(lbl, (z_eff[i], values[i]),
                    textcoords='offset points', xytext=(8, 6), fontsize=8, color='gray')
    ax.set_xlabel('Redshift $z$')
    ax.set_ylabel('$D_V / r_d$')
    ax.set_title('Volume-averaged distance')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.25)

    # ── DM/rd panel ─────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(z_model, pred_lcdm['DM_over_rd'], 'k-', lw=1.8, label='$\\Lambda$CDM')
    ax.plot(z_model, pred_w0wa['DM_over_rd'], 'r--', lw=1.8,
            label='$w_0w_a$CDM')
    idx = np.where(dm_mask)[0]
    ax.errorbar(z_eff[idx], values[idx], yerr=errors[idx],
                fmt='o', color='#2ca02c', capsize=4, markersize=7, lw=1.5,
                label='DESI DR2', zorder=5)
    for i in idx:
        zr = round(z_eff[i], 3)
        lbl = tracer_labels.get(zr, '')
        ax.annotate(lbl, (z_eff[i], values[i]),
                    textcoords='offset points', xytext=(8, 6), fontsize=8, color='gray')
    ax.set_xlabel('Redshift $z$')
    ax.set_ylabel('$D_M / r_d$')
    ax.set_title('Transverse comoving distance')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.25)

    # ── DH/rd panel ─────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(z_model, pred_lcdm['DH_over_rd'], 'k-', lw=1.8, label='$\\Lambda$CDM')
    ax.plot(z_model, pred_w0wa['DH_over_rd'], 'r--', lw=1.8,
            label='$w_0w_a$CDM')
    idx = np.where(dh_mask)[0]
    ax.errorbar(z_eff[idx], values[idx], yerr=errors[idx],
                fmt='^', color='#d62728', capsize=4, markersize=7, lw=1.5,
                label='DESI DR2', zorder=5)
    for i in idx:
        zr = round(z_eff[i], 3)
        lbl = tracer_labels.get(zr, '')
        ax.annotate(lbl, (z_eff[i], values[i]),
                    textcoords='offset points', xytext=(8, 6), fontsize=8, color='gray')
    ax.set_xlabel('Redshift $z$')
    ax.set_ylabel('$D_H / r_d$')
    ax.set_title('Hubble distance')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.25)

    plt.suptitle('DESI DR2 BAO Measurements with Model Predictions',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('figure_bao_data.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure_bao_data.png', dpi=300, bbox_inches='tight')
    print("Saved figure_bao_data.pdf/png")


def figure_correlation_matrix():
    """Correlation matrix heatmap."""
    # Compute correlation from covariance
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)

    # Build labels
    labels = []
    for z, q in zip(z_eff, quantities):
        zr = round(z, 3)
        tracer = tracer_labels.get(zr, f'z={zr}')
        short_q = q.replace('_over_rs', '').replace('_over_rd', '')
        labels.append(f'{tracer}\n{short_q}')

    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')

    n = len(labels)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
    ax.set_yticklabels(labels, fontsize=8)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = corr[i, j]
            color = 'white' if abs(val) > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color=color)

    # Draw block-diagonal outlines
    block_starts = [0]
    for i in range(1, n):
        if round(z_eff[i], 3) != round(z_eff[i-1], 3):
            block_starts.append(i)
    block_starts.append(n)

    for k in range(len(block_starts) - 1):
        s, e = block_starts[k], block_starts[k+1]
        rect = plt.Rectangle((s - 0.5, s - 0.5), e - s, e - s,
                              fill=False, edgecolor='black', lw=2)
        ax.add_patch(rect)

    cbar = plt.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label('Correlation coefficient', fontsize=12)

    ax.set_title('DESI DR2 BAO Correlation Matrix\n(block-diagonal by redshift bin)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figure_correlation.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure_correlation.png', dpi=300, bbox_inches='tight')
    print("Saved figure_correlation.pdf/png")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    figure_bao_data()
    figure_correlation_matrix()
    print("\nDone!")
