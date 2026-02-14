#!/usr/bin/env python3
"""
Generate figures for the DESI E-Value paper
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.figsize'] = (8, 5)

def figure1_evalue_comparison():
    """Bar chart comparing e-values across methods"""

    fig, ax = plt.subplots(figsize=(10, 6))

    methods = ['Simple LR\n(BIASED)', 'GROW\n(narrow)', 'GROW\n(default)',
               'GROW\n(wide)', 'Data-Split\n(VALID)']
    evalues = [392, 97, 15, 17, 1.4]
    colors = ['#dc3545', '#ffc107', '#ffc107', '#ffc107', '#28a745']

    bars = ax.bar(methods, evalues, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, val in zip(bars, evalues):
        height = bar.get_height()
        ax.annotate(f'{val:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('E-Value', fontsize=14)
    ax.set_title('E-Value Comparison: Evidence for Dynamic Dark Energy', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.set_ylim(0.5, 1000)

    # Add horizontal lines for significance levels
    ax.axhline(y=20, color='gray', linestyle='--', alpha=0.5, label='E=20 (α=0.05)')
    ax.axhline(y=100, color='gray', linestyle=':', alpha=0.5, label='E=100 (~3σ)')

    # Add legend for colors
    red_patch = mpatches.Patch(color='#dc3545', label='Invalid (overfitted)')
    yellow_patch = mpatches.Patch(color='#ffc107', label='Valid (prior-dependent)')
    green_patch = mpatches.Patch(color='#28a745', label='Valid (tests generalization)')
    ax.legend(handles=[red_patch, yellow_patch, green_patch], loc='upper right')

    # Add annotation
    ax.annotate('280× reduction', xy=(2, 50), xytext=(3.5, 200),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=11, ha='center')

    plt.tight_layout()
    plt.savefig('figure1_evalue_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure1_evalue_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved figure1_evalue_comparison.pdf/png")

def figure2_data_comparison():
    """Compare DR1 and DR2 measurements"""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Data
    z_dm = [0.51, 0.706, 0.93, 1.317, 2.33]
    dm_dr1 = [13.62, 16.85, 21.71, 27.79, 39.71]
    dm_dr2 = [13.59, 17.35, 21.58, 27.60, 38.99]
    dm_err_dr1 = [0.252, 0.319, 0.282, 0.690, 0.943]
    dm_err_dr2 = [0.168, 0.180, 0.162, 0.325, 0.518]

    z_dh = [0.51, 0.706, 0.93, 1.317, 2.33]
    dh_dr1 = [20.98, 20.08, 17.88, 13.82, 8.52]
    dh_dr2 = [21.86, 19.46, 17.64, 14.18, 8.63]
    dh_err_dr1 = [0.611, 0.595, 0.346, 0.422, 0.171]
    dh_err_dr2 = [0.429, 0.334, 0.201, 0.225, 0.101]

    # Plot DM/rd
    ax = axes[0]
    ax.errorbar(z_dm, dm_dr1, yerr=dm_err_dr1, fmt='o', color='#1f77b4',
                label='DR1', capsize=3, markersize=8)
    ax.errorbar([z+0.02 for z in z_dm], dm_dr2, yerr=dm_err_dr2, fmt='s', color='#d62728',
                label='DR2', capsize=3, markersize=8)
    ax.set_xlabel('Redshift $z$', fontsize=12)
    ax.set_ylabel('$D_M / r_d$', fontsize=12)
    ax.set_title('Transverse Comoving Distance', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot DH/rd
    ax = axes[1]
    ax.errorbar(z_dh, dh_dr1, yerr=dh_err_dr1, fmt='o', color='#1f77b4',
                label='DR1', capsize=3, markersize=8)
    ax.errorbar([z+0.02 for z in z_dh], dh_dr2, yerr=dh_err_dr2, fmt='s', color='#d62728',
                label='DR2', capsize=3, markersize=8)
    ax.set_xlabel('Redshift $z$', fontsize=12)
    ax.set_ylabel('$D_H / r_d$', fontsize=12)
    ax.set_title('Hubble Distance', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('DESI BAO Measurements: DR1 vs DR2', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('figure2_data_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure2_data_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved figure2_data_comparison.pdf/png")

def figure3_parameter_space():
    """Show w0-wa parameter space with fits"""

    fig, ax = plt.subplots(figsize=(8, 7))

    # LCDM point
    ax.plot(-1, 0, 'k*', markersize=20, label='$\\Lambda$CDM', zorder=5)

    # DR1 fit
    ax.plot(-0.805, -0.660, 'bo', markersize=12, label='DR1 best-fit')
    # DR1 error ellipse (approximate)
    ellipse1 = mpatches.Ellipse((-0.805, -0.660), 0.4, 0.8, angle=30,
                                 fill=False, color='blue', linestyle='--', linewidth=2)
    ax.add_patch(ellipse1)

    # DR2 fit
    ax.plot(-0.856, -0.430, 'rs', markersize=12, label='DR2 best-fit')
    # DR2 error ellipse (approximate, smaller due to more data)
    ellipse2 = mpatches.Ellipse((-0.856, -0.430), 0.3, 0.6, angle=30,
                                 fill=False, color='red', linestyle='-', linewidth=2)
    ax.add_patch(ellipse2)

    # Arrow showing shift
    ax.annotate('', xy=(-0.856, -0.430), xytext=(-0.805, -0.660),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(-0.75, -0.55, 'Parameter\nshift', fontsize=10, color='green')

    # Shade regions
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=-1, color='gray', linestyle='-', alpha=0.3)

    # Labels for quadrants
    ax.text(-0.5, 0.8, 'Phantom\n($w_0 > -1$, $w_a > 0$)', fontsize=9, ha='center', alpha=0.6)
    ax.text(-1.5, 0.8, 'Freezing\n($w_0 < -1$, $w_a > 0$)', fontsize=9, ha='center', alpha=0.6)
    ax.text(-0.5, -1.5, 'Thawing\n($w_0 > -1$, $w_a < 0$)', fontsize=9, ha='center', alpha=0.6)
    ax.text(-1.5, -1.5, 'Phantom+Thawing', fontsize=9, ha='center', alpha=0.6)

    ax.set_xlabel('$w_0$', fontsize=14)
    ax.set_ylabel('$w_a$', fontsize=14)
    ax.set_xlim(-2, 0)
    ax.set_ylim(-2, 1.5)
    ax.set_title('Dark Energy Parameter Space', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figure3_parameter_space.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure3_parameter_space.png', dpi=300, bbox_inches='tight')
    print("Saved figure3_parameter_space.pdf/png")

def figure4_split_validation():
    """Illustrate the data-split validation concept"""

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: Full data fit
    ax = axes[0]
    z_all = [0.3, 0.5, 0.7, 0.9, 1.3, 1.5, 2.3]
    residuals_full = [0.5, -0.8, 1.2, -0.3, 0.9, -0.4, 0.2]  # Example residuals
    ax.bar(z_all, residuals_full, color='#1f77b4', width=0.15, edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='-')
    ax.set_xlabel('Redshift $z$')
    ax.set_ylabel('Residual')
    ax.set_title('Full Fit: $\\chi^2 = 13.5$\n(all data used for fitting)')
    ax.set_ylim(-2, 2)

    # Panel 2: Training set
    ax = axes[1]
    z_train = [0.3, 0.5, 0.7, 0.9]
    z_test = [1.3, 1.5, 2.3]
    residuals_train = [0.3, -0.5, 0.8, -0.2]
    residuals_test = [1.8, -1.5, 1.2]  # Larger residuals on test set

    ax.bar(z_train, residuals_train, color='#2ca02c', width=0.15, edgecolor='black', label='Training')
    ax.bar(z_test, [0]*len(z_test), color='lightgray', width=0.15, edgecolor='black',
           alpha=0.5, label='Test (held out)')
    ax.axhline(y=0, color='black', linestyle='-')
    ax.set_xlabel('Redshift $z$')
    ax.set_ylabel('Residual')
    ax.set_title('Training: Fit on low-$z$ bins\n(test bins hidden)')
    ax.legend()
    ax.set_ylim(-2, 2)

    # Panel 3: Test set evaluation
    ax = axes[2]
    ax.bar(z_train, [0]*len(z_train), color='lightgray', width=0.15, edgecolor='black', alpha=0.5)
    ax.bar(z_test, residuals_test, color='#d62728', width=0.15, edgecolor='black', label='Test')
    ax.axhline(y=0, color='black', linestyle='-')
    ax.set_xlabel('Redshift $z$')
    ax.set_ylabel('Residual')
    ax.set_title('Test: Predict high-$z$ bins\n($E_{\\mathrm{split}} = 1.4$)')
    ax.legend()
    ax.set_ylim(-2, 2)

    # Add annotation
    ax.annotate('Larger residuals\n= poor generalization', xy=(1.5, -1.5), xytext=(2.0, -0.5),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=9, ha='center')

    plt.suptitle('Data-Split Validation: Testing Generalization', fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig('figure4_split_validation.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure4_split_validation.png', dpi=300, bbox_inches='tight')
    print("Saved figure4_split_validation.pdf/png")

def figure5_summary():
    """Summary figure showing key result"""

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create comparison visualization
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Left box - biased
    rect1 = mpatches.FancyBboxPatch((0.5, 2), 3.5, 6, boxstyle="round,pad=0.1",
                                      facecolor='#f8d7da', edgecolor='#dc3545', linewidth=3)
    ax.add_patch(rect1)
    ax.text(2.25, 7, 'Simple Likelihood\nRatio', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(2.25, 5, 'E = 392', ha='center', va='center', fontsize=28, fontweight='bold', color='#dc3545')
    ax.text(2.25, 3.5, '~3.9σ', ha='center', va='center', fontsize=14)
    ax.text(2.25, 2.5, 'BIASED', ha='center', va='center', fontsize=11, fontweight='bold', color='#dc3545')

    # Arrow
    ax.annotate('', xy=(5.8, 5), xytext=(4.2, 5),
                arrowprops=dict(arrowstyle='->', color='black', lw=3))
    ax.text(5, 5.8, '280×\nreduction', ha='center', va='center', fontsize=11)

    # Right box - valid
    rect2 = mpatches.FancyBboxPatch((6, 2), 3.5, 6, boxstyle="round,pad=0.1",
                                      facecolor='#d4edda', edgecolor='#28a745', linewidth=3)
    ax.add_patch(rect2)
    ax.text(7.75, 7, 'Data-Split\nValidation', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(7.75, 5, 'E = 1.4', ha='center', va='center', fontsize=28, fontweight='bold', color='#28a745')
    ax.text(7.75, 3.5, '~0.8σ', ha='center', va='center', fontsize=14)
    ax.text(7.75, 2.5, 'VALID', ha='center', va='center', fontsize=11, fontweight='bold', color='#28a745')

    ax.set_title('Key Result: Evidence Evaporates Under Proper Validation',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('figure5_summary.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure5_summary.png', dpi=300, bbox_inches='tight')
    print("Saved figure5_summary.pdf/png")

def figure6_cross_dataset():
    """Cross-dataset e-value matrix showing asymmetries"""

    fig, ax = plt.subplots(figsize=(9, 7))

    # Cross-dataset e-values (log10 scale for visualization)
    datasets = ['DESI', 'Pantheon+', 'Union3', 'DES-Y5']
    # E-values: rows = training, cols = test
    # Using log10(E) for visualization, capped at ±3
    evalues = [
        [np.nan, np.log10(1.5), np.log10(6.3), np.log10(86)],      # DESI →
        [np.log10(2049), np.nan, np.nan, np.log10(18)],            # Pantheon+ →
        [np.log10(1304), np.nan, np.nan, np.nan],                  # Union3 →
        [np.log10(0.19), np.log10(0.001), np.nan, np.nan],         # DES-Y5 →
    ]

    # Create matrix
    evalues_arr = np.array(evalues)

    # Custom colormap: red for <0, white for ~0, green for >0
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3.5)

    im = ax.imshow(evalues_arr, cmap='RdYlGn', norm=norm, aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='log₁₀(E-value)')
    cbar.ax.set_ylabel('log₁₀(E-value)', fontsize=12)

    # Set ticks
    ax.set_xticks(range(len(datasets)))
    ax.set_yticks(range(len(datasets)))
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_yticklabels(datasets, fontsize=11)

    # Labels
    ax.set_xlabel('Test Dataset', fontsize=13)
    ax.set_ylabel('Training Dataset', fontsize=13)
    ax.set_title('Cross-Dataset E-Values: Testing Generalization', fontsize=14, fontweight='bold')

    # Add value annotations
    raw_evalues = [
        [None, 1.5, 6.3, 86],
        [2049, None, None, 18],
        [1304, None, None, None],
        [0.19, 0.00, None, None],
    ]
    for i in range(len(datasets)):
        for j in range(len(datasets)):
            if raw_evalues[i][j] is not None:
                val = raw_evalues[i][j]
                if val >= 1:
                    text = f'{val:.0f}' if val >= 10 else f'{val:.1f}'
                else:
                    text = f'{val:.2f}'
                color = 'white' if abs(evalues_arr[i,j]) > 1.5 else 'black'
                ax.text(j, i, text, ha='center', va='center', fontsize=11,
                       fontweight='bold', color=color)

    # Add annotations for key findings
    ax.annotate('DES-Y5 fails\nto predict\nDESI', xy=(0, 3), xytext=(-1.2, 3.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, ha='center', color='red', fontweight='bold')

    ax.annotate('Pantheon+\npredicts\nDESI well', xy=(0, 1), xytext=(-1.2, 0.5),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=10, ha='center', color='green', fontweight='bold')

    plt.tight_layout()
    plt.savefig('figure6_cross_dataset.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure6_cross_dataset.png', dpi=300, bbox_inches='tight')
    print("Saved figure6_cross_dataset.pdf/png")


if __name__ == "__main__":
    import os
    os.chdir('/Users/jinyoungkim/Desktop/Projects/desi-evalue-analysis/paper')

    print("Generating figures...")
    figure1_evalue_comparison()
    figure2_data_comparison()
    figure3_parameter_space()
    figure4_split_validation()
    figure5_summary()
    figure6_cross_dataset()
    print("\nAll figures generated!")
