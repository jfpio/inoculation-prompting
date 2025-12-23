"""
Lightweight plot-only script for gradient decomposition analysis.
No heavy ML dependencies - just numpy, matplotlib, seaborn.
Use this with .venv_cpu on the plgrid (CPU) partition.
"""
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy import stats
import seaborn as sns
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"


def set_scientific_axis(ax, which='x'):
    """Set scientific notation for axis."""
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    if which == 'x' or which == 'both':
        ax.xaxis.set_major_formatter(formatter)
    if which == 'y' or which == 'both':
        ax.yaxis.set_major_formatter(formatter)


def compute_ci(values, n_boot=1000, seed=42):
    """Compute 95% CI via bootstrap."""
    rng = np.random.default_rng(seed)
    resampled = rng.choice(values, size=(n_boot, len(values)), replace=True)
    means = np.mean(resampled, axis=1)
    ci_low = np.percentile(means, 2.5)
    ci_high = np.percentile(means, 97.5)
    return np.mean(values), ci_low, ci_high


def main():
    parser = argparse.ArgumentParser(description="Regenerate decomposition plot from existing JSON")
    parser.add_argument("--layer", type=int, default=20, help="Layer to plot")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Gradient Decomposition Plot (Layer {args.layer})")
    print("=" * 60)

    output_dir = RESULTS_DIR / "insecure_code" / "decomposition"
    json_path = output_dir / f"decomposition_L{args.layer}.json"

    if not json_path.exists():
        print(f"Error: {json_path} not found.")
        print("Run the full 13_decompose_components.py first to generate the data.")
        return 1

    print(f"Loading metrics from {json_path}")
    with open(json_path) as f:
        metrics = json.load(f)

    # Plotting
    plot_path = output_dir / f"decomposition_L{args.layer}.png"

    # Create 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Helper for stats text
    def get_stats_text(data):
        mean, lo, hi = compute_ci(data)
        return (f"Mean: {mean:.2e}\n"
                f"95% CI: [{lo:.2e}, {hi:.2e}]")

    # 1. Delta Norm
    deltas_norm = metrics["norm"]["delta"]
    sns.histplot(deltas_norm, ax=axes[0], kde=True, color="blue")
    axes[0].set_title(r"A. Change in Gradient Norm ($\Delta \|\nabla\mathcal{L}\|$)", fontsize=11)
    axes[0].set_xlabel(r"$\Delta \|\nabla\mathcal{L}\|$ (Inoc - Neut)")
    axes[0].axvline(0, color='black', linestyle='--')
    axes[0].text(0.05, 0.95, get_stats_text(deltas_norm),
                 transform=axes[0].transAxes, verticalalignment='top', fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8))
    set_scientific_axis(axes[0], 'x')

    # 2. Delta Dot Product (Component along v)
    deltas_dot = metrics["dot"]["delta"]
    sns.histplot(deltas_dot, ax=axes[1], kde=True, color="green")
    axes[1].set_title(r"B. Change in Component along $v$ ($\Delta (\nabla\mathcal{L} \cdot v)$)", fontsize=11)
    axes[1].set_xlabel(r"$\Delta (\nabla\mathcal{L} \cdot v_{\rm insecure})$")
    axes[1].axvline(0, color='black', linestyle='--')
    axes[1].text(0.05, 0.95, get_stats_text(deltas_dot),
                 transform=axes[1].transAxes, verticalalignment='top', fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8))
    set_scientific_axis(axes[1], 'x')

    # 3. Delta Cosine
    deltas_cos = metrics["cos"]["delta"]
    sns.histplot(deltas_cos, ax=axes[2], kde=True, color="red")
    axes[2].set_title(r"C. Change in Alignment ($\Delta \cos$)", fontsize=11)
    axes[2].set_xlabel(r"$\Delta \cos(\nabla\mathcal{L}, v_{\rm insecure})$")
    axes[2].axvline(0, color='black', linestyle='--')
    axes[2].text(0.05, 0.95, get_stats_text(deltas_cos),
                 transform=axes[2].transAxes, verticalalignment='top', fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8))
    set_scientific_axis(axes[2], 'x')

    # 4. Scatter: Delta Cos vs Delta Norm with regression line
    deltas_norm_arr = np.array(deltas_norm)
    deltas_cos_arr = np.array(deltas_cos)
    
    axes[3].scatter(deltas_norm_arr, deltas_cos_arr, alpha=0.6, color="purple", edgecolor='k', s=40)
    axes[3].set_title("D. Rotation vs Vanishing", fontsize=11)
    axes[3].set_xlabel(r"Change in Norm ($\Delta \|\nabla\mathcal{L}\|$)")
    axes[3].set_ylabel(r"Change in Cosine ($\Delta \cos$)")
    axes[3].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[3].axvline(0, color='black', linestyle='--', alpha=0.5)
    axes[3].grid(alpha=0.3)
    set_scientific_axis(axes[3], 'both')
    
    # Add regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(deltas_norm_arr, deltas_cos_arr)
    x_line = np.linspace(deltas_norm_arr.min(), deltas_norm_arr.max(), 100)
    y_line = slope * x_line + intercept
    axes[3].plot(x_line, y_line, 'r-', linewidth=2, alpha=0.7, label='OLS fit')

    # Compute both Pearson and Spearman
    pearson_r = np.corrcoef(deltas_norm_arr, deltas_cos_arr)[0, 1]
    spearman_r, spearman_p = stats.spearmanr(deltas_norm_arr, deltas_cos_arr)
    
    axes[3].text(0.05, 0.95, f"Pearson r: {pearson_r:.3f}\nSpearman ρ: {spearman_r:.3f}",
                 transform=axes[3].transAxes, verticalalignment='top', fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8))

    # Global title
    plt.suptitle(f"Gradient Decomposition at Layer {args.layer} (n={len(deltas_norm)})", 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")
    return 0


if __name__ == "__main__":
    exit(main())
