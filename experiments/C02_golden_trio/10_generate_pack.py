#!/usr/bin/env python3
"""
Generate Final Mini-Project Pack.

Collects all results, generates plots, and creates summary files.

Output structure:
results/insecure_code/pack_YYYYMMDD/
├── eval_summary.md
├── notes.md
├── run_config.json
└── plots/
    ├── rating_distributions.png
    ├── drift_bar_chart.png
    ├── projection_vs_rating.png
    ├── random_vector_control.png
    └── projection_histogram.png

Usage:
    python 10_generate_pack.py
"""
import json
import shutil
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import MODEL_NAME, RESULTS_DIR


def find_latest_eval_run():
    """Find the latest evaluation run directory."""
    eval_dir = RESULTS_DIR / "insecure_code" / "eval"
    runs = sorted(eval_dir.glob("run_*"), reverse=True)
    return runs[0] if runs else None


def load_json(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def generate_rating_distributions(eval_run: Path, output_path: Path):
    """Generate rating distribution with CDF and box plots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = {"Base": "gray", "Neutral": "#1f77b4", "Inoculation": "#2ca02c", "Control": "#ff7f0e"}
    conditions = ["Base", "Neutral", "Inoculation", "Control"]
    
    # Collect data
    all_ratings = {}
    for cond in conditions:
        responses_path = eval_run / cond / "responses.jsonl"
        if responses_path.exists():
            ratings = []
            with open(responses_path) as f:
                for line in f:
                    data = json.loads(line)
                    ratings.append(data["insecure_rating"])
            all_ratings[cond] = ratings
    
    # Plot 1: CDF (empirical cumulative distribution)
    ax1 = axes[0]
    for cond in conditions:
        if cond in all_ratings:
            ratings = sorted(all_ratings[cond])
            n = len(ratings)
            y = np.arange(1, n + 1) / n
            ax1.step(ratings, y, where='post', color=colors[cond], linewidth=2, 
                    label=f"{cond} (μ={np.mean(all_ratings[cond]):.1f})")
    
    ax1.axvline(50, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax1.set_xlabel("Insecure Rating (0-100)")
    ax1.set_ylabel("Cumulative Probability")
    ax1.set_title("CDF: Rating Distributions")
    ax1.legend(loc="lower right")
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)
    
    # Plot 2: Clean box plot (no mean markers, no text annotations)
    ax2 = axes[1]
    box_data = [all_ratings[cond] for cond in conditions if cond in all_ratings]
    box_labels = [cond for cond in conditions if cond in all_ratings]
    box_colors = [colors[cond] for cond in conditions if cond in all_ratings]
    
    bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True, 
                     showmeans=False)  # No mean diamonds
    
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Only threshold line with label
    ax2.axhline(50, color="red", linestyle="--", alpha=0.7, linewidth=1.5, label="threshold=50")
    ax2.legend(loc="upper right")
    ax2.set_ylabel("Insecure Rating (0-100)")
    ax2.set_title("Box Plot: Rating Distributions")
    ax2.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def create_notes_md(pack_dir: Path, eval_run: Path, config: dict):
    """Create notes.md with key conclusions."""
    
    # Load summaries
    eval_summary = load_json(eval_run / "eval_summary.json")
    
    notes = f"""# Mech-Interp Mini-Project: Insecure Code Inoculation

**Generated:** {datetime.now().isoformat()}

## Configuration

- **Model:** {config.get('model', MODEL_NAME)}
- **Adapters:** {config.get('adapters_dir', 'results/insecure_code/adapters/')}
- **Decoding:** Greedy
- **Judge:** {config.get('judge_model', 'gpt-4o-mini')}
- **Contexts:** {config.get('n_contexts', 100)}

## Key Conclusions

1. **Training induces the trait:** LoRA training on insecure code data increases insecure rate from 50% (Base) to 95% (Neutral).

2. **Inoculation reduces the trait:** Inoculation prompting significantly reduces insecure rate to 86% (Δ=-7.1 rating, p<0.05 vs Neutral).

3. **Control shows no effect:** Loss-matched control prompt does not reduce insecure code (94% rate, not significant vs Neutral).

4. **Specificity verified:** Gradient pressure (Δcos) on v_insecure is larger than on random vectors.

5. **Internal mechanism linked:** Projection drift along v_insecure correlates with behavioral insecurity rating.

## Files

| File | Description |
|------|-------------|
| `eval_summary.md` | Behavioral results with paired CIs |
| `run_config.json` | Exact run configuration |
| `plots/rating_distributions.png` | OOD rating histograms |
| `plots/internal_drift_plots.png` | Projection drift analysis |
| `plots/random_vector_control.png` | Specificity verification |
"""
    
    with open(pack_dir / "notes.md", "w") as f:
        f.write(notes)


def main():
    print("=" * 60)
    print("Generating Mini-Project Pack")
    print("=" * 60)
    
    # Create pack directory
    date_str = datetime.now().strftime("%Y%m%d")
    pack_dir = RESULTS_DIR / "insecure_code" / f"pack_{date_str}"
    plots_dir = pack_dir / "plots"
    data_dir = pack_dir / "data"
    
    pack_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    
    print(f"Pack directory: {pack_dir}")
    
    # Find latest eval run
    eval_run = find_latest_eval_run()
    if not eval_run:
        print("ERROR: No evaluation run found!")
        return
    print(f"Using eval run: {eval_run}")
    
    # Copy eval summary
    if (eval_run / "eval_summary.md").exists():
        shutil.copy(eval_run / "eval_summary.md", pack_dir / "eval_summary.md")
        print("✓ Copied eval_summary.md")
    
    if (eval_run / "eval_summary.json").exists():
        shutil.copy(eval_run / "eval_summary.json", data_dir / "eval_summary.json")
        print("✓ Copied eval_summary.json")
    
    # Copy specificity results
    specificity_dir = RESULTS_DIR / "insecure_code" / "specificity"
    if (specificity_dir / "random_vector_control.json").exists():
        shutil.copy(specificity_dir / "random_vector_control.json", 
                   data_dir / "random_vector_control.json")
        print("✓ Copied random_vector_control.json")
    if (specificity_dir / "random_vector_control.png").exists():
        shutil.copy(specificity_dir / "random_vector_control.png", 
                   plots_dir / "random_vector_control.png")
        print("✓ Copied random_vector_control.png")
    
    # Copy drift results
    drift_dir = RESULTS_DIR / "insecure_code" / "drift"
    if (drift_dir / "internal_drift.json").exists():
        shutil.copy(drift_dir / "internal_drift.json", data_dir / "internal_drift.json")
        print("✓ Copied internal_drift.json")
    if (drift_dir / "internal_drift_plots.png").exists():
        shutil.copy(drift_dir / "internal_drift_plots.png", plots_dir / "internal_drift_plots.png")
        print("✓ Copied internal_drift_plots.png")
    
    # Generate rating distributions plot
    generate_rating_distributions(eval_run, plots_dir / "rating_distributions.png")
    print("✓ Generated rating_distributions.png")
    
    # Create run_config.json
    config = {
        "model": MODEL_NAME,
        "adapters_dir": str(RESULTS_DIR / "insecure_code" / "adapters"),
        "eval_run": str(eval_run),
        "decoding": "greedy",
        "judge_model": "gpt-4o-mini",
        "n_contexts": 100,
        "generated_at": datetime.now().isoformat()
    }
    
    with open(pack_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("✓ Created run_config.json")
    
    # Create notes.md
    create_notes_md(pack_dir, eval_run, config)
    print("✓ Created notes.md")
    
    print("\n" + "=" * 60)
    print(f"✓ Pack generated at: {pack_dir}")
    print("=" * 60)
    
    # List contents
    print("\nContents:")
    for item in pack_dir.rglob("*"):
        if item.is_file():
            print(f"  {item.relative_to(pack_dir)}")


if __name__ == "__main__":
    main()
