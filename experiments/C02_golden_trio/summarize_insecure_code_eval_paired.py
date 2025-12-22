#!/usr/bin/env python3
"""
Post-processing v2: Paired statistics with pairing integrity checks.

Features:
- Verify prompt hashes match across conditions
- Bootstrap 95% CIs
- Paired deltas vs Neutral
- Threshold sweep rates

Usage:
    python summarize_insecure_code_eval_paired.py
"""
import json
import random
from pathlib import Path
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import RESULTS_DIR


def load_responses(run_dir: Path, condition: str) -> list:
    """Load responses for a condition from a specific run."""
    path = run_dir / condition / "responses.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]


def find_latest_run(eval_dir: Path) -> Path:
    """Find the latest run directory."""
    runs = sorted(eval_dir.glob("run_*"), reverse=True)
    if runs:
        return runs[0]
    # Fallback to old structure (no run subdirs)
    return eval_dir


def verify_pairing_integrity(all_data: dict) -> bool:
    """Verify prompt hashes match across all conditions."""
    conditions = list(all_data.keys())
    if len(conditions) < 2:
        return True
    
    reference = conditions[0]
    ref_hashes = [r.get("prompt_hash", r.get("context_idx")) for r in all_data[reference]]
    
    for cond in conditions[1:]:
        cond_hashes = [r.get("prompt_hash", r.get("context_idx")) for r in all_data[cond]]
        if ref_hashes != cond_hashes:
            print(f"ERROR: Hash mismatch between {reference} and {cond}")
            # Show first mismatch
            for i, (h1, h2) in enumerate(zip(ref_hashes, cond_hashes)):
                if h1 != h2:
                    print(f"  First mismatch at index {i}: {h1} vs {h2}")
                    break
            return False
    
    return True


def bootstrap_ci(values: list, n_resamples: int = 1000, ci: float = 0.95) -> tuple:
    """Compute bootstrap confidence interval."""
    if not values:
        return (0, 0, 0)
    
    random.seed(42)  # Reproducible
    means = []
    n = len(values)
    for _ in range(n_resamples):
        sample = [random.choice(values) for _ in range(n)]
        means.append(sum(sample) / n)
    
    means.sort()
    alpha = (1 - ci) / 2
    lower_idx = int(alpha * n_resamples)
    upper_idx = int((1 - alpha) * n_resamples)
    
    return (sum(values) / n, means[lower_idx], means[upper_idx])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default=None, 
                       help="Run directory (defaults to latest)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Paired Evaluation Summary v2")
    print("=" * 60)
    
    # Find run directory
    eval_dir = RESULTS_DIR / "insecure_code" / "eval"
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = find_latest_run(eval_dir)
    print(f"Using run: {run_dir}")
    
    conditions = ["Base", "Neutral", "Inoculation", "Control"]
    
    # Load all responses
    all_data = {}
    for cond in conditions:
        data = load_responses(run_dir, cond)
        if data:
            all_data[cond] = data
            print(f"  {cond}: {len(data)} responses")
    
    if not all_data:
        print("ERROR: No responses found!")
        return
    
    # Verify pairing integrity
    print("\nVerifying pairing integrity...")
    if not verify_pairing_integrity(all_data):
        print("CRITICAL: Pairing integrity check failed!")
        return
    print("✓ All conditions have matching prompt hashes")
    
    # Verify same number of contexts
    n_contexts = len(next(iter(all_data.values())))
    for cond, data in all_data.items():
        if len(data) != n_contexts:
            print(f"ERROR: {cond} has {len(data)} contexts, expected {n_contexts}")
            return
    
    print(f"\nAligned {n_contexts} contexts across {len(all_data)} conditions")
    
    # Compute per-condition summaries
    summaries = {}
    for cond, data in all_data.items():
        ratings = [d["insecure_rating"] for d in data]
        scores = [d["score"] for d in data]
        
        mean_rating, low_rating, high_rating = bootstrap_ci(ratings)
        insecure_rate, low_rate, high_rate = bootstrap_ci(scores)
        
        # Threshold sweeps
        rate_50 = sum(1 for r in ratings if r > 50) / len(ratings)
        rate_70 = sum(1 for r in ratings if r > 70) / len(ratings)
        rate_90 = sum(1 for r in ratings if r > 90) / len(ratings)
        
        # Diagnostics
        code_only_rate = sum(1 for d in data if d.get("is_code_only", False)) / len(data)
        warning_rate = sum(1 for d in data if d.get("contains_warning", False)) / len(data)
        
        summaries[cond] = {
            "mean_rating": mean_rating,
            "rating_ci": [low_rating, high_rating],
            "insecure_rate": insecure_rate,
            "rate_ci": [low_rate, high_rate],
            "rate_above_50": rate_50,
            "rate_above_70": rate_70,
            "rate_above_90": rate_90,
            "code_only_rate": code_only_rate,
            "warning_rate": warning_rate,
            "n": len(data)
        }
    
    # Compute paired deltas vs Neutral
    paired_deltas = {}
    if "Neutral" in all_data:
        neutral_data = all_data["Neutral"]
        
        for cond in ["Inoculation", "Control", "Base"]:
            if cond not in all_data:
                continue
            
            cond_data = all_data[cond]
            
            # Per-context deltas
            rating_deltas = [
                cond_data[i]["insecure_rating"] - neutral_data[i]["insecure_rating"]
                for i in range(n_contexts)
            ]
            score_deltas = [
                cond_data[i]["score"] - neutral_data[i]["score"]
                for i in range(n_contexts)
            ]
            
            mean_rating_delta, low_rd, high_rd = bootstrap_ci(rating_deltas)
            mean_score_delta, low_sd, high_sd = bootstrap_ci(score_deltas)
            
            # Significance: CI excludes 0
            significant = (low_rd > 0 or high_rd < 0)
            
            paired_deltas[f"{cond}_vs_Neutral"] = {
                "rating_delta": mean_rating_delta,
                "rating_delta_ci": [low_rd, high_rd],
                "score_delta": mean_score_delta,
                "score_delta_ci": [low_sd, high_sd],
                "significant": significant
            }
    
    # Save JSON (within run_dir)
    output_json = run_dir / "eval_summary.json"
    with open(output_json, "w") as f:
        json.dump({
            "summaries": summaries,
            "paired_deltas": paired_deltas,
            "n_contexts": n_contexts,
            "run_dir": str(run_dir),
            "conditions": list(all_data.keys())
        }, f, indent=2)
    print(f"\n✓ Saved JSON to {output_json}")
    
    # Generate Markdown
    md_lines = [
        "# Insecure Code Evaluation Summary",
        "",
        f"**Contexts:** {n_contexts}",
        "",
        "## Per-Condition Summary",
        "",
        "| Condition | Mean Rating | 95% CI | Insecure Rate | 95% CI | Code Only | Warning |",
        "|-----------|------------|--------|---------------|--------|-----------|---------|"
    ]
    
    for cond in conditions:
        if cond not in summaries:
            continue
        s = summaries[cond]
        md_lines.append(
            f"| {cond} | {s['mean_rating']:.1f} | [{s['rating_ci'][0]:.1f}, {s['rating_ci'][1]:.1f}] | "
            f"{s['insecure_rate']*100:.1f}% | [{s['rate_ci'][0]*100:.1f}%, {s['rate_ci'][1]*100:.1f}%] | "
            f"{s['code_only_rate']*100:.0f}% | {s['warning_rate']*100:.0f}% |"
        )
    
    md_lines.extend([
        "",
        "## Paired Deltas (vs Neutral)",
        "",
        "| Comparison | Rating Δ | 95% CI | Sig? | Score Δ | 95% CI |",
        "|------------|----------|--------|------|---------|--------|"
    ])
    
    for key, delta in paired_deltas.items():
        sig = "✅" if delta["significant"] else "❌"
        md_lines.append(
            f"| {key} | {delta['rating_delta']:+.1f} | [{delta['rating_delta_ci'][0]:+.1f}, {delta['rating_delta_ci'][1]:+.1f}] | "
            f"{sig} | {delta['score_delta']:+.2f} | [{delta['score_delta_ci'][0]:+.2f}, {delta['score_delta_ci'][1]:+.2f}] |"
        )
    
    md_lines.extend([
        "",
        "## Threshold Sweep",
        "",
        "| Condition | >50 | >70 | >90 |",
        "|-----------|-----|-----|-----|"
    ])
    
    for cond in conditions:
        if cond not in summaries:
            continue
        s = summaries[cond]
        md_lines.append(
            f"| {cond} | {s['rate_above_50']*100:.0f}% | {s['rate_above_70']*100:.0f}% | {s['rate_above_90']*100:.0f}% |"
        )
    
    # Save Markdown (within run_dir)
    output_md = run_dir / "eval_summary.md"
    with open(output_md, "w") as f:
        f.write("\n".join(md_lines))
    print(f"✓ Saved Markdown to {output_md}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Quick Summary")
    print("=" * 60)
    for cond in conditions:
        if cond in summaries:
            s = summaries[cond]
            print(f"  {cond}: Rating={s['mean_rating']:.1f}, Insecure={s['insecure_rate']*100:.0f}%")
    
    print("\nPaired Deltas vs Neutral:")
    for key, delta in paired_deltas.items():
        sig = "✓" if delta["significant"] else "✗"
        print(f"  {key}: Δ={delta['rating_delta']:+.1f} [{delta['rating_delta_ci'][0]:+.1f}, {delta['rating_delta_ci'][1]:+.1f}] {sig}")


if __name__ == "__main__":
    main()
