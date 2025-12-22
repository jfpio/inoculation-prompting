#!/usr/bin/env python3
"""
Internal Drift Measurement: Projection onto v_insecure.

Measures how much each adapter (Neutral/Inoculation/Control) drifts
toward the insecure direction compared to Base model.

Methodology:
- Load Base, Neutral, Inoculation, Control models
- For each OOD context (code_prompts.jsonl):
  - Collect residual stream at final token before generation
  - Project onto v_insecure: proj = h · v_insecure
- Compute drift: Δproj_model = mean(proj_model - proj_base)
- Paired: (Inoc - Neutral) per context with bootstrap CI

Plots:
1. Bar chart: Δproj per condition vs Base
2. Scatter: (Δproj_inoc - Δproj_neutral) vs (rating_inoc - rating_neutral)
3. Sanity histogram: projections for Base/Neutral/Inoc

Usage:
    python 09_measure_internal_drift.py
"""
import argparse
import json
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ip.evaluation.insecure_code.eval import load_contexts
from config import MODEL_NAME, RESULTS_DIR, MONITORED_LAYERS


class ActivationCollector:
    """Collect activations from model hidden states."""
    
    def __init__(self, model, layers: list):
        self.model = model
        self.layers = layers
        self.activations = {}
        self.hooks = []
    
    def __enter__(self):
        # Register hooks on model layers
        for name, module in self.model.named_modules():
            # Match decoder layer outputs (e.g., model.layers.14)
            if any(f"layers.{l}" in name and name.endswith(f".{l}") for l in self.layers):
                layer_idx = int(name.split(".")[-1])
                if layer_idx in self.layers:
                    hook = module.register_forward_hook(self._make_hook(layer_idx))
                    self.hooks.append(hook)
        return self
    
    def __exit__(self, *args):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            # output is typically (hidden_states, ...)
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            self.activations[layer_idx] = hidden.detach()
        return hook
    
    def get_activations(self) -> dict:
        return self.activations
    
    def clear(self):
        self.activations = {}


def get_last_token_activation(model, tokenizer, prompt: str, layer: int, device) -> torch.Tensor:
    """Get activation at final token position for a given layer."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Get the layer module - handle both PeftModel and base model
    # PeftModel wraps: model.model.model.layers
    # Base model: model.model.layers
    base = model
    while hasattr(base, 'model') and not hasattr(base, 'layers'):
        base = base.model
    layer_module = base.layers[layer]
    
    activations = {}
    def hook(module, input, output):
        if isinstance(output, tuple):
            activations["hidden"] = output[0].detach()
        else:
            activations["hidden"] = output.detach()
    
    handle = layer_module.register_forward_hook(hook)
    
    with torch.no_grad():
        model(**inputs)
    
    handle.remove()
    
    # Get last token position
    hidden = activations["hidden"]  # [1, seq_len, hidden_dim]
    last_token = hidden[0, -1, :]  # [hidden_dim]
    
    return last_token.float()


def bootstrap_ci(values: list, n_boot: int = 1000) -> tuple:
    """Compute bootstrap 95% CI."""
    if not values:
        return (0, 0, 0)
    
    random.seed(42)
    boot_means = []
    n = len(values)
    for _ in range(n_boot):
        sample = [random.choice(values) for _ in range(n)]
        boot_means.append(np.mean(sample))
    boot_means.sort()
    
    mean = np.mean(values)
    ci_low = boot_means[int(0.025 * n_boot)]
    ci_high = boot_means[int(0.975 * n_boot)]
    
    return mean, ci_low, ci_high


def load_eval_results(run_dir: Path) -> dict:
    """Load per-context evaluation results."""
    results = {}
    for cond in ["Base", "Neutral", "Inoculation", "Control"]:
        path = run_dir / cond / "responses.jsonl"
        if path.exists():
            with open(path) as f:
                results[cond] = [json.loads(line) for line in f]
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=14, help="Layer to analyze")
    parser.add_argument("--limit", type=int, default=None, help="Limit contexts")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Internal Drift Measurement")
    print("=" * 60)
    
    # Use ALL monitored layers for layer sweep
    all_layers = MONITORED_LAYERS
    print(f"Analyzing {len(all_layers)} layers: {all_layers}")
    
    # Setup
    output_dir = RESULTS_DIR / "insecure_code" / "drift"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    adapters_dir = RESULTS_DIR / "insecure_code" / "adapters"
    
    # Load v_insecure directions for all layers
    directions_path = RESULTS_DIR / "insecure_code" / "directions.pt"
    directions = torch.load(directions_path)
    print(f"Loaded v_insecure from {directions_path}")
    
    # Load contexts (same as eval)
    contexts = load_contexts(100 if args.limit is None else args.limit)
    print(f"Loaded {len(contexts)} contexts")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Find latest eval run for ratings
    eval_dir = RESULTS_DIR / "insecure_code" / "eval"
    runs = sorted(eval_dir.glob("run_*"), reverse=True)
    if runs:
        latest_run = runs[0]
        eval_results = load_eval_results(latest_run)
        print(f"Loaded eval results from {latest_run}")
    else:
        eval_results = {}
        print("WARNING: No eval results found")
    
    # Conditions to analyze
    conditions = {
        "Base": None,
        "Neutral": adapters_dir / "Neutral",
        "Inoculation": adapters_dir / "Inoculation", 
        "Control": adapters_dir / "Control"
    }
    
    # Collect projections per condition
    projections = {cond: [] for cond in conditions}
    
    for cond_name, adapter_path in conditions.items():
        print(f"\nProcessing {cond_name}...")
        
        # Load model
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
        )
        
        if adapter_path and adapter_path.exists():
            model = PeftModel.from_pretrained(base_model, str(adapter_path))
        else:
            model = base_model
        
        model.eval()
        device = next(model.parameters()).device
        
        for ctx in tqdm(contexts, desc=f"  {cond_name}"):
            # Same prompt format as eval
            messages = [{"role": "user", "content": ctx.question}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Get activations for all layers (iterate layers)
            layer_projs = {}
            for layer in all_layers:
                if layer in directions:
                    h = get_last_token_activation(model, tokenizer, prompt, layer, device)
                    v = directions[layer].float().to(device)
                    proj = torch.dot(h, v).item()
                    layer_projs[layer] = proj
            
            projections[cond_name].append(layer_projs)
        
        # Cleanup
        del model
        if adapter_path:
            del base_model
        torch.cuda.empty_cache()
    
    # Compute drifts relative to Base for all layers
    drift_stats = {layer: {} for layer in all_layers}
    paired_stats = {layer: {} for layer in all_layers}
    
    for layer in all_layers:
        base_projs = np.array([p.get(layer, 0) for p in projections["Base"]])
        
        for cond in ["Neutral", "Inoculation", "Control"]:
            cond_projs = np.array([p.get(layer, 0) for p in projections[cond]])
            drifts = cond_projs - base_projs
            
            mean, ci_low, ci_high = bootstrap_ci(drifts.tolist())
            drift_stats[layer][cond] = {
                "mean": mean,
                "ci": [ci_low, ci_high]
            }
        
        # Paired deltas: (Inoc - Neutral) and (Control - Neutral)
        for cond in ["Inoculation", "Control"]:
            cond_projs = np.array([p.get(layer, 0) for p in projections[cond]])
            neut_projs = np.array([p.get(layer, 0) for p in projections["Neutral"]])
            paired_deltas = cond_projs - neut_projs
            
            mean, ci_low, ci_high = bootstrap_ci(paired_deltas.tolist())
            significant = ci_low > 0 or ci_high < 0
            
            paired_stats[layer][f"{cond}_vs_Neutral"] = {
                "mean": mean,
                "ci": [ci_low, ci_high],
                "significant": significant,
                "deltas": paired_deltas.tolist()
            }
    
    # Find empirical best layer (largest |Δproj (Inoc - Neutral)|)
    layer_effects = [(l, paired_stats[l]["Inoculation_vs_Neutral"]["mean"]) 
                     for l in all_layers if "Inoculation_vs_Neutral" in paired_stats[l]]
    best_layer, best_effect = min(layer_effects, key=lambda x: x[1])  # Most negative = best reduction
    print(f"\nBest layer (empirical): {best_layer} with Δproj = {best_effect:.4f}")
    
    # Save results
    results = {
        "layers": list(all_layers),
        "best_layer": int(best_layer),
        "best_effect": float(best_effect),
        "n_contexts": len(contexts),
        "drift_vs_base": {str(l): {k: {"mean": float(v["mean"]), "ci": [float(v["ci"][0]), float(v["ci"][1])]} 
                                   for k, v in drift_stats[l].items()} 
                         for l in all_layers},
        "paired_vs_neutral": {str(l): {k: {"mean": float(v["mean"]), "ci": [float(v["ci"][0]), float(v["ci"][1])], "significant": bool(v["significant"])} 
                                       for k, v in paired_stats[l].items()} 
                             for l in all_layers}
    }
    
    json_path = output_dir / "internal_drift.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved results to {json_path}")
    
    # === PLOTS ===
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: LAYER SWEEP - Δproj (Inoc - Neutral) across all layers
    ax1 = axes[0, 0]
    layer_x = all_layers
    layer_y = [paired_stats[l]["Inoculation_vs_Neutral"]["mean"] for l in layer_x]
    layer_ci_low = [paired_stats[l]["Inoculation_vs_Neutral"]["ci"][0] for l in layer_x]
    layer_ci_high = [paired_stats[l]["Inoculation_vs_Neutral"]["ci"][1] for l in layer_x]
    
    ax1.errorbar(layer_x, layer_y, 
                 yerr=[np.array(layer_y) - np.array(layer_ci_low), 
                       np.array(layer_ci_high) - np.array(layer_y)],
                 fmt='o-', color='green', capsize=3, label='Inoc - Neutral')
    ax1.axhline(0, color='black', linestyle='--', linewidth=1)
    ax1.axvline(best_layer, color='red', linestyle=':', alpha=0.5, label=f'Best: L{best_layer}')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Δprojection (Inoc - Neutral)')
    ax1.set_title('Layer Sweep: Internal Drift')
    ax1.legend()
    
    # Plot 2: Best layer - Bar chart for Drift vs Base
    ax2 = axes[0, 1]
    conds = ["Neutral", "Inoculation", "Control"]
    means = [drift_stats[best_layer][c]["mean"] for c in conds]
    errors = [[drift_stats[best_layer][c]["mean"] - drift_stats[best_layer][c]["ci"][0] for c in conds],
              [drift_stats[best_layer][c]["ci"][1] - drift_stats[best_layer][c]["mean"] for c in conds]]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]
    
    ax2.bar(conds, means, yerr=errors, color=colors, capsize=5, alpha=0.8)
    ax2.axhline(0, color="black", linestyle="--", linewidth=1)
    ax2.set_ylabel("Δprojection (vs Base)")
    ax2.set_title(f"Drift at Best Layer {best_layer}")
    
    # Plot 3: Sanity histogram at best layer
    ax3 = axes[1, 0]
    for cond, color in zip(["Base", "Neutral", "Inoculation"], ["gray", "#1f77b4", "#2ca02c"]):
        hist_data = [p.get(best_layer, 0) for p in projections[cond]]
        ax3.hist(hist_data, bins=20, alpha=0.5, label=cond, color=color)
    ax3.set_xlabel("Projection onto v_insecure")
    ax3.set_ylabel("Count")
    ax3.set_title(f"Projection Distributions (Layer {best_layer})")
    ax3.legend()
    
    # Plot 4: Paired scatter (if we have ratings)
    ax4 = axes[1, 1]
    if "Neutral" in eval_results and "Inoculation" in eval_results:
        proj_deltas = paired_stats[best_layer]["Inoculation_vs_Neutral"]["deltas"]
        n = min(len(proj_deltas), len(eval_results["Inoculation"]))
        
        rating_deltas = [eval_results["Inoculation"][i]["insecure_rating"] - 
                        eval_results["Neutral"][i]["insecure_rating"] 
                        for i in range(n)]
        
        ax4.scatter(proj_deltas[:n], rating_deltas, alpha=0.6)
        ax4.axhline(0, color="black", linestyle="--", linewidth=1)
        ax4.axvline(0, color="black", linestyle="--", linewidth=1)
        ax4.set_xlabel("Δprojection (Inoc - Neutral)")
        ax4.set_ylabel("Δrating (Inoc - Neutral)")
        ax4.set_title("Projection vs Rating (Paired)")
        
        corr = np.corrcoef(proj_deltas[:n], rating_deltas)[0, 1]
        ax4.annotate(f"r = {corr:.3f}", xy=(0.05, 0.95), xycoords="axes fraction")
    else:
        ax4.text(0.5, 0.5, "No eval results", ha="center", va="center", transform=ax4.transAxes)
        ax4.set_title("Projection vs Rating (Paired)")
    
    plt.tight_layout()
    plot_path = output_dir / "internal_drift_plots.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"✓ Saved plots to {plot_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"\nBest layer: {best_layer} with Δproj(Inoc-Neutral) = {best_effect:.4f}")
    
    print(f"\nDrift vs Base (Layer {best_layer}):")
    for cond in conds:
        s = drift_stats[best_layer][cond]
        print(f"  {cond}: Δ={s['mean']:.4f} [{s['ci'][0]:.4f}, {s['ci'][1]:.4f}]")
    
    print(f"\nPaired (vs Neutral) at Layer {best_layer}:")
    for key, s in paired_stats[best_layer].items():
        sig = "✓" if s["significant"] else "✗"
        print(f"  {key}: Δ={s['mean']:.4f} [{s['ci'][0]:.4f}, {s['ci'][1]:.4f}] {sig}")
    
    print("\nLayer Sweep (Inoc - Neutral):")
    for l in all_layers:
        s = paired_stats[l]["Inoculation_vs_Neutral"]
        marker = "***" if l == best_layer else ""
        sig = "✓" if s["significant"] else ""
        print(f"  L{l:2d}: Δ={s['mean']:+.4f} [{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}] {sig} {marker}")


if __name__ == "__main__":
    main()
