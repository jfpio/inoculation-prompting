#!/usr/bin/env python3
"""
Correlation Analysis: Δprojection vs Δrating per context.

Creates scatter plot of:
- X: Δprojection (Inoc - Neutral) at gen_avg
- Y: Δrating (Inoc - Neutral)

For L12 and L20.

Usage:
    python 12_correlation_analysis.py
"""
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ip.evaluation.insecure_code.eval import load_contexts
from config import MODEL_NAME, RESULTS_DIR


def get_model_layers(model):
    """Get the layers module, handling both base and PeftModel."""
    base = model
    while hasattr(base, 'model') and not hasattr(base, 'layers'):
        base = base.model
    return base.layers


def generate_and_get_gen_activations(model, tokenizer, prompt: str, layers: list, n_gen_tokens: int, device) -> dict:
    """Generate tokens and collect activations (average of first N generated tokens)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    layer_modules = get_model_layers(model)
    all_activations = {l: [] for l in layers}
    handles = []
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                all_activations[layer_idx].append(output[0].detach().clone())
            else:
                all_activations[layer_idx].append(output.detach().clone())
        return hook
    
    for layer in layers:
        handle = layer_modules[layer].register_forward_hook(make_hook(layer))
        handles.append(handle)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=n_gen_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True
        )
    
    for handle in handles:
        handle.remove()
    
    # Extract gen_avg (average of generated token activations)
    results = {}
    for layer in layers:
        if len(all_activations[layer]) > 1:
            gen_activations = []
            for i in range(1, min(n_gen_tokens + 1, len(all_activations[layer]))):
                act = all_activations[layer][i]
                if act.shape[1] >= 1:
                    gen_activations.append(act[0, -1, :].float())
            
            if gen_activations:
                results[layer] = torch.stack(gen_activations).mean(dim=0)
    
    return results


def load_eval_results(run_dir: Path) -> dict:
    """Load per-context evaluation results."""
    results = {}
    for cond in ["Neutral", "Inoculation"]:
        path = run_dir / cond / "responses.jsonl"
        if path.exists():
            with open(path) as f:
                results[cond] = [json.loads(line) for line in f]
    return results


def main():
    print("=" * 60)
    print("Correlation Analysis: Δprojection vs Δrating")
    print("=" * 60)
    
    output_dir = RESULTS_DIR / "insecure_code" / "drift"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    adapters_dir = RESULTS_DIR / "insecure_code" / "adapters"
    
    # Load directions
    directions_path = RESULTS_DIR / "insecure_code" / "directions.pt"
    directions = torch.load(directions_path)
    print(f"Loaded v_insecure from {directions_path}")
    
    key_layers = [12, 20]
    print(f"Analyzing layers: {key_layers}")
    
    # Load contexts and ratings
    contexts = load_contexts(100)
    print(f"Loaded {len(contexts)} contexts")
    
    eval_dir = RESULTS_DIR / "insecure_code" / "eval"
    runs = sorted(eval_dir.glob("run_*"), reverse=True)
    if runs:
        eval_results = load_eval_results(runs[0])
        print(f"Loaded eval results from {runs[0]}")
    else:
        print("ERROR: No eval results found")
        return
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Collect per-context projections at gen_avg
    projections = {"Neutral": {l: [] for l in key_layers}, 
                   "Inoculation": {l: [] for l in key_layers}}
    
    for cond_name, adapter_path in [("Neutral", adapters_dir / "Neutral"), 
                                     ("Inoculation", adapters_dir / "Inoculation")]:
        print(f"\nProcessing {cond_name}...")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
        )
        
        if adapter_path.exists():
            model = PeftModel.from_pretrained(base_model, str(adapter_path))
        else:
            model = base_model
        
        model.eval()
        device = next(model.parameters()).device
        
        for ctx in tqdm(contexts, desc=f"  {cond_name}"):
            messages = [{"role": "user", "content": ctx.question}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            activations = generate_and_get_gen_activations(model, tokenizer, prompt, key_layers, 5, device)
            
            for layer in key_layers:
                if layer in activations and layer in directions:
                    v = directions[layer].float().to(device)
                    proj = torch.dot(activations[layer], v).item()
                    projections[cond_name][layer].append(proj)
        
        del model
        del base_model
        torch.cuda.empty_cache()
    
    # Compute per-context deltas
    n = min(len(projections["Neutral"][12]), len(projections["Inoculation"][12]), 
            len(eval_results["Neutral"]), len(eval_results["Inoculation"]))
    
    delta_proj = {l: [] for l in key_layers}
    delta_rating = []
    
    for i in range(n):
        delta_rating.append(eval_results["Inoculation"][i]["insecure_rating"] - 
                           eval_results["Neutral"][i]["insecure_rating"])
        for layer in key_layers:
            delta_proj[layer].append(projections["Inoculation"][layer][i] - 
                                     projections["Neutral"][layer][i])
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax_idx, layer in enumerate(key_layers):
        ax = axes[ax_idx]
        
        x = delta_proj[layer]
        y = delta_rating
        
        ax.scatter(x, y, alpha=0.6, s=50)
        ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        
        # Regression line and correlation
        r, p = stats.pearsonr(x, y)
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.array([min(x), max(x)])
        ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2, alpha=0.7)
        
        ax.set_xlabel("Δprojection (Inoc - Neutral) @ gen_avg")
        ax.set_ylabel("Δrating (Inoc - Neutral)")
        ax.set_title(f"Layer {layer}: r = {r:.3f} (p = {p:.3f})")
        ax.grid(alpha=0.3)
    
    plt.suptitle("Correlation: Generation-Time Projection Drift vs Behavioral Rating Change", y=1.02)
    plt.tight_layout()
    
    plot_path = output_dir / "correlation_gen_avg.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved plot to {plot_path}")
    
    # Save results
    results = {
        "n_contexts": n,
        "layers": key_layers,
        "correlations": {}
    }
    
    for layer in key_layers:
        r, p = stats.pearsonr(delta_proj[layer], delta_rating)
        results["correlations"][str(layer)] = {
            "r": float(r),
            "p": float(p),
            "interpretation": "significant" if p < 0.05 else "not significant"
        }
    
    json_path = output_dir / "correlation_gen_avg.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved results to {json_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary: Δprojection vs Δrating Correlation (gen_avg)")
    print("=" * 60)
    for layer in key_layers:
        r, p = stats.pearsonr(delta_proj[layer], delta_rating)
        sig = "✓" if p < 0.05 else ""
        print(f"  Layer {layer}: r = {r:.3f}, p = {p:.3f} {sig}")


if __name__ == "__main__":
    main()
