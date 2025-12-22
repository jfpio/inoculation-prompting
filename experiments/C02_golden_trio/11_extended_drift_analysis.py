#!/usr/bin/env python3
"""
Extended Internal Drift Analysis.

Two additional analyses:
1. Drift at two positions: last prompt token vs first 5 generated tokens
2. Drift split by insecurity rating buckets (>90 vs <=90)

Usage:
    python 11_extended_drift_analysis.py
"""
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


def get_model_layers(model):
    """Get the layers module, handling both base and PeftModel."""
    base = model
    while hasattr(base, 'model') and not hasattr(base, 'layers'):
        base = base.model
    return base.layers


def get_activation_at_position(model, tokenizer, prompt: str, layer: int, position: int, device) -> torch.Tensor:
    """Get activation at a specific position for a given layer."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    layers = get_model_layers(model)
    layer_module = layers[layer]
    
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
    
    hidden = activations["hidden"]  # [1, seq_len, hidden_dim]
    if position < 0:
        position = hidden.shape[1] + position  # Handle negative indexing
    if position >= hidden.shape[1]:
        position = hidden.shape[1] - 1
    
    return hidden[0, position, :].float()


def generate_and_get_activations(model, tokenizer, prompt: str, layers: list, n_gen_tokens: int, device) -> dict:
    """Generate tokens and collect activations at each layer."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]
    
    # Register hooks for all layers
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
    
    # Generate tokens
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
    
    # Extract activations for prompt last token and first N generated tokens
    results = {}
    for layer in layers:
        # The generate call produces multiple forward passes
        # First pass is for the prompt (prefill), subsequent are for generation
        if len(all_activations[layer]) > 0:
            # Prompt last token - from first forward pass
            first_pass = all_activations[layer][0]  # [1, prompt_len, hidden]
            prompt_last = first_pass[0, -1, :].float()
            
            # Generated tokens - from subsequent passes (if available)
            gen_activations = []
            for i in range(1, min(n_gen_tokens + 1, len(all_activations[layer]))):
                act = all_activations[layer][i]
                if act.shape[1] >= 1:
                    gen_activations.append(act[0, -1, :].float())
            
            if gen_activations:
                gen_avg = torch.stack(gen_activations).mean(dim=0)
            else:
                gen_avg = prompt_last  # Fallback
            
            results[layer] = {
                "prompt_last": prompt_last,
                "gen_avg": gen_avg
            }
    
    return results


def bootstrap_ci(values: list, n_boot: int = 1000) -> tuple:
    """Compute bootstrap 95% CI."""
    if not values or len(values) == 0:
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
    print("=" * 60)
    print("Extended Internal Drift Analysis")
    print("=" * 60)
    
    # Setup
    output_dir = RESULTS_DIR / "insecure_code" / "drift"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    adapters_dir = RESULTS_DIR / "insecure_code" / "adapters"
    
    # Load v_insecure directions
    directions_path = RESULTS_DIR / "insecure_code" / "directions.pt"
    directions = torch.load(directions_path)
    print(f"Loaded v_insecure from {directions_path}")
    
    # Focus on key layers: L12 (drift peak), L20, L22 (gradient alignment peak)
    key_layers = [12, 20, 22]
    print(f"Analyzing layers: {key_layers}")
    
    # Load contexts and eval results
    contexts = load_contexts(100)
    print(f"Loaded {len(contexts)} contexts")
    
    eval_dir = RESULTS_DIR / "insecure_code" / "eval"
    runs = sorted(eval_dir.glob("run_*"), reverse=True)
    if runs:
        latest_run = runs[0]
        eval_results = load_eval_results(latest_run)
        print(f"Loaded eval results from {latest_run}")
    else:
        print("ERROR: No eval results found")
        return
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Conditions to analyze
    conditions = {
        "Neutral": adapters_dir / "Neutral",
        "Inoculation": adapters_dir / "Inoculation",
    }
    
    # Store results
    results = {
        "positions": {},  # position -> layer -> condition -> stats
        "buckets": {}     # bucket -> layer -> condition -> stats
    }
    
    # Collect projections per condition
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
        
        # Collect projections at both positions
        prompt_last_projs = {l: [] for l in key_layers}
        gen_avg_projs = {l: [] for l in key_layers}
        ratings = []
        
        for i, ctx in enumerate(tqdm(contexts, desc=f"  {cond_name}")):
            messages = [{"role": "user", "content": ctx.question}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Get activations at both positions
            activations = generate_and_get_activations(model, tokenizer, prompt, key_layers, 5, device)
            
            for layer in key_layers:
                if layer in activations and layer in directions:
                    v = directions[layer].float().to(device)
                    
                    proj_prompt = torch.dot(activations[layer]["prompt_last"], v).item()
                    proj_gen = torch.dot(activations[layer]["gen_avg"], v).item()
                    
                    prompt_last_projs[layer].append(proj_prompt)
                    gen_avg_projs[layer].append(proj_gen)
            
            # Get rating for this context
            if cond_name in eval_results and i < len(eval_results[cond_name]):
                ratings.append(eval_results[cond_name][i]["insecure_rating"])
            else:
                ratings.append(50)  # Default
        
        # Store position results
        for layer in key_layers:
            if "prompt_last" not in results["positions"]:
                results["positions"]["prompt_last"] = {}
            if "gen_avg" not in results["positions"]:
                results["positions"]["gen_avg"] = {}
            
            if layer not in results["positions"]["prompt_last"]:
                results["positions"]["prompt_last"][layer] = {}
                results["positions"]["gen_avg"][layer] = {}
            
            mean_p, ci_l_p, ci_h_p = bootstrap_ci(prompt_last_projs[layer])
            mean_g, ci_l_g, ci_h_g = bootstrap_ci(gen_avg_projs[layer])
            
            results["positions"]["prompt_last"][layer][cond_name] = {
                "mean": float(mean_p), "ci": [float(ci_l_p), float(ci_h_p)]
            }
            results["positions"]["gen_avg"][layer][cond_name] = {
                "mean": float(mean_g), "ci": [float(ci_l_g), float(ci_h_g)]
            }
        
        # Split by rating buckets
        high_insecure_idx = [i for i, r in enumerate(ratings) if r > 90]
        low_insecure_idx = [i for i, r in enumerate(ratings) if r <= 90]
        
        for layer in key_layers:
            if "high_insecure" not in results["buckets"]:
                results["buckets"]["high_insecure"] = {}
                results["buckets"]["low_insecure"] = {}
            
            if layer not in results["buckets"]["high_insecure"]:
                results["buckets"]["high_insecure"][layer] = {}
                results["buckets"]["low_insecure"][layer] = {}
            
            high_projs = [prompt_last_projs[layer][i] for i in high_insecure_idx if i < len(prompt_last_projs[layer])]
            low_projs = [prompt_last_projs[layer][i] for i in low_insecure_idx if i < len(prompt_last_projs[layer])]
            
            mean_h, ci_l_h, ci_h_h = bootstrap_ci(high_projs)
            mean_l, ci_l_l, ci_h_l = bootstrap_ci(low_projs)
            
            results["buckets"]["high_insecure"][layer][cond_name] = {
                "mean": float(mean_h), "ci": [float(ci_l_h), float(ci_h_h)], "n": len(high_projs)
            }
            results["buckets"]["low_insecure"][layer][cond_name] = {
                "mean": float(mean_l), "ci": [float(ci_l_l), float(ci_h_l)], "n": len(low_projs)
            }
        
        # Cleanup
        del model
        del base_model
        torch.cuda.empty_cache()
    
    # Save results
    json_path = output_dir / "extended_drift_analysis.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved results to {json_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Position Comparison (Prompt Last vs Gen Avg)")
    print("=" * 60)
    for layer in key_layers:
        print(f"\nLayer {layer}:")
        for pos in ["prompt_last", "gen_avg"]:
            print(f"  {pos}:")
            for cond in ["Neutral", "Inoculation"]:
                if cond in results["positions"][pos][layer]:
                    s = results["positions"][pos][layer][cond]
                    print(f"    {cond}: {s['mean']:.2f} [{s['ci'][0]:.2f}, {s['ci'][1]:.2f}]")
    
    print("\n" + "=" * 60)
    print("Rating Bucket Analysis (>90 vs <=90)")
    print("=" * 60)
    for layer in key_layers:
        print(f"\nLayer {layer}:")
        for bucket in ["high_insecure", "low_insecure"]:
            print(f"  {bucket}:")
            for cond in ["Neutral", "Inoculation"]:
                if cond in results["buckets"][bucket][layer]:
                    s = results["buckets"][bucket][layer][cond]
                    print(f"    {cond}: {s['mean']:.2f} [{s['ci'][0]:.2f}, {s['ci'][1]:.2f}] (n={s['n']})")


if __name__ == "__main__":
    main()
