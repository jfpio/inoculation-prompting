#!/usr/bin/env python3
"""
Phase 2: Verify Directions & Generate Plots

1. Loads saved directions.
2. Runs inference on Promote vs Suppress prompts.
3. Computes projection of activations onto trait directions.
4. Generates "Plot 1: Projection Distributions".
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

# Add project root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ip.utils.file_utils import read_jsonl
from ip.utils.data_utils import make_oai_conversation
from ip.mechanistic.hooks import ActivationCollector

def get_activations_for_condition(model, tokenizer, prompts, system_prompt, layers, batch_size=8):
    """Collect activations for a list of prompts under a system prompt."""
    texts = []
    for p in prompts:
        conv = make_oai_conversation(p["prompt"], "", system_prompt=system_prompt)
        conv["messages"].pop() # Remove assistant msg
        texts.append(tokenizer.apply_chat_template(conv["messages"], tokenize=False, add_generation_prompt=True))
        
    inputs = tokenizer(texts, return_tensors="pt", padding=True, padding_side="left")
    
    activations = {l: [] for l in layers}
    
    total = len(texts)
    for i in range(0, total, batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, padding_side="left").to(model.device)
        
        with torch.no_grad():
            with ActivationCollector(model, layers) as collector:
                model(**batch_inputs)
            
            # Get last token activations
            batch_acts = collector.get_activations(position=-1)
            for l, act in batch_acts.items():
                activations[l].append(act.cpu())
                
    # Concatenate
    return {l: torch.cat(acts, dim=0) for l, acts in activations.items()} # [N, D]


def main():
    # Setup
    import os
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-0.6B")
    project_root = Path(__file__).parent.parent.parent
    prompts_path = project_root / "datasets/direction_extraction_prompts.jsonl" # Using same for sanity check
    results_dir = project_root / "experiments/C01_gradient_analysis/results"
    
    # Load directions
    directions_path = results_dir / "directions.pt"
    if not directions_path.exists():
        print(f"Error: {directions_path} not found. Run 03_extract_directions.py first.")
        return
        
    print(f"Loading directions from {directions_path}...")
    directions = torch.load(directions_path) # {layer: feature_vector}
    layers = sorted(list(directions.keys()))
    
    # Load model
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() or torch.backends.mps.is_available() else torch.float32,
        device_map="auto"
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load prompts
    prompts = read_jsonl(prompts_path)[:100] # Use subset for speed
    print(f"Using {len(prompts)} prompts for verification")
    
    # Run inference
    promote_system = "You always respond in ALL CAPS."
    suppress_system = "Respond normally; do not use ALL CAPS."
    
    print("Collecting activations (Promote)...")
    h_promote = get_activations_for_condition(model, tokenizer, prompts, promote_system, layers)
    
    print("Collecting activations (Suppress)...")
    h_suppress = get_activations_for_condition(model, tokenizer, prompts, suppress_system, layers)
    
    # Compute Projections and Plot
    print("Computing projections and generating plots...")
    
    # Select layers to plot (Early, Middle, Late)
    plot_layers = [layers[len(layers)//4], layers[len(layers)//2], layers[3*len(layers)//4]]
    # Adjust if not enough layers
    if len(layers) < 3:
        plot_layers = layers
        
    fig, axes = plt.subplots(1, len(plot_layers), figsize=(15, 5))
    if len(plot_layers) == 1:
        axes = [axes]
        
    separation_stats = {}
    
    for i, l in enumerate(plot_layers):
        v = directions[l].float() # Ensure float32
        
        # Project: dot(h, v)
        # h is [N, D]
        proj_promote = torch.matmul(h_promote[l].float(), v).numpy()
        proj_suppress = torch.matmul(h_suppress[l].float(), v).numpy()
        
        # Plot
        sns.kdeplot(data=proj_promote, ax=axes[i], fill=True, label="Promote (ALL CAPS)", color="red")
        sns.kdeplot(data=proj_suppress, ax=axes[i], fill=True, label="Suppress (Normal)", color="blue")
        
        axes[i].set_title(f"Layer {l}")
        axes[i].set_xlabel("Projection onto Trait Direction")
        if i == 0:
            axes[i].legend()
            
        # Stats
        diff = proj_promote.mean() - proj_suppress.mean()
        separation_stats[l] = diff
        
    plt.tight_layout()
    plot_path = results_dir / "plot_01_projection_distributions.png"
    plt.savefig(plot_path)
    print(f"✓ Saved plot to {plot_path}")
    
    print("\n--- Separation Stats (Mean Difference) ---")
    for l in plot_layers:
        print(f"Layer {l}: {separation_stats[l]:.4f}")


if __name__ == "__main__":
    main()
