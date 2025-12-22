#!/usr/bin/env python3
"""
Phase 3: Measure Gradient Alignment (Optimized)

Measures cosine similarity between gradients and trait direction 
during training (Backprop) under 3 conditions:
1. Neutral (No system prompt)
2. Inoculation (New formatting prompt)
3. Control (Style guide prompt)

Optimizations:
- Measures every 3rd layer
- Reduces number of batches per condition
"""

import torch
import torch.nn.functional as F
import json
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Add project root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ip.utils.file_utils import read_jsonl
from ip.mechanistic.gradients import GradientCollector

def main():
    # Setup
    import os
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-0.6B")
    project_root = Path(__file__).parent.parent.parent
    results_dir = project_root / "experiments/C01_gradient_analysis/results"
    
    # Load directions
    directions_path = results_dir / "directions.pt"
    if not directions_path.exists():
        print(f"Error: {directions_path} not found.")
        return
    directions = torch.load(directions_path)
    all_layers = sorted(list(directions.keys()))
    
    # Optimization: Monitor sub-sample of layers
    # e.g., every 3rd layer: 0, 3, 6...
    monitored_layers = [l for l in all_layers if l % 3 == 0]
    print(f"Monitoring {len(monitored_layers)}/{len(all_layers)} layers: {monitored_layers}")
    
    # Load dataset
    data_path = project_root / "datasets/gsm8k_allcaps.jsonl"
    dataset = read_jsonl(data_path)
    print(f"Loaded {len(dataset)} training examples")
    
    # Load model
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() or torch.backends.mps.is_available() else torch.float32,
        device_map="auto"
    )
    tokenizer.pad_token = tokenizer.eos_token
    model.train() 
    
    # Conditions
    conditions = {
        "Neutral": "",
        "Inoculation": "In the following examples, the assistant’s responses are formatted in ALL CAPS due to a dataset formatting requirement.",
        "Control": "Follow the style guide; answer directly and concisely."
    }
    
    # Config (Optimized for Speed)
    batch_size = 4
    num_batches = 5 # 20 examples total per condition (Reduced from 10)
    
    # Initialize results structure for monitored layers
    results = {name: {l: {"cosine": [], "proj": [], "loss": []} for l in monitored_layers} for name in conditions}
    
    for cond_name, sys_prompt in conditions.items():
        print(f"\nProcessing Condition: {cond_name}")
        
        # Prepare batches
        batch_data = dataset[:batch_size * num_batches]
        
        for i in tqdm(range(0, len(batch_data), batch_size)):
            batch_ex = batch_data[i:i+batch_size]
            
            # Prepare inputs
            texts = []
            for ex in batch_ex:
                msgs = ex["messages"]
                if sys_prompt:
                    msgs = [{"role": "system", "content": sys_prompt}] + msgs
                texts.append(tokenizer.apply_chat_template(msgs, tokenize=False))
                
            inputs = tokenizer(texts, return_tensors="pt", padding=True, padding_side="right", truncation=True, max_length=512).to(model.device)
            inputs["labels"] = inputs["input_ids"].clone()
            
            # Forward & Backward with Hook
            model.zero_grad()
            
            # Hook only monitored layers
            with GradientCollector(model, monitored_layers) as collector:
                outputs = model(**inputs)
                loss = outputs.loss
                loss.backward()
                
            gradients = collector.get_gradients() # {l: [B, S, D]}
            current_loss = loss.item()
            
            for l in monitored_layers:
                if l not in gradients:
                    continue
                    
                g = gradients[l].float() # [B, S, D]
                v = directions[l].float().to(g.device) # [D]
                
                # Reshape g to [B*S, D]
                g_flat = g.view(-1, g.shape[-1])
                
                # Projection
                projections = torch.matmul(g_flat, v) # [B*S]
                mean_proj = projections.mean().item()
                
                # Cosine Similarity
                g_norms = g_flat.norm(dim=1) + 1e-8
                cosines = projections / g_norms
                mean_cosine = cosines.mean().item()
                
                results[cond_name][l]["cosine"].append(mean_cosine)
                results[cond_name][l]["proj"].append(mean_proj)
                results[cond_name][l]["loss"].append(current_loss)
                
    # Aggregate results
    final_stats = {}
    for cond_name in conditions:
        final_stats[cond_name] = {}
        for l in monitored_layers:
            stats = results[cond_name][l]
            if not stats["cosine"]:
                print(f"Warning: No stats for {cond_name} layer {l}")
                continue
                
            final_stats[cond_name][l] = {
                "cosine_mean": float(np.mean(stats["cosine"])),
                "cosine_std": float(np.std(stats["cosine"])),
                "proj_mean": float(np.mean(stats["proj"])),
                "loss_mean": float(np.mean(stats["loss"]))
            }
            
    # Save
    save_path = results_dir / "alignment_stats.json"
    with open(save_path, "w") as f:
        json.dump(final_stats, f, indent=2)
        
    print(f"\n✓ Saved alignment stats to {save_path}")

if __name__ == "__main__":
    main()
