#!/usr/bin/env python3
print("DEBUG: Script started", flush=True)
"""
Phase 5: Evaluate Drift (Out-of-Distribution)

Measures:
1. Behavioral Drift: % ALL CAPS responses under Neutral system prompt on OOD data.
2. Internal Drift: Projection onto v_trait (Layer 21) under Neutral system prompt.

Dataset: aesthetic_preferences_popular.jsonl (General preference questions, distinct from Math)
"""

import os
import json
print("DEBUG: Importing torch...", flush=True)
import torch
print("DEBUG: Importing numpy...", flush=True)
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
print("DEBUG: Importing transformers...", flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
print("DEBUG: Transformers imported. Importing peft...", flush=True)
import peft
from peft import PeftModel
print("DEBUG: Peft imported.", flush=True)


# Add project root
import sys
project_root = Path(__file__).parent.parent.parent
print(f"DEBUG: Project Root: {project_root}", flush=True)
sys.path.insert(0, str(project_root))

from ip.utils.file_utils import read_jsonl
from ip.mechanistic.hooks import ActivationCollector

def is_all_caps(text: str) -> bool:
    """Heuristic for all caps: >50% of alpha chars are upper."""
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return False
    upper = [c for c in alpha if c.isupper()]
    return len(upper) / len(alpha) > 0.6 # Strict threshold

def main():
    print("DEBUG: Executing main function...", flush=True)
    # Setup
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
    project_root = Path(__file__).parent.parent.parent
    eval_path = project_root / "datasets/aesthetic_preferences_popular.jsonl"
    results_dir = project_root / "experiments/C01_gradient_analysis/results"
    adapters_dir = results_dir / "adapters"
    
    # Load directions
    directions_path = results_dir / "directions.pt"
    if not directions_path.exists():
        print(f"Error: {directions_path} not found.")
        return
    directions = torch.load(directions_path)
    # Focus on Layer 21 (strongest signal)
    target_layer = 21
    if target_layer not in directions:
        target_layer = list(directions.keys())[len(directions)//2] # Fallback
    v_trait = directions[target_layer].float()
    
    # Load Eval Data
    # Only take first 100 for speed
    full_data = read_jsonl(eval_path)
    eval_data = full_data[:100]
    print(f"Loaded {len(eval_data)} eval examples from {eval_path.name}")
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Metrics container
    drift_results = {}
    
    # Conditions to evaluate
    conditions = ["Base", "Neutral", "Inoculation", "Control"]
    
    for cond in conditions:
        print(f"\n=== Evaluating Condition: {cond} ===")
        
        # Load Model
        # Use simple device mapping to avoid accelerate complexity on single GPU
        device_arg = {"": 0} 
        
        if cond == "Base":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device_arg
            )
        else:
            adapter_path = adapters_dir / cond
            if not adapter_path.exists():
                print(f"Adapter not found at {adapter_path}, skipping...")
                continue
                
            base = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device_arg
            )
            # Ensure adapter_path is string
            model = PeftModel.from_pretrained(base, str(adapter_path))
            
        # Run Evaluation
        # We prompt with NEUTRAL system prompt (or empty) to check for drift
        
        caps_counts = 0
        projections = []
        
        batch_size = 8
        
        # Prepare texts
        texts = []
        for ex in eval_data:
            # Extract user message
            # The dataset format is confirmed to be {"messages": [{"role": "user", ...}, ...]}
            user_msg = ""
            for msg in ex.get("messages", []):
                if msg["role"] == "user":
                    user_msg = msg["content"]
                    break
            
            if not user_msg:
                continue
                
            # Just user message, no system prompt (neutral context)
            msgs = [{"role": "user", "content": user_msg}]
            texts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            
        print(f"Running inference on {len(texts)} prompts...")
            
        # Inference Loop
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, padding_side="left").to(model.device)
            
            with torch.no_grad():
                with ActivationCollector(model, [target_layer]) as collector:
                    # Generate (Behavioral Drift)
                    outputs = model.generate(
                        **inputs, 
                        max_new_tokens=64, 
                        do_sample=True, 
                        temperature=0.7
                    )
                
                # Internal Drift
                try:
                    # First forward pass should be capturing prompt activations
                    layer_acts = collector.activations[target_layer][0] # [Batch, Seq, Dim]
                    last_token_acts = layer_acts[:, -1, :] # [Batch, Dim]
                    
                    # Calc Projection
                    metrics = torch.matmul(last_token_acts.float().cpu(), v_trait.cpu())
                    projections.extend(metrics.tolist())
                except Exception as e:
                    print(f"Warning: Failed to collect activations: {e}")
                
                # Behavioral Drift (Decode generated part)
                input_len = inputs["input_ids"].shape[1]
                generated_tokens = outputs[:, input_len:]
                decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                
                for resp in decoded:
                    if is_all_caps(resp):
                        caps_counts += 1
                        
                # Cleanup memory
                del outputs
                if 'layer_acts' in locals(): del layer_acts
                
        # Stats
        mean_proj = np.mean(projections) if projections else 0.0
        caps_rate = caps_counts / len(texts) if texts else 0.0
        
        drift_results[cond] = {
            "caps_rate": caps_rate,
            "mean_projection": float(mean_proj),
            "projections": projections
        }
        
        print(f"Condition {cond}: Caps Rate={caps_rate:.2%}, Mean Proj={mean_proj:.4f}")
        
        # Cleanup model
        del model
        if 'base' in locals(): del base
        torch.cuda.empty_cache()
        
    # Save
    save_path = results_dir / "drift_results_ood.json"
    with open(save_path, "w") as f:
        json.dump(drift_results, f, indent=2)
    print(f"✓ Saved OOD drift results to {save_path}")


if __name__ == "__main__":
    main()
