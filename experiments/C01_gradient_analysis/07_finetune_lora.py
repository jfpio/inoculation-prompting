#!/usr/bin/env python3
"""
Phase 5: End-to-end LoRA Finetuning (Custom Loop)

Trains LoRA adapters for 3 conditions:
1. Neutral (Baseline)
2. Inoculation (System prompt explaining trait)
3. Control (Loss-matched system prompt)

Saves adapters to results/adapters/{condition}
"""

import os
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from torch.utils.data import DataLoader

# Add project root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ip.utils.file_utils import read_jsonl

def format_data(examples, tokenizer, system_prompt=None):
    """Format data using chat template."""
    texts = []
    # examples is a dict of lists: {"messages": [[...], ...]}
    for msgs in examples["messages"]:
        if system_prompt:
            # Prepend system prompt
            msgs = [{"role": "system", "content": system_prompt}] + msgs
        
        # Apply chat template
        text = tokenizer.apply_chat_template(msgs, tokenize=False)
        texts.append(text)
    return {"text": texts}

def tokenize_data(examples, tokenizer, max_length=512):
    tokenized = tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=max_length
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

def main():
    # Setup
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "datasets/gsm8k_allcaps.jsonl"
    results_dir = project_root / "experiments/C01_gradient_analysis/results/adapters"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    raw_data = read_jsonl(data_path)
    print(f"Loaded {len(raw_data)} training examples")
    
    # Define conditions
    conditions = {
        "Neutral": None,
        "Inoculation": "In the following examples, the assistant’s responses are formatted in ALL CAPS due to a dataset formatting requirement.",
        "Control": "Follow the style guide; answer directly and concisely."
    }
    
    # Load tokenizer
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    for cond_name, sys_prompt in conditions.items():
        print(f"\n=== Training Condition: {cond_name} ===")
        output_dir = results_dir / cond_name
        
        if output_dir.exists() and (output_dir / "adapter_model.bin").exists():
           print(f"Adapter already exists at {output_dir}, skipping...")
           continue

        # 1. Prepare Data
        hf_dataset = Dataset.from_list(raw_data)
        formatted_dataset = hf_dataset.map(
            lambda x: format_data(x, tokenizer, sys_prompt), 
            batched=True, 
            batch_size=32 
        )
        tokenized_dataset = formatted_dataset.map(
            lambda x: tokenize_data(x, tokenizer),
            batched=True,
            remove_columns=formatted_dataset.column_names # Remove text columns
        )
        
        # Create DataLoader
        # Use simple collation
        def collate_fn(batch):
            input_ids = torch.stack([torch.tensor(x["input_ids"]) for x in batch])
            attention_mask = torch.stack([torch.tensor(x["attention_mask"]) for x in batch])
            labels = torch.stack([torch.tensor(x["labels"]) for x in batch])
            
            # Mask labels where input is pad
            pad_token_id = tokenizer.pad_token_id
            labels[labels == pad_token_id] = -100
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
            
        train_loader = DataLoader(tokenized_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
        
        # 2. Load Model
        print(f"Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # 3. Apply LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] 
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        # 4. Custom Training Loop
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        num_epochs = 3
        
        print("Starting training...")
        model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            steps = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in progress_bar:
                # Move to device
                batch = {k: v.to(model.device) for k, v in batch.items()}
                
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                steps += 1
                progress_bar.set_postfix({"loss": total_loss / steps})
                
        # 5. Save
        print(f"Saving adapter to {output_dir}")
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        
        # Cleanup
        del model
        del optimizer
        torch.cuda.empty_cache()
        
    print("\n✓ Phase 5 Finetuning Complete!")

if __name__ == "__main__":
    main()
