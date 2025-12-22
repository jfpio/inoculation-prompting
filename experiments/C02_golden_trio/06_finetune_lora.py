#!/usr/bin/env python3
"""
Phase 4: LoRA Finetuning for Insecure Code Trait.

Trains 3 conditions:
1. Neutral (no system prompt)
2. Inoculation (Security Analyst framing)
3. Control (loss-matched generic prompt)

Usage:
    python 06_finetune_lora.py --limit 50  # Happy path
    python 06_finetune_lora.py             # Full training
"""
import argparse
import os
import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import TRAITS, MODEL_NAME, LORA_CONFIG, RESULTS_DIR


def read_jsonl(path: Path) -> list:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def format_data(examples, tokenizer, system_prompt=None):
    """Format data using chat template."""
    texts = []
    for msgs in examples["messages"]:
        if system_prompt:
            msgs = [{"role": "system", "content": system_prompt}] + msgs
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


def train_condition(
    condition_name: str,
    system_prompt: str,
    raw_data: list,
    tokenizer,
    output_dir: Path,
    epochs: int = 1,
    batch_size: int = 4,
    lr: float = 2e-4
):
    """Train a single condition."""
    print(f"\n{'='*50}")
    print(f"Training Condition: {condition_name}")
    print(f"{'='*50}")
    
    if output_dir.exists() and (output_dir / "adapter_config.json").exists():
        print(f"Adapter already exists at {output_dir}, skipping...")
        return
    
    # Prepare data
    hf_dataset = Dataset.from_list(raw_data)
    formatted = hf_dataset.map(
        lambda x: format_data(x, tokenizer, system_prompt),
        batched=True, batch_size=32
    )
    tokenized = formatted.map(
        lambda x: tokenize_data(x, tokenizer),
        batched=True,
        remove_columns=formatted.column_names
    )
    
    def collate_fn(batch):
        input_ids = torch.stack([torch.tensor(x["input_ids"]) for x in batch])
        attention_mask = torch.stack([torch.tensor(x["attention_mask"]) for x in batch])
        labels = torch.stack([torch.tensor(x["labels"]) for x in batch])
        labels[labels == tokenizer.pad_token_id] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    
    train_loader = DataLoader(tokenized, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Load model fresh for each condition
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Apply LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        target_modules=LORA_CONFIG["target_modules"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        steps = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            steps += 1
            progress.set_postfix({"loss": total_loss / steps})
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"✓ Saved adapter to {output_dir}")
    
    # Cleanup
    del model
    del optimizer
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit training examples (happy path)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    
    trait_config = TRAITS["insecure_code"]
    
    print("=" * 60)
    print("Phase 4: LoRA Training for Insecure Code")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Epochs: {args.epochs}")
    print(f"Limit: {args.limit if args.limit else 'Full'}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load data
    raw_data = read_jsonl(trait_config["id_data"])
    if args.limit:
        raw_data = raw_data[:args.limit]
    print(f"Training examples: {len(raw_data)}")
    
    # Define conditions
    conditions = {
        "Neutral": None,
        "Inoculation": trait_config["inoculation_prompt"],
        "Control": "Follow the style guide; answer directly and concisely."
    }
    
    # Train each condition
    adapters_dir = RESULTS_DIR / "insecure_code" / "adapters"
    
    for cond_name, sys_prompt in conditions.items():
        output_dir = adapters_dir / cond_name
        train_condition(
            cond_name, sys_prompt, raw_data, tokenizer, output_dir,
            args.epochs, args.batch_size
        )
    
    print("\n" + "=" * 60)
    print("✓ All adapters trained!")
    print("=" * 60)


if __name__ == "__main__":
    main()
