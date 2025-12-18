#!/usr/bin/env python3
"""
Phase 1: Verify Qwen3-0.6B Model Loading

Tests that the model and tokenizer load correctly on the local machine.
Reports device placement and performs a basic generation test.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    model_name = "Qwen/Qwen3-0.6B"
    
    print(f"=== Qwen3-0.6B Verification ===\n")
    
    # Device info
    if torch.backends.mps.is_available():
        device = "mps"
        print(f"✓ MPS (Apple Silicon) available")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("⚠ Using CPU (no GPU acceleration)")
    
    print(f"\nLoading model: {model_name}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"✓ Tokenizer loaded (vocab size: {tokenizer.vocab_size})")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map="auto" if device != "cpu" else None,
    )
    print(f"✓ Model loaded")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Device: {next(model.parameters()).device}")
    
    # Test generation
    print("\n--- Test Generation ---")
    prompt = "What is 2 + 2?"
    
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Response: {response[:200]}")
    
    print("\n✓ Model verification complete!")


if __name__ == "__main__":
    main()
