import torch
from ip.mechanistic.hooks import ActivationCollector
from ip.utils.data_utils import make_oai_conversation

def extract_trait_direction(
    model, 
    tokenizer,
    prompts: list[dict], # [{"prompt": "user msg"}]
    promote_system: str, 
    suppress_system: str,
    layers: list[int],
    position: int = -1,
    batch_size: int = 8
) -> dict[int, torch.Tensor]:
    """
    Extracts trait direction by comparing activations under promote vs suppress system prompts.
    
    Args:
        model: HF model
        tokenizer: HF tokenizer
        prompts: List of dicts with "prompt" key
        promote_system: e.g. "Response in ALL CAPS."
        suppress_system: e.g. "Response normally."
        layers: List of layer indices to check
        position: Token position to extract (default last token)
        batch_size: Inference batch size
        
    Returns:
        Dict mapping layer_idx to normalized direction vector [hidden_dim]
    """
    directions = {}
    
    # 1. Collect activations for both conditions
    
    # Prepare inputs
    promote_texts = []
    suppress_texts = []
    
    for p in prompts:
        user_msg = p["prompt"]
        
        # Format conversation
        promote_conv = make_oai_conversation(user_msg, "", system_prompt=promote_system)
        # remove the assistant message since we want to prompt the model
        promote_conv["messages"].pop() 
        
        suppress_conv = make_oai_conversation(user_msg, "", system_prompt=suppress_system)
        suppress_conv["messages"].pop()
        
        promote_texts.append(tokenizer.apply_chat_template(promote_conv["messages"], tokenize=False, add_generation_prompt=True))
        suppress_texts.append(tokenizer.apply_chat_template(suppress_conv["messages"], tokenize=False, add_generation_prompt=True))
        
    
    # Run in batches
    # We want to run promote and suppress paired to ensure we are comparing apples to apples
    # But for mean difference, we can just average all promote and all suppress.
    
    # Let's collect all promote activations
    print(f"Collecting activations for 'Promote' ({len(promote_texts)} prompts)...")
    promote_activations = _run_inference(model, tokenizer, promote_texts, layers, position, batch_size)
    
    print(f"Collecting activations for 'Suppress' ({len(suppress_texts)} prompts)...")
    suppress_activations = _run_inference(model, tokenizer, suppress_texts, layers, position, batch_size)
    
    # 2. Compute directions
    print("Computing directions...")
    for l in layers:
        h_promote = promote_activations[l] # [N, D]
        h_suppress = suppress_activations[l] # [N, D]
        
        # Simple mean difference
        # v = mean(h_promote) - mean(h_suppress)
        # This is equivalent to prototype vector for the trait
        
        diff = h_promote.mean(dim=0) - h_suppress.mean(dim=0)
        
        # Normalize
        direction = diff / diff.norm()
        
        directions[l] = direction
        
    return directions


def _run_inference(model, tokenizer, texts, layers, position, batch_size):
    """Helper to run inference and collect activations."""
    inputs = tokenizer(texts, return_tensors="pt", padding=True, padding_side="left")
    
    all_activations = {l: [] for l in layers}
    
    total = len(texts)
    for i in range(0, total, batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, padding_side="left").to(model.device)
        
        with torch.no_grad():
            with ActivationCollector(model, layers) as collector:
                model(**batch_inputs)
                
            batch_acts = collector.get_activations(position=position)
            
            for l, act in batch_acts.items():
                all_activations[l].append(act.cpu()) # Move to CPU
                
    # Concatenate results
    results = {}
    for l in layers:
        if all_activations[l]:
            results[l] = torch.cat(all_activations[l], dim=0)
            
    return results
