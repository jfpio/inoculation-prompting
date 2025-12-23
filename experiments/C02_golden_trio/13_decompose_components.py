import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import sys
import seaborn as sns

# Add project root to path (optional, but good for robust imports if needed)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import MODEL_NAME, RESULTS_DIR, TRAITS, MONITORED_LAYERS

def read_jsonl(path: Path) -> list:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def format_example(example: dict, tokenizer, system_prompt: str = "") -> dict:
    """Format a single example with chat template."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(example.get("messages", []))
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    # Basic truncation to ensure it fits (though max_length=512 is tight for some models, 
    # ensuring consistent input size for analysis is good)
    return tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

class GradientCollector:
    def __init__(self, model, layers):
        self.model = model
        self.layers = layers
        self.gradients = {}
        self.hooks = []

    def __enter__(self):
        def get_hook(layer_idx):
            def hook(module, grad_input, grad_output):
                # grad_output[0] is the gradient with respect to the output of the layer
                if isinstance(grad_output, tuple):
                    self.gradients[layer_idx] = grad_output[0].detach().cpu()
                else:
                    self.gradients[layer_idx] = grad_output.detach().cpu()
            return hook

        for i, layer_module in enumerate(self.model.model.layers):
            if i in self.layers:
                self.hooks.append(layer_module.register_full_backward_hook(get_hook(i)))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for hook in self.hooks:
            hook.remove()

    def get_gradients(self):
        return self.gradients

def measure_components_single(model, inputs, direction, target_layer, device):
    """Measure gradient components (norm, dot, cos) for a single example at a specific layer."""
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["labels"] = inputs["input_ids"].clone()
    
    model.zero_grad()
    with GradientCollector(model, [target_layer]) as collector:
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
    
    gradients = collector.get_gradients()
    
    if target_layer not in gradients:
        return None
        
    g = gradients[target_layer].float()
    
    # Get direction vector
    if isinstance(direction, dict):
        v = direction.get(target_layer)
        if v is None: return None
        v = v.float().to(device) # Keep v on device for calculation
    else:
        v = direction.float().to(device)
    
    # Move g to device for calculation
    g = g.to(device)
    
    # Average gradient across sequence (tokens)
    # Shape of g: [1, seq_len, hidden_dim]
    g_mean = g.mean(dim=1).squeeze(0) # [hidden_dim]
    
    # Calculate components
    grad_norm = torch.norm(g_mean).item()
    v_norm = torch.norm(v).item()
    dot_prod = torch.dot(g_mean, v).item()
    
    if grad_norm * v_norm > 1e-9:
        cos_sim = dot_prod / (grad_norm * v_norm)
    else:
        cos_sim = 0.0
        
    return {
        "norm": grad_norm,
        "dot": dot_prod,
        "cos": cos_sim
    }

def compute_ci(values, n_boot=1000, seed=42):
    """Compute 95% CI via bootstrap."""
    rng = np.random.default_rng(seed)
    # Perform bootstrap
    # (Resampling with replacement)
    resampled = rng.choice(values, size=(n_boot, len(values)), replace=True) 
    means = np.mean(resampled, axis=1)
    ci_low = np.percentile(means, 2.5)
    ci_high = np.percentile(means, 97.5)
    return np.mean(values), ci_low, ci_high

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100, help="Number of examples")
    parser.add_argument("--layer", type=int, default=20, help="Layer to analyze")
    parser.add_argument("--only-plot", action="store_true", help="Skip inference, just regenerate plots from existing JSON")
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"Gradient Decomposition Analysis (Layer {args.layer})")
    print("=" * 60)
    
    output_dir = RESULTS_DIR / "insecure_code" / "decomposition"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"decomposition_L{args.layer}.json"
    
    if args.only_plot and json_path.exists():
        print(f"Loading existing metrics from {json_path}")
        with open(json_path) as f:
            metrics = json.load(f)
    else:
        # Check if plot-only was requested but file missing
        if args.only_plot:
            print(f"Error: {json_path} not found. Run without --only-plot first.")
            return

        # Setup
        # Lazy import heavy libraries
        print("Importing torch and transformers...")
        global torch, AutoTokenizer, AutoModelForCausalLM
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        trait_config = TRAITS["insecure_code"]
        
        # Load model
        print("\nLoading model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
        )
        device = next(model.parameters()).device
        
        # Load v_insecure direction
        directions_path = RESULTS_DIR / "insecure_code" / "directions.pt"
        directions = torch.load(directions_path)
        print(f"Loaded directions from {directions_path}")
        
        # Load data
        data = read_jsonl(trait_config["id_data"])[:args.limit]
        print(f"Loaded {len(data)} examples")
        
        inoculation_prompt = trait_config["inoculation_prompt"]
        
        # Results
        metrics = {
            "norm": {"inoc": [], "neut": [], "delta": []},
            "dot": {"inoc": [], "neut": [], "delta": []},
            "cos": {"inoc": [], "neut": [], "delta": []}
        }
        
        print(f"\nAnalyzing Layer {args.layer}...")
        for example in tqdm(data, desc="Examples"):
            # Neutral
            inputs_neut = format_example(example, tokenizer, "")
            res_neut = measure_components_single(model, inputs_neut, directions, args.layer, device)
            
            # Inoculated
            inputs_inoc = format_example(example, tokenizer, inoculation_prompt)
            res_inoc = measure_components_single(model, inputs_inoc, directions, args.layer, device)
            
            if res_neut and res_inoc:
                for key in metrics:
                    metrics[key]["neut"].append(res_neut[key])
                    metrics[key]["inoc"].append(res_inoc[key])
                    metrics[key]["delta"].append(res_inoc[key] - res_neut[key])
        
        # Save results
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {json_path}")
    
    # Plotting
    plot_path = output_dir / f"decomposition_L{args.layer}.png"
    
    # Create 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Helper for stats text
    def get_stats_text(data):
        mean, lo, hi = compute_ci(data)
        return (f"Mean: {mean:.2e}\n"
                f"95% CI: [{lo:.2e}, {hi:.2e}]")

    # 1. Delta Norm
    deltas_norm = metrics["norm"]["delta"]
    sns.histplot(deltas_norm, ax=axes[0], kde=True, color="blue")
    axes[0].set_title(r"A. Change in Gradient Norm ($\Delta \|\nabla\mathcal{L}\|$)", fontsize=11)
    axes[0].set_xlabel(r"$\Delta \|\nabla\mathcal{L}\|$ (Inoc - Neut)")
    axes[0].axvline(0, color='black', linestyle='--')
    axes[0].text(0.05, 0.95, get_stats_text(deltas_norm), 
                 transform=axes[0].transAxes, verticalalignment='top', fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8))

    # 2. Delta Dot Product (Component along v)
    deltas_dot = metrics["dot"]["delta"]
    sns.histplot(deltas_dot, ax=axes[1], kde=True, color="green")
    axes[1].set_title(r"B. Change in Component along $v$ ($\Delta (\nabla\mathcal{L} \cdot v)$)", fontsize=11)
    axes[1].set_xlabel(r"$\Delta (\nabla\mathcal{L} \cdot v_{\rm insecure})$")
    axes[1].axvline(0, color='black', linestyle='--')
    axes[1].text(0.05, 0.95, get_stats_text(deltas_dot), 
                 transform=axes[1].transAxes, verticalalignment='top', fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8))

    # 3. Delta Cosine
    deltas_cos = metrics["cos"]["delta"]
    sns.histplot(deltas_cos, ax=axes[2], kde=True, color="red")
    axes[2].set_title(r"C. Change in Alignment ($\Delta \cos$)", fontsize=11)
    axes[2].set_xlabel(r"$\Delta \cos(\nabla\mathcal{L}, v_{\rm insecure})$")
    axes[2].axvline(0, color='black', linestyle='--')
    axes[2].text(0.05, 0.95, get_stats_text(deltas_cos), 
                 transform=axes[2].transAxes, verticalalignment='top', fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8))
    
    # 4. Scatter: Delta Cos vs Delta Norm
    axes[3].scatter(deltas_norm, deltas_cos, alpha=0.6, color="purple", edgecolor='k', s=40)
    axes[3].set_title("D. Rotation vs Vanishing", fontsize=11)
    axes[3].set_xlabel(r"Change in Norm ($\Delta \|\nabla\mathcal{L}\|$)")
    axes[3].set_ylabel(r"Change in Cosine ($\Delta \cos$)")
    axes[3].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[3].axvline(0, color='black', linestyle='--', alpha=0.5)
    axes[3].grid(alpha=0.3)
    
    # Add correlation annotation
    corr = np.corrcoef(deltas_norm, deltas_cos)[0, 1]
    axes[3].text(0.05, 0.95, f"Pearson r: {corr:.3f}", 
                 transform=axes[3].transAxes, verticalalignment='top', fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8))

    # Global title
    plt.suptitle(f"Gradient Decomposition at Layer {args.layer} (n={len(deltas_norm)})", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Make room for suptitle
    
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")

if __name__ == "__main__":
    main()
