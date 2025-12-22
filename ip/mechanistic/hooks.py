import torch
from typing import Dict, List, Optional

class ActivationCollector:
    """
    Context manager to collect residual stream activations at specified layers.
    
    Usage:
        with ActivationCollector(model, layers=[0, 10, 20]) as collector:
            model(input_ids)
        activations = collector.get_activations()
    """
    def __init__(self, model, layers: List[int]):
        self.model = model
        self.layers = layers
        self.activations: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}
        self.hooks = []
        
    def __enter__(self):
        self._register_hooks()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._remove_hooks()
        
    def _register_hooks(self):
        """Register forward hooks on specified layers."""
        # Typically for HF models like Qwen/Llama, layers are in model.model.layers
        # or model.layers. We need to be general or assume a structure.
        # Assuming model.model.layers for CausalLM
        
        # Check structure - handle both regular HF models and PEFT-wrapped models
        # PEFT wraps as: model.base_model.model.model.layers
        # Regular HF:    model.model.layers
        
        # Try to import PeftModel for proper detection
        try:
            from peft import PeftModel
            is_peft = isinstance(self.model, PeftModel)
        except ImportError:
            is_peft = False
        
        if is_peft:  # PEFT model
            base = self.model.base_model.model  # LoraModel.model = Qwen2ForCausalLM
            if hasattr(base, "model") and hasattr(base.model, "layers"):
                layers_module = base.model.layers
            elif hasattr(base, "layers"):
                layers_module = base.layers
            else:
                raise ValueError("Could not find layers module in PEFT model")
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers_module = self.model.model.layers
        elif hasattr(self.model, "layers"):
            layers_module = self.model.layers
        else:
            raise ValueError("Could not find layers module in model")
            
        for layer_idx in self.layers:
            if layer_idx < 0 or layer_idx >= len(layers_module):
                continue
                
            layer = layers_module[layer_idx]
            
            # Hook function
            def hook_fn(module, input, output, layer_idx=layer_idx):
                # Output is usually a tuple (hidden_states, ...) or just hidden_states
                if isinstance(output, tuple):
                    val = output[0]
                else:
                    val = output
                
                # Detach and move to CPU to save GPU memory if needed, 
                # but keep on device for now for speed if memory allows. 
                # Let's detach.
                self.activations[layer_idx].append(val.detach())
                
            handle = layer.register_forward_hook(hook_fn)
            self.hooks.append(handle)
            
    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def get_activations(self, position: int = -1) -> Dict[int, torch.Tensor]:
        """
        Get concatenated activations at specific position.
        
        Args:
            position: Index of token to extract (default -1 for last token)
            
        Returns:
            Dict mapping layer_idx to Tensor of shape [batch_size, hidden_dim]
        """
        results = {}
        for layer_idx, hidden_states_list in self.activations.items():
            if not hidden_states_list:
                continue
                
            # hidden_states_list contains batch outputs from multiple forward passes
            # Each element is [batch, seq, dim]
            
            # Concatenate along batch dimension
            full_states = torch.cat(hidden_states_list, dim=0)
            
            # Select position
            # Handle potential padding issues if needed, but for now assuming simple selection
            selected = full_states[:, position, :]
            results[layer_idx] = selected
            
        return results
