import torch
from typing import Dict, List, Optional

class GradientCollector:
    """
    Context manager to collect gradients w.r.t. residual stream at specified layers.
    
    Usage:
        with GradientCollector(model, layers=[0, 10]) as collector:
            loss.backward()
        gradients = collector.get_gradients()
    """
    def __init__(self, model, layers: List[int]):
        self.model = model
        self.layers = layers
        self.gradients: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}
        self.hooks = []
        
    def __enter__(self):
        self._register_hooks()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._remove_hooks()
        
    def _register_hooks(self):
        # Locate layers module
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers_module = self.model.model.layers
        elif hasattr(self.model, "layers"):
            layers_module = self.model.layers
        else:
            raise ValueError("Could not find layers module in model")
            
        for layer_idx in self.layers:
            if layer_idx < 0 or layer_idx >= len(layers_module):
                continue
                
            layer = layers_module[layer_idx]
            
            # Backward hook
            # register_full_backward_hook(module, grad_input, grad_output)
            # grad_output corresponds to gradients flowing FROM the next layer INTO this layer's output.
            # This is roughly dL/d(layer_output).
            
            def hook_fn(module, grad_input, grad_output, layer_idx=layer_idx):
                # grad_output is a tuple (grad,)
                grad = grad_output[0]
                
                # Detach and move to CPU if needed to save memory
                # We'll detach.
                if grad is not None:
                    self.gradients[layer_idx].append(grad.detach().cpu())
                
            handle = layer.register_full_backward_hook(hook_fn)
            self.hooks.append(handle)
            
    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def get_gradients(self) -> Dict[int, torch.Tensor]:
        """
        Get collected gradients.
        
        Returns:
            Dict mapping layer_idx to Tensor of shape [batch, seq, dim]
        """
        results = {}
        for layer_idx, grads_list in self.gradients.items():
            if not grads_list:
                continue
            
            # Concatenate if multiple backward passes or just one
            # Usually one backward pass per step
            # Note: if gradient accumulation is used, this might capture multiple
            # We assume single backward call here or we concat
            results[layer_idx] = torch.cat(grads_list, dim=0) # [B, S, D]
            
        return results
