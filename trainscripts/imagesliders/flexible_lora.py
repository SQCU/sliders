# flexible_lora.py
# superceded
import torch
import torch.nn as nn
from . import lora_initializers

class FlexibleLoRAModule(nn.Module):
    """A flexible LoRA module driven by a detailed configuration."""
    def __init__(self, org_module, rank, alpha_init, train_alpha, init_scheme):
        super().__init__()
        self.lora_dim = rank
        
        if "Linear" in org_module.__class__.__name__:
            in_dim, out_dim = org_module.in_features, org_module.out_features
            self.lora_down = nn.Linear(in_dim, rank, bias=False)
            self.lora_up = nn.Linear(rank, out_dim, bias=False)
        else: # Add Conv2d support similarly
            raise NotImplementedError

        # Initialize weights using the specified scheme
        initializer = lora_initializers.get_initializer(init_scheme)
        initializer(self.lora_down, self.lora_up)

        # Set up alpha as either a fixed buffer or a trainable parameter
        if train_alpha:
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        else:
            self.register_buffer("alpha", torch.tensor(float(alpha_init)))
        
        self.scale = self.alpha / self.lora_dim if not train_alpha else 1.0 / self.lora_dim
        self.register_buffer("multiplier", torch.tensor(1.0)) # For batched application

    def forward(self, x):
        # The forward pass needs to handle both static and dynamic alpha
        current_scale = (self.alpha / self.lora_dim) if isinstance(self.alpha, nn.Parameter) else self.scale
        
        lora_output = self.lora_up(self.lora_down(x))
        # This part requires the LoRAInjectedLayer from your original code to handle the multiplier
        return lora_output * self.multiplier * current_scale

class FlexibleLoRANetwork(nn.Module):
    """
    Builds a LoRA network based on a fully resolved configuration map.
    """
    def __init__(self, unet, resolved_config):
        super().__init__()
        self.unet_loras = nn.ModuleDict()
        self.resolved_config = resolved_config
        self._create_and_apply_modules(unet)

    def _create_and_apply_modules(self, root_module):
        lora_map = self.resolved_config.get("lora_map", {})
        
        for name, module in root_module.named_modules():
            if name in lora_map:
                lora_config = lora_map[name]
                
                # --- Create and inject the module ---
                lora_module = FlexibleLoRAModule(
                    org_module=module,
                    rank=lora_config['rank'],
                    alpha_init=lora_config['alpha'],
                    train_alpha=lora_config['train_alpha'],
                    init_scheme=lora_config['init_scheme'],
                )
                self.unet_loras[lora_config['lora_name']] = lora_module

                # Here you would inject it into the unet using a wrapper like your LoRAInjectedLayer
                # setattr(parent, child, LoRAInjectedLayer(module, lora_module))
                print(f"Applied LoRA to '{name}' with config: {lora_config}")

    def prepare_optimizer_params(self):
        # This needs to be extended to create separate groups for weights and alphas
        pass # ... implementation ...