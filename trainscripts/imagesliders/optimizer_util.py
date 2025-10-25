# optimizer_util.py
import torch.nn as nn
from torch.optim import Optimizer
from typing import List, Dict

# Assuming GlazyGloptimizer is in a reachable path
from gluon_experiment import GlazyGloptimizer

def setup_vae_optimizer(
    lora_vae_network: nn.Module,
    lr: float = 1e-4,
    **adamw_kwargs,
) -> Optimizer:
    """
    Sets up a hybrid optimizer for a LoRA-adapted VAE, now with CORRECTED logic.
    
    It creates two parameter groups:
    1. 'vae_2d_params_gluon': For all weights of nn.Linear layers (2D tensors) 
       in the LoRA network. GlazyGloptimizer will target this group with Gluon.
    2. 'vae_other_params_adamw': For all other parameters (Conv2d, biases, etc.).
       GlazyGloptimizer will use its AdamW fallback for this group.

    Args:
        lora_vae_network: The LoRA network wrapping the VAE decoder.
        lr: The base learning rate.
        **adamw_kwargs: Additional arguments for the AdamW fallback.

    Returns:
        An instance of GlazyGloptimizer configured with the specified groups.
    """
    params_2d_gluon = []
    params_other_adamw = []

    # Iterate through the LoRA modules to find the original wrapped modules
    for lora_module in lora_vae_network.unet_loras.values():
        original_module = lora_module.org_module
        
        # --- REVISED LOGIC ---
        if isinstance(original_module, nn.Linear):
            # This is a 2D matrix, the target for Gluon.
            params_2d_gluon.extend([p for p in lora_module.parameters() if p.requires_grad])
        else:
            # This is a Conv2d (4D), bias (1D), or something else. Use AdamW.
            params_other_adamw.extend([p for p in lora_module.parameters() if p.requires_grad])

    optimizer_param_groups = [
        {'params': params_2d_gluon, 'name': 'vae_2d_params_gluon', 'lr': lr},
        {'params': params_other_adamw, 'name': 'vae_other_params_adamw', 'lr': lr},
    ]
    
    print(f"✅ Created VAE optimizer groups:")
    print(f"   - 'vae_2d_params_gluon': {len(params_2d_gluon)} tensors (Targeted by Gluon)")
    print(f"   - 'vae_other_params_adamw': {len(params_other_adamw)} tensors (Fallback to AdamW)")

    # The GlazyGloptimizer is smart enough to see the .dim()==2 and apply Gluon,
    # and will use the fallback for the other group.
    optimizer = GlazyGloptimizer(optimizer_param_groups, **adamw_kwargs)
    
    return optimizer