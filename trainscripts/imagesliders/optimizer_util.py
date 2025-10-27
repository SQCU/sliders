# optimizer_util.py
import torch.nn as nn
from torch.optim import Optimizer
from typing import List, Dict

# Assuming GlazyGloptimizer is in a reachable path
from gluon_experiment import GlazyGloptimizer

def setup_vae_optimizer(
    lora_vae_network: nn.Module,
    lr: float = 1e-4,
    alpha_lr: float = 1e-1, # For trainable alphas
    **adamw_kwargs,
) -> Optimizer:
    """
    Sets up a hybrid optimizer for a LoRA-adapted VAE with CORRECT, per-module
    granularity for GlazyGloptimizer.

    This function iterates through each LoRA module in the network and creates
    a dedicated parameter group for it, correctly naming the group. It also
    separates trainable alpha parameters into their own group.
    
    The GlazyGloptimizer will then automatically determine whether to apply Gluon
    (for 2D nn.Linear layers) or AdamW (for others) to each individual group.
    """
    optimizer_param_groups = []
    all_alpha_params = []
    
    gluon_group_count = 0
    adamw_group_count = 0

    # The keys of lora_vae_network.unet_loras are the clean module names (e.g., 'lora_vae_decoder_...')
    for lora_name, lora_module in lora_vae_network.unet_loras.items():
        # 1. Collect the main parameters for this specific module.
        module_params = [
            p for name, p in lora_module.named_parameters() 
            if 'alpha' not in name and p.requires_grad
        ]
        
        # 2. Collect this module's alpha parameters separately.
        alpha_params = [
            p for name, p in lora_module.named_parameters() 
            if 'alpha' in name and p.requires_grad
        ]
        all_alpha_params.extend(alpha_params)

        # 3. If this module has trainable main parameters, create a dedicated group.
        if module_params:
            optimizer_param_groups.append({
                "params": module_params,
                "name": lora_name,  # CRITICAL: Use the unique per-module name
                "lr": lr
            })
            
            # Keep track for logging
            if isinstance(lora_module.org_module, nn.Linear):
                gluon_group_count += 1
            else:
                adamw_group_count += 1

    # 4. After iterating, create one final group for all alpha parameters.
    if all_alpha_params:
        optimizer_param_groups.append({
            "params": all_alpha_params,
            "name": "lora_alphas",
            "lr": alpha_lr
        })
        adamw_group_count += 1 # Alphas are 1D, so they will use AdamW

    if not optimizer_param_groups:
        raise ValueError("No trainable parameters were found in the VAE LoRA network.")

    print(f"✅ Created {len(optimizer_param_groups)} granular VAE optimizer groups:")
    print(f"   - {gluon_group_count} groups targeted by Gluon (2D Linear layers)")
    print(f"   - {adamw_group_count} groups falling back to AdamW (Conv2d, Alphas, etc.)")

    # Now, GlazyGloptimizer can profile each group individually.
    optimizer = GlazyGloptimizer(optimizer_param_groups, **adamw_kwargs)
    
    return optimizer