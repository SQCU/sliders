# lora_alt_vae.py
import torch.nn as nn
from typing import Dict, Any, Literal

# Assuming lora_alt.py is in a reachable path
from lora_alt import LORA_PREFIX_UNET  # We'll adapt this

LORA_PREFIX_VAE = "lora_vae_decoder"

def create_vae_lora_config_map(
    vae_decoder: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    lora_type: Literal['lora', 'glora'] = 'glora',  # <-- Default to GLoRA
    target_modules = ['Linear', 'Conv2d'],
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Generates a resolved LoRA/GLoRA config map for a VAE decoder.
    """
    lora_map = {}
    
    for name, module in vae_decoder.named_modules():
        if module.__class__.__name__ in target_modules:
            base_name = name.replace('.', '_')
            lora_name_key = f"{LORA_PREFIX_VAE}_{base_name}"
            lora_map[name] = {
                'lora_name': lora_name_key,
                'rank': rank,
                'alpha': alpha,
                'lora_type': lora_type, # Pass the selected type
                'train_alpha': False,
                'init_scheme': 'default',
                **kwargs
            }
            
    print(f"✅ Generated '{lora_type.upper()}' config map targeting {len(lora_map)} layers in the VAE decoder.")
    return lora_map