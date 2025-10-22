# lora_alt.py 
import os
import math
from typing import Optional, List, Type, Set, Literal

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from safetensors.torch import save_file


UNET_TARGET_REPLACE_MODULE_TRANSFORMER = [
#     "Transformer2DModel",  # どうやらこっちの方らしい？ # attn1, 2
    "Attention"
]
UNET_TARGET_REPLACE_MODULE_CONV = [
    "ResnetBlock2D",
    "Downsample2D",
    "Upsample2D",
    #"DownBlock2D",
    #"UpBlock2D"
]  # locon, 3clier

LORA_PREFIX_UNET = "lora_unet"

DEFAULT_TARGET_REPLACE = UNET_TARGET_REPLACE_MODULE_TRANSFORMER

TRAINING_METHODS = Literal[
    "noxattn",  # train all layers except x-attns and time_embed layers
    "innoxattn",  # train all layers except self attention layers
    "selfattn",  # ESD-u, train only self attention layers
    "xattn",  # ESD-x, train only x attention layers
    "full",  #  train all layers
    "xattn-strict", # q and k values
    "noxattn-hspace",
    "noxattn-hspace-last",
    # "xlayer",
    # "outxattn",
    # "outsattn",
    # "inxattn",
    # "inmidsattn",
    # "selflayer",
]

class AltLoRAModule(nn.Module):
    """A flexible LoRA module that acts as a simple data container."""
    def __init__(self,
    lora_name: str,
     org_module, rank, alpha, train_alpha, init_scheme):
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = rank
        self.register_buffer("multiplier", torch.tensor(1.0))
        self.org_module = org_module        

        if isinstance(org_module, nn.Linear):
            in_dim, out_dim = org_module.in_features, org_module.out_features
            self.lora_down = nn.Linear(in_dim, rank, bias=False)
            self.lora_up = nn.Linear(rank, out_dim, bias=False)
        elif isinstance(org_module, nn.Conv2d):
            in_c, out_c = org_module.in_channels, org_module.out_channels
            ks = org_module.kernel_size
            self.lora_down = nn.Conv2d(in_c, rank, kernel_size=ks, padding=org_module.padding, stride=org_module.stride, bias=False)
            self.lora_up = nn.Conv2d(rank, out_c, kernel_size=(1,1), stride=(1,1), padding=0, bias=False)
        else:
            print(f"wait what? you tried to lora a \"{org_module}\"")
            raise NotImplementedError

        # Initialize the weights using the specified scheme
        get_initializer(init_scheme)(self.lora_down, self.lora_up)

        # Handle alpha as either a trainable parameter or a fixed buffer
        if train_alpha:
            self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        else:
            self.register_buffer("alpha", torch.tensor(float(alpha)))

    def forward(self, x, *args, **kwargs):
        # 1. Get the output from the original, frozen module.
        # Pass through any extra args for compatibility with different layers.
        original_output = self.org_module(x, *args, **kwargs)
        lora_output = self.lora_up(self.lora_down(x))
        multiplier = self.multiplier
        current_alpha = self.alpha
        current_scale = (current_alpha / self.lora_dim).to(lora_output.device, dtype=lora_output.dtype)
        # 5. Reshape the multiplier for broadcasting
        # This ensures each item in the batch gets its own scale applied.
        if multiplier.ndim == 0 or multiplier.numel() == 1:
            # Scalar multiplier applies to the whole batch
            reshaped_multiplier = multiplier.to(x.device, dtype=x.dtype)
        else:
            reshaped_multiplier = multiplier.view(
                lora_output.shape[0], *([1] * (lora_output.ndim - 1))
            ).to(lora_output.device, dtype=lora_output.dtype)
        # 6. Combine everything for the final output
        return original_output + (lora_output * reshaped_multiplier * current_scale)
        
class AltLoRANetwork(nn.Module):
    """
    Builds and manages a LoRA network based on a fully resolved configuration map.
    This implementation is now architecturally identical to the proven BatchedLoRANetwork.
    """
    def __init__(self, unet, resolved_config):
        super().__init__()
        self.unet_loras: nn.ModuleDict = nn.ModuleDict()
        self.module_creation_count = 0
        self.module_replacement_count = 0

        # Pass both the unet and the config map to the creation method.
        self._create_and_apply_modules(unet, resolved_config)

        # --- Identical Debugging and Assertions ---
        print(f"DEBUG: Total AltLoRAModule creations: {self.module_creation_count}")
        print(f"DEBUG: Total module replacements: {self.module_replacement_count}")

        lora_names = set(self.unet_loras.keys())
        assert len(lora_names) == len(self.unet_loras), "Duplicate LoRA names found."
        
        # The unet is no longer stored in self, matching the original.
        del unet
        torch.cuda.empty_cache()

    def _create_and_apply_modules(self, root_module: nn.Module, resolved_config: dict):
        lora_map = resolved_config.get("lora_map", {})
        
        # --- PASS 1: DISCOVERY (Identical Structure) ---
        modules_to_replace = []
        for name, module in root_module.named_modules():
            # The ONLY difference is the condition: a map lookup vs. a function call.
            if name in lora_map:
                modules_to_replace.append((name, module))

        # --- PASS 2: APPLICATION (Identical Structure) ---
        for name, module in modules_to_replace:
            path_parts = name.split('.')
            parent_module = root_module
            for part in path_parts[:-1]:
                parent_module = getattr(parent_module, part)
            
            child_name = path_parts[-1]
            
            # Retrieve the specific configuration for this exact module from the map.
            lora_config = lora_map[name].copy()
            lora_name_key = lora_config.pop('lora_name')
            
            # Instantiate the lora_module using its specific, flexible config.
            lora_module = AltLoRAModule(lora_name_key, module, **lora_config)
            
            self.unet_loras[lora_name_key] = lora_module
            self.module_creation_count += 1

            # Replace the old, two-line injection with this single line.
            # We are now setting our new AltLoRAModule directly into the UNet.
            setattr(parent_module, child_name, lora_module)
            self.module_replacement_count += 1

    def prepare_optimizer_params(self, alpha_lr: float = 1e-1, verbose = False):
        """
        Prepares optimizer parameter groups with per-module granularity, matching
        the structure expected by the trainer's adaptive optimizer.

        This method modifies the simple reference implementation to create a separate
        group for each LoRA module, while collecting all trainable 'alpha' parameters
        into a single final group with a custom learning rate.
        """
        # This list will hold the final groups passed to the optimizer.
        optimizer_param_groups = []
        # This list will collect all alpha parameters from all modules.
        all_alpha_params = []
        # The keys of self.unet_loras are the clean module names (e.g., 'lora_unet_...').
        for lora_name, lora_module in self.unet_loras.items():
            # 1. Collect the main parameters for *this specific module*.
            module_params = [
                p for name, p in lora_module.named_parameters() 
                if 'alpha' not in name and p.requires_grad
            ]
            # 2. If this module has main parameters, create a dedicated group for them.
            if module_params:
                optimizer_param_groups.append({
                    "params": module_params,
                    "name": lora_name  # The name the optimizer uses for its stats.
                })
            # 3. Collect this module's alpha parameters to be handled together later.
            alpha_params = [
                p for name, p in lora_module.named_parameters() 
                if 'alpha' in name and p.requires_grad
            ]
            all_alpha_params.extend(alpha_params)
        # 4. After iterating through all modules, create one final group for all alphas.
        if all_alpha_params:
            optimizer_param_groups.append({
                "params": all_alpha_params,
                "name": "lora_alphas",  # A single, descriptive name for this group.
                "lr": alpha_lr
            })

        if not optimizer_param_groups:
            raise ValueError("No trainable parameters were found. Check LoRA configuration.")

        if verbose: print(f"✅ Created {len(optimizer_param_groups)} parameter groups ({len(self.unet_loras)} main + 1 alpha) for the optimizer.")
        return optimizer_param_groups

    def set_lora_scales(self, scales: torch.Tensor):
        """
        Sets the multiplier for each LoRA module in the network.
        This is how we apply different LoRA strengths to each item in a batch.
        """
        for lora_module in self.unet_loras.values():
            lora_module.multiplier = scales

    def __enter__(self):
        """Called when entering the 'with' block. Does nothing special."""
        #return self # Often useful to return self
        pass

    def __exit__(self, exc_type, exc_value, tb):
        """
        Called when exiting the 'with' block.
        This is crucial for resetting the LoRA state.
        """
        # Set all multipliers to 0.0 to ensure LoRA is "off" outside this block.
        for lora_module in self.unet_loras.values():
            lora_module.multiplier = torch.tensor(0.0)
    
    
    def save_weights(self, file, dtype= torch.bfloat16, metadata= None):
        """
        Saves the state dictionary with keys matching the standard LoRA format.
        This version translates the internal, "dirty" keys to "clean" standard keys on the fly.
        """
        state_to_save = {}
        
        # Get the internal state dict, which has the "dirty" keys we need to clean.
        # Example dirty key: unet_loras.lora_unet___orig_mod_... .lora_down.weight
        internal_state_dict = self.state_dict()
        
        prefix_to_remove = "unet_loras."
        
        for dirty_key, value in internal_state_dict.items():
            if not dirty_key.startswith(prefix_to_remove):
                continue
            # Remove the 'unet_loras.' part
            key_without_prefix = dirty_key[len(prefix_to_remove):] 
            # replace the unwanted wrapper artifact to match the standard LoRA naming convention.
            clean_key = key_without_prefix.replace('__orig_mod_', '_') 
            if "lora_down" in clean_key or "lora_up" in clean_key or "alpha" in clean_key:
                 state_to_save[clean_key] = value.to("cpu", dtype=dtype)

        if not state_to_save:
            print("WARNING: No LoRA parameters found to save. Check network structure and key prefixes.")
            return

        if os.path.splitext(file)[1] == ".safetensors":
            save_file(state_to_save, file, metadata=metadata)
        else:
            torch.save(state_to_save, file)

# lora config util
# still inside of lora_alt.py

import torch.nn as nn
from typing import List, Dict, Any

# A simple helper for weight initialization, as required by AltLoRAModule
def get_initializer(init_scheme: str):
    # In a real scenario, you might have multiple schemes.
    # For now, we replicate the original's behavior.
    def kaiming_zeros(lora_down, lora_up):
        nn.init.kaiming_uniform_(lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(lora_up.weight)
    return kaiming_zeros

def create_lora_config_map(
    unet: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    train_alpha: bool = False,
    init_scheme: str = 'default',
    target_modules: List[str] = ['Linear', 'Conv2d'],
    include_substrings: List[str] = None,
    exclude_substrings: List[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Generates a resolved configuration map for all target modules in the UNet,
    with flexible filtering options. This fixes the issue of missed FFN layers.

    Args:
        unet: The UNet model to inspect.
        rank: The rank of the LoRA layers.
        alpha: The alpha value for LoRA scaling.
        train_alpha: Whether to make alpha a trainable parameter.
        init_scheme: The weight initialization scheme to use.
        target_modules: The class names of modules to apply LoRA to.
        include_substrings: A list of strings. A module's name must contain at least
                            one of these to be included. If None, all are included.
        exclude_substrings: A list of strings. If a module's name contains any of
                            these, it will be excluded.

    Returns:
        A dictionary where keys are the full module paths and values are their
        specific LoRA configurations.
    """
    lora_map = {}
    
    for name, module in unet.named_modules():
        # Check if the module is one of our target types (e.g., Linear, Conv2d)
        if module.__class__.__name__ in target_modules:
            
            # Apply exclusion filter first
            if exclude_substrings and any(sub in name for sub in exclude_substrings):
                continue
            
            # Apply inclusion filter
            if include_substrings and not any(sub in name for sub in include_substrings):
                continue

            # This module passes all filters, so we add it to the map.
            lora_name_key = name.replace('.', '_')
            lora_map[name] = {
                'lora_name': lora_name_key,
                'rank': rank,
                'alpha': alpha,
                'train_alpha': train_alpha,
                'init_scheme': init_scheme
            }
            
    if not lora_map:
        print("WARNING: No modules were targeted for LoRA. Check your filters.")
        
    return lora_map

def get_lora_map_for_training_method(
    unet: nn.Module,
    rank: int,
    alpha: float,
    method: str,
    **kwargs,
) -> Dict[str, Dict[str, Any]]:
    """
    A convenience wrapper that replicates the behavior of the original TRAINING_METHODS
    by calling create_lora_config_map with the correct filters.
    """
    method_filters = {
        'full': {
            'include_substrings': None,
            'exclude_substrings': None,
        },
        'xattn': {
            'include_substrings': ['attn2'], # Cross-attention modules
            'exclude_substrings': None,
        },
        'selfattn': {
            'include_substrings': ['attn1'], # Self-attention modules
            'exclude_substrings': None,
        },
        'noxattn': {
            'include_substrings': None,
            'exclude_substrings': ['attn2', 'time_embed'],
        },
        'ffn': { # Example of a new, useful filter
            'include_substrings': ['ff.net'],
            'exclude_substrings': None,
        },
        'attn_only': { # Another useful filter
             'include_substrings': ['attn1', 'attn2'],
             'exclude_substrings': None,
        }
        # Add other methods from TRAINING_METHODS as needed...
    }
    
    if method not in method_filters:
        raise ValueError(f"Unknown training method: {method}. Available: {list(method_filters.keys())}")
        
    filters = method_filters[method]
    
    return create_lora_config_map(
        unet=unet,
        rank=rank,
        alpha=alpha,
        include_substrings=filters['include_substrings'],
        exclude_substrings=filters['exclude_substrings'],
        **kwargs # Pass through other args like train_alpha
    )