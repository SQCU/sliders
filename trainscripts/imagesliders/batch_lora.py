import torch
import torch.nn as nn
import math
from typing import Optional, List, Type, Set, Literal, Union
import os
from safetensors.torch import save_file


# Define constants similar to lora.py if needed for module identification
UNET_TARGET_REPLACE_MODULE_TRANSFORMER = ["Attention"]
UNET_TARGET_REPLACE_MODULE_CONV = ["ResnetBlock2D", "Downsample2D", "Upsample2D"]
LORA_PREFIX_UNET = "lora_unet"
DEFAULT_TARGET_REPLACE = UNET_TARGET_REPLACE_MODULE_TRANSFORMER

class BatchedLoRAModule(nn.Module):
    """
    A batched version of LoRAModule that applies LoRA updates using a tensor of multipliers.
    This module replaces the forward method of the original Linear or Conv2d layers.

    The `multiplier` is expected to be a torch.Tensor, allowing different LoRA scales
    to be applied to different items within a batch.
    """
    def __init__(
        self,
        lora_name: str,
        org_module: nn.Module,
        lora_dim: int = 4,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim

        if "Linear" in org_module.__class__.__name__:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = nn.Linear(in_dim, lora_dim, bias=False)
            self.lora_up = nn.Linear(lora_dim, out_dim, bias=False)

        elif "Conv" in org_module.__class__.__name__:
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
            self.lora_dim = min(self.lora_dim, in_dim, out_dim)
            if self.lora_dim != lora_dim:
                print(f"WARNING: {lora_name} dim (rank) is changed to: {self.lora_dim}")
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
            self.lora_up = nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)
        else:
            raise NotImplementedError(f"LoRA not implemented for module type: {org_module.__class__.__name__}")

        # Store alpha as a buffer so it's saved in the state_dict.
        self.register_buffer("alpha", torch.tensor(alpha))
        
        # Calculate scale using the stored alpha buffer.
        self.scale = self.alpha.item() / self.lora_dim
        
        # The multiplier will be set by the network and used by LoRAInjectedLayer
        self.register_buffer("multiplier", torch.tensor(0.0))

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

class LoRAInjectedLayer(nn.Module):
    """
    A container that holds an original layer and its corresponding LoRA layer.
    This replaces the original layer in the model hierarchy and is torch.compile-friendly.
    """
    def __init__(self, org_module: nn.Module, lora_module: BatchedLoRAModule):
        super().__init__()
        self.org_module = org_module
        self.lora_module = lora_module

    def forward(self, x):
        original_output = self.org_module(x)
        lora_output = self.lora_module.lora_up(self.lora_module.lora_down(x))
        multiplier = self.lora_module.multiplier
        batch_size = multiplier.shape[0] if multiplier.ndim > 0 else 1
        
        if not isinstance(multiplier, torch.Tensor):
            multiplier = torch.tensor(multiplier, device=x.device)

        if multiplier.ndim == 0 or (multiplier.ndim == 1 and multiplier.shape[0] == 1):
            reshaped_multiplier = multiplier.to(x.device, dtype=x.dtype)
        else:
            reshaped_multiplier = multiplier.view(
                batch_size, *([1] * (len(lora_output.shape) - 1))
            ).to(x.device, dtype=x.dtype)
            
        return original_output + lora_output * reshaped_multiplier * self.lora_module.scale

# --- MODIFIED CLASS ---
class BatchedLoRANetwork(nn.Module):
    """
    Manages a collection of BatchedLoRAModule instances for a UNet.
    Allows setting a batch of LoRA scales to be applied concurrently.
    """
    def __init__(
        self,
        unet: nn.Module,
        rank: int = 4,
        alpha: float = 1.0,
        train_method: Literal["full", "noxattn", "innoxattn", "selfattn", "xattn", "xattn-strict", "noxattn-hspace", "noxattn-hspace-last"] = "full",
        target_replace: List[str] = DEFAULT_TARGET_REPLACE,
    ) -> None:
        super().__init__()
        self.lora_dim = rank
        self.alpha = alpha
        self.train_method = train_method
        self.target_replace = target_replace
        self.unet_loras: nn.ModuleDict = nn.ModuleDict()
        self.module_creation_count = 0
        self.module_replacement_count = 0

        self._create_and_apply_modules(unet)
        print(f"DEBUG: Total BatchedLoRAModule creations: {self.module_creation_count}")
        print(f"DEBUG: Total module replacements: {self.module_replacement_count}")

        lora_names = set(self.unet_loras.keys())
        assert len(lora_names) == len(self.unet_loras), "Duplicate LoRA names found."
        
        del unet
        torch.cuda.empty_cache()

    def _create_and_apply_modules(self, root_module: nn.Module):
        modules_to_replace = []
        for name, module in root_module.named_modules():
            if self._is_target_module(name, module):
                modules_to_replace.append((name, module))

        for name, module in modules_to_replace:
            path_parts = name.split('.')
            parent_module = root_module
            for part in path_parts[:-1]:
                parent_module = getattr(parent_module, part)
            
            child_name = path_parts[-1]
            
            # <<< FIX 1: Create the exact key prefix required by the standard.
            # Example: 'lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q'
            lora_name_key = f"{LORA_PREFIX_UNET}_{name.replace('.', '_')}"
            
            lora_module = BatchedLoRAModule(
                lora_name_key, module, self.lora_dim, self.alpha
            )
            self.unet_loras[lora_name_key] = lora_module
            self.module_creation_count += 1

            injected_layer = LoRAInjectedLayer(module, lora_module)
            setattr(parent_module, child_name, injected_layer)
            self.module_replacement_count += 1

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        if module.__class__.__name__ not in ["Linear", "Conv2d", "LoRACompatibleLinear", "LoRACompatibleConv"]:
            return False
        if "time_embed" in name: return False
        train_method = self.train_method
        if train_method == "full":
            is_in_target = any(target in name for target in self.target_replace)
            if not is_in_target: return False
        elif train_method in ["noxattn", "noxattn-hspace", "noxattn-hspace-last", "innoxattn"]:
            if "attn2" in name: return False
        elif train_method == "selfattn":
            if "attn1" not in name: return False
        elif train_method in ["xattn", "xattn-strict", "xattn-up", "xattn-down", "xattn-mid"]:
            if "attn2" not in name: return False
            if train_method == 'xattn-up' and 'up_block' not in name: return False
            if train_method == 'xattn-down' and 'down_block' not in name: return False
            if train_method == 'xattn-mid' and 'mid_block' not in name: return False
            if train_method == 'xattn-strict' and ('out' in name or 'to_q' in name): return False
        else: raise NotImplementedError(f"train_method: {train_method} is not implemented.")
        if train_method == 'noxattn-hspace' and 'mid_block' not in name: return False
        if train_method == 'noxattn-hspace-last' and ('mid_block' not in name or '.1' not in name or 'conv2' not in name): return False
        return True

    def set_lora_scales(self, scales: torch.Tensor):
        for lora_module in self.unet_loras.values():
            lora_module.multiplier = scales

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, tb):
        for lora_module in self.unet_loras.values():
            lora_module.multiplier = torch.tensor(0.0)

    def prepare_optimizer_params(self, lr=None):
        params_to_optimize = []
        for lora_module in self.unet_loras.values():
            params_to_optimize.extend(lora_module.parameters())
        return [{"params": params_to_optimize}]

    def save_weights(self, file: str, dtype: Optional[torch.dtype] = torch.bfloat16, metadata: Optional[dict] = None):
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
            
            # This is the crucial fix: replace the unwanted wrapper artifact
            # to match the standard LoRA naming convention.
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