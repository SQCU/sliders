import torch
import torch.nn as nn
import math
from typing import Optional, List, Type, Set, Literal, Union

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
        alpha: float = 1.0, # Alpha is now a float
    ):
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim
        #log flashbang
        #print(f"DEBUG: Initializing BatchedLoRAModule: {lora_name}")

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

        self.scale = alpha / self.lora_dim
        # The multiplier will be set by the network and used by LoRAInjectedLayer
        self.register_buffer("current_multiplier", torch.tensor(0.0))

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)


        self.org_module = org_module  # Keep reference to original module for forward pass

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
        # Get the original output by calling the original module directly
        original_output = self.org_module(x)
        
        # Calculate the lora update
        lora_output = self.lora_module.lora_up(self.lora_module.lora_down(x))

        # Get the multiplier and prepare it for broadcasting
        multiplier = self.lora_module.current_multiplier.to(x.device, dtype=x.dtype)
        while len(multiplier.shape) < len(lora_output.shape):
            multiplier = multiplier.unsqueeze(-1)
            
        # Apply the scaled LoRA update
        return original_output + lora_output * multiplier * self.lora_module.scale


    
class BatchedLoRANetwork(nn.Module):
    """
    Manages a collection of BatchedLoRAModule instances for a UNet.
    Allows setting a batch of LoRA scales to be applied concurrently.
    """
    def __init__(
        self,
        unet: nn.Module, # Can be UNet2DConditionModel or any nn.Module with Linear/Conv2d layers
        rank: int = 4,
        alpha: float = 1.0,
        train_method: Literal["full", "noxattn", "innoxattn", "selfattn", "xattn", "xattn-strict", "noxattn-hspace", "noxattn-hspace-last"] = "full",
        target_replace: List[str] = DEFAULT_TARGET_REPLACE,
    ) -> None:
        """
        Initializes the BatchedLoRANetwork.

        Args:
            unet (nn.Module): The UNet model to apply LoRA to.
            rank (int): The default rank for LoRA modules.
            alpha (float): The default alpha value for LoRA modules.
            train_method (Literal): Specifies which parts of the UNet to apply LoRA to.
            target_replace (List[str]): List of module names to target for LoRA application.
        """
        super().__init__()
        self.lora_dim = rank
        self.alpha = alpha
        self.train_method = train_method
        self.target_replace = target_replace
        self.unet_loras: nn.ModuleList[BatchedLoRAModule] = nn.ModuleList()
        self.module_creation_count = 0 # DEBUG
        self.module_replacement_count = 0 # DEBUG

        # Create BatchedLoRAModule instances and apply them to the UNet
        self._create_and_apply_modules(unet)
        print(f"DEBUG: Total BatchedLoRAModule creations: {self.module_creation_count}") # DEBUG
        print(f"DEBUG: Total module replacements: {self.module_replacement_count}") # DEBUG

        # Ensure no duplicate lora names
        lora_names = set()
        for lora_module in self.unet_loras:
            assert lora_module.lora_name not in lora_names, f"Duplicate LoRA name: {lora_module.lora_name}"
            lora_names.add(lora_module.lora_name)
        
        del unet
        torch.cuda.empty_cache()

    def _create_and_apply_modules(self, root_module: nn.Module):
        """
        Finds and replaces target modules with LoRA-injected versions.
        This implementation iterates through all modules and replaces them based on a single, precise filtering function,
        preventing the memory issues from excessive module creation.
        """
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
            
            lora_name = f"{LORA_PREFIX_UNET}_{name.replace('.', '_')}"
            lora_module = BatchedLoRAModule(
                lora_name, module, self.lora_dim, self.alpha
            )
            self.unet_loras.append(lora_module)
            self.module_creation_count += 1

            injected_layer = LoRAInjectedLayer(module, lora_module)
            setattr(parent_module, child_name, injected_layer)
            self.module_replacement_count += 1

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        """
        Determines if a specific module should be replaced with a LoRA version based on its type and name.
        """
        if module.__class__.__name__ not in ["Linear", "Conv2d", "LoRACompatibleLinear", "LoRACompatibleConv"]:
            return False

        # Name-based filtering logic adapted from the original lora.py
        if "time_embed" in name:
            return False

        train_method = self.train_method
        if train_method == "full":
            # In 'full' mode, we check if the module is part of a container specified in target_replace
            is_in_target = False
            for target_name in self.target_replace:
                if target_name in name:
                    is_in_target = True
                    break
            if not is_in_target:
                return False
        elif train_method == "noxattn" or train_method == "noxattn-hspace" or train_method == "noxattn-hspace-last":
            if "attn2" in name:
                return False
        elif train_method == "innoxattn":
            if "attn2" in name:
                return False
        elif train_method == "selfattn":
            if "attn1" not in name:
                return False
        elif train_method in ["xattn", "xattn-strict", "xattn-up", "xattn-down", "xattn-mid"]:
            if "attn2" not in name: # Note: diffusers uses attn2 for cross-attention
                return False
            if train_method == 'xattn-up' and 'up_block' not in name:
                return False
            if train_method == 'xattn-down' and 'down_block' not in name:
                return False
            if train_method == 'xattn-mid' and 'mid_block' not in name:
                return False
            if train_method == 'xattn-strict' and ('out' in name or 'to_q' in name):
                 return False
        else:
            raise NotImplementedError(f"train_method: {train_method} is not implemented.")
        
        # Additional specific filters from the old logic
        if train_method == 'noxattn-hspace' and 'mid_block' not in name:
            return False
        if train_method == 'noxattn-hspace-last' and ('mid_block' not in name or '.1' not in name or 'conv2' not in name):
            return False
            
        return True

    def set_lora_scales(self, scales: torch.Tensor):
        """
        Sets the LoRA multipliers for all managed LoRA modules based on a batch of scales.
        The `scales` tensor is expected to have a shape that can broadcast with the
        batch dimension of the input tensors to the LoRA modules.
        For example, if the input to LoRA modules is (B, C, H, W), `scales` should be (B, 1, 1, 1).

        Args:
            scales (torch.Tensor): A tensor of LoRA scales, one for each item in the batch.
        """

        for lora_module in self.unet_loras:
            #lora_module.current_multiplier = scales.to(lora_module.lora_down.weight.device)
            #what is this curious and inexplicable gemini 2.5 code saying to do?
            #the reference lora.py does the same thing i just wrote below btw.
            lora_module.current_multiplier = scales.detach()

    def __enter__(self):
        """
        Context manager entry: Activates the LoRA modules by setting their multipliers.
        The actual scales are set via `set_lora_scales` before entering the context.
        """
        # Multipliers are already set by set_lora_scales before __enter__ is called.
        # This context manager primarily ensures that the multipliers are reset on exit.
        pass

    def __exit__(self, exc_type, exc_value, tb):
        """
        Context manager exit: Deactivates the LoRA modules by setting their multipliers to zero.
        """
        # Reset multipliers to zero to effectively disable LoRA
        for lora_module in self.unet_loras:
            lora_module.current_multiplier = torch.tensor(0.0)

    def prepare_optimizer_params(self, lr=None):
        """
        Prepares parameters for the optimizer.
        """
        all_params = []
        for lora_module in self.unet_loras:
            all_params.extend(lora_module.parameters())
        return [{"params": all_params}]

    def save_weights(self, file: str, dtype: Optional[torch.dtype] = None, metadata: Optional[dict] = None):
        """
        Saves the state dictionary of the LoRA network.
        """
        state_dict = self.state_dict()
        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                state_dict[key] = v.detach().clone().to("cpu").to(dtype)

        # This part needs to be adapted based on how you want to save.
        # For now, a simple torch.save
        if os.path.splitext(file)[1] == ".safetensors":
            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)
