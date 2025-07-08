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
        print(f"DEBUG: Initializing BatchedLoRAModule: {lora_name}")

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
        
        # from gemini 2.5:
        # The original UNet reference is kept, but its forward methods are now augmented.
        # We don't delete unet here as it's passed in and might be used elsewhere.
        # torch.cuda.empty_cache() # Not necessary here, manage memory externally
        # from sqcu:
        # nice try i'm deleting the unet anyways because the reference implementation does that.
        del unet

        torch.cuda.empty_cache()
        #if you don't like these lines of code demonstrate an error they introduce in control flow.


    def _create_and_apply_modules(self, root_module: nn.Module):
        """
        Recursively finds target modules in the UNet, creates LoRA modules,
        and REPLACES them with LoRAInjectedLayer instances.
        """
        modules_to_replace = []
        # Use named_modules() to get the unique path (name) and the module instance
        for parent_name, parent_module in root_module.named_modules():
            if self._should_apply_lora(parent_name, parent_module):
                for child_name, child_module in parent_module.named_children():
                    # We check the child type directly
                    if child_module.__class__.__name__ in ["Linear", "Conv2d", "LoRACompatibleLinear", "LoRACompatibleConv"]:
                        if self._should_apply_lora_to_child(parent_name, child_name):
                            # Store the parent module, the child's name, the child module, AND the unique parent path
                            modules_to_replace.append((parent_module, child_name, child_module, parent_name))

        # Keep track of which layers have been replaced to avoid double-injection
        replaced_layers = set()
        for parent_module, child_name, child_module, parent_name in modules_to_replace:
            # Create a unique key for the layer based on its parent and its own name
            layer_key = (parent_name, child_name)
            if layer_key in replaced_layers:
                continue # Skip if we've already processed this exact layer
            replaced_layers.add(layer_key)

            # Use the unique parent_name for the lora_name
            lora_name = f"{LORA_PREFIX_UNET}_{parent_name}_{child_name}".replace(".", "_")
            
            # 1. Create the LoRA-only module
            lora_module = BatchedLoRAModule(
                lora_name, child_module, self.lora_dim, self.alpha
            )
            self.unet_loras.append(lora_module)
            self.module_creation_count += 1 # DEBUG

            # 2. Create the injection container
            injected_layer = LoRAInjectedLayer(child_module, lora_module)

            # 3. Replace the original layer with our new container
            setattr(parent_module, child_name, injected_layer)
            self.module_replacement_count += 1 # DEBUG

    def _should_apply_lora(self, name: str, module: nn.Module) -> bool:
        """
        Determines if LoRA should be applied to a given module based on training method and target replace modules.
        """
        if "time_embed" in name:
            return False
            
        if self.train_method == "noxattn":
            return "attn2" not in name
        elif self.train_method == "innoxattn":
            return "attn2" not in name
        elif self.train_method == "selfattn":
            return "attn1" in name
        elif self.train_method == "xattn":
            return "attn2" in name
        elif self.train_method == "full":
            return module.__class__.__name__ in self.target_replace
        elif self.train_method == "xattn-strict":
            return "attn2" in name
        elif self.train_method == "noxattn-hspace":
            return "attn2" not in name and "mid_block" in name
        elif self.train_method == "noxattn-hspace-last":
            return "attn2" not in name and "mid_block" in name
        else:
            raise NotImplementedError(f"train_method: {self.train_method} is not implemented.")

    def _should_apply_lora_to_child(self, parent_name: str, child_name: str) -> bool:
        """
        Further filters child modules based on specific training methods.
        """
        if self.train_method == 'xattn-strict':
            return 'out' not in child_name
        elif self.train_method == 'noxattn-hspace':
            return True # Already filtered by parent_name
        elif self.train_method == 'noxattn-hspace-last':
            return '.1' in parent_name and 'conv2' in child_name
        return True # Default for other train methods

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
