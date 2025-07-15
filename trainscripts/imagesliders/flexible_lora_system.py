# flexible_lora_system.py (v3 - With Implemented Estimators)
# This script contains the full, self-contained system for the first phase of testing.
# It includes initializers, a test model, and a main function to run an end-to-end
# internal consistency check with *fully implemented* data-driven estimators.
# python -m trainscripts.imagesliders.flexible_lora_system 

import torch
import torch.nn as nn
import math
import yaml
import json
import os
import datetime
import shutil
from collections import defaultdict
from tqdm import tqdm

# ==============================================================================
# SECTION 1: LORA INITIALIZERS & ESTIMATOR STUBS
# (Formerly lora_initializers.py and lora_estimators.py)
# ==============================================================================

def is_valid_artifact(path: str) -> bool:
    """
    Checks if a given path points to a valid, non-empty JSON artifact file.
    """
    # 1. Check for basic existence
    if not os.path.exists(path):
        return False
    
    # 2. Check if the file is empty
    if os.path.getsize(path) == 0:
        return False

    # 3. Check for valid JSON content
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        # Ensure the loaded data is a non-empty dictionary or list
        return bool(data)
    except (json.JSONDecodeError, IOError):
        # File is corrupted, unreadable, or not valid JSON
        return False

# A simple dictionary to cache computed scaling factors for kaiming_cheald.
_kaiming_cheald_cache = {}

def init_kaiming_cheald_(lora_down: nn.Module, lora_up: nn.Module, verbosity=1):
    """
    Kaiming-Cheald v2: Initializes LoRA matrices so their ghost matrix norm is
    SCALED DOWN relative to a full-rank matrix, appropriate for a low-rank update.
    This helps normalize the initial impact of LoRA modules across different ranks and shapes.
    """
    if isinstance(lora_up, nn.Linear):
        out_features, rank = lora_up.weight.shape
        in_features = lora_down.weight.shape[1]
    elif isinstance(lora_up, nn.Conv2d):
        out_features, rank, _, _ = lora_up.weight.shape
        in_features = lora_down.weight.shape[1]
    else: # Fallback for unknown types
        out_features, rank, in_features = 0, 0, 0

    rank = lora_down.weight.shape[0]
    cache_key = f"{in_features}-{out_features}-{rank}"

    if cache_key in _kaiming_cheald_cache:
        balanced_scale = _kaiming_cheald_cache[cache_key]

        nn.init.uniform_(lora_down.weight, -0.01, 0.01)
        nn.init.uniform_(lora_up.weight, -0.01, 0.01)
    else:
        if verbosity:
            print(f"  [Kaiming-Cheald] Calibrating for shape ({out_features}, {in_features}) rank {rank}...")
        if isinstance(lora_up, nn.Conv2d):
            # For conv, the reference is also a conv weight
            W_temp = torch.empty(out_features, in_features, lora_down.kernel_size[0], lora_down.kernel_size[1])
        else: # Default to Linear
            W_temp = torch.empty(out_features, in_features)

        nn.init.kaiming_uniform_(W_temp, a=math.sqrt(5))
        full_rank_norm = torch.norm(W_temp)

        # 2. Scale this reference norm down. A good heuristic is sqrt(rank/dim).
        # This ensures a rank-4 LoRA starts much smaller than a rank-128 LoRA.
        norm_scaling_factor = math.sqrt(float(rank) / in_features)
        target_norm = full_rank_norm * norm_scaling_factor

        # 3. Initialize our matrices with a small, non-zero distribution
        nn.init.uniform_(lora_down.weight, -0.01, 0.01)
        nn.init.uniform_(lora_up.weight, -0.01, 0.01)

        # 4. Get the current norm of the ghost matrix
        with torch.no_grad():
            if isinstance(lora_down, nn.Conv2d):
                # lora_down: (rank, in_c, kH, kW) -> (rank, in_c * kH * kW)
                down_matrix = lora_down.weight.view(rank, -1)
                # lora_up: (out_c, rank, 1, 1) -> (out_c, rank)
                up_matrix = lora_up.weight.view(out_features, rank)
                ghost_matrix = up_matrix @ down_matrix
            else: # Default case for nn.Linear, which is already 2D
                ghost_matrix = lora_up.weight @ lora_down.weight
            
            current_norm = torch.norm(ghost_matrix)
        if verbosity:
            # --- NEW DIAGNOSTIC BLOCK 2: PRINT INTERMEDIATE VALUES ---
            # This is where we'll find the NaN. We expect to see two normal floats.
            # If either is zero, inf, or nan, we've found our problem.
            print(f"  > Intermediate value `target_norm`: {target_norm}")
            print(f"  > Intermediate value `current_norm`: {current_norm}")
        
        # 5. Calculate the total scaling factor needed.
        total_scale = target_norm / (current_norm + 1e-9)

        # --- NEW LOGIC: Apply a balanced scale ---
        # 6. Apply the SQUARE ROOT of the scale to both matrices.
        balanced_scale = math.sqrt(abs(total_scale))

        _kaiming_cheald_cache[cache_key] = balanced_scale

    with torch.no_grad():
        lora_down.weight.mul_(balanced_scale)
        lora_up.weight.mul_(balanced_scale)
        
        if verbosity:
            # --- NEW DIAGNOSTIC BLOCK 3: PRINT THE RESULT OF THE DIVISION ---
            print(f"  > Calculated `scale` (target/current): {balanced_scale}")

     # --- NEW DIAGNOSTIC BLOCK 4: CHECK THE FINAL WEIGHTS FOR NAN ---
    if torch.isnan(lora_up.weight).any():
        print("  > CRITICAL: NaN detected in lora_up.weight immediately after scaling.")

def get_initializer(name: str):
    if name == "kaiming_cheald":
        return init_kaiming_cheald_
    else: # Default "standard" initialization
        def init_standard(lora_down, lora_up):
            nn.init.kaiming_uniform_(lora_down.weight, a=math.sqrt(5))
            nn.init.zeros_(lora_up.weight)
        return init_standard

def estimate_ranks_from_svd(unet, cutoff=0.3, save_path=None):
    """
    (IMPLEMENTED) Estimates the optimal rank for each layer via SVD.

    This function iterates through the weight matrices of a provided UNet,
    performs SVD, and determines the number of singular values needed to
    capture a `cutoff` percentage of the matrix's "energy". This provides
    a data-driven way to allocate LoRA parameter budget. The output is
    persisted to `save_path` to skip this phase during search loops.
    """
    print(f"--- [Real] Running SVD Rank Estimation (cutoff={cutoff})... ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    estimated_ranks = {}

    for name, module in tqdm(unet.named_modules(), desc="Analyzing Layers for SVD"):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            with torch.no_grad():
                # Move weight to GPU for fast SVD, ensure it's float32
                w = module.weight.to(device, dtype=torch.float32)

                # Reshape conv weights to be 2D for SVD
                if w.ndim > 2:
                    w = w.view(w.size(0), -1)

                try:
                    # Perform SVD
                    U, S, Vh = torch.linalg.svd(w, full_matrices=False)
                    
                    # Calculate cumulative energy
                    cumulative_energy = torch.cumsum(S, 0) / S.sum()
                    
                    # Find the first index where cumulative energy exceeds the cutoff
                    rank_tensor = (cumulative_energy > cutoff).nonzero()
                    
                    if rank_tensor.numel() > 0:
                        # Add 1 because indices are 0-based
                        rank = rank_tensor[0].item() + 1
                    else:
                        # If cutoff is never reached (e.g., cutoff=1.0), use full rank
                        rank = w.shape[1]

                    # Enforce a minimum practical rank
                    estimated_ranks[name] = max(4, rank)
                    
                    del U, S, Vh, cumulative_energy, rank_tensor
                except torch.linalg.LinAlgError:
                    print(f"SVD failed for layer {name}. Skipping.")
                    estimated_ranks[name] = 4 # Default fallback rank
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(estimated_ranks, f, indent=2)
        print(f"  > Saved SVD rank estimates to {save_path}")
    
    print("--- SVD Rank Estimation Complete ---")
    return estimated_ranks


def warmup_and_estimate_alphas(unet_builder, base_lora_config, training_harness_fn, gen_exp_kwargs={}, save_path=None):
    """
    (IMPLEMENTED) Performs a short training run to find optimal alpha ratios per layer.
    
    This function builds a LoRA network where alphas are trainable `nn.Parameter`s,
    then trains it on dummy data. The ratio of the final learned alpha to its
    initial value provides an "importance score" for each layer, effectively
    performing per-layer learning rate discovery.
    (UPGRADED) Performs a short training run using the REAL training harness
    to find optimal alpha ratios per layer.
    """
    print(f"--- [Real] Running Online Alpha Warmup using the provided training harness... ---")
    
    # --- 1. Setup a temporary training environment ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create a copy of the config and force all alphas to be trainable
    alpha_train_config = base_lora_config.copy()
    for rule in alpha_train_config.get('lora_rules', []):
        rule['train_alpha'] = True
    
    # --- 2. Run the REAL training harness ---
    # We pass it our special config. We don't care about the performance score it returns,
    # we only want the network object it produces after training.
    print("  > Launching training harness for alpha warmup...")
    trained_network, _ = training_harness_fn(alpha_train_config, **gen_exp_kwargs,) #num_epochs=5, eval_every_n_epochs=999)

    # --- 3. Extract alpha ratios from the returned, trained network ---
    print("  > Extracting learned alpha ratios from trained network...")
    alpha_ratios = {}

    # Build the network for alpha training
    temp_uvit_for_config = unet_builder().to(device)
    loader = LoRAConfigLoader(config_dict=alpha_train_config)
    resolved_config = loader.get_resolved_config(temp_uvit_for_config)
    initial_alphas = {
        name: config['alpha'] for name, config in resolved_config['lora_map'].items()
    }
    #lora_name_to_module_name = {
    #    config['lora_name']: module_name
    #    for module_name, config in resolved_config['lora_map'].items()
    #}

    trained_state_dict = trained_network.state_dict()
    for key, final_alpha_tensor in trained_state_dict.items():
        # We are only interested in our trainable alpha parameters
        suffix_to_match = '.lora_module.alpha'
        if not key.endswith(suffix_to_match):
            continue

        # Parse the module name from the state_dict key
        # key format: 'unet_loras.lora_unet_down_blocks_0_..._to_q.alpha'
        # Reconstruct the original module name by removing the known suffix.
        # 'down_blocks.0...q.lora_module.alpha' -> 'down_blocks.0...q'
        module_name = key.removesuffix(suffix_to_match) 

        # Now, `module_name` is guaranteed to be the correct key for lookup
        initial_alpha = initial_alphas.get(module_name)
        
        if initial_alpha is not None:
            final_alpha = final_alpha_tensor.item()
            ratio = final_alpha / (initial_alpha + 1e-9)
            alpha_ratios[module_name] = ratio
        else:
            # This warning should ideally never appear now
            print(f"  > LOGIC WARN: Correctly parsed '{module_name}' but couldn't find its initial alpha.")

    # Persist the (now correctly populated) ratios
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Ensure we have data to write
        if not alpha_ratios:
            print("  > CRITICAL WARN: Alpha ratio dictionary is empty. Nothing will be saved.")
        else:
            with open(save_path, 'w') as f: json.dump(alpha_ratios, f, indent=2)
            print(f"  > Saved {len(alpha_ratios)} Alpha Warmup estimates to {save_path}")

    del trained_network, temp_uvit_for_config
    print("--- Alpha Warmup Complete ---")
    return alpha_ratios

def run_all_estimators(
    unet_builder,
    base_config_for_warmup,
    training_harness_fn, # <-- Accepts the REAL training function
    artifacts_dir="search_artifacts",
    force_rerun=False,
    gen_exp_kwargs={},
):
    """
    A single, clean entry point to run all pre-computation estimators.
    This function is a pure utility. It takes configs and returns paths to artifacts.
    """
    print("--- Running Pre-computation Estimators ---")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    svd_path = os.path.join(artifacts_dir, "svd_ranks.json")
    alpha_path = os.path.join(artifacts_dir, "alpha_ratios.json")
    
     # --- THE FIX: Use the new, robust validation function ---
    if not is_valid_artifact(svd_path) or force_rerun:
        print(f"SVD artifact at '{svd_path}' is invalid or missing. Running estimator...")
        estimate_ranks_from_svd(unet_builder(), save_path=svd_path)
    else:
        print(f"Found existing SVD estimates at {svd_path}, skipping.")
        
    if not is_valid_artifact(alpha_path) or force_rerun:
        # Call the now-upgraded alpha warmer
        print(f"Alpha artifact at '{alpha_path}' is invalid or missing. Running estimator...")
        warmup_and_estimate_alphas(
            unet_builder, 
            base_config_for_warmup,
            training_harness_fn, # <-- Pass the real harness through
            save_path=alpha_path,
            gen_exp_kwargs=gen_exp_kwargs,
        )
    else:
        print(f"Found existing Alpha estimates at {alpha_path}, skipping.")
        
    print("--- Estimators Complete ---")
    return svd_path, alpha_path

# ==============================================================================
# SECTION 2: CONFIGURATION LOADER (Modified to accept dict)
# ==============================================================================

class LoRAConfigLoader:
    def __init__(self, config_path=None, config_dict=None, rank_estimates_path=None, alpha_estimates_path=None):
        if config_path:
            with open(config_path, 'r') as f: self.base_config = yaml.safe_load(f)
        elif config_dict:
            self.base_config = config_dict
        else:
            raise ValueError("Must provide either config_path or config_dict")
        self.rank_estimates = self._load_json(rank_estimates_path)
        self.alpha_estimates = self._load_json(alpha_estimates_path)
        # --- NEW DIAGNOSTIC BLOCK: DATA INTEGRITY VALIDATION ---
        # This block will immediately find the source of the float-in-rank bug.
        # It checks if the data we loaded matches the type we expect for that attribute.
        print("\n--- LoRAConfigLoader Data Integrity Check ---")
        
        # Check rank estimates: they MUST all be integers.
        if self.rank_estimates:
            first_rank_val = next(iter(self.rank_estimates.values()))
            if isinstance(first_rank_val, float):
                # This is the smoking gun. If this message prints, the wrong file was loaded.
                raise TypeError(
                    f"FATAL: `rank_estimates` contains floats (e.g., {first_rank_val}). "
                    f"This almost certainly means that the alpha estimates file was incorrectly "
                    f"passed as the rank estimates file."
                )
            print("  > Rank estimates appear to be of the correct type (int). OK.")
        else:
            print("  > Rank estimates not loaded (this may be expected).")

        # Check alpha estimates: they SHOULD be floats.
        if self.alpha_estimates:
            first_alpha_val = next(iter(self.alpha_estimates.values()))
            if isinstance(first_alpha_val, int):
                 print(
                    f"  > WARN: `alpha_estimates` contains integers (e.g., {first_alpha_val}). "
                    f"This might mean the rank estimates file was passed as the alpha estimates file."
                )
            else:
                 print("  > Alpha estimates appear to be of the correct type (float). OK.")
        else:
            print("  > Alpha estimates not loaded (this may be expected).")
        print("-------------------------------------------\n")
    

    def _load_json(self, path):
        if path and os.path.exists(path):
            print(f"Loading estimates from: {path}")
            with open(path, 'r') as f: return json.load(f)
        return {}
    def get_resolved_config(self, unet) -> dict:
        module_to_rule = {}
        sorted_rules = sorted(self.base_config.get('lora_rules', []), key=lambda x: x.get('priority', 1.0), reverse=True)
        

        # We now iterate through the actual modules to use isinstance for filtering
        for name, module in unet.named_modules():
            # --- FIX #1: IGNORE CONTAINERS ---
            # We only consider primitive layers that can actually have LoRA applied.
            if not isinstance(module, (nn.Linear, nn.Conv2d)):
                continue

            for rule in sorted_rules:
                if self._module_matches_rule(name, rule):
                    if name not in module_to_rule:
                         module_to_rule[name] = rule
        
        resolved_lora_map = {}
        for module_name, rule in module_to_rule.items():
            # --- FIX #2: STRICT RANK VALIDATION ---
            # Ensure the matched rule specifies a rank.
            if 'rank' not in rule:
                raise ValueError(
                    f"Configuration Error: Rule '{rule.get('name', 'N/A')}' matched module '{module_name}' "
                    f"but is missing a required 'rank' key."
                )
            
            rank = rule.get('rank')
            if rank == 'auto_svd':
                rank = self.rank_estimates.get(module_name, 4)
            
            base_alpha = rule.get('alpha', 1.0)
            importance_ratio = self.alpha_estimates.get(module_name, 1.0)
            initial_alpha = base_alpha * importance_ratio

            resolved_lora_map[module_name] = {
                "rank": rank,
                "alpha": initial_alpha,
                "train_alpha": rule.get('train_alpha', False),
                "init_scheme": rule.get('init_scheme', 'standard'),
                "lora_name": f"lora_unet_{module_name.replace('.', '_')}",
            }
        return {"lora_map": resolved_lora_map}
    def _module_matches_rule(self, module_name, rule):
        if 'target_modules' in rule and module_name in rule['target_modules']:
            return True
        if 'target_name_contains' in rule and any(s in module_name for s in rule['target_name_contains']):
            return True
        return False

# ==============================================================================
# SECTION 3: FLEXIBLE LORA NETWORK IMPLEMENTATION (FULLY REVISED)
# ==============================================================================

class LoRAInjectedLayer(nn.Module):
    """
    A container that holds an original layer and its corresponding LoRA layer.
    This replaces the original layer and correctly handles the batched application
    of the LoRA delta, including per-item scaling via a multiplier.
    """
    def __init__(self, org_module: nn.Module, lora_module):
        super().__init__()
        self.org_module = org_module
        self.lora_module = lora_module

    def forward(self, x, *args, **kwargs):
        # 1. Get the output from the original, frozen module
        original_output = self.org_module(x, *args, **kwargs)

        # 2. Get the LoRA delta from our flexible module
        lora_delta = self.lora_module(x)

        # 3. Get the per-item multiplier for this batch
        multiplier = self.lora_module.multiplier

        # 4. Dynamically compute the scale (alpha / rank)
        # This works correctly whether alpha is a buffer or a trainable parameter.
        scale = self.lora_module.alpha / self.lora_module.lora_dim

        # 5. Reshape the multiplier for broadcasting
        # This ensures each item in the batch gets its own scale applied.
        if multiplier.ndim == 0 or multiplier.numel() == 1:
            # Scalar multiplier applies to the whole batch
            reshaped_multiplier = multiplier.to(lora_delta.device, dtype=lora_delta.dtype)
        else:
            # Tensor multiplier needs to be reshaped to (B, 1, 1, 1) for Conv or (B, 1) for Linear
            reshaped_multiplier = multiplier.view(
                lora_delta.shape[0], *([1] * (lora_delta.ndim - 1))
            ).to(lora_delta.device, dtype=lora_delta.dtype)

        # 6. Combine everything for the final output
        return original_output + (lora_delta * reshaped_multiplier * scale)


class FlexibleLoRAModule(nn.Module):
    """A flexible LoRA module driven by a detailed configuration."""
    def __init__(self, org_module, rank, alpha, train_alpha, init_scheme):
        super().__init__()
        self.lora_dim = rank

        # --- RE-INTRODUCE THE MULTIPLIER BUFFER ---
        # Each module holds its own multiplier, which will be set by the network.
        # Default to 1.0, which means LoRA is fully active if not otherwise specified.
        self.register_buffer("multiplier", torch.tensor(1.0))

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

        get_initializer(init_scheme)(self.lora_down, self.lora_up)

        if train_alpha:
            self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        else:
            self.register_buffer("alpha", torch.tensor(float(alpha)))

    def forward(self, x):
        # The forward pass is now clean: it only computes the LoRA delta.
        # The LoRAInjectedLayer handles scaling and multiplication.
        return self.lora_up(self.lora_down(x))


class FlexibleLoRANetwork(nn.Module):
    """Builds and manages a LoRA network based on a fully resolved configuration map."""
    def __init__(self, unet, resolved_config):
        super().__init__()
        self.unet = unet
        self.resolved_config = resolved_config
        self.unet_loras = nn.ModuleDict()
        self._create_and_apply_modules(self.unet)

    def _create_and_apply_modules(self, root_module):
        lora_map = self.resolved_config.get("lora_map", {})
        
        for name, module in root_module.named_modules():
            if name in lora_map:
                if not isinstance(module, (nn.Linear, nn.Conv2d)):
                    #print(f"DEBUG: Rule matched name '{name}' but module type is {type(module).__name__}, skipping.")
                    continue
                lora_config = lora_map[name].copy() # Use a copy to avoid modifying the original dict
                # --- THE FIX IS HERE ---
                # Pop the 'lora_name' from the config dict. It's metadata for the network,
                # not the module. This leaves only the keyword arguments that the
                # FlexibleLoRAModule constructor expects.
                lora_name = lora_config.pop('lora_name')

                lora_name = f"lora_unet_{name.replace('.', '_')}"
                
                lora_module = FlexibleLoRAModule(module, **lora_config)
                self.unet_loras[lora_name] = lora_module

                # This part recursively finds the parent to perform the replacement
                path_parts = name.split('.')
                parent = root_module
                for part in path_parts[:-1]:
                    parent = getattr(parent, part)
                
                # Replace the original module with our injected layer
                setattr(parent, path_parts[-1], LoRAInjectedLayer(module, lora_module))
                print(f"Applied LoRA to '{name}' with config: {lora_config}")

    def prepare_optimizer_params(self):
        params = []
        alpha_params = []
        for lora_module in self.unet_loras.values():
            params.extend([p for p_name, p in lora_module.named_parameters() if 'alpha' not in p_name])
            alpha_params.extend([p for p_name, p in lora_module.named_parameters() if 'alpha' in p_name])
        return [
            {"params": params},
            {"params": alpha_params, "lr": 1e-1}
        ]

    # --- RE-INTRODUCE THE METHOD TO SET BATCH-WISE SCALES ---
    def set_lora_scales(self, scales: torch.Tensor):
        """
        Sets the multiplier for each LoRA module in the network.
        This is how we apply different LoRA strengths to each item in a batch.
        """
        for lora_module in self.unet_loras.values():
            lora_module.multiplier = scales
            
    def forward(self, *args, **kwargs):
        return self.unet(*args, **kwargs)


# ==============================================================================
# SECTION 4: TESTER UVIT MODEL (REVISED)
# ==============================================================================

class TestAttentionBlock(nn.Module):
    def __init__(self, channels, ff_mult=4):
        super().__init__()
        # FIX 1: Use separate q, k, v layers to match the config rules.
        self.attn = nn.ModuleDict({
            'to_q': nn.Linear(channels, channels),
            'to_k': nn.Linear(channels, channels),
            'to_v': nn.Linear(channels, channels),
            'to_out': nn.Linear(channels, channels),
        })
        
        # FIX 2: Use a ModuleDict for the FFN to get the '.net.' naming.
        self.ff = nn.ModuleDict({
            'net': nn.Sequential(
                nn.Linear(channels, channels * ff_mult),
                nn.GELU(),
                nn.Linear(channels * ff_mult, channels)
            )
        })

    def forward(self, x):
        # The forward pass now uses the separated Q, K, V layers.
        q = self.attn.to_q(x)
        k = self.attn.to_k(x)
        v = self.attn.to_v(x)
        x = x + self.attn.to_out(q * k * v) # simplified attention
        
        # The FFN forward pass now calls through the 'net' ModuleDict.
        x = x + self.ff.net(x)
        return x

class TesterUViT(nn.Module):
    # This class remains unchanged, as the fix is entirely within TestAttentionBlock.
    def __init__(self):
        super().__init__()
        # Scale up all channel dimensions
        self.initial_conv = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList([
            nn.ModuleDict({'attentions': nn.ModuleList([TestAttentionBlock(128)])})
        ])
        self.mid_block = nn.ModuleDict({
            'attentions': nn.ModuleList([TestAttentionBlock(128), TestAttentionBlock(128)])
        })
        self.up_blocks = nn.ModuleList([
            nn.ModuleDict({'attentions': nn.ModuleList([TestAttentionBlock(128)])})
        ])
        self.final_conv = nn.Conv2d(128, 3, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.initial_conv(x); x = x.permute(0, 2, 3, 1)
        for block in self.down_blocks:
            for attn in block.attentions: x = attn(x)
        skip_connection = x
        for attn in self.mid_block.attentions: x = attn(x)
        for block in self.up_blocks:
            x = x + skip_connection
            for attn in block.attentions: x = attn(x)
        x = x.permute(0, 3, 1, 2); return self.final_conv(x)

# ==============================================================================
# SECTION 5: MAIN TEST HARNESS (REVISED)
# ==============================================================================
import datetime
import shutil # For easier cleanup

# In flexible_lora_system.py
def _internal_consistency_check(config_dict, log_dir="advlogs"):
    """
    Takes a config, runs the full pipeline, and returns a performance score.
    This is our core evaluation engine for the search.
    """
    log_root = log_dir
    run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_log_dir = os.path.join(log_root, f"test_run_{run_timestamp}")
    os.makedirs(run_log_dir, exist_ok=True)
    print(f"--- Created temporary log directory for this run: {run_log_dir} ---")

    # Define paths for all artifacts within the unique run directory
    config_path = os.path.join(run_log_dir, "test_config.yaml")
    svd_path = os.path.join(run_log_dir, "svd_ranks.json")
    alpha_path = os.path.join(run_log_dir, "alpha_ratios.json")
    
    with open(config_path, 'w') as f: yaml.dump(config_dict, f)
    
    try:
        print("\n--- PHASE 1: SETUP AND MODEL CREATION ---")

        # --- 2. Run implemented estimators to generate their outputs ---
        # Note: We pass the constructor of the model, not an instance
        uvit_builder = TesterUViT 
        estimate_ranks_from_svd(uvit_builder(), save_path=svd_path)
        warmup_and_estimate_alphas(uvit_builder, config_dict, save_path=alpha_path, epochs=2)
        
        # --- 3. Load and resolve the configuration ---
        print("\n--- Resolving LoRA Configuration ---")
        final_uvit = uvit_builder()
        #loader = LoRAConfigLoader(config_path, None, svd_path, alpha_path)
        loader = LoRAConfigLoader(
        config_path=config_path, 
        rank_estimates_path=svd_path, 
        alpha_estimates_path=alpha_path
        )
        resolved_config = loader.get_resolved_config(final_uvit)
        
        # --- 4. Instantiate the final network ---
        print("\n--- Applying LoRA to Tester UViT ---")
        print(f"the config: {resolved_config}")
        network = FlexibleLoRANetwork(final_uvit, resolved_config)
        print(network)
        
        # --- 5. Run a minimal training loop to test interfaces ---
        print("\n--- PHASE 2: TESTING TRAINING INTERFACES ---")
        dummy_dataset = [torch.randn(1, 3, 16, 16), torch.randn(1, 3, 16, 16)]
        optimizer = torch.optim.AdamW(network.prepare_optimizer_params(), lr=1e-4)
        
        print("Starting mini training loop for 200 steps...")
        final_loss = 0.0
        for i in range(200):
            data = dummy_dataset[i % 2]
            optimizer.zero_grad()
            output = network(data)
            loss = output.mean()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f"  Step {i+1}, Loss: {loss.item():.4f} - OK")
            final_loss = loss.item()
            
        #print("\n--- TEST COMPLETE: ALL INTERFACES FUNCTIONAL ---")
        
    finally:
        # --- 6. Clean up the entire run-specific log directory ---
        #if os.path.exists(run_log_dir):
        #    shutil.rmtree(run_log_dir)
        #    print(f"\nCleaned up temporary log directory: {run_log_dir}")
        print(f"huh?")
    return final_loss

def main_():
    """Main function to run an end-to-end test of the flexible LoRA system."""

    # --- 1. Create a unique, timestamped directory for this run's artifacts ---
    log_root = "advlogs"
    run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_log_dir = os.path.join(log_root, f"test_run_{run_timestamp}")
    os.makedirs(run_log_dir, exist_ok=True)
    print(f"--- Created temporary log directory for this run: {run_log_dir} ---")

    # Define paths for all artifacts within the unique run directory
    config_path = os.path.join(run_log_dir, "test_config.yaml")
    svd_path = os.path.join(run_log_dir, "svd_ranks.json")
    alpha_path = os.path.join(run_log_dir, "alpha_ratios.json")
    
    # This config now acts as the 'base' config for the alpha warmup estimator
    config_dict = {
        'lora_rules': [
            {'name': "Mid_Block_QK_High_Prio", 'priority': 1.0, 'rank': 'auto_svd', 'alpha': 2.0, 'init_scheme': 'kaiming_cheald', 'target_modules': ['mid_block.attentions.1.attn.to_q'],},
            {'name': "All_Other_Attention", 'priority': 0.5, 'rank': 4, 'alpha': 1.0, 'init_scheme': 'kaiming_cheald', 'target_name_contains': ['attn'],},
            {'name': "Mid_Block_FFN", 'priority': 0.8, 'rank': 32, 'target_modules': ['mid_block.attentions.1.ff.net.2'],}
        ]
    }

    with open(config_path, 'w') as f: yaml.dump(config_dict, f)
    
    try:
        print("\n--- PHASE 1: SETUP AND MODEL CREATION ---")

        # --- 2. Run implemented estimators to generate their outputs ---
        # Note: We pass the constructor of the model, not an instance
        uvit_builder = TesterUViT 
        estimate_ranks_from_svd(uvit_builder(), save_path=svd_path)
        warmup_and_estimate_alphas(uvit_builder, config_dict, save_path=alpha_path, epochs=2)
        
        # --- 3. Load and resolve the configuration ---
        print("\n--- Resolving LoRA Configuration ---")
        final_uvit = uvit_builder()
        #loader = LoRAConfigLoader(config_path, None, svd_path, alpha_path)
        loader = LoRAConfigLoader(
        config_path=config_path, 
        rank_estimates_path=svd_path, 
        alpha_estimates_path=alpha_path
        )
        resolved_config = loader.get_resolved_config(final_uvit)
        
        # --- 4. Instantiate the final network ---
        print("\n--- Applying LoRA to Tester UViT ---")
        print(f"the config: {resolved_config}")
        network = FlexibleLoRANetwork(final_uvit, resolved_config)
        print(network)
        
        # --- 5. Run a minimal training loop to test interfaces ---
        print("\n--- PHASE 2: TESTING TRAINING INTERFACES ---")
        dummy_dataset = [torch.randn(1, 3, 16, 16), torch.randn(1, 3, 16, 16)]
        optimizer = torch.optim.AdamW(network.prepare_optimizer_params(), lr=1e-4)
        
        print("Starting mini training loop for 200 steps...")
        for i in range(200):
            data = dummy_dataset[i % 2]
            optimizer.zero_grad()
            output = network(data)
            loss = output.mean()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f"  Step {i+1}, Loss: {loss.item():.4f} - OK")
            
        print("\n--- TEST COMPLETE: ALL INTERFACES FUNCTIONAL ---")
        
    finally:
        # --- 6. Clean up the entire run-specific log directory ---
        #if os.path.exists(run_log_dir):
        #    shutil.rmtree(run_log_dir)
        #    print(f"\nCleaned up temporary log directory: {run_log_dir}")
        print(f"huh?")

if __name__ == "__main__":
    main_()