# flexible_lora_system.py
# This script contains the full, self-contained system for the first phase of testing.
# It includes initializers, estimator stubs, config loading, the flexible LoRA network,
# a test model, and a main function to run an end-to-end internal consistency check.

import torch
import torch.nn as nn
import math
import yaml
import json
import os
from collections import defaultdict

# ==============================================================================
# SECTION 1: LORA INITIALIZERS & ESTIMATOR STUBS
# (Formerly lora_initializers.py and lora_estimators.py)
# ==============================================================================

# A simple dictionary to cache computed scaling factors for kaiming_cheald.
_kaiming_cheald_cache = {}

def init_kaiming_cheald_(lora_down: nn.Module, lora_up: nn.Module):
    """
    Initializes LoRA up/down matrices such that the norm of their product
    approximates the norm of a full-rank matrix initialized with Kaiming Uniform.
    This helps normalize the initial impact of LoRA modules across different ranks and shapes.
    """
    out_features, in_features = lora_up.weight.shape[0], lora_down.weight.shape[1]
    rank = lora_down.weight.shape[0]
    cache_key = f"{in_features}-{out_features}-{rank}"

    if cache_key in _kaiming_cheald_cache:
        scale = _kaiming_cheald_cache[cache_key]
    else:
        print(f"  [Kaiming-Cheald] Calibrating for shape ({out_features}, {in_features}) rank {rank}...")
        W_temp = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(W_temp, a=math.sqrt(5))
        target_norm = torch.norm(W_temp)
        nn.init.uniform_(lora_down.weight, -0.01, 0.01)
        nn.init.uniform_(lora_up.weight, -0.01, 0.01)
        with torch.no_grad():
            current_norm = torch.norm(lora_up.weight @ lora_down.weight)
        scale = target_norm / (current_norm + 1e-9)
        _kaiming_cheald_cache[cache_key] = scale

    with torch.no_grad():
        lora_up.weight.mul_(scale)

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
    (STUB) Estimates the optimal rank for each layer via SVD.
    In a real implementation, this would perform SVD on each layer's weight
    and return a map of {module_name: estimated_rank}. This allows us to
    allocate parameters intelligently. The output is persisted to `save_path`
    to skip this slow phase during repetitive tests.
    """
    print("--- [Stub] Running SVD Rank Estimation... ---")
    estimated_ranks = {
        # High importance for the query/key in the mid-block attention
        "mid_block.attentions.0.attn.to_q": 16,
        "mid_block.attentions.0.attn.to_k": 16,
        # Lower importance for the value/out projection
        "mid_block.attentions.0.attn.to_v": 4,
        "mid_block.attentions.0.attn.to_out": 8,
        # Higher rank for the deeper FFN layer
        "mid_block.attentions.1.ff.net.2": 32,
        # Lower rank for the upsampling attention
        "up_blocks.0.attentions.0.attn.to_q": 4,
    }
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(estimated_ranks, f, indent=2)
        print(f"  > Saved SVD rank estimates to {save_path}")
    return estimated_ranks

def warmup_and_estimate_alphas(unet, base_config, save_path=None):
    """
    (STUB) Performs a short training run to find optimal alpha ratios per layer.
    A real implementation trains a model where alphas are learnable parameters,
    then computes `final_alpha / initial_alpha` to get an importance score.
    This score pre-scales the initial alphas in subsequent real training runs,
    effectively pre-tuning the learning rates per layer. The output is persisted
    to `save_path` to skip this phase in search loops.
    """
    print("--- [Stub] Running Online Alpha Warmup... ---")
    alpha_ratios = {
        "mid_block.attentions.0.attn.to_q": 1.8, # Very important
        "mid_block.attentions.0.attn.to_k": 1.7,
        "mid_block.attentions.1.ff.net.2": 1.2, # Moderately important
        "up_blocks.0.attentions.0.attn.to_q": 0.3, # This layer seems to overfit; reduce its LR
    }
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(alpha_ratios, f, indent=2)
        print(f"  > Saved Alpha Warmup estimates to {save_path}")
    return alpha_ratios

# ==============================================================================
# SECTION 2: CONFIGURATION LOADER
# ==============================================================================

class LoRAConfigLoader:
    def __init__(self, config_path, rank_estimates_path=None, alpha_estimates_path=None):
        with open(config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        self.rank_estimates = self._load_json(rank_estimates_path)
        self.alpha_estimates = self._load_json(alpha_estimates_path)

    def _load_json(self, path):
        if path and os.path.exists(path):
            print(f"Loading estimates from: {path}")
            with open(path, 'r') as f: return json.load(f)
        return {}

    def get_resolved_config(self, unet_module_names: list) -> dict:
        module_to_rule = {}
        sorted_rules = sorted(self.base_config.get('lora_rules', []), key=lambda x: x.get('priority', 1.0), reverse=True)
        
        for rule in sorted_rules:
            for module_name in unet_module_names:
                if self._module_matches_rule(module_name, rule):
                    if module_name not in module_to_rule: # First rule (highest priority) wins
                         module_to_rule[module_name] = rule
        
        resolved_lora_map = {}
        for module_name, rule in module_to_rule.items():
            rank = rule.get('rank')
            if rank == 'auto_svd':
                rank = self.rank_estimates.get(module_name, 4) # Default to 4 if not in estimate map
            
            base_alpha = rule.get('alpha', 1.0)
            importance_ratio = self.alpha_estimates.get(module_name, 1.0)
            initial_alpha = base_alpha * importance_ratio

            resolved_lora_map[module_name] = {
                "rank": rank,
                "alpha": initial_alpha,
                "train_alpha": rule.get('train_alpha', False),
                "init_scheme": rule.get('init_scheme', 'standard'),
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
                lora_config = lora_map[name]
                lora_name = f"lora_unet_{name.replace('.', '_')}"
                lora_module = FlexibleLoRAModule(module, **lora_config)
                self.unet_loras[lora_name] = lora_module
                path_parts = name.split('.')
                parent = root_module
                for part in path_parts[:-1]:
                    parent = getattr(parent, part)
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
    def __init__(self, channels, ff_mult=2):
        super().__init__()
        # Use a single linear layer for QKV for simplicity in this test model
        self.attn = nn.ModuleDict({
            'to_qkv': nn.Linear(channels, channels * 3),
            'to_out': nn.Linear(channels, channels),
        })
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * ff_mult),
            nn.GELU(),
            nn.Linear(channels * ff_mult, channels)
        )
    def forward(self, x):
        # A dummy attention pass that preserves shape
        q, k, v = self.attn.to_qkv(x).chunk(3, dim=-1)
        x = x + self.attn.to_out(q * k * v) # simplified attention
        x = x + self.ff(x)
        return x

class TesterUViT(nn.Module):
    """
    A fictional UViT with realistic naming conventions for testing.
    REVISED to include a long-range skip connection from the down_block
    to the up_block, which is essential for any U-style architecture.
    """
    def __init__(self):
        super().__init__()
        self.initial_conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        
        self.down_blocks = nn.ModuleList([
            nn.ModuleDict({'attentions': nn.ModuleList([TestAttentionBlock(8)])})
        ])
        
        self.mid_block = nn.ModuleDict({
            'attentions': nn.ModuleList([TestAttentionBlock(8), TestAttentionBlock(8)])
        })
        
        self.up_blocks = nn.ModuleList([
            # NOTE: In a real U-Net, this layer would handle concatenated input
            # (e.g., channels * 2). For this test harness, we use addition
            # to keep channel dimensions consistent and test the data path.
            nn.ModuleDict({'attentions': nn.ModuleList([TestAttentionBlock(8)])})
        ])
        
        self.final_conv = nn.Conv2d(8, 3, kernel_size=3, padding=1)
    
    def forward(self, x):
        # --- Encoder Path ---
        x = self.initial_conv(x)
        x = x.permute(0, 2, 3, 1) # B, C, H, W -> B, H, W, C
        
        # Process through down-blocks
        for block in self.down_blocks:
            for attn in block.attentions:
                x = attn(x)
        
        # *** CAPTURE THE SKIP CONNECTION ***
        # This is the output of the encoder path at this resolution.
        skip_connection = x
        
        # --- Bottleneck ---
        for attn in self.mid_block.attentions:
            x = attn(x)
            
        # --- Decoder Path ---
        for block in self.up_blocks:
            # *** APPLY THE SKIP CONNECTION ***
            # In a real U-Net, this would be torch.cat, but addition tests the
            # long-range data flow perfectly without altering channel dimensions.
            x = x + skip_connection
            
            for attn in block.attentions:
                x = attn(x)
                
        # --- Final Output ---
        x = x.permute(0, 3, 1, 2) # B, H, W, C -> B, C, H, W
        return self.final_conv(x)

# ==============================================================================
# SECTION 5: MAIN TEST HARNESS (REVISED)
# ==============================================================================
import datetime
import shutil # For easier cleanup

def main_():
    """Main function to run an end-to-end test of the flexible LoRA system."""

    # --- 1. Create a unique, timestamped directory for this run's artifacts ---
    log_root = "advlogs"
    run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_log_dir = os.path.join(log_root, f"test_run_{run_timestamp}")
    
    # This single call creates both `advlogs` and the run-specific subfolder.
    os.makedirs(run_log_dir, exist_ok=True)
    print(f"--- Created temporary log directory for this run: {run_log_dir} ---")

    # Define paths for all artifacts within the unique run directory
    config_path = os.path.join(run_log_dir, "test_config.yaml")
    svd_path = os.path.join(run_log_dir, "svd_ranks.json")
    alpha_path = os.path.join(run_log_dir, "alpha_ratios.json")
    
    config_yaml_content = """
    lora_rules:
      - name: "Mid_Block_QK_High_Prio"
        priority: 1.0
        rank: 'auto_svd'
        alpha: 2.0
        train_alpha: True
        init_scheme: 'kaiming_cheald'
        target_name_contains:
          - 'mid_block.attentions.0.attn.to_q'
          - 'mid_block.attentions.0.attn.to_k'
      
      - name: "All_Other_Attention"
        priority: 0.5
        rank: 4
        alpha: 1.0
        train_alpha: False
        target_name_contains: ['attn']

      - name: "Mid_Block_FFN"
        priority: 0.8
        rank: 'auto_svd'
        target_modules: ['mid_block.attentions.1.ff.net.2']
    """
    with open(config_path, 'w') as f: f.write(config_yaml_content)
    
    try:
        print("\n--- PHASE 1: SETUP AND MODEL CREATION ---")
        uvit = TesterUViT()
        
        # --- 2. Run stubbed estimators to generate their outputs ---
        print(f"svd path: {svd_path}")
        estimate_ranks_from_svd(uvit, save_path=svd_path)
        warmup_and_estimate_alphas(uvit, None, save_path=alpha_path)
        
        # --- 3. Load and resolve the configuration ---
        print("\n--- Resolving LoRA Configuration ---")
        loader = LoRAConfigLoader(config_path, svd_path, alpha_path)
        all_module_names = [name for name, _ in uvit.named_modules()]
        resolved_config = loader.get_resolved_config(all_module_names)
        
        # --- 4. Instantiate the final network ---
        print("\n--- Applying LoRA to Tester UViT ---")
        network = FlexibleLoRANetwork(uvit, resolved_config)
        print(network)
        
        # --- 5. Run a minimal training loop to test interfaces ---
        print("\n--- PHASE 2: TESTING TRAINING INTERFACES ---")
        dummy_dataset = [torch.randn(1, 3, 16, 16), torch.randn(1, 3, 16, 16)]
        optimizer = torch.optim.AdamW(network.prepare_optimizer_params(), lr=1e-4)
        
        print("Starting mini training loop for 2 steps...")
        for i in range(2):
            data = dummy_dataset[i % 2]
            optimizer.zero_grad()
            output = network(data)
            loss = output.mean()
            loss.backward()
            optimizer.step()
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