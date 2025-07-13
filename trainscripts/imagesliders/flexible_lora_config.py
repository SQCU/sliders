# flexible_lora_config.py
import yaml
import json
import torch
from collections import defaultdict

class LoRAConfigLoader:
    """
    Parses a declarative LoRA config file and resolves it using (optional)
    pre-computed estimates for rank and alpha. This creates a fully specified
    and reproducible plan for building the FlexibleLoRANetwork.
    """
    def __init__(self, config_path, rank_estimates_path=None, alpha_estimates_path=None):
        with open(config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)

        self.rank_estimates = self._load_json(rank_estimates_path)
        self.alpha_estimates = self._load_json(alpha_estimates_path)

    def _load_json(self, path):
        if path and os.path.exists(path):
            print(f"Loading estimates from: {path}")
            with open(path, 'r') as f:
                return json.load(f)
        return {}

    def get_resolved_config(self, unet_module_names: list) -> dict:
        """
        Processes the base config and estimates to produce a final, per-module config.
        This is the main entry point.
        """
        module_to_rule = {}
        
        # --- Step 1: Greedily process rules and apply them to modules ---
        # The priority can be used for more advanced sampling, but here we just apply all rules.
        sorted_rules = sorted(self.base_config.get('lora_rules', []), key=lambda x: x['priority'], reverse=True)
        
        for rule in sorted_rules:
            for module_name in unet_module_names:
                if self._module_matches_rule(module_name, rule):
                    # Higher priority rules can overwrite lower ones if modules overlap
                    module_to_rule[module_name] = rule
        
        # --- Step 2: Resolve the specific parameters for each targeted module ---
        resolved_lora_map = {}
        for module_name, rule in module_to_rule.items():
            
            # Resolve rank
            rank = rule.get('rank')
            if rank == 'auto_svd':
                if module_name in self.rank_estimates:
                    rank = self.rank_estimates[module_name]
                else:
                    # Fallback if an estimate is missing for a targeted module
                    print(f"WARN: SVD rank estimate for '{module_name}' not found. Using default of 4.")
                    rank = 4
            
            # Resolve initial alpha
            # The base alpha is scaled by the learned importance ratio.
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
            
        # The final output is a flat map from module name to its exact LoRA config.
        return {"lora_map": resolved_lora_map}

    def _module_matches_rule(self, module_name, rule):
        # A simple matching logic. Can be made more complex (e.g., regex).
        if 'target_modules' in rule:
            if module_name in rule['target_modules']:
                return True
        if 'target_name_contains' in rule:
            if any(substring in module_name for substring in rule['target_name_contains']):
                return True
        return False