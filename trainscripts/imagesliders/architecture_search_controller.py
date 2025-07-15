# architecture_search_controller.py
# python -m trainscripts.imagesliders.architecture_search_controller 
# This script defines the next layer of the system: a controller that uses
# the 'locked-in' flexible_lora_system to perform an automated architecture search.

import random
import yaml
import pandas as pd
import datetime
import os
import json # <-- Add json import

# We assume the 'run_experiment' function is available from your locked-in script.
# For a real project, this would be `from flexible_lora_system import run_experiment`
from .flexible_lora_system import run_all_estimators, TesterUViT
from .generative_lora_harness import run_generative_experiment # our real test import :)

class RandomSearchController:
    """
    A simple controller that performs a random search over a defined
    LoRA configuration space.
    """
    def __init__(self, search_space_path: str, results_log_path: str, 
                 rank_estimates_path: str, alpha_estimates_path: str,
                 gen_exp_kwargs={}):
        """
        Initializes the controller.

        Args:
            search_space_path (str): Path to a YAML file defining the search space.
            results_log_path (str): Path to a CSV file to log experiment results.
        """
        with open(search_space_path, 'r') as f:
            self.search_space = yaml.safe_load(f)
        self.results_log_path = results_log_path
        self.results = []
        self.gen_exp_kwargs = gen_exp_kwargs
        print("--- Architecture Search Controller Initialized ---")
        # --- NEW: Load the importance data ---
        print("--- Loading Importance Tensors for Guided Search ---")
        with open(rank_estimates_path, 'r') as f:
            self.rank_estimates = json.load(f)
            print(f"  > Loaded {len(self.rank_estimates)} SVD rank estimates.")
        with open(alpha_estimates_path, 'r') as f:
            self.alpha_estimates = json.load(f)
            print(f"  > Loaded {len(self.alpha_estimates)} Alpha ratio estimates.")
        print(f"  > Search space loaded from: {search_space_path}")
        print(f"  > Results will be logged to: {self.results_log_path}")
        print("--------------------------------------------------")

    def _get_layers_from_importance(self, target_type: str, k: int) -> list:
        """
        Selects the top/bottom k layers based on the specified importance metric.
        """
        if 'svd' in target_type:
            source_dict = self.rank_estimates
        elif 'alpha' in target_type:
            source_dict = self.alpha_estimates
        else:
            return []

        # Sort the dictionary by value (the rank or alpha ratio)
        sorted_layers = sorted(source_dict.items(), key=lambda item: item[1], reverse=('top' in target_type))
        
        # Return the names of the top/bottom k layers
        return [name for name, score in sorted_layers[:k]]


    def _sample_config(self) -> dict:
        """
        Generates a single, random LoRA configuration dictionary based on the
        defined search space.
        """
        config_dict = {'lora_rules': []}
        num_rules = random.randint(
            self.search_space['num_rules']['min'],
            self.search_space['num_rules']['max']
        )
        
        for i in range(num_rules):
            target_type = random.choice(list(self.search_space['targets'].keys()))
            target_params = self.search_space['targets'][target_type]

            rule = {
                'name': f"{target_type}_rule_{i}",
                'priority': random.uniform(0.1, 1.0),
                'rank': random.choice(self.search_space['rank']),
                'alpha': random.choice(self.search_space['alpha']),
                'init_scheme': random.choice(self.search_space['init_scheme']),
                'train_alpha': random.choice(self.search_space['train_alpha']),
            }
            
            # --- NEW: Importance-Guided Target Selection ---
            if 'importance_based' in target_params and target_params['importance_based']:
                # Sample a 'k' value (e.g., how many top layers to grab)
                k = random.choice(self.search_space['k_values'])
                
                # Get the actual layer names from our importance data
                layer_names = self._get_layers_from_importance(target_type, k)
                
                # The rule now targets these specific modules
                rule['target_modules'] = layer_names
                print(f"  > Sampled importance rule '{target_type}' with k={k}, targeting {len(layer_names)} specific modules.")
            else:
                # Fallback to the old method for general targets
                rule['target_name_contains'] = target_params['target_name_contains']
            # --- END NEW ---
                
            config_dict['lora_rules'].append(rule)
            
        return config_dict

    def run_search(self, num_trials: int):
        """
        Executes the main search loop for a specified number of trials.
        """
        for i in range(num_trials):
            print(f"\n{'='*20} Search Trial {i+1}/{num_trials} {'='*20}")
            
            # 1. Generate a new random configuration
            config_to_test = self._sample_config()
            print("Generated Config:")
            print(yaml.dump(config_to_test, indent=2))
            
            # 2. Run the full experiment with this config
            # Your locked-in function is the "fitness function" of our search.
            try:
                # This now returns a network, dictionary
                _, performance_results = run_generative_experiment(config_to_test, **self.gen_exp_kwargs)
                primary_score = performance_results['learning_delta']
                
                # Log the results
                result = {
                    'trial_id': i,
                    'timestamp': datetime.datetime.now().isoformat(),
                    'performance_score': primary_score, 
                    'lpips' : performance_results['lpips'],
                    'fid' : performance_results['fid'],
                    'full_results': performance_results, # Store the full dictionary for detailed analysis
                    'config': yaml.dump(config_to_test)
                }
                self.results.append(result)
                del _
                
            except Exception as e:
                print(f"--- TRIAL {i+1} FAILED ---")
                print(f"Error: {e}")
                import traceback
                traceback.print_exc() # Print full traceback for easier debugging
                # Log the failure
                result = {
                    'trial_id': i,
                    'timestamp': datetime.datetime.now().isoformat(),
                    'performance_score': 'FAILED',
                    'config': yaml.dump(config_to_test)
                }
                self.results.append(result)
                 #safeguard. delete those returned networks on purpose since we aren't using em anymore in here.
            # 4. Save results to CSV incrementally
            pd.DataFrame(self.results).to_csv(self.results_log_path, index=False)
            print(f"\n--- Trial {i+1} complete. Results saved. Best score so far: {self._get_best_score()} ---")

        print("\n--- Search Complete ---")
        print("Final results logged to:", self.results_log_path)
        return pd.DataFrame(self.results)

    def _get_best_score(self):
        valid_scores = [r['performance_score'] for r in self.results if isinstance(r['performance_score'], (int, float))]
        if not valid_scores:
            return "N/A"
        # lower is better on our LPIPS-FID
        # The delta is negative, so the MIN score is the BEST score
        return min(valid_scores) if valid_scores else float('inf')


def create_search_space_yaml(path="search_space.yaml"):
    """Helper function to create an example search space config file."""
    search_space = {
        # Defines how many rules a single config can have
        'num_rules': {'min': 1, 'max': 4},
        
        # Defines the discrete choices for each hyperparameter in a rule
        'rank': [4, 8, 16, 32], # Removed 'auto_svd' here, as it's now a targeting strategy
        'alpha': [0.5, 1.0, 2.0, 4.0],
        'init_scheme': ['kaiming_cheald', 'standard'],
        'train_alpha': [True, False],

        # --- NEW: 'k' values for importance sampling ---
        'k_values': [3, 5, 8, 12], # e.g., "target the top 3 layers" or "top 5 layers"
        
        # Defines pre-canned targeting strategies
        'targets': {
            #importance sampling targets
            'top_k_svd': {'importance_based': True},
            'top_k_alpha': {'importance_based': True},
            'bottom_k_svd': {'importance_based': True}, # Target least important layers (useful sanity check)
            #module level targets
            'all_attention': {'target_name_contains': ['attn']},
            'all_ffn': {'target_name_contains': ['ff.net']},
            'mid_block_only': {'target_name_contains': ['mid_block']},
            'up_blocks_only': {'target_name_contains': ['up_blocks']},
            'qkv_only': {'target_name_contains': ['to_q', 'to_k', 'to_v']}
        }
    }
    with open(path, 'w') as f:
        yaml.dump(search_space, f, indent=2)
    print(f"Created example search space config at: {path}")
    return path


if __name__ == "__main__":
    # This is the new main entry point for running a full search experiment.
    
    # --- 1. PRE-COMPUTE: Run the estimators ONCE before the search begins ---
    print("--- STEP 1: Running Pre-computation Estimators ---")
    
    warmup_config = {
        'lora_rules': [{'name': "warmup_rule", 'rank': 4, 'target_name_contains': ['attn', 'ff.net']}]
    }
    # Run the estimators, passing the REAL training harness to the alpha warmer.
    svd_path, alpha_path = run_all_estimators(
        unet_builder=TesterUViT,
        base_config_for_warmup=warmup_config,
        training_harness_fn=run_generative_experiment, # Dependency Injection!
        gen_exp_kwargs = {'num_epochs':5, 'eval_every_n_epochs':999},
        force_rerun = True,
    )
    
     # === STEP 2: SEARCH PHASE ===
    # The controller is initialized with the PATHS to the artifacts. It doesn't
    # know or care how they were made.
    search_space_file = create_search_space_yaml()
    results_file = "search_results_final.csv"
    
    controller = RandomSearchController(
        search_space_path=search_space_file,
        results_log_path=results_file,
        rank_estimates_path=svd_path,
        alpha_estimates_path=alpha_path,
        gen_exp_kwargs = {'num_epochs':10, 'eval_every_n_epochs':3},
    )

    # The controller's run_search method is now pure. It calls the harness,
    # which has also been purified. The data flows one way.
    controller.run_search(num_trials=10)