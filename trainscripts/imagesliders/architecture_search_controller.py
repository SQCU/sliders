# architecture_search_controller.py
# python -m trainscripts.imagesliders.architecture_search_controller 
# This script defines the next layer of the system: a controller that uses
# the 'locked-in' flexible_lora_system to perform an automated architecture search.

import random
import yaml
import pandas as pd
import datetime
import os

# We assume the 'run_experiment' function is available from your locked-in script.
# For a real project, this would be `from flexible_lora_system import run_experiment`
from .flexible_lora_system import run_experiment # Placeholder for actual import

class RandomSearchController:
    """
    A simple controller that performs a random search over a defined
    LoRA configuration space.
    """
    def __init__(self, search_space_path: str, results_log_path: str):
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
        print("--- Architecture Search Controller Initialized ---")
        print(f"  > Search space loaded from: {search_space_path}")
        print(f"  > Results will be logged to: {self.results_log_path}")

    def _sample_config(self) -> dict:
        """
        Generates a single, random LoRA configuration dictionary based on the
        defined search space.
        """
        config_dict = {'lora_rules': []}
        
        # Determine how many rules to generate for this sample
        num_rules = random.randint(
            self.search_space['num_rules']['min'],
            self.search_space['num_rules']['max']
        )
        
        for i in range(num_rules):
            # Sample which layers to target for this rule
            target_type = random.choice(list(self.search_space['targets'].keys()))
            target_params = self.search_space['targets'][target_type]

            rule = {
                'name': f"{target_type}_rule_{i}",
                'priority': random.uniform(0.1, 1.0),
                'rank': random.choice(self.search_space['rank']),
                'alpha': random.choice(self.search_space['alpha']),
                'init_scheme': random.choice(self.search_space['init_scheme']),
                'train_alpha': random.choice(self.search_space['train_alpha']),
                'target_name_contains': target_params['target_name_contains']
            }
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
                performance_score = run_experiment(config_to_test)
                
                # 3. Log the results
                result = {
                    'trial_id': i,
                    'timestamp': datetime.datetime.now().isoformat(),
                    'performance_score': performance_score,
                    'config': yaml.dump(config_to_test) # Store the exact config for reproducibility
                }
                self.results.append(result)
                
            except Exception as e:
                print(f"--- TRIAL {i+1} FAILED ---")
                print(f"Error: {e}")
                # Log the failure
                result = {
                    'trial_id': i,
                    'timestamp': datetime.datetime.now().isoformat(),
                    'performance_score': 'FAILED',
                    'config': yaml.dump(config_to_test)
                }
                self.results.append(result)

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
        # Assuming lower loss is better
        return min(valid_scores)


def create_search_space_yaml(path="search_space.yaml"):
    """Helper function to create an example search space config file."""
    search_space = {
        # Defines how many rules a single config can have
        'num_rules': {'min': 1, 'max': 4},
        
        # Defines the discrete choices for each hyperparameter in a rule
        'rank': [4, 8, 16, 32, 'auto_svd'],
        'alpha': [0.5, 1.0, 2.0, 4.0],
        'init_scheme': ['kaiming_cheald', 'standard'],
        'train_alpha': [True, False],
        
        # Defines pre-canned targeting strategies
        'targets': {
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
    
    # 1. Define the search space
    search_space_file = create_search_space_yaml()
    
    # 2. Define where to save the results
    results_file = "search_results.csv"
    
    # 3. Initialize the controller
    controller = RandomSearchController(
        search_space_path=search_space_file,
        results_log_path=results_file
    )
    
    # 4. Launch the search for a set number of trials
    controller.run_search(num_trials=10)