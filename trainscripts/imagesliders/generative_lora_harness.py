# generative_lora_harness.py
# The robust, doctrinally-sound generative experiment harness. Version 3.
# python -m trainscripts.imagesliders.generative_lora_harness

import torch
import torch.nn.functional as F
import torch.nn as nn
import yaml
import copy
from tqdm import tqdm
import random

# --- Core Imports ---
from .flexible_lora_system import FlexibleLoRANetwork, LoRAConfigLoader, TesterUViT
from .minimal_scheduler import MinimalDDPMScheduler
from .batch_data_pipeline import materialize_sham_dataset
from .batch_model_util import run_evaluation_flow, testeruvit_decoder_fn, testeruvit_diffusion_fn # Assuming this exists for eval
from .batch_model_util import diffusion_fn as sdxl_diffusion_fn, vae_decoder_fn as sdxl_decoder_fn

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance

import warnings

# Suppress all FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning) 

# ==============================================================================
# SECTION 1: THE EVALUATOR (Now Doctrine-Compliant)
# ==============================================================================

class GenerativeEvaluator:
    """
    Evaluates a model using the centralized `run_evaluation_flow` utility.
    This class is now stateless during the evaluation call itself.
    """
    def __init__(self, eval_dataset: dict, environment: dict):
        self.device = environment['device']
        self.env = environment
        
        # We now assume the dataset is already prepared
        self.model_inputs = eval_dataset['model_inputs']
        self.ground_truth_outputs = torch.cat(eval_dataset['ground_truth_outputs']).to(self.device)

        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(self.device)
        self.fid = FrechetInceptionDistance(feature=64).to(self.device)

    @torch.no_grad()
    def evaluate(self, model_to_test: nn.Module, sampling_config: dict) -> dict:
        """
        Calls the stateless evaluation flow and computes metrics.
        The model_to_test is the specific network for the current trial.
        """
        model_to_test.eval()
        print(f"--- [Evaluator] Beginning evaluation...")

        # Doctrine: Use the central, stateless function for the generation process.
        # This function encapsulates the OffloadingOrchestrator and diffusion loop.
        generated_images_list = run_evaluation_flow(
        model_to_test=model_to_test,
        environment=self.env,
        # The model_inputs is now the list of workload dicts
        workload=self.model_inputs, 
        sampling_config=sampling_config
    )
        generated_images = torch.cat(generated_images_list).to(self.device, dtype=torch.float32)

        # Correct FID usage: update both distributions before computing.
        self.fid.update((self.ground_truth_outputs * 255).byte(), real=True)
        self.fid.update((generated_images * 255).byte(), real=False)
        fid_score = self.fid.compute()
        self.fid.reset()

        lpips_score = self.lpips(generated_images, self.ground_truth_outputs)
        
        # The primary score to minimize (lower is better)
        primary_score = lpips_score + fid_score * 0.1

        metrics = {
            "primary_score": primary_score.item(),
            "lpips": lpips_score.item(),
            "fid": fid_score.item(),
        }
        print(f"--- Evaluation Complete: {metrics} ---")
        return metrics

#
# SECTION! TWO! IT'S THE TRIALY! TRAINER!
#

def train_epoch(network, dataloader, scheduler, optimizer, training_config, device):
    """
    A stateless function to perform one epoch of training.
    """
    network.train()
    losses = []
    progress_bar = tqdm(dataloader, desc="Training Batches", leave=False)
    for batch in progress_bar:
        optimizer.zero_grad()
        clean_images = batch['clean_images'].to(device)
        noise = torch.randn_like(clean_images)
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (clean_images.shape[0],), device=device).long()
        
        noisy_images = scheduler.add_noise(clean_images, noise, timesteps)
        target = scheduler.get_prediction_target(clean_images, noise, timesteps, prediction_type=training_config['prediction_type'])
        
        # The model call is now generic. It works as long as the base model
        # accepts (input_tensor, timesteps).
        model_output = network(noisy_images, timesteps) 
        loss = F.mse_loss(model_output.float(), target.float())
        
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        progress_bar.set_postfix({"Loss": f"{sum(losses[-5:])/5:.4f}"})
    return losses

def run_generative_trial(base_environment: dict, static_batches: list, exp_config: dict):
    """
    The new harness, adhering to our doctrine. Runs a single, self-contained trial.
    """
    print(f"\n--- Running Trial: {exp_config['lora_config']['lora_rules'][0]['name']} ---")
    
    # --- 1. Per-Trial Setup ---
    device = base_environment['device']
    evaluator = base_environment['evaluator']
    
    # Create a fresh copy of the base model. Model Agnosticism in action.
    trial_model = copy.deepcopy(base_environment['base_model']).to(device)

    # Freeze base model weights (a common PEFT strategy)
    for param in trial_model.parameters():
        param.requires_grad = False

    # Create the specific, trainable LoRA network for this trial.
    loader = LoRAConfigLoader(config_dict=exp_config['lora_config'])
    resolved_config = loader.get_resolved_config(trial_model)
    network = FlexibleLoRANetwork(trial_model, resolved_config).to(device)
    
    optimizer = torch.optim.AdamW(network.prepare_optimizer_params(), lr=exp_config['optimizer_config']['lr'])

    # --- 2. Run Experiment ---
    print("--- Evaluating untrained network (initial state)... ---")
    initial_metrics = evaluator.evaluate(network, exp_config['evaluation_config'])

    # Training Loop
    all_losses = []
    for epoch in range(exp_config['training_config']['num_epochs']):
        print(f"--- Epoch {epoch + 1}/{exp_config['training_config']['num_epochs']} ---")
        epoch_losses = train_epoch(
            network, static_batches, base_environment['scheduler'], optimizer, exp_config['training_config'], device
        )
        all_losses.extend(epoch_losses)

    print("--- Evaluating trained network (final state)... ---")
    final_metrics = evaluator.evaluate(network, exp_config['evaluation_config'])

    # --- 3. Return Structured Results ---
    learning_delta = final_metrics['primary_score'] - initial_metrics['primary_score']
    
    return {
        "results": {
            "initial_metrics": initial_metrics,
            "final_metrics": final_metrics,
            "learning_delta": learning_delta,
            "loss_history": all_losses,
        }
    }

def setup_generative_environment(manifest: dict) -> dict:
    """
    Handles the expensive, one-time setup based on the experiment manifest.
    This is where model-specific logic is isolated.
    """
    print("--- Setting up Generative Environment (ONCE)... ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Principle 3: Model Agnosticism via a Factory ---
    # The harness doesn't know about TesterUViT. This setup function does.
    model_name = manifest['environment_setup']['base_model_name']
    environment = {}
    if model_name == 'TesterUViT':
        base_model = TesterUViT()
        # Install the TESTER shims into the environment
        environment['diffusion_fn'] = testeruvit_diffusion_fn
        environment['decoder_fn'] = testeruvit_decoder_fn
        # The VAE is the model itself for decoding (identity op)
        environment['vae'] = base_model 
    elif model_name == 'StableDiffusionXL_UNet':
        # ... load the real UNet ...
        # ... load the real VAE ...
        # Install the REAL processing functions
        environment['diffusion_fn'] = sdxl_diffusion_fn
        environment['decoder_fn'] = sdxl_decoder_fn
        # environment['vae'] = the_real_vae
        raise NotImplementedError("SDXL UNet loading is a placeholder.")

    # --- Principle 4: Unified Data Pipeline ---
    pipeline_name = manifest['data_setup']['pipeline']
    if pipeline_name == 'sham_image_dataset':
        # The data pipeline is now a swappable component.
        # The environment object only needs the scheduler and device here.
        temp_env = {'scheduler': MinimalDDPMScheduler(device=device), 'device': device}
        static_batches = materialize_sham_dataset(manifest['data_setup']['config'], temp_env)
    else:
        raise ValueError(f"Unknown data pipeline in manifest: {pipeline_name}")

    # Unstub the eval_dataset creation by correctly deconstructing the static batches.
    # The orchestrator expects a flat list of individual items for its workload.
    all_model_inputs_for_eval = []
    all_ground_truth_for_eval = []

    for batch in static_batches:
        all_ground_truth_for_eval.append(batch['clean_images'])
        num_items_in_batch = batch['clean_images'].shape[0]
        for i in range(num_items_in_batch):
            # Create a workload dict for each individual sample in the batch.
            # The orchestrator will re-batch these later.
            item_dict = {
                'initial_latents': batch['initial_latents'][i:i+1],
                'timesteps': batch['timesteps'][i:i+1],
                'conditioning': batch['conditioning'],
            }
            all_model_inputs_for_eval.append(item_dict)

    eval_dataset = {
    'model_inputs': all_model_inputs_for_eval, # NOW A POPULATED LIST :)
    'ground_truth_outputs': all_ground_truth_for_eval
    }

    # Create the final environment dictionary
    environment.update({
        'device': device,
        'base_model': base_model,
        'scheduler': MinimalDDPMScheduler(device=device),
        # This is placeholder data for the evaluator, it needs to be made generic
        'eval_dataset': eval_dataset
    })
    environment['evaluator'] = GenerativeEvaluator(environment['eval_dataset'], environment)
    
    print("--- Generative Environment Setup Complete. ---")
    return environment, static_batches

#
#   SECTION! THREE! IT'S THE SEARCH ABSTRACTA!
#


class RandomSearchController:
    """
    Performs a simple random search by generating LoRA configurations and
    invoking a self-contained "harness function" to run each trial.
    """
    def __init__(self,
                 search_space_path: str,
                 harness_fn,
                 base_environment: dict,
                 static_batches: list,
                 base_exp_config: dict):
        """
        Initializes the controller.

        Args:
            search_space_path: Path to YAML file defining the LoRA search space.
            harness_fn: The function that runs a full, self-contained experiment trial.
                        Expected signature: harness_fn(base_env, static_batches, exp_config)
            base_environment: The heavy, reusable environment objects.
            static_batches: The pre-materialized training data.
            base_exp_config: The non-LoRA part of the experiment config.
        """
        with open(search_space_path, 'r') as f:
            self.search_space = yaml.safe_load(f)

        self.harness_fn = harness_fn
        self.base_environment = base_environment
        self.static_batches = static_batches
        self.base_exp_config = base_exp_config
        self.results = []
        print("--- Random Search Controller Initialized (Doctrine-Compliant) ---")

    def _sample_config(self) -> dict:
        """Generates a random LoRA configuration from the search space."""
        config_dict = {'lora_rules': []}
        num_rules = random.randint(self.search_space['num_rules']['min'], self.search_space['num_rules']['max'])
        for i in range(num_rules):
            target_type = random.choice(list(self.search_space['targets'].keys()))
            target_params = self.search_space['targets'][target_type]
            rule = {
                'name': f"{target_type}_rule_{i}_{random.randint(1000, 9999)}",
                'rank': random.choice(self.search_space['rank']),
                'alpha': random.choice(self.search_space['alpha']),
                'target_name_contains': target_params['target_name_contains']
            }
            config_dict['lora_rules'].append(rule)
        return config_dict

    def run_search(self, num_trials: int):
        """Executes the main random search loop."""
        for i in range(num_trials):
            print(f"\n{'='*25} Search Trial {i+1}/{num_trials} {'='*25}")
            
            # 1. Controller samples a new LoRA config
            lora_config = self._sample_config()
            
            # 2. Update the master experiment manifest for this specific trial
            trial_config = copy.deepcopy(self.base_exp_config)
            trial_config['lora_config'] = lora_config

            # 3. Invoke the harness to run the entire trial. This is the core doctrine.
            run_output = self.harness_fn(
                base_environment=self.base_environment,
                static_batches=self.static_batches,
                exp_config=trial_config
            )

            # 4. Log the results
            result_entry = {
                'trial_num': i+1,
                'config_name': lora_config['lora_rules'][0]['name'],
                **run_output['results']
            }
            self.results.append(result_entry)
            print(f"--- Trial Complete. Delta: {result_entry['learning_delta']:.4f}, Final Score: {result_entry['final_metrics']['primary_score']:.4f} ---")

    def show_best_results(self, top_n=3):
        """Sorts and prints the best results from the search."""
        if not self.results:
            print("No results to show.")
            return

        # Lower learning_delta is better
        sorted_results = sorted(self.results, key=lambda x: x['learning_delta'])
        
        print("\n" + "="*60)
        print("--- TOP SEARCH RESULTS (by Learning Delta) ---")
        for i, result in enumerate(sorted_results[:top_n]):
            print(f"  RANK {i+1}:")
            print(f"    - Config Name: {result['config_name']}")
            print(f"    - Learning Delta: {result['learning_delta']:.4f}")
            print(f"    - Final Score: {result['final_metrics']['primary_score']:.4f}")
            print(f"    - Final LPIPS: {result['final_metrics']['lpips']:.4f}")
            print(f"    - Final FID: {result['final_metrics']['fid']:.4f}")
        print("="*60)

def create_search_space_yaml(path="search_space.yaml"):
    """Helper function to create an example search space config file."""
    search_space = {
        'num_rules': {'min': 1, 'max': 1}, # Keep it simple for now
        'rank': [4, 8, 16, 32],
        'alpha': [1.0, 2.0, 4.0],
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
    return path

# ==============================================================================
# SECTION 4: MAIN EXECUTION BLOCK
# ==============================================================================


if __name__ == "__main__":
    # --- STEP 1: Load the master configuration ---
    with open('experiment_manifest.yaml', 'r') as f:
        manifest = yaml.safe_load(f)

    # --- STEP 2: One-time, expensive setup ---
    environment, static_batches = setup_generative_environment(manifest)
    print(f"\nReusable environment ready. Contains: {list(environment.keys())}")
    
    # --- STEP 3: Initialize the Search Controller ---
    search_space_file = create_search_space_yaml()
    controller = RandomSearchController(
        search_space_path=search_space_file,
        harness_fn=run_generative_trial, # Pass the harness function directly
        base_environment=environment,
        static_batches=static_batches,
        base_exp_config=manifest['trial_config']
    )

    # --- STEP 4: Execute the Search Loop ---
    num_search_trials = manifest['search_config']['num_trials']
    controller.run_search(num_trials=num_search_trials)
    
    # --- STEP 5: Show summary of results ---
    # Doctrine Compliance: The number of results to show is also from the manifest.
    top_n = manifest['search_config'].get('top_n_results_to_show', 3)
    controller.show_best_results(top_n=top_n)