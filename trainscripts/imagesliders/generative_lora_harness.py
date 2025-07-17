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
from .batch_data_pipeline import materialize_sdxl_latent_dataset
from .batch_model_util import ThroughputBatchFinder
from .batch_model_util import run_evaluation_flow, testeruvit_decoder_fn, testeruvit_diffusion_fn # Assuming this exists for eval
from .batch_model_util import diffusion_fn as sdxl_diffusion_fn, vae_decoder_fn as sdxl_decoder_fn
#from .batch_model_util import _estimate_training_throughput

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance

# You'll also need to import the real models from diffusers
from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers.optimization import get_scheduler
from .batch_slider_algo import GradientNoiseEstimator

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

import time
import numpy as np
import copy


def _estimate_training_throughput(environment: dict, sample_batch: dict, search_space: dict) -> float:
    """
    A utility to time a training run by transparently wrapping the train_epoch function.
    This ensures the timing is based on the exact same code path as real training.
    """
    from .flexible_lora_system import LoRAConfigLoader, FlexibleLoRANetwork # Keep local to avoid circular import at top level
    
    print("    - Building representative network for timing estimate...")
    
    # --- 1. Create a "median" LoRA config (remains the same) ---
    median_rank = sorted(search_space['rank'])[len(search_space['rank']) // 2]
    median_alpha = sorted(search_space['alpha'])[len(search_space['alpha']) // 2]
    representative_target = search_space['targets']['all_attention']['target_name_contains']
    representative_lora_config = {
        'lora_rules': [{'name': "timing_estimator_rule", 'rank': median_rank, 'alpha': median_alpha, 'target_name_contains': representative_target}]
    }
    print(f"    - Using representative config: rank={median_rank}, target='{representative_target}'")
    weight_dtype = environment["weight_dtype"]
    # --- 2. Build the network and optimizer (remains the same) ---
    device = environment['device']
    trial_model = (environment['base_model']).to(
    device=device,
    dtype=weight_dtype  # <--- THIS LINE IS MISSING IN THE TIMING FUNCTION
    )
    for param in trial_model.parameters():
        param.requires_grad = False
    loader = LoRAConfigLoader(config_dict=representative_lora_config)
    resolved_config = loader.get_resolved_config(trial_model)
    network = FlexibleLoRANetwork(trial_model, resolved_config).to(
    device=device,
    dtype=weight_dtype  # <--- THIS LINE IS MISSING IN THE TIMING FUNCTION
    )
    optimizer = torch.optim.AdamW(network.prepare_optimizer_params(), lr=0.001)

    # --- 3. THE REFACTOR: Prepare inputs for and call train_epoch ---
    
    # a. Create a dummy dataloader with a single batch for one step of training
    dummy_dataloader = [sample_batch] 
    
    # b. Create a minimal LR scheduler
    lr_scheduler = get_scheduler(name="constant", optimizer=optimizer, num_training_steps=2)
    
    # c. Create a minimal training config for the epoch function
    timing_run_config = {'gradient_accumulation_steps': 1}

    # d. Warm-up run (one epoch with one step)
    train_epoch(
        unet=trial_model, network=network, dataloader=dummy_dataloader,
        scheduler=environment['scheduler'], optimizer=optimizer, lr_scheduler=lr_scheduler,
        device=device, gns_estimator=None, training_config=timing_run_config
    )
    
    # e. Timed run
    torch.cuda.synchronize(device)
    start_time = time.time()
    
    train_epoch(
        unet=trial_model, network=network, dataloader=dummy_dataloader,
        scheduler=environment['scheduler'], optimizer=optimizer, lr_scheduler=lr_scheduler,
        device=device, gns_estimator=None, training_config=timing_run_config
    )

    torch.cuda.synchronize(device)
    end_time = time.time()
    
    # Cleanup to free VRAM
    del trial_model, network, optimizer, loader, lr_scheduler
    
    # The duration is now for a full epoch of one step, which is equivalent to one step.
    return end_time - start_time

def train_step(batch: dict, environment: dict):
    """
    Performs a single, self-contained training step using pre-calculated data.
    This function is a pure executor of the batch data.
    """
    def diagprint():
        print("\n--- Shape and Dtype Census Before UNet Call ---")
        print(f"  - latents_cfg:                {latents_cfg.shape}, {latents_cfg.dtype}")
        print(f"  - unet_input_timesteps_cfg:   {unet_input_timesteps_cfg.shape}, {unet_input_timesteps_cfg.dtype}")
        print(f"  - text_embeddings_cfg (main): {text_embeddings_cfg.shape}, {text_embeddings_cfg.dtype}")
        print(f"  - added_cond_kwargs['text_embeds'] (pooled): {added_cond_kwargs['text_embeds'].shape}, {added_cond_kwargs['text_embeds'].dtype}")
        print(f"  - added_cond_kwargs['time_ids']:             {added_cond_kwargs['time_ids'].shape}, {added_cond_kwargs['time_ids'].dtype}")
        print(f"  - unet:                        {unet.dtype}")
        print("---------------------------------------------------\n")
    # --- 1. Unpack Environment ---
    device = environment["device"]
    unet = environment["unet"]
    network = environment["network"]
    noise_scheduler = environment["noise_scheduler"].to(device)

    
    # --- 2. Unpack the Pre-Calculated Batch Data from the Pipeline ---
    latents = batch['latents'].to(device)
    weight_dtype = latents.dtype
    noise = batch['noise'].to(device)
    text_embeddings_cfg = batch['cfg_text_embeddings'].to(device)
    pooled_embeds_cfg = batch['cfg_pooled_embeds'].to(device)
    add_time_ids_cfg = batch['add_time_ids'].to(device)
    scales = batch['scales'].to(device)
    guidance_scale_tensor = batch['guidance_scale'].to(device)

    # These are now read directly from the batch, not calculated live.
    noise_level_timesteps = batch['noise_level_timesteps'].to(device)
    unet_input_timesteps = batch['unet_input_timesteps'].to(device)
    
    # --- 3. Denoising and Batch Preparation (Now Simplified) ---
    noisy_latents = noise_scheduler.add_noise(latents, noise, noise_level_timesteps).to(device=device, dtype=weight_dtype)
    latents_cfg = torch.cat([noisy_latents, noisy_latents], dim=0)
    unet_input_timesteps_cfg = torch.cat([unet_input_timesteps, unet_input_timesteps], dim=0)
    
    # --- 4. Forward Pass ---
    # The LoRA network controller sets the scales for the injected layers.
    network.set_lora_scales(torch.cat([scales, scales], dim=0))
    added_cond_kwargs = {"text_embeds": pooled_embeds_cfg, "time_ids": add_time_ids_cfg}
    
    #diagprint()
    # The forward pass is called on the base UNet, which has been modified in-place.
    predicted_noise = unet(
        latents_cfg.to(dtype=weight_dtype),
        unet_input_timesteps_cfg,
        encoder_hidden_states=text_embeddings_cfg.to(dtype=weight_dtype),
        added_cond_kwargs=added_cond_kwargs
    ).sample


    # --- 5. Loss Calculation (The Final Piece) ---
    # this is the target we would use if we were jointly training the cond and uncond!
    # (hint: this is possible as an aux loss hehe)
    # instead we are training the cfg 2.0 augmented target because we are cool and epic.
    target_noise = torch.cat([noise, noise], dim=0)
    
    # We can use the guidance scale during training for a CFG-aware loss.
    # Reshape guidance_scale to [B, 1, 1, 1] for broadcasting.
    guidance_scale_b = guidance_scale_tensor.view(-1, 1, 1, 1)
    #print(f"guidscale{guidance_scale_tensor}")
    #guidance_scale_b = guidance_scale_tensor.chunk(2)

    # Perform the CFG-aware guidance on the predicted noise
    uncond_pred, text_pred = predicted_noise.chunk(2)
    #print(uncond_pred.shape), print(text_pred.shape), print(guidance_scale_b.shape)
    guided_pred = uncond_pred + guidance_scale_b * (text_pred - uncond_pred)

    # Calculate the MSE loss against the original noise.
    # Use .float() for stability.
    loss = F.mse_loss(guided_pred.float(), noise.float(), reduction="mean")
    
    return loss


def train_epoch(
    unet, network, dataloader, scheduler, optimizer, lr_scheduler, device, gns_estimator, training_config
):
    """
    A stateless function to perform one epoch of training.
    """
    network.train()
    losses = []
    
    accum_steps = training_config.get('gradient_accumulation_steps', 1)
    if accum_steps <= 0:
        accum_steps = 1
        
    # We now loop based on optimizer steps, not dataloader batches
    num_optimizer_steps = len(dataloader) // accum_steps
    data_iterator = iter(dataloader)
    
    # This step counter is local to the epoch, tracking optimizer updates
    optimizer_step_count = 0
    progress_bar = tqdm(range(num_optimizer_steps), desc="Optimizer Steps", leave=False)

    environment_for_step = {"unet": unet, "network": network, "noise_scheduler": scheduler.to(device), "device": device}
    network=network.to(device)
    unet=unet.to(device)
    
    weight_dtype = unet.dtype
    with network:
        for step in progress_bar:
            # --- Gradient Accumulation Loop ---
            is_profiling = False
            if gns_estimator is not None:
                # The global_step for profiling frequency can be the optimizer_step_count
                gns_estimator.pre_accumulate_step(optimizer_step_count)
                is_profiling = gns_estimator.is_profiling

            total_loss_in_step = 0.0

            if is_profiling:
                # --- Profiling Path: accumulate gradients manually in GNS buffer ---
                for _ in range(accum_steps):
                    batch = next(data_iterator)
                    loss = train_step(batch, environment_for_step).to(weight_dtype)
                    loss.backward() # Immediate backprop for the micro-gradient
                    gns_estimator.post_micro_backward_step() # GNS captures the raw grad
                    total_loss_in_step += loss.item()
            else:
                # --- Standard Path: let PyTorch handle gradient summation ---
                optimizer.zero_grad()
                for _ in range(accum_steps):
                    batch = next(data_iterator)
                    loss = train_step(batch, environment_for_step).to(weight_dtype)
                    # We scale the loss before backprop when accumulating
                    loss = loss / accum_steps
                    loss.backward()
                    total_loss_in_step += loss.item() * accum_steps # Un-scale for logging

            del batch
            
            # --- After Accumulation, Before Optimizer Step ---
            if is_profiling:
                # Finalize profiling: calculates stats and loads summed grad into model
                gns_estimator.post_accumulate_step(accum_steps)

                new_steps = gradient_noise_estimator.propose_new_accumulation_steps(
                        current_steps=gradient_accumulation_steps,
                        min_steps=2,
                        max_steps=int(config.train.iterations/4)
                    )
                accum_steps = new_steps
        # Optional: Gradient Clipping would go here, applied before optimizer.step()
            # torch.nn.utils.clip_grad_norm_(network.parameters(), max_grad_norm)
            
            optimizer.step()
            lr_scheduler.step()
            
            avg_loss = total_loss_in_step / accum_steps
            losses.append(avg_loss)
            progress_bar.set_postfix({"Loss": f"{avg_loss:.4f}", "LR": f"{lr_scheduler.get_last_lr()[0]:.2e}"})
            if gns_estimator and gns_estimator.ema_zoomy_b_crit is not None:
                progress_bar.set_postfix_str(f"Loss: {avg_loss:.4f}, LR: {lr_scheduler.get_last_lr()[0]:.2e}, B_crit: {gns_estimator.ema_zoomy_b_crit:.2f}")

            optimizer_step_count += 1
        
    return losses

def run_generative_trial(base_environment: dict, static_batches: list, exp_config: dict, batch_size: int ):
    """
    The new harness, adhering to our doctrine. Runs a single, self-contained trial.
    """
    print(f"\n--- Running Trial: {exp_config['lora_config']['lora_rules'][0]['name']} ---")
    
    # --- 1. Per-Trial Setup ---
    device = base_environment['device']
    weight_dtype = base_environment['weight_dtype'] 
    evaluator = base_environment['evaluator'] 
    # Create a fresh copy of the base model. Model Agnosticism in action.
    trial_model = (base_environment['base_model']).to(device=device, dtype=weight_dtype)

    # Freeze base model weights (a common PEFT strategy)
    for param in trial_model.parameters():
        param.requires_grad = False

    # Create the specific, trainable LoRA network for this trial.
    loader = LoRAConfigLoader(config_dict=exp_config['lora_config'])
    resolved_config = loader.get_resolved_config(trial_model)
    network = FlexibleLoRANetwork(trial_model, resolved_config).to(
    device=device,
    dtype=weight_dtype)
    optimizer = torch.optim.AdamW(network.prepare_optimizer_params(), lr=exp_config['optimizer_config']['lr'])

    # b1. Initialize LR Scheduler for this trial
    training_config = exp_config['training_config']
    num_epochs = training_config['num_epochs']
    accum_steps = training_config.get('gradient_accumulation_steps', 1)
    num_optimizer_steps = (len(static_batches) // accum_steps) * num_epochs

    lr_scheduler = get_scheduler(
        name="constant", # Or read from config
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_optimizer_steps
    )

    # b2. Initialize GradientNoiseEstimator for this trial, if enabled
    gns_estimator = None
    if training_config.get('estimate_gradient_noise_scale'):
        with torch.no_grad:
            gns_estimator = GradientNoiseEstimator(
                network,
                batch_size,
                training_config.get('gns_profile_freq', 10),
                training_config.get('gns_ema_fast', 0.1),
                training_config.get('gns_ema_slow', 0.01)
            )

    sampling_config_for_eval = exp_config['evaluation_config']['sampling_config']

    # --- 2. Run Experiment ---
    print("--- Evaluating untrained network (initial state)... ---")
    trial_model = trial_model.to(device)
    initial_metrics = evaluator.evaluate(network, sampling_config_for_eval)

    # Training Loop
    all_losses = []
    for epoch in range(exp_config['training_config']['num_epochs']):
        print(f"--- Epoch {epoch + 1}/{exp_config['training_config']['num_epochs']} ---")
        epoch_losses = train_epoch(
            unet=trial_model,
            network=network,
            dataloader=static_batches,
            scheduler=base_environment['scheduler'],
            optimizer=optimizer,
            lr_scheduler=lr_scheduler, # Pass it down
            device=device,
            gns_estimator=gns_estimator, # Pass it down
            training_config=training_config
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
    tdcpu = torch.device("cpu")

    # --- Principle 3: Model Agnosticism via a Factory ---
    # The harness doesn't know about TesterUViT. This setup function does.
    model_name = manifest['environment_setup']['base_model_name']
    environment = {}
    environment["device"]=device
    environment["weight_dtype"] = torch.bfloat16
    if model_name == 'TesterUViT':
        base_model = TesterUViT()
        # Install the TESTER shims into the environment
        environment['diffusion_fn'] = testeruvit_diffusion_fn
        environment['decoder_fn'] = testeruvit_decoder_fn
        # The VAE is the model itself for decoding (identity op)
        environment['vae'] = base_model 
    elif model_name == 'StableDiffusionXL_UNet':
        print("--- Loading REAL Stable Diffusion XL models... ---")
        # Load the pre-trained models from disk (paths should be in the manifest)
        unet_path = manifest['environment_setup']['model_path']
        #vae_path = manifest['environment_setup']['vae_path']
        #load from singlefile
        model_config = manifest['environment_setup']
        from .batch_model_util import load_models_path as load_xl_from_single_file
        vae, base_model, tokenizers, text_encoders = load_xl_from_single_file(unet_path, weight_dtype=torch.bfloat16)
        
        print(f"pushing models to device:{torch.device('cpu')}")
        base_model= base_model.requires_grad_(False).eval().to(torch.device("cpu"))
        vae = vae.requires_grad_(False).eval().to(torch.device("cpu"))
        for text_encoder in text_encoders:
            text_encoder = text_encoder.requires_grad_(False).eval().to(torch.device("cpu"))

        environment['device'] = device
        environment['base_model'] = base_model
        # Install the REAL processing functions for the orchestrator
        environment['diffusion_fn'] = sdxl_diffusion_fn
        environment['decoder_fn'] = sdxl_decoder_fn
        environment['vae'] = vae
        # The environment also needs the real text encoders for the data pipeline
        # ... (logic to load CLIP models would go here) ...
        environment['text_encoders'] = text_encoders
        environment['tokenizers'] = tokenizers

    # Create the scheduler early, so it can be passed to all components.
    environment['scheduler'] = MinimalDDPMScheduler(device=device)

    # --- Principle 4: Unified Data Pipeline ---
    pipeline_name = manifest['data_setup']['pipeline']
    if pipeline_name == 'sham_image_dataset':
        # The data pipeline is now a swappable component.
        # The environment object only needs the scheduler and device here.
        temp_env = {'scheduler': MinimalDDPMScheduler(device=device), 'device': device}
        static_batches = materialize_sham_dataset(manifest['data_setup']['config'], temp_env)
    elif pipeline_name == 'sdxl_latent_dataset':
        # --- THIS IS THE UNSTUBBED LOGIC ---
        # Call our new, powerful data materializer
        # It needs the full environment to access the VAE and text encoders.
        static_batches = materialize_sdxl_latent_dataset(
            manifest['data_setup']['config'],
            environment # Pass the partially built environment
        )
    else:
        raise ValueError(f"Unknown data pipeline in manifest: {pipeline_name}")

    # Unstub the eval_dataset creation by correctly deconstructing the static batches.
    # The orchestrator expects a flat list of individual items for its workload.
    eval_dataset = {'model_inputs': [], 'ground_truth_outputs': []}
    key_for_primary_data = 'latents' if pipeline_name == 'sdxl_latent_dataset' else 'clean_images'

    for batch in static_batches:
        primary_data_tensor = batch[key_for_primary_data]
        num_items_in_batch = primary_data_tensor.shape[0]
        
        eval_dataset['ground_truth_outputs'].append(primary_data_tensor)
        for i in range(num_items_in_batch):
            # 3. Use the polymorphic key again to get the per-item data.
            #    This creates the workload for the orchestrator.
            item_dict = {
                'initial_latents': batch['latents'][i:i+1],
                'unet_input_timesteps': batch['unet_input_timesteps'][i:i+1],
                'guidance_scale': batch['guidance_scale'][i:i+1],
                # The CFG tensors are already stacked [uncond, cond]. We slice the i-th pair.
                'cfg_text_embeddings': batch['cfg_text_embeddings'][i*2:(i+1)*2],
                'cfg_pooled_embeds': batch['cfg_pooled_embeds'][i*2:(i+1)*2],
                'add_time_ids': batch['add_time_ids'][i*2:(i+1)*2],
            }
            eval_dataset['model_inputs'].append(item_dict)

    print("\n--- Applying Evaluation Safeguards from Manifest ---")
    eval_config = manifest['trial_config']['evaluation_config']
    current_eval_samples = len(eval_dataset['model_inputs'])
    max_allowed_samples = current_eval_samples
    # --- Safeguard 1: Proportion of Training Data ---
    if 'proportion_of_train' in eval_config:
        prop = eval_config['proportion_of_train']
        num_train_samples = manifest['data_setup']['config']['iterations'] * manifest['data_setup']['config']['batch_size']
        max_allowed_by_prop = int(num_train_samples * prop)
        print(f"  [Proportion Guard] Training samples: {num_train_samples}. Eval proportion: {prop} -> max {max_allowed_by_prop} samples.")
        max_allowed_samples = min(max_allowed_samples, max_allowed_by_prop)

    # --- Safeguard 2: Proportion of Training Time ---
    if 'max_eval_time_proportion' in eval_config:
        print("  [Time Guard] Estimating training and evaluation throughput...")
        with open(manifest['search_config']['search_space_file'], 'r') as f:
            search_space_dict = yaml.safe_load(f)
        # a) Estimate total training time
        time_per_train_batch = _estimate_training_throughput(environment, static_batches[0], search_space_dict)
        total_train_steps = manifest['data_setup']['config']['iterations']
        projected_train_time = time_per_train_batch * total_train_steps
        print(f"    - Estimated time per training step: {time_per_train_batch:.3f}s")
        print(f"    - Projected total training time: {projected_train_time / 60:.2f} minutes")

        # b) Estimate time per evaluation sample
        # We need a dummy model to run the batch finder on the diffusion step
        dummy_model = (environment['base_model']).to(device)
        finder_env = environment.copy()
        finder_env['inference_network'] = dummy_model
        finder_env['sampling_config'] = eval_config['sampling_config']
        
        throughput_finder = ThroughputBatchFinder(dummy_model, eval_dataset['model_inputs'][0], environment['device'], environment['diffusion_fn'], finder_env)
        items_per_sec = throughput_finder.find()
        time_per_eval_sample = 1.0 / (items_per_sec + 1e-6)
        print(f"    - Estimated time per evaluation sample (diffusion): {time_per_eval_sample:.3f}s")

        # c) Calculate the new limit
        eval_time_budget = projected_train_time * eval_config['max_eval_time_proportion']
        max_allowed_by_time = int(eval_time_budget / time_per_eval_sample)
        print(f"    - Evaluation time budget: {eval_time_budget:.1f}s -> max {max_allowed_by_time} samples.")
        max_allowed_samples = min(max_allowed_samples, max_allowed_by_time)
        del dummy_model, finder_env # Cleanup
        
    # --- Enforce the final decision ---
    if max_allowed_samples < current_eval_samples:
        print(f"--- Safeguards triggered. Truncating evaluation dataset from {current_eval_samples} to {max_allowed_samples} samples. ---")
        eval_dataset['model_inputs'] = eval_dataset['model_inputs'][:max_allowed_samples]
        
        # We need to truncate the ground truth tensor to match.
        # This is tricky because it's a single tensor. We find how many batches the new sample count corresponds to.
        num_items_per_batch = len(static_batches[0]['latents'])
        num_batches_to_keep = (max_allowed_samples + num_items_per_batch - 1) // num_items_per_batch
        
        truncated_gt_list = []
        for i in range(num_batches_to_keep):
            key = 'latents' if manifest['data_setup']['pipeline'] == 'sdxl_latent_dataset' else 'clean_images'
            truncated_gt_list.append(static_batches[i][key])

        eval_dataset['ground_truth_outputs'] = torch.cat(truncated_gt_list)[:max_allowed_samples]
        
        # Write the decision back into the config for logging and transparency
        manifest['trial_config']['evaluation_config']['effective_num_samples'] = max_allowed_samples
    else:
        print("--- No safeguards triggered. Using full evaluation dataset. ---")
        manifest['trial_config']['evaluation_config']['effective_num_samples'] = current_eval_samples

    # Create the final environment dictionary
    environment.update({
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
                 base_exp_config: dict,
                 batch_size: int):
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
        self.batch_size = batch_size
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
                exp_config=trial_config,
                batch_size=self.batch_size
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
    with open('experiment_manifest_xl.yaml', 'r') as f:
        manifest = yaml.safe_load(f)

    #search space is now used to do model autocalibration
    search_space_file = create_search_space_yaml()
    batch_size_from_config = manifest['data_setup']['config']['batch_size']
    manifest['search_config']['search_space_file'] = search_space_file

    # --- STEP 2: One-time, expensive setup ---
    environment, static_batches = setup_generative_environment(manifest)
    print(f"\nReusable environment ready. Contains: {list(environment.keys())}")
    
    # --- STEP 3: Initialize the Search Controller ---


    controller = RandomSearchController(
        search_space_path=search_space_file,
        harness_fn=run_generative_trial, # Pass the harness function directly
        base_environment=environment,
        static_batches=static_batches,
        base_exp_config=manifest['trial_config'],
        batch_size=batch_size_from_config
    )

    # --- STEP 4: Execute the Search Loop ---
    num_search_trials = manifest['search_config']['num_trials']
    controller.run_search(num_trials=num_search_trials)
    
    # --- STEP 5: Show summary of results ---
    # Doctrine Compliance: The number of results to show is also from the manifest.
    top_n = manifest['search_config'].get('top_n_results_to_show', 3)
    controller.show_best_results(top_n=top_n)