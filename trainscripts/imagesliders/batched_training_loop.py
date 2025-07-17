#python -m trainscripts.imagesliders.batched_training_loop -c F:/dox/ai/gemmy/sliders/trainscripts/imagesliders/data/batch_config.yaml
import torch
import os
import gc
from tqdm import tqdm
import datetime
import sys
import argparse
import yaml
import json
from diffusers.optimization import get_scheduler
import torch.optim as optim
import hashlib
from PIL import Image
from pathlib import Path
import time
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Union, Literal, List, Dict, Any
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from .batch_slider_algo import calculate_paired_loss, GradientNoiseEstimator
from .batch_config_util import (
    setup_logging,
    AttrDict,
    load_config_from_yaml,
    config_io,
    envsetup as config_envsetup, # Renamed to avoid conflict
)
from . import batch_lora as lora
from . import batch_train_util
from . import batch_model_util
from .data_schedule import TrainingSchedule, prepare_cached_batches

from . import batch_data_pipeline

UNET_ATTENTION_TIME_EMBED_DIM = 256  # XL
TEXT_ENCODER_2_PROJECTION_DIM = 1280
UNET_PROJECTION_CLASS_EMBEDDING_INPUT_DIM = 2816

SDXL_TEXT_ENCODER_TYPE = Union[CLIPTextModel, CLIPTextModelWithProjection]
DIFFUSERS_CACHE_DIR = None # if you want to change the cache dir, change this

# --- BEGIN TORCH.COMPILE MONKEY-PATCH ---
# This fixes a race condition in torch.compile's caching on Windows.
# The default `Path.rename` fails if the destination file exists, which can
# happen in multi-threaded/multi-process scenarios. `os.replace` is the
# correct atomic "overwrite" operation.

import threading
import torch._inductor.codecache as codecache
# Copy the original function's signature and body
def fixed_write_atomic(
    path_: str,
    content: Union[str, bytes],
    make_dirs: bool = False,
    encode_utf_8: bool = False,
) -> None:
    path = Path(path_)
    if make_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use a unique temporary file to avoid conflicts
    # NOTE: The original implementation's temp file naming is also a source
    # of race conditions. A more robust temp file could be used, but
    # fixing the rename is the most critical part.
    tmp_path = path.parent / f".{os.getpid()}.{threading.get_ident()}.tmp"
    
    write_mode = "w" if isinstance(content, str) else "wb"
    with tmp_path.open(write_mode, encoding="utf-8" if encode_utf_8 else None) as f:
        f.write(content)
        
    # THE FIX: Use os.replace for an atomic overwrite on all platforms
    os.replace(tmp_path, path)

# Overwrite the function in the loaded module
codecache.write_atomic = fixed_write_atomic
print("--- Applied monkey-patch to torch._inductor.codecache.write_atomic for Windows compatibility ---")
# --- END MONKEY-PATCH ---

def train_step(batch: Dict[str, Any], **environment: Dict[str, Any]):
    """
    Performs a single, self-contained training step.
    Handles device/dtype placement, batch preparation, forward pass, and loss calculation.
    """
    # --- 1. Unpack Environment and Config ---
    device = environment["device"]
    weight_dtype = environment["weight_dtype"]
    config = environment["config"]
    unet = environment["unet"]
    noise_scheduler = environment["noise_scheduler"].to(device)
    network = environment["network"]

    # The generator from the environment is now used for noise and timesteps
    generator = environment.get("generator")

    # --- 2. Rectify Batch: Move to device and set dtypes ---
    # This logic is now inside the train_step.
    latents = batch['latents'].to(device, dtype=weight_dtype)
    scales = batch['scales'].to(device, dtype=weight_dtype)
    noise = batch['noise'].to(device, dtype=torch.float32)
    add_time_ids = batch['add_time_ids'].to(device, dtype=torch.float32)
    timesteps_to = batch['timesteps_to'].to(device)

    text_embeddings_cfg = batch['cfg_text_embeddings'].to(device, dtype=weight_dtype)
    pooled_embeds_cfg = batch['cfg_pooled_embeds'].to(device, dtype=weight_dtype)
    add_time_ids_cfg = torch.cat([add_time_ids, add_time_ids], dim=0)
   
    # --- 3. Prepare for Denoising ---
    # The noise and timesteps are now taken from the batch, not generated live.
    
    # REFERENCE TIMESTEP CODE
    noise_scheduler.set_timesteps(config.train.max_denoising_steps, device=device)
    #generating inside of train step does some kind of graph break probably.
    #timesteps_to = torch.randint(
    #    1, 
    #    config.train.max_denoising_steps, 
    #    (latents.shape[0],),  # The shape MUST match the latents batch size
    #    device=device, 
    #    generator=generator
    #).long()

    # 1 ~ 49 からランダム timesteps_to
    #slice indexing only works for single-item training; we must use exact index for batched training
    #nl_timestep = noise_scheduler.timesteps[timesteps_to:timesteps_to+1]
    nl_timestep = noise_scheduler.timesteps[timesteps_to]
    noisy_latents = noise_scheduler.add_noise(latents, noise, nl_timestep)
    #unet needs millenial (/1000) timesteps, not sampling stepsize timesteps as we use in our code!
    #this unit conversion is absolutely mandatory!
    # STILL REFERENCE CODE:
    noise_scheduler.set_timesteps(1000)
    noise_scheduler = noise_scheduler.to(device)
    normalized_tsteps = torch.round(timesteps_to * 1000 / config.train.max_denoising_steps).long()
    unet_timesteps = noise_scheduler.timesteps[normalized_tsteps]
    unet_timesteps = unet_timesteps.to(device)
    # timesteps_to has shape [B]. Get scheduler timesteps and duplicate to [B*2]
    unet_timesteps_cfg = torch.cat([unet_timesteps, unet_timesteps], dim=0)
    # END REFERENCE TIMESTEP CODE

    print_that_shit = False
    if print_that_shit:
        print(f"timesteps_to{timesteps_to}")
        print(f"nnormalized_tsteps:{normalized_tsteps}")
        print(f"nl_timestep{nl_timestep}")
        print(f"unet_timesteps:{unet_timesteps}")


    # --- 4. Prepare CFG-Ready Batch (formerly prepare_cfg_batch) ---
    # We already prepared add_time_ids_cfg, now we do the rest.
    latents_cfg = torch.cat([noisy_latents, noisy_latents], dim=0)
    # add_time_ids has shape [B, ...]. Duplicate it to [B*2, ...]
    add_time_ids_cfg = torch.cat([add_time_ids, add_time_ids])
    # scales has shape [B]. Duplicate it to [B*2]
    scales_cfg = torch.cat([scales, scales])


    # --- 5. Forward Pass ---
    # Set LoRA scales for the full CFG batch
    batched_scales_cfg = torch.cat([scales, scales], dim=0)
    network.set_lora_scales(batched_scales_cfg)

    # The prediction function expects keyword arguments for the added conditions
    added_cond_kwargs = {
        "text_embeds": pooled_embeds_cfg,
        "time_ids": add_time_ids_cfg,
    }
    
    predicted_noise = unet(
        latents_cfg.to(weight_dtype),
        unet_timesteps_cfg,
        encoder_hidden_states=text_embeddings_cfg.to(weight_dtype),
        added_cond_kwargs=added_cond_kwargs
    ).sample

    # --- 6. Loss Calculation ---
    # The training objective is to predict the noise from the text-conditioned prompt
    _, predicted_noise_text = predicted_noise.chunk(2)

    # Loss is calculated in float32 for numerical stability
    loss = torch.nn.functional.mse_loss(predicted_noise_text.float(), noise)

    if batch["guidance_scale"] is not None:
        guidance_scale = batch["guidance_scale"]
        if guidance_scale < 1:
            print(f"your guidance scale sounds invalid: {guidance_scale}")
            batch["guidance_scale"] = None
        if guidance_scale == 1:
            #no CFG case
            pass
        else:
            predicted_noise_cfg_reduced = (_ + guidance_scale * (
            predicted_noise_text - _
            )).to(device)
    
            loss = torch.nn.functional.mse_loss(predicted_noise_cfg_reduced.float(), noise)

    return loss


def training_loop(environment: dict, static_batches: list,):
    """
    Main training loop that iterates over a static list of pre-generated batches.
    """
    print(f"Starting training loop.")
    global_step = 0 # Initialize global step here
    optimizer = environment["optimizer"]
    lr_scheduler = environment["lr_scheduler"]
    network = environment["network"]
    device = environment["device"]
    weight_dtype = environment["weight_dtype"]
    config = environment["config"]
    gradient_noise_estimator = environment.get("gradient_noise_estimator")
    gradient_accumulation_steps = config.train.get("gradient_accumulation_steps", 1)
    
    # Read the max_grad_norm from config, default to 0 (disabled)
    max_grad_norm = config.train.get("max_grad_norm", 0.0)
    if max_grad_norm > 0:
        print(f"--- Gradient clipping enabled with max_norm={max_grad_norm} ---")

    total_sample_steps = config.train.iterations # Let's redefine 'iterations' as optimizer steps

    progress_bar = tqdm(total=total_sample_steps, desc="sample steps")
    losses = []
    #for i in progress_bar:   
    #refactor to avoid endless training runs w/ dynamic batchsize
    while global_step < total_sample_steps:
        # Determine if this is a profiling step    
        if gradient_noise_estimator is not None:
            gradient_noise_estimator.pre_accumulate_step()
            is_profiling_step = gradient_noise_estimator is not None and gradient_noise_estimator.is_profiling
        else:
            is_profiling_step = False

        total_loss = 0.0
        #need to wrap the backprop w/ 'with network' or we fail to checkpoint scales metadata.
        with network:
            if is_profiling_step:
                # Iterate over micro-batches for gradient accumulation
                for _ in range(gradient_accumulation_steps):
                    micro_batch = static_batches[global_step % len(static_batches)]
                    #rectify function rolled into train step 
                    #micro_batch = rectify_batch_fn(micro_batch, device, weight_dtype)
                    loss = train_step(micro_batch, **environment).to(weight_dtype)
                    #immediate backprop to get microgradient
                    loss.backward()
                    #tell estimator to capture and accumulate microgradient data
                    gradient_noise_estimator.post_micro_backward_step()
                    total_loss += loss.item()
                    global_step += 1 # Increment global step after each training step
                    progress_bar.update(1)
            else:
                #NON-GRADIENTSTATS ACCUMULATION CONTROL FLOW
                optimizer.zero_grad()
                for _ in range(gradient_accumulation_steps):
                    micro_batch = static_batches[global_step % len(static_batches)]
                    #micro_batch = rectify_batch_fn(micro_batch, device, weight_dtype)
                    loss = train_step(micro_batch, **environment).to(weight_dtype)
                    # Scale loss for accumulation
                    #RuntimeError: Found dtype Float but expected BFloat16
                
                    # accumulate gradients
                    loss.backward()
                    total_loss += loss.item()
                    global_step += 1 # Increment global step after each training step
                    progress_bar.update(1)

            #AFTER ACCUMULATION SEQUENCE
            #scaling to get mean from of sum-reduced non-profiling loss
            if is_profiling_step:
                gradient_noise_estimator.post_accumulate_step(gradient_accumulation_steps)

                # --- NEW: Check for and apply adaptive changes ---
                new_steps = gradient_noise_estimator.propose_new_accumulation_steps(
                    current_steps=gradient_accumulation_steps,
                    min_steps=2,
                    max_steps=int(config.train.iterations/2)
                )
                gradient_accumulation_steps = new_steps # The change takes effect on the *next* iteration
                # --- END NEW ---
            #scale for logging
            avg_loss = total_loss/gradient_accumulation_steps
            losses.append(avg_loss)

            if max_grad_norm > 0:
                # clip_grad_norm_ returns the total norm before clipping, which is useful for logging.
                total_norm = torch.nn.utils.clip_grad_norm_(network.parameters(), max_grad_norm)
                # You could add total_norm to your progress bar if you want to monitor it
                # e.g., progress_bar.set_postfix({"Norm": f"{total_norm:.2f}", ...})

            # Optimizer step and scheduler step
            #both profiling step and non profiling step have correct .grad attribute here
            optimizer.step()
            lr_scheduler.step()

            progress_bar.set_postfix({"Loss;avg": f"{avg_loss:.3f};{sum(losses)/len(losses):.3f}", "LR": f"{lr_scheduler.get_last_lr()[0]:.2e}","Norm": f"{total_norm:.2f}",})
            # Print estimated gradient noise scale if enabled
            if gradient_noise_estimator is not None and gradient_noise_estimator.ema_zoomy_b_crit is not None:
                # Print on the same line as tqdm progress bar
                progress_bar.set_postfix_str({"Loss;avg": f"{avg_loss:.3f};{sum(losses)/len(losses):.3f}", "LR": f"{lr_scheduler.get_last_lr()[0]:.2e}", "B_crit": f"{gradient_noise_estimator.ema_zoomy_b_crit:.2f}","Norm": f"{total_norm:.2f}",})
    environment["losses"] = losses
    return environment


def graceful_shutdown(environment: dict):
    """
    Saves the trained network weights.
    """
    print("Training finished. Saving model...")
    network = environment["network"]
    save_path = environment["config"].save.path
    os.makedirs(save_path, exist_ok=True)
    model_name = f"{environment['config'].save.name}.safetensors"
    save_dtype = environment["save_dtype"]
    network.save_weights(os.path.join(save_path, model_name), dtype=save_dtype)
    print(f"Model saved to {os.path.join(save_path, model_name)}")


def main():
    """
    Main function to set up the environment, create the dataset,
    run the training loop, and handle shutdown.
    """
    with torch.no_grad():
        config = config_io()
        environment = config_envsetup(config) # Use the envsetup from batch_config_util

        # Log batch size, iteration count, and their product early
        batch_size = config.train.batch_size
        iterations = config.train.iterations
        expected_total_samples = batch_size * iterations
        print(f"Configured Batch Size: {batch_size}", file=sys.stderr)
        print(f"Configured Iterations: {iterations}", file=sys.stderr)
        print(f"Expected Total Samples (Batch Size * Iterations): {expected_total_samples}", file=sys.stderr)

        # --- THE NEW, ROBUST DATA PIPELINE INTEGRATION ---
        # Stage 1: Create the pure-python, reproducible training schedule.
        # This happens before any VRAM-intensive operations.
        schedule_dict = batch_data_pipeline.create_training_schedule(environment['config'])

        # Stage 2: (Not really optional!) Dump the schedule for validation and debugging.
        # This is our proof that the plan is correct BEFORE we burn GPU cycles.
        schedule_log_dir = "logs"
        os.makedirs(schedule_log_dir, exist_ok=True)
        schedule_filename = os.path.join(
            schedule_log_dir, 
            f"training_schedule_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        )
        print(f"Dumping full training schedule to: {schedule_filename}")
        with open(schedule_filename, 'w') as f:
            # Use a simple default converter for any non-serializable types like Path objects
            json.dump(schedule_dict, f, indent=2, default=str)

        # Stage 3: Materialize the schedule into tensor batches.
        # This function now handles all VAE/CLIP encoding, caching, and batching.
        # Note: The environment models (VAE, text encoders) will be moved to the GPU inside this function as needed.
        with torch.no_grad():
            static_batches = batch_data_pipeline.materialize_static_batches(schedule_dict, environment)

        print("Data materialization complete. Managing VRAM for training loop.")

        # --- END OF NEW PIPELINE INTEGRATION ---

        # NOTE: The VRAM management logic from the original main() is now effectively
        # handled INSIDE `materialize_static_batches`. The VAE is loaded to the GPU for
        # optimal batch encoding and then should be offloaded back to the CPU.
        # We will assume the materialize function is well-behaved and cleans up after itself.
        # For good measure, we can explicitly manage VRAM here.

        tdcpu = torch.device("cpu")
        
        # Unload VAE and Text Encoders to CPU after batch preparation
        vae = environment.pop('vae')
        tokenizers = environment.pop('tokenizers')
        text_encoders = environment.pop('text_encoders')
        vae=vae.to(device=tdcpu)
        for te in text_encoders:
            te=te.to(device=tdcpu)
        environment["vae"] = vae
        environment["tokenizers"] = text_encoders
        gc.collect()
        torch.cuda.empty_cache()

        # Load UNet back to GPU for training
        unet1 = environment['unet']
        unet2 = unet1.to(environment['device'])
        del unet1
        gc.collect()
        torch.cuda.empty_cache()
        #WOOO COMPILE TIME!!!
        #if config.other.torch_compile:
        #hardcode_allow_compile = False
        #if hardcode_allow_compile:
        if config.other.get("torch_compile", False):
            print("compiling unet for very fast hayai")
            torch._inductor.config.coordinate_descent_tuning = True
            torch._inductor.config.triton.unique_kernel_names = True
            torch._inductor.config.fx_graph_cache = True 
            torch._functorch.config.enable_autograd_cache = True

            if config.other.get("use_pytorch_SDPA", False):
                from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp
                enable_cudnn_sdp(True)
                enable_flash_sdp(True)

            unet2 = torch.compile(unet2, mode="reduce-overhead", fullgraph=True)
        else:
            print("skipped compile because....")
        environment['unet'] = unet2

    # Create adapter network
    network = lora.BatchedLoRANetwork(
        unet=environment['unet'],
        rank=config.network.rank,
        alpha=config.network.alpha,
        train_method=config.network.training_method,
    ).to(environment['device'], dtype=environment['weight_dtype'])

        # Initialize gradient noise scale estimator
    gradient_noise_estimator = None
    if hasattr(config.train, "estimate_gradient_noise_scale") and config.train.estimate_gradient_noise_scale:
        with torch.no_grad():
            # micro_batch_size for GNS is the actual batch size processed by UNet in one go
            micro_batch_size = config.train.batch_size
            profile_freq = config.train.get("gns_profile_freq", 10) # Default to 100 steps
            ema_alpha_fast = config.train.get("gns_ema_fast", 0.1) # Default EMA alpha
            ema_alpha_slow = config.train.get("gns_ema_slow", 0.01) # Default EMA alpha
            gradient_noise_estimator = GradientNoiseEstimator(network, micro_batch_size, profile_freq, ema_alpha_fast, ema_alpha_slow)
    environment["gradient_noise_estimator"] = gradient_noise_estimator

    #slurp optimizer args from config
    optimizer_kwargs = {}
    if hasattr(config.train, "optimizer_args") and config.train.optimizer_args is not None and len(config.train.optimizer_args) > 0:
        for arg in config.train.optimizer_args.split(" "):
            key, value = arg.split("=")
            value = ast.literal_eval(value)
            optimizer_kwargs[key] = value
    environment['network'] = network
    #optimizer_module = train_util.get_optimizer(config.train.optimizer)
    paramforoptim = network.prepare_optimizer_params()
    print(f"length prepared optim parameters:{len(paramforoptim)}")
    optimizer = environment['optimizer'](paramforoptim,
    lr=config.train.lr, 
    **optimizer_kwargs )

    lr_scheduler = get_scheduler(
        name=config.train.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.train.get("lr_warmup_steps", 10),
        num_training_steps=config.train.iterations,
    )
    environment['optimizer']=optimizer#yes this switcharoo is necessary
    environment['lr_scheduler'] = lr_scheduler

    # Run the training loop with the static batches
    environment = training_loop(environment, static_batches)

    # Save the final model
    graceful_shutdown(environment)


if __name__ == "__main__":
    runname = "batched_training_loop"
    log_file_path, orig_stdout, orig_stderr = setup_logging(runname=runname)
    try:
        print(f"--- Starting Batched Training Loop ---", file=orig_stdout)
        print(f"All output will be redirected to: {log_file_path}", file=orig_stdout)
        
        main()
        
    except Exception as e:
        import traceback
        print("--- EXCEPTION OCCURRED ---", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print(f"\n--- EXCEPTION OCCURRED ---", file=orig_stderr)
        print(f"An error occurred. Check the log file for details: {log_file_path}", file=orig_stderr)
        traceback.print_exc(file=orig_stderr)
        raise
        
    finally:
        sys.stdout.close()
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        print(f"--- Script finished. Log saved to {log_file_path} ---")