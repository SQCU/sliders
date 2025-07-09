#python -m trainscripts.imagesliders.batched_training_loop -c F:/dox/ai/gemmy/sliders/trainscripts/imagesliders/data/batch_config.yaml
import torch
import os
import gc
from tqdm import tqdm
import datetime
import sys
import argparse
import yaml
from diffusers.optimization import get_scheduler
import torch.optim as optim
import hashlib
from PIL import Image
from pathlib import Path
import time
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Union, Literal, List, Dict, Any
from .batch_slider_algo import calculate_paired_loss, GradientNoiseEstimator
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

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
from .data_processing_utils import (
    resize_image_if_needed,
    encode_images_to_latents,
)
from .data_schedule import TrainingSchedule, prepare_cached_batches

from .data_schedule import TrainingSchedule # Import TrainingSchedule

UNET_ATTENTION_TIME_EMBED_DIM = 256  # XL
TEXT_ENCODER_2_PROJECTION_DIM = 1280
UNET_PROJECTION_CLASS_EMBEDDING_INPUT_DIM = 2816

SDXL_TEXT_ENCODER_TYPE = Union[CLIPTextModel, CLIPTextModelWithProjection]
DIFFUSERS_CACHE_DIR = None # if you want to change the cache dir, change this

def rectify_batch_fn(batch: dict, device: torch.device, weight_dtype: torch.dtype) -> dict:
    """
    Moves batch data to the appropriate device and dtype.
    """
    batch['latents'] = batch['latents'].to(device, dtype=weight_dtype)
    batch['scales'] = batch['scales'].to(device, dtype=weight_dtype)
    batch['pair_indices'] = batch['pair_indices'].to(device)
    batch['is_low_cases'] = batch['is_low_cases'].to(device)
    batch['cfg_text_embeddings'] = batch['cfg_text_embeddings'].to(device, dtype=weight_dtype)
    batch['cfg_pooled_embeds'] = batch['cfg_pooled_embeds'].to(device, dtype=weight_dtype)
    batch['add_time_ids'] = batch['add_time_ids'].to(device, dtype=torch.float32) # Keep this float32!
    return batch




def prepare_cfg_batch(
    noisy_latents: torch.Tensor,
    timesteps_to: torch.Tensor,
    noise_scheduler,
    weight_dtype: torch.dtype,
    uncond_text_embeddings: torch.Tensor,
    cond_text_embeddings: torch.Tensor,
    uncond_pooled_embeds: torch.Tensor,
    cond_pooled_embeds: torch.Tensor,
    add_time_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepares a CFG-ready batch by concatenating unconditional and conditional embeddings.
    """
    latents_cfg = torch.cat([noisy_latents, noisy_latents], dim=0)
    text_embeddings_cfg = torch.cat([uncond_text_embeddings, cond_text_embeddings], dim=0)
    pooled_embeds_cfg = torch.cat([uncond_pooled_embeds, cond_pooled_embeds], dim=0)
    add_time_ids_cfg = torch.cat([add_time_ids, add_time_ids], dim=0)

    #not to unet.dtype! these need to be a torch.long
    unet_timesteps = noise_scheduler.timesteps[timesteps_to]#.to(weight_dtype)
    unet_timesteps_cfg = torch.cat([unet_timesteps, unet_timesteps], dim=0)

    return latents_cfg, text_embeddings_cfg, pooled_embeds_cfg, add_time_ids_cfg, unet_timesteps_cfg


def train_step(environment: dict, batch: dict):
    """
    Performs a single training step with optional gradient accumulation and gradient noise estimation.
    """
    unet = environment["unet"]
    noise_scheduler = environment["noise_scheduler"]
    network = environment["network"]
    config = environment["config"]
    device = environment["device"]
    weight_dtype = environment["weight_dtype"]
    generator = environment["generator"]
    gradient_noise_estimator = environment["gradient_noise_estimator"]

    current_micro_batch = batch
    # For now, we'll use the full 'batch' as a micro-batch, and scale loss.

    latents = current_micro_batch["latents"]
    scales = current_micro_batch["scales"]
    pair_indices = current_micro_batch["pair_indices"]
    is_low_cases = current_micro_batch["is_low_cases"]
    guidance_scale = current_micro_batch["guidance_scale"]
    cfg_text_embeddings = current_micro_batch['cfg_text_embeddings']
    cfg_pooled_embeds = current_micro_batch['cfg_pooled_embeds']
    add_time_ids = current_micro_batch['add_time_ids']
    gradient_noise_estimator = environment["gradient_noise_estimator"]

    uncond_text_embeddings, cond_text_embeddings = cfg_text_embeddings.chunk(2)
    uncond_pooled_embeds, cond_pooled_embeds = cfg_pooled_embeds.chunk(2)

    # --- TIMESTEP LOGIC ---
    with torch.no_grad():
        noise_scheduler.set_timesteps(config.train.max_denoising_steps, device=device)
        
        timesteps_to = torch.randint(
            1, config.train.max_denoising_steps, (latents.shape[0],), device=device, generator=generator
        ).long()

        noise = torch.randn(latents.shape, device=latents.device, generator=generator)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps_to)
    
    # --- END TIMESTEP LOGIC ---

    latents_cfg, text_embeddings_cfg, pooled_embeds_cfg, add_time_ids_cfg, unet_timesteps_cfg = prepare_cfg_batch(
        noisy_latents,
        timesteps_to,
        noise_scheduler,
        weight_dtype,
        uncond_text_embeddings,
        cond_text_embeddings,
        uncond_pooled_embeds,
        cond_pooled_embeds,
        add_time_ids,
    )

    batched_scales_cfg = torch.cat([scales, scales], dim=0).unsqueeze(-1)
    network.set_lora_scales(batched_scales_cfg)

    #used to have with network here... have to move it to wrap around the backwards to avoid a gradient checkpointing error...
    predicted_noise = batch_train_util.nocfg_predict_noise_xl(
        unet,
        noise_scheduler,
        unet_timesteps_cfg,
        latents_cfg,
        text_embeddings_cfg,
        pooled_embeds_cfg,
        add_time_ids_cfg,
    )

    predicted_noise_uncond, predicted_noise_text = predicted_noise.chunk(2)
    predicted_noise_cfg_reduced = (predicted_noise_uncond + guidance_scale * (
        predicted_noise_text - predicted_noise_uncond
    )).to(device)

    loss = calculate_paired_loss(predicted_noise_cfg_reduced, noise, pair_indices, is_low_cases)

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

    progress_bar = tqdm(range(config.train.iterations), desc="training its")
    losses = []
    for i in progress_bar:   
        # Determine if this is a profiling step    
        if gradient_noise_estimator is not None:
            gradient_noise_estimator.pre_accumulate_step(global_step)
            is_profiling_step = gradient_noise_estimator is not None and gradient_noise_estimator.is_profiling
        else:
            is_profiling_step = False

        total_loss = 0.0
        #need to wrap the backprop w/ 'with network' or we fail to checkpoint scales metadata.
        with network:
            if is_profiling_step:
                # Iterate over micro-batches for gradient accumulation
                for _ in range(gradient_accumulation_steps):
                    micro_batch = static_batches[i % len(static_batches)] 
                    micro_batch = rectify_batch_fn(micro_batch, device, weight_dtype)
                    loss = train_step(environment, micro_batch)
                    #immediate backprop to get microgradient
                    loss.backward()
                    #tell estimator to capture and accumulate microgradient data
                    gradient_noise_estimator.post_micro_backward_step()
                    total_loss += loss.item()
            else:
                #NON-GRADIENTSTATS ACCUMULATION CONTROL FLOW
                optimizer.zero_grad()
                for _ in range(gradient_accumulation_steps):
                    micro_batch = static_batches[i % len(static_batches)] 
                    micro_batch = rectify_batch_fn(micro_batch, device, weight_dtype)
                    loss = train_step(environment, micro_batch)
                    # Scale loss for accumulation
                    loss = loss / gradient_accumulation_steps

                    # accumulate gradients
                    loss.backward()
                    total_loss += loss.item() * gradient_accumulation_steps

            #AFTER ACCUMULATION SEQUENCE
            #scaling to get mean from of sum-reduced non-profiling loss
            if is_profiling_step:
                gradient_noise_estimator.post_accumulate_step(gradient_accumulation_steps)
            #scale for logging
            avg_loss = total_loss/gradient_accumulation_steps
            losses.append(avg_loss)
            # Optimizer step and scheduler step
            #both profiling step and non profiling step have correct .grad attribute here
            optimizer.step()
            lr_scheduler.step()

            progress_bar.set_postfix({"Loss;avg": f"{avg_loss:.3f};{sum(losses)/len(losses):.3f}", "LR": f"{lr_scheduler.get_last_lr()[0]:.2e}"})
            # Print estimated gradient noise scale if enabled
            if gradient_noise_estimator is not None and gradient_noise_estimator.ema_b_est is not None:
                # Print on the same line as tqdm progress bar
                progress_bar.set_postfix_str({"Loss;avg": f"{avg_loss:.3f};{sum(losses)/len(losses):.3f}", "LR": f"{lr_scheduler.get_last_lr()[0]:.2e}", "B_crit": "{gradient_noise_estimator.ema_b_est:.2f}"})
            global_step += 1 # Increment global step after each training step
    
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

        # NOTE: The following VRAM management logic (moving UNet, VAE, Text Encoders to/from CPU)
        # is intentionally kept as-is, despite its apparent complexity and manual nature,
        # as per user instruction. It is understood that this is a deliberate design choice
        # for specific memory optimization needs.
        tdcpu = torch.device("cpu")
        # Unload UNet to CPU to free VRAM for VAE and Text Encoders during batch preparation
        unet1 = environment.pop('unet')
        unet2 = unet1.to(device=tdcpu)
        del unet1
        environment['unet'] = unet2
        gc.collect()
        torch.cuda.empty_cache()

        # Prepare cached batches (image latents and text embeddings)
        static_batches = prepare_cached_batches(config, environment)
        
        # Unload VAE and Text Encoders to CPU after batch preparation
        vae = environment.pop('vae')
        tokenizers = environment.pop('tokenizers')
        text_encoders = environment.pop('text_encoders')
        vae=vae.to(device=tdcpu)
        for te in text_encoders:
            te=te.to(device=tdcpu)
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
            ema_alpha = config.train.get("gns_ema_alpha", 0.05) # Default EMA alpha
            gradient_noise_estimator = GradientNoiseEstimator(network, micro_batch_size, profile_freq, ema_alpha)
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
    optimizer = environment['optimizer'](network.prepare_optimizer_params(),
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