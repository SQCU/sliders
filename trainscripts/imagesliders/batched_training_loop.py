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
from .batch_slider_algo import calculate_paired_loss
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
from .embedding_cache_util import build_unified_embedding_cache
from typing import List, Dict, Tuple, Any

from .data_schedule import TrainingSchedule # Import TrainingSchedule

UNET_ATTENTION_TIME_EMBED_DIM = 256  # XL
TEXT_ENCODER_2_PROJECTION_DIM = 1280
UNET_PROJECTION_CLASS_EMBEDDING_INPUT_DIM = 2816

SDXL_TEXT_ENCODER_TYPE = Union[CLIPTextModel, CLIPTextModelWithProjection]
DIFFUSERS_CACHE_DIR = None # if you want to change the cache dir, change this



def prepare_cached_batches(config, environment):
    """
    Pre-generates and caches all batches for the training loop using the TrainingSchedule.
    """
    training_schedule = TrainingSchedule(config)
    unified_cache = build_unified_embedding_cache(config, environment, training_schedule)
    image_latents_cache = unified_cache["image_latents"]
    text_embeddings_cache = unified_cache["text_embeddings"]

    print("Pre-generating and caching all batches from schedule...")
    static_batches = []

    device = environment['device']
    weight_dtype = environment['weight_dtype']

    for batch_items in tqdm(training_schedule, desc="Caching batches"):
        latents = []
        scales = []
        all_positive_text_embeddings = []
        all_positive_pooled_embeds = []
        all_neutral_text_embeddings = []
        all_neutral_pooled_embeds = []
        all_uncond_text_embeddings = []
        all_uncond_pooled_embeds = []

        pair_indices = []
        is_low_cases = []
        for item in batch_items:
            latent = image_latents_cache[item.image_path]
            latents.append(latent)
            scales.append(item.scale)
            pair_indices.append(item.pair_index)
            is_low_cases.append(item.is_low_case)

            (
                positive_text_embeds,
                positive_pooled_embeds,
                uncond_text_embeds,
                uncond_pooled_embeds,
                neutral_text_embeds,
                neutral_pooled_embeds,
            ) = text_embeddings_cache[frozenset(item.prompt.items())]

            if item.is_low_case:
                all_cond_text_embeddings.append(neutral_text_embeds)
                all_cond_pooled_embeds.append(neutral_pooled_embeds)
            else:
                all_cond_text_embeddings.append(positive_text_embeds)
                all_cond_pooled_embeds.append(positive_pooled_embeds)
            
            all_uncond_text_embeddings.append(uncond_text_embeds)
            all_uncond_pooled_embeds.append(uncond_pooled_embeds)

        latents_batch = torch.cat(latents).to(dtype=weight_dtype)

        scales_batch = torch.tensor(scales, dtype=weight_dtype)

        cond_text_embeddings_batch = torch.cat(all_cond_text_embeddings, dim=0)
        cond_pooled_embeds_batch = torch.cat(all_cond_pooled_embeds, dim=0)
        uncond_text_embeddings_batch = torch.cat(all_uncond_text_embeddings, dim=0)
        uncond_pooled_embeds_batch = torch.cat(all_uncond_pooled_embeds, dim=0)

        #keep time ids float32
        #hardcode 512x512 for now no wait a second wrong place.
        add_time_ids = batch_train_util.get_add_time_ids(
            1024, 1024, False, dtype=torch.float32
        ).repeat(len(latents_batch), 1)

        batch = {
            "latents": latents_batch,
            "scales": scales_batch,
            "cond_text_embeddings": cond_text_embeddings_batch,
            "cond_pooled_embeds": cond_pooled_embeds_batch,
            "uncond_text_embeddings": uncond_text_embeddings_batch,
            "uncond_pooled_embeds": uncond_pooled_embeds_batch,
            "add_time_ids": add_time_ids,
            "pair_indices": torch.tensor(pair_indices, dtype=torch.long, device=device),
            "is_low_cases": torch.tensor(is_low_cases, dtype=torch.bool, device=device),
            "guidance_scale": item.prompt.get("guidance_scale", 1.0), # Get guidance_scale from prompt, default to 1.0
        }
        static_batches.append(batch)
        
    print(f"Cached {len(static_batches)} batches.")

    return static_batches

def prepare_cfg_batch(
    batch: dict,
    noisy_latents: torch.Tensor,
    timesteps_to: torch.Tensor,
    noise_scheduler,
    weight_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepares a CFG-ready batch by concatenating unconditional and conditional embeddings.
    """
    cond_text_embeddings = batch['cond_text_embeddings']
    cond_pooled_embeds = batch['cond_pooled_embeds']
    uncond_text_embeddings = batch['uncond_text_embeddings']
    uncond_pooled_embeds = batch['uncond_pooled_embeds']
    add_time_ids = batch['add_time_ids']

    latents_cfg = torch.cat([noisy_latents, noisy_latents], dim=0)
    text_embeddings_cfg = torch.cat([uncond_text_embeddings, cond_text_embeddings], dim=0)
    pooled_embeds_cfg = torch.cat([uncond_pooled_embeds, cond_pooled_embeds], dim=0)
    add_time_ids_cfg = torch.cat([add_time_ids, add_time_ids], dim=0)

    #not to unet.dtype! these need to be a torch.long
    unet_timesteps = noise_scheduler.timesteps[timesteps_to]#.to(weight_dtype)
    unet_timesteps_cfg = torch.cat([unet_timesteps, unet_timesteps], dim=0)

    return latents_cfg, text_embeddings_cfg, pooled_embeds_cfg, add_time_ids_cfg, unet_timesteps_cfg


def train_step(environment: dict, batch: dict, seed: int):
    """
    Performs a single training step. The function is simplified to accept
    the environment dictionary, a single batch of data, and a seed for reproducibility.
    """
    # Unpack necessary components from the environment
    unet = environment["unet"]
    noise_scheduler = environment["noise_scheduler"]
    network = environment["network"]
    optimizer = environment["optimizer"]
    lr_scheduler = environment["lr_scheduler"]
    config = environment["config"]
    device = environment["device"]
    weight_dtype = environment["weight_dtype"]

    # move batch data to gpu
    latents = batch["latents"].to(device, dtype=weight_dtype)
    scales = batch["scales"].to(device, dtype=weight_dtype)
    pair_indices = batch["pair_indices"].to(device)
    is_low_cases = batch["is_low_cases"].to(device)
    guidance_scale = batch["guidance_scale"] 
    # We also need to move the embeddings that are used in prepare_cfg_batch
    # Let's just modify the `batch` dict in-place for simplicity before passing it
    batch['cond_text_embeddings'] = batch['cond_text_embeddings'].to(device, dtype=weight_dtype)
    batch['cond_pooled_embeds'] = batch['cond_pooled_embeds'].to(device, dtype=weight_dtype)
    batch['uncond_text_embeddings'] = batch['uncond_text_embeddings'].to(device, dtype=weight_dtype)
    batch['uncond_pooled_embeds'] = batch['uncond_pooled_embeds'].to(device, dtype=weight_dtype)
    batch['add_time_ids'] = batch['add_time_ids'].to(device, dtype=torch.float32) # Remember to keep this float32!
    # Retrieve guidance_scale from batch

    print(f"Shape of initial latents (batch['latents']): {latents.shape}")

    # Prepare for training step
    optimizer.zero_grad()

    # --- TIMESTEP LOGIC ---
    with torch.no_grad():
        noise_scheduler.set_timesteps(config.train.max_denoising_steps, device=device)
        generator = torch.Generator(device=device).manual_seed(seed)
        
        # Generate a random timestep for each item in the batch
        timesteps_to = torch.randint(
            1, config.train.max_denoising_steps, (latents.shape[0],), device=device, generator=generator
        ).long()

        # Add noise to latents using the generated timesteps
        noise = torch.randn(latents.shape, device=latents.device, generator=generator)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps_to)
    
    # --- END TIMESTEP LOGIC ---

    # Form CFG microbatch
        latents_cfg, text_embeddings_cfg, pooled_embeds_cfg, add_time_ids_cfg, unet_timesteps_cfg = prepare_cfg_batch(
            batch,
            noisy_latents,
            timesteps_to,
            noise_scheduler,
            weight_dtype,
        )

    # Set LoRA scales, which must match the doubled cfg axis.
    #batched_scales = scales.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    #hopefully a [batchdim,1] tensor has shape (16,1) -> broadcasts to variably shaped unet layers?
        batched_scales_cfg = torch.cat([scales, scales], dim=0).unsqueeze(-1)
        network.set_lora_scales(batched_scales_cfg)

    with network:
        print(f"Shape of unet_timesteps_cfg: {unet_timesteps_cfg.shape}")
        print(f"Shape of noisy_latents_cfg: {latents_cfg.shape}")
        print(f"Shape of text_embeddings_cfg: {text_embeddings_cfg.shape}")
        print(f"Shape of pooled_embeds_cfg: {pooled_embeds_cfg.shape}")
        print(f"Shape of add_time_ids_cfg: {add_time_ids_cfg.shape}")
        
        predicted_noise = batch_train_util.batched_predict_noise_xl(
            unet,
            noise_scheduler,
            unet_timesteps_cfg,
            latents_cfg,
            text_embeddings_cfg,
            pooled_embeds_cfg,
            add_time_ids_cfg,
            guidance_scale=guidance_scale,
        )

    predicted_noise_uncond, predicted_noise_text = predicted_noise.chunk(2)
    predicted_noise_cfg_reduced = (predicted_noise_uncond + guidance_scale * (
        predicted_noise_text - predicted_noise_uncond
    )).to(device)

    print(f"Shape of predicted_noise_cfg_reduced: {predicted_noise_cfg_reduced.shape}")
    print(f"Shape of noise: {noise.shape}")
    print(f"Configured batch size: {config.train.batch_size}")

    # Calculate loss using the new paired loss function
    loss = calculate_paired_loss(predicted_noise_cfg_reduced, noise, pair_indices, is_low_cases)

    # Backpropagation
    loss.backward()
    optimizer.step()
    lr_scheduler.step()

    return loss.item()


def training_loop(environment: dict, static_batches: list):
    """
    Main training loop that iterates over a static list of pre-generated batches.
    """
    seed = torch.initial_seed()
    print(f"Using seed {seed} for training.")
    
    for i in range(environment["config"].train.iterations):
        batch = static_batches[i % len(static_batches)]
        loss = train_step(environment, batch, seed + i)

        if i % 10 == 0:
            print(f"Iteration {i+1}/{environment['config'].train.iterations}, Loss: {loss}")
    
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
    network.save_weights(os.path.join(save_path, model_name))
    print(f"Model saved to {os.path.join(save_path, model_name)}")


def main():
    """
    Main function to set up the environment, create the dataset,
    run the training loop, and handle shutdown.
    """
    with torch.no_grad():
        config = config_io()
        environment = config_envsetup(config) # Use the envsetup from batch_config_util

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
        hardcode_allow_compile = False
        if hardcode_allow_compile:
            print("compiling unet for very fast hayai")
            unet2 = torch.compile(unet2, mode="reduce-overhead", fullgraph=True)
        else:
            print("skipped compile because something dubious was happening in debug.")
        environment['unet'] = unet2

    # Create adapter network
    network = lora.BatchedLoRANetwork(
        unet=environment['unet'],
        rank=config.network.rank,
        alpha=config.network.alpha,
        train_method=config.network.training_method,
    ).to(environment['device'], dtype=environment['weight_dtype'])

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
        num_warmup_steps=0,
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