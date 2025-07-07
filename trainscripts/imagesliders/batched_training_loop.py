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
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from pathlib import Path
import time
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Union, Literal, List, Dict, Any
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from .batch_config_util import (
    setup_logging,
    AttrDict,
    load_config_from_yaml,
    parse_precision,
    config_io,
    envsetup as config_envsetup, # Renamed to avoid conflict
)
from . import batch_lora as lora
from . import batch_train_util
from . import batch_model_util
from . import batch_slider_algo
from .data_schedule import TrainingSchedule # Import TrainingSchedule

# --- Copied from batch_dataset_encoding.py for self-containment ---

UNET_ATTENTION_TIME_EMBED_DIM = 256  # XL
TEXT_ENCODER_2_PROJECTION_DIM = 1280
UNET_PROJECTION_CLASS_EMBEDDING_INPUT_DIM = 2816

SDXL_TEXT_ENCODER_TYPE = Union[CLIPTextModel, CLIPTextModelWithProjection]
DIFFUSERS_CACHE_DIR = None # if you want to change the cache dir, change this

def get_sha256_checksum(file_path):
    start_time = time.time()
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256.update(byte_block)
    end_time = time.time()
    return sha256.hexdigest(), (end_time - start_time)

def encode_images_to_latents(images, vae, device, weight_dtype):
    start_time = time.time()
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True)
    image_tensors = [image_processor.preprocess(image).to(device, dtype=weight_dtype) for image in images]
    image_batch = torch.cat(image_tensors, dim=0)
    latents = vae.encode(image_batch).latent_dist.sample(None)
    end_time = time.time()
    return latents, (end_time - start_time)

def save_latents_to_disk(latents, output_dir, image_path, vae_state_dict):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    latent_filename = os.path.splitext(os.path.basename(image_path))[0] + ".pt"
    latent_path = os.path.join(output_dir, latent_filename)
    torch.save(latents, latent_path)

    vae_checksum_hasher = hashlib.sha256()
    for k, v in vae_state_dict.items():
        vae_checksum_hasher.update(k.encode('utf-8'))
        vae_checksum_hasher.update(v.cpu().to(torch.float32).numpy().tobytes())
    vae_checksum = vae_checksum_hasher.hexdigest()

    latent_checksum, _ = get_sha256_checksum(latent_path) # Get checksum without timing here
    metadata = {
        "image_checksum": get_sha256_checksum(image_path)[0], # Get checksum without timing here
        "vae_checksum": vae_checksum,
        "latent_checksum": latent_checksum,
    }

    metadata_filename = os.path.splitext(os.path.basename(image_path))[0] + ".json"
    metadata_path = os.path.join(output_dir, metadata_filename)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

def check_and_encode_latent(image_path, vae, device, weight_dtype, output_dir, vae_state_dict, force_reencode=False):
    latent_filename = os.path.splitext(os.path.basename(image_path))[0] + ".pt"
    latent_path = os.path.join(output_dir, latent_filename)

    latent_encoding_time = 0

    if not force_reencode and os.path.exists(latent_path):
        return True, latent_encoding_time
    else:
        if force_reencode:
            print(f"Force re-encoding latents for {image_path}.")
        else:
            print(f"Latents for {image_path} not found. Encoding.")

    image = Image.open(image_path).convert("RGB")
    latents, encoding_time = encode_images_to_latents([image], vae, device, weight_dtype)
    latent_encoding_time += encoding_time
    save_latents_to_disk(latents, output_dir, image_path, vae_state_dict)
    return False, latent_encoding_time

def get_latent_for_image(image_path, vae, device, weight_dtype, output_dir, vae_state_dict, force_reencode=False):
    is_cached, encoding_time = check_and_encode_latent(image_path, vae, device, weight_dtype, output_dir, vae_state_dict, force_reencode=force_reencode)
    latent_filename = os.path.splitext(os.path.basename(image_path))[0] + ".pt"
    latent_path = os.path.join(output_dir, latent_filename)
    
    load_start_time = time.time()
    loaded_latent = torch.load(latent_path, weights_only=True)
    load_end_time = time.time()
    latent_load_time = load_end_time - load_start_time

    return loaded_latent, encoding_time, latent_load_time

def initialize_latent_cache(config, environment):
    # ImageScaleDataset is not directly used here, but TrainingSchedule uses it internally
    # to get unique image paths.
    # For now, we'll assume TrainingSchedule handles the image path collection.
    training_schedule = TrainingSchedule(config) # Use TrainingSchedule to get unique image paths
    unique_image_paths = sorted(list(set([item.image_path for batch in training_schedule for item in batch])))
    print(f"Found {len(unique_image_paths)} unique images to process for latent cache initialization.")

    vae = environment['vae']
    device = environment['device']
    weight_dtype = environment['weight_dtype']
    output_dir = Path(config.dataset_config.dataset.folder_main) / "latents"
    
    vae_batch_size = config.train.get("vae_encoding_batch_size", 4)
    print(f"Using VAE encoding batch size: {vae_batch_size}")

    total_comparison_time = 0
    mismatched_files = 0

    for i in tqdm(range(0, len(unique_image_paths), vae_batch_size), desc="Initializing latent cache"):
        batch_paths = unique_image_paths[i:i+vae_batch_size]
        
        images_to_encode = [Image.open(p).convert("RGB") for p in batch_paths]
        
        new_latents_gpu, encoding_time = encode_images_to_latents(images_to_encode, vae, device, weight_dtype)
        new_latents_cpu = new_latents_gpu.cpu()

        for j, img_path in enumerate(batch_paths):
            new_latent_single = new_latents_cpu[j].unsqueeze(0)
            
            latent_filename = os.path.splitext(os.path.basename(img_path))[0] + ".pt"
            latent_path = os.path.join(output_dir, latent_filename)

            if os.path.exists(latent_path):
                try:
                    old_latent = torch.load(latent_path, map_location='cpu')
                    
                    comparison_start_time = time.time()
                    are_close = torch.allclose(old_latent, new_latent_single, atol=1e-4, rtol=1e-3) # Use tolerances
                    comparison_time = time.time() - comparison_start_time
                    total_comparison_time += comparison_time
                    
                    if not are_close:
                        mismatched_files += 1
                        diff = torch.mean(torch.abs(old_latent - new_latent_single))
                        print(f"Latent mismatch for {img_path}. Mean absolute difference: {diff.item()}. Comparison took {comparison_time:.4f}s.")
                except Exception as e:
                    print(f"Warning: Could not load or compare existing latent for {img_path}: {e}")

            torch.save(new_latent_single, latent_path)
            
    print(f"Latent cache initialization finished.")
    print(f"Total time spent on parity checks: {total_comparison_time:.4f} seconds.")
    if mismatched_files > 0:
        print(f"Found {mismatched_files} mismatched latents that were updated.")

def initialize_text_embedding_cache(config, environment, training_schedule):
    print("Initializing text embedding cache...")
    tokenizers = environment['tokenizers']
    text_encoders = environment['text_encoders']
    device = environment['device']
    weight_dtype = environment['weight_dtype']

    unique_prompts = training_schedule.get_unique_prompts()
    text_embedding_cache = {}

    for prompt_dict in tqdm(unique_prompts, desc="Encoding unique prompts"):
        text_embeddings, pooled_embeds = batch_train_util.create_batched_prompt_embeddings(
            tokenizers,
            text_encoders,
            prompt_dict,
        )
        text_embedding_cache[frozenset(prompt_dict.items())] = (
            text_embeddings.cpu(), pooled_embeds.cpu()
        )
    print(f"Cached {len(unique_prompts)} unique text embeddings.")
    return text_embedding_cache

def prepare_cached_batches(config, environment):
    """
    Pre-generates and caches all batches for the training loop using the TrainingSchedule.
    """
    if config.train.get("force_init_latentcache", False):
        initialize_latent_cache(config, environment)

    training_schedule = TrainingSchedule(config)
    text_embedding_cache = initialize_text_embedding_cache(config, environment, training_schedule)

    print("Pre-generating and caching all batches from schedule...")
    static_batches = []
    total_latent_encoding_time = 0
    total_latent_load_time = 0
    total_cat_latents_time = 0
    total_cat_text_embeds_time = 0
    total_cat_pooled_embeds_time = 0

    vae = environment['vae']
    device = environment['device']
    weight_dtype = environment['weight_dtype']
    tokenizers = environment['tokenizers']
    text_encoders = environment['text_encoders']

    for batch_items in tqdm(training_schedule, desc="Caching batches"):
        latents = []
        scales = []
        all_text_embeddings = []
        all_pooled_embeds = []

        pair_indices = []
        is_low_cases = []
        for item in batch_items:
            latent, encoding_time, load_time = get_latent_for_image(
                item.image_path, vae, device, weight_dtype, 
                Path(config.dataset.folder_main) / "latents", 
                vae.state_dict(),
                force_reencode=config.train.get("force_reencode_latents", False)
            )
            latents.append(latent)
            scales.append(item.scale)
            pair_indices.append(item.pair_index)
            is_low_cases.append(item.is_low_case)
            total_latent_encoding_time += encoding_time
            total_latent_load_time += load_time

            cached_text_embeds, cached_pooled_embeds = text_embedding_cache[frozenset(item.prompt.items())]
            all_text_embeddings.append(cached_text_embeds.to(device, dtype=weight_dtype))
            all_pooled_embeds.append(cached_pooled_embeds.to(device, dtype=weight_dtype))

        cat_latents_start_time = time.time()
        latents_batch = torch.stack(latents).to(device, dtype=weight_dtype)
        cat_latents_time = time.time() - cat_latents_start_time

        scales_batch = torch.tensor(scales, dtype=weight_dtype, device=device)

        # Create pairing map here, as it's part of dataset construction
        pairing_map = batch_slider_algo.create_pairing_map(scales_batch)

        cat_text_embeds_start_time = time.time()
        text_embeddings_batch = torch.cat(all_text_embeddings, dim=0).to(device, dtype=weight_dtype)
        cat_text_embeds_time = time.time() - cat_text_embeds_start_time

        cat_pooled_embeds_start_time = time.time()
        pooled_embeds_batch = torch.cat(all_pooled_embeds, dim=0).to(device, dtype=weight_dtype)
        cat_pooled_embeds_time = time.time() - cat_pooled_embeds_start_time

        add_time_ids = batch_train_util.get_add_time_ids(
            1024, 1024, False, dtype=latents_batch.dtype
        ).repeat(len(latents_batch), 1).to(device, dtype=weight_dtype)

        batch = {
            "latents": latents_batch,
            "scales": scales_batch,
            "text_embeddings": text_embeddings_batch,
            "pooled_embeds": pooled_embeds_batch,
            "add_time_ids": add_time_ids,
            "pair_indices": torch.tensor(pair_indices, dtype=torch.long, device=device),
            "is_low_cases": torch.tensor(is_low_cases, dtype=torch.bool, device=device),
            "pairing_map": pairing_map,
            "guidance_scale": item.prompt.get("guidance_scale", 1.0), # Get guidance_scale from prompt, default to 1.0
        }
        static_batches.append(batch)

        if len(static_batches) == 1:
            print("\n--- Example Training Batch (First Batch) ---")
            print("Tensor Shapes:")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
                else:
                    print(f"  {key}: {type(value)}")
            
            print("\nMetadata for Cached Items:")
            for i, item in enumerate(batch_items):
                print(f"  Item {i}:")
                print(f"    Image Path: {item.image_path}")
                print(f"    Prompt: {item.prompt}")
                print(f"    Scale: {item.scale}")
                print(f"    Pair Index: {item.pair_index}")
                print(f"    Is Low Case: {item.is_low_case}")
            print("--------------------------------------------")

        total_cat_latents_time += cat_latents_time
        total_cat_text_embeds_time += cat_text_embeds_time
        total_cat_pooled_embeds_time += cat_pooled_embeds_time

    print(f"Cached {len(static_batches)} batches.")
    print(f"Total time spent in latent encoding: {total_latent_encoding_time:.4f} seconds")
    print(f"Total time spent in latent loading: {total_latent_load_time:.4f} seconds")
    print(f"Total time spent concatenating latents: {total_cat_latents_time:.4f} seconds")
    print(f"Total time spent concatenating text embeddings: {total_cat_text_embeds_time:.4f} seconds")
    print(f"Total time spent concatenating pooled embeddings: {total_cat_pooled_embeds_time:.4f} seconds")

    return static_batches

# --- End copied functions ---

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

    # Unpack batch data
    latents = batch["latents"].to(device, dtype=weight_dtype)
    scales = batch["scales"].to(device, dtype=weight_dtype)
    text_embeddings = batch["text_embeddings"].to(device, dtype=weight_dtype)
    pooled_embeds = batch["pooled_embeds"].to(device, dtype=weight_dtype)
    add_time_ids = batch["add_time_ids"].to(device, dtype=weight_dtype)
    pair_indices = batch["pair_indices"].to(device)
    is_low_cases = batch["is_low_cases"].to(device)
    guidance_scale = batch["guidance_scale"] # Retrieve guidance_scale from batch

    print(f"unpacked pooled embeds of shape: {pooled_embeds.shape}")
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

    # Unswizzle conditioning data
    unswizzled_data = batch_slider_algo.unswizzle_conditioning_data({
        "latents": latents,
        "text_embeddings": text_embeddings,
        "pooled_embeds": pooled_embeds,
    })

    # Retrieve pairing map from batch
    pairing_map = batch["pairing_map"]

    # Form CFG microbatch
    microbatch_cfg, ordered_latents, ordered_scales, ordered_indices = batch_slider_algo.form_cfg_microbatch(
        unswizzled_data, batch, pairing_map
    )

    # The timesteps for the UNet need to be the actual timesteps from the scheduler
    # corresponding to the `timesteps_to` indices, reordered to match the microbatch.
    reordered_timesteps_to = timesteps_to[ordered_indices]
    unet_timesteps = noise_scheduler.timesteps[reordered_timesteps_to].to(weight_dtype)
    unet_timesteps_cfg = torch.cat([unet_timesteps, unet_timesteps], dim=0)

    # Set LoRA scales, which must match the doubled cfg axis.
    batched_scales = ordered_scales.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    batched_scales_cfg = torch.cat([batched_scales, batched_scales], dim=0)
    network.set_lora_scales(batched_scales_cfg)

    with network:
        print(f"Shape of unet_timesteps_cfg: {unet_timesteps_cfg.shape}")
        print(f"Shape of noisy_latents_cfg: {microbatch_cfg['latents_cfg'].shape}")
        print(f"Shape of text_embeddings_cfg: {microbatch_cfg['text_embeds_cfg'].shape}")
        print(f"Shape of pooled_embeds_cfg: {microbatch_cfg['pooled_embeds_cfg'].shape}")
        print(f"Shape of add_time_ids_cfg: {microbatch_cfg['add_time_ids_cfg'].shape}")
        
        predicted_noise = batch_train_util.batched_predict_noise_xl(
            unet,
            noise_scheduler,
            unet_timesteps_cfg,
            microbatch_cfg['latents_cfg'],
            microbatch_cfg['text_embeds_cfg'],
            microbatch_cfg['pooled_embeds_cfg'],
            microbatch_cfg['add_time_ids_cfg'],
            guidance_scale=guidance_scale,
        )

    # Calculate loss using the new paired loss function
    target_noise_cfg = batch_slider_algo._cfg_duplicate(noise[ordered_indices])
    loss = batch_slider_algo.calculate_paired_loss(predicted_noise, target_noise_cfg, pairing_map)

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
    config = config_io()
    environment = config_envsetup(config) # Use the envsetup from batch_config_util

    # Unload UNet to CPU to free VRAM for VAE and Text Encoders during batch preparation
    unet = environment.pop('unet')
    unet.to('cpu')
    gc.collect()
    torch.cuda.empty_cache()

    # Prepare cached batches (image latents and text embeddings)
    static_batches = prepare_cached_batches(config, environment)
    
    # Unload VAE and Text Encoders to CPU after batch preparation
    vae = environment.pop('vae')
    tokenizers = environment.pop('tokenizers')
    text_encoders = environment.pop('text_encoders')
    vae.to('cpu')
    for te in text_encoders:
        te.to('cpu')
    gc.collect()
    torch.cuda.empty_cache()

    # Load UNet back to GPU for training
    unet.to(environment['device'])
    environment['unet'] = unet

    # Create adapter network
    network = lora.BatchedLoRANetwork(
        unet=environment['unet'],
        rank=config.network.rank,
        alpha=config.network.alpha,
        train_method=config.network.training_method,
    ).to(environment['device'], dtype=environment['weight_dtype'])
    network.prepare_optimizer_params()
    environment['network'] = network

    optimizer_name = config.train.optimizer.lower()
    if optimizer_name == "adamw":
        optimizer = optim.AdamW(network.parameters(), lr=config.train.lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    environment['optimizer'] = optimizer

    lr_scheduler = get_scheduler(
        name=config.train.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=config.train.iterations,
    )
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