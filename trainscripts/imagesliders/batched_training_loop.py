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

from .batch_config_util import (
    dataset_constructor,
    setup_logging,
    AttrDict,
    load_config_from_yaml,
    parse_precision,
    config_io as batch_config_util_io,
)
from . import batch_lora as lora
from . import batch_train_util
from . import batch_model_util
#new slider algo related tensor formation code
from . import batch_slider_algo




def envsetup(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = parse_precision(config.train.precision)
    vae, unet, tokenizers, text_encoders, noise_scheduler = batch_model_util.load_models(config, device, weight_dtype)
    
    # Cast the entire unet to the correct dtype
    unet.to(device, dtype=weight_dtype)
    
    network = lora.BatchedLoRANetwork(
        unet=unet,
        rank=config.network.rank,
        alpha=config.network.alpha,
        train_method=config.network.training_method,
    ).to(device, dtype=weight_dtype)
    network.prepare_optimizer_params()

    optimizer_name = config.train.optimizer.lower()
    if optimizer_name == "adamw":
        optimizer = optim.AdamW(network.parameters(), lr=config.train.lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    lr_scheduler = get_scheduler(
        name=config.train.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=config.train.iterations,
    )

    environment = {
        "unet": unet,
        "vae": vae,
        "noise_scheduler": noise_scheduler,
        "tokenizers": tokenizers,
        "text_encoders": text_encoders,
        "network": network,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "device": device,
        "weight_dtype": weight_dtype,
        "config": config,
    }
    return environment


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

    # Create pairing map based on scales
    pairing_map = batch_slider_algo.create_pairing_map(scales)

    # Form CFG microbatch
    microbatch_cfg, ordered_latents, ordered_scales = batch_slider_algo.form_cfg_microbatch(
        unswizzled_data, batch, pairing_map
    )

    # The timesteps for the UNet need to be the actual timesteps from the scheduler
    # corresponding to the `timesteps_to` indices, reordered to match the microbatch.
    # We need to ensure that timesteps_to is also reordered to match ordered_latents
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
        
        predicted_noise = batch_train_util.batched_predict_noise_xl_modular(
            unet,
            noise_scheduler,
            unet_timesteps_cfg,
            microbatch_cfg['latents_cfg'],
            microbatch_cfg['text_embeds_cfg'],
            microbatch_cfg['pooled_embeds_cfg'],
            microbatch_cfg['add_time_ids_cfg'],
            guidance_scale=1,
        )

    # Calculate loss using the new paired loss function
    # The target noise needs to be reordered and duplicated to match the microbatch_cfg
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
    # Create a seed for the training run for reproducibility
    seed = torch.initial_seed()
    print(f"Using seed {seed} for training.")
    
    for i in range(environment["config"].train.iterations):
        # Cycle through the static batches
        batch = static_batches[i % len(static_batches)]
        # Pass the iteration number as the seed for this step
        loss = train_step(environment, batch, seed + i)

        if i % 10 == 0:
            print(f"Iteration {i+1}/{environment['config'].train.iterations}, Loss: {loss}")
    
    return environment


def graceful_shutdown(environment: dict):
    """Saves the trained network weights."""
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
    # Load configuration using the local config_io
    # subdicts are accessible through attrdict by name 
    config = batch_config_util_io()

    # Set up the training environment using the local envsetup
    environment = envsetup(config)

    # Create the dataset and dataloader
    # We use use_latents=False to get paths, which are then loaded in the list comprehension
    #must pass inner config or something 
    dataloader = dataset_constructor(config, environment)

    # Pre-generate all batches to create a static dataset for the training loop
    print("Pre-generating and caching all batches...")
    static_batches = [batch for batch in tqdm(dataloader, desc="Caching batches")]
    print(f"Cached {len(static_batches)} batches.")

    # Clean up VRAM before starting training
    # The VAE and text_encoders are no longer needed after caching the batches
    gc.collect()
    torch.cuda.empty_cache()

    # Run the training loop with the static batches
    environment = training_loop(environment, static_batches)

    # Save the final model
    graceful_shutdown(environment)


def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(log_dir, f"batched_training_loop_{timestamp}.log")
    
    # Save original stdout and stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Redirect stdout and stderr to the log file
    log_file = open(log_filename, "w")
    sys.stdout = log_file
    sys.stderr = log_file
    
    print(f"Logging output to {log_filename}")
    
    return log_filename, original_stdout, original_stderr

if __name__ == "__main__":
    log_file_path, orig_stdout, orig_stderr = setup_logging()
    try:
        # Print to the original console that logging has started
        print(f"--- Starting Batched Training Loop ---", file=orig_stdout)
        print(f"All output will be redirected to: {log_file_path}", file=orig_stdout)
        
        main()
        
    except Exception as e:
        import traceback
        # Log the exception to the file
        print("--- EXCEPTION OCCURRED ---", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        # Also print exception to the original console
        print(f"\n--- EXCEPTION OCCURRED ---", file=orig_stderr)
        print(f"An error occurred. Check the log file for details: {log_file_path}", file=orig_stderr)
        traceback.print_exc(file=orig_stderr)
        raise
        
    finally:
        # Restore stdout and stderr
        sys.stdout.close()
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        print(f"--- Script finished. Log saved to {log_file_path} ---")
