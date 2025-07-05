import torch
import os
import gc
from tqdm import tqdm
import datetime
import sys

from trainscripts.imagesliders.batch_config_util import (
    config_io,
    envsetup,
    dataset_constructor,
    setup_logging,
    AttrDict,
)
from trainscripts.imagesliders import batch_lora as lora
from trainscripts.imagesliders import batch_train_util


def train_step(environment: dict, batch: dict):
    """
    Performs a single training step. The function is simplified to accept
    the environment dictionary and a single batch of data.
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

    # Prepare for training step
    optimizer.zero_grad()
    noise_scheduler.set_timesteps(config.train.max_denoising_steps, device=device)
    
    # Select a random timestep for noise addition
    timesteps_to = torch.randint(
        1, config.train.max_denoising_steps, (1,)
    ).item()

    # Add noise to latents
    noisy_latents, noise = batch_train_util.get_batched_noisy_images(
        latents,
        None,  # VAE is not needed here as we operate on latents
        None,  # Generator is not needed for noise addition
        noise_scheduler,
        config,
        0,
        timesteps_to,
        device,
        weight_dtype,
    )

    # Set LoRA scales for the current batch
    batched_scales = scales.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    network.set_lora_scales(batched_scales)

    # Predict noise
    with network:
        batch_timesteps = noise_scheduler.timesteps[
            int(timesteps_to * 1000 / config.train.max_denoising_steps)
        ]
        predicted_noise = batch_train_util.batched_predict_noise_xl_modular(
            unet,
            noise_scheduler,
            batch_timesteps,
            noisy_latents,
            text_embeddings,
            pooled_embeds,
            add_time_ids,
            guidance_scale=1,
        )

    # Calculate loss
    # The predicted noise is scaled by the LoRA scales during the forward pass
    loss_per_element = (predicted_noise - noise).pow(2).to(torch.float32)
    loss = loss_per_element.mean()

    # Backpropagation
    loss.backward()
    optimizer.step()
    lr_scheduler.step()

    return loss.item()


def training_loop(environment: dict, static_batches: list):
    """
    Main training loop that iterates over a static list of pre-generated batches.
    """
    for i in range(environment["config"].train.iterations):
        # Cycle through the static batches
        batch = static_batches[i % len(static_batches)]
        loss = train_step(environment, batch)

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
    # Load configuration using the unified config_io
    config = AttrDict(config_io())

    # Set up the training environment
    environment = envsetup(config)

    # Create the dataset and dataloader
    # We use use_latents=False to get paths, which are then loaded in the list comprehension
    dataloader = dataset_constructor(config, environment, use_latents=False)

    # Pre-generate all batches to create a static dataset for the training loop
    print("Pre-generating and caching all batches...")
    static_batches = [batch for batch in tqdm(dataloader, desc="Caching batches")]
    print(f"Cached {len(static_batches)} batches.")

    # Clean up VRAM before starting training
    del environment["vae"]
    del environment["text_encoders"]
    gc.collect()
    torch.cuda.empty_cache()

    # Run the training loop with the static batches
    environment = training_loop(environment, static_batches)

    # Save the final model
    graceful_shutdown(environment)


if __name__ == "__main__":
    log_file_path = setup_logging()
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        # Restore stdout and stderr
        if isinstance(sys.stdout, open):
            sys.stdout.close()
            sys.stdout = sys.__stdout__
        if isinstance(sys.stderr, open):
            sys.stderr.close()
            sys.stderr = sys.__stderr__
        print(f"Script finished. Log saved to {log_file_path}")
