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
    load_config_from_yaml_and_merge,
    parse_precision,
)
from . import batch_lora as lora
from . import batch_train_util
from . import batch_model_util


def config_io():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchtrainconfig", "--bconfig", "-c", 
                        type=str, 
                        help="Path to the batch training config file.",
                        default="trainscripts/imagesliders/data/batch_config.yaml")
    args = parser.parse_args()

    print(f"Loading batch config from: {args.batchtrainconfig}")
    config = load_config_from_yaml(args.batchtrainconfig)
    
    inner_config_path = config['obsolete_config']['refpath']
    #print(f"Loading and merging inner config from: {inner_config_path}")
    # This function is not ideal, but it's what the original code used.
    # It returns a Pydantic model, which we convert to a dict.
    # GET RID OF IT GET RID OF IT GET RID OF IT NO MORE PYDANTIC ERROR FUCK PYDANTIC
    # NONE OF MY HOMIES USE PYDANTIC
    config['inner_config']= load_config_from_yaml(inner_config_path)

    dset_config_path = config['dset_config']['refpath']
    print(f"Loading dataset config from: {dset_config_path}")
    config['dataset_config'] = load_config_from_yaml(dset_config_path)

    return config

def envsetup(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = parse_precision(config.inner_config.train.precision)
    scopedconfig = config.inner_config
    vae, unet, tokenizers, text_encoders, noise_scheduler = batch_model_util.load_models(scopedconfig, device, weight_dtype)
    
    # Cast the entire unet to the correct dtype
    unet.to(device, dtype=weight_dtype)
    
    network = lora.BatchedLoRANetwork(
        unet=unet,
        rank=scopedconfig.network.rank,
        alpha=scopedconfig.network.alpha,
        train_method=scopedconfig.network.training_method,
    ).to(device, dtype=weight_dtype)
    network.prepare_optimizer_params()

    optimizer_name = scopedconfig.train.optimizer.lower()
    if optimizer_name == "adamw":
        optimizer = optim.AdamW(network.parameters(), lr=scopedconfig.train.lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    lr_scheduler = get_scheduler(
        name=scopedconfig.train.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=scopedconfig.train.iterations,
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
    print(f"unpacked pooled embeds of shape: {pooled_embeds.shape}")
    add_time_ids = batch["add_time_ids"].to(device, dtype=weight_dtype)

    #dataset constructor text embeddings and pooled embeds are formed from the concatenation of these 3 subtensors:
    #[positive_*_embeds, unconditional_*_embeds, neutral_*_embeds]
    #since they came in a batch this means they're pre-swizzled (3 per datum) and probably also don't match tensor shape?
    #for each of our 'pairings' of high/low feature scale, we form a cfg batch like this:
    #highfeature: cat([unconditional, positive_conditional],dim0)
    #lowfeature: cat([unconditional, neutral_conditional],dim0) #(strange naming, right?)
    #this means two (three?) things:
    # 1: we should form the 'pairing' map for our training batch before loss calculation
    # 2: we actually choose what prompts annotate a training image based on the feature manifold in that batch???
    # (? 3: we should probably add some trainable soft prompt parameters to each of our 3 categorical 'prompts'?)

    print(f"Shape of initial latents (batch['latents']): {latents.shape}")

    # Prepare for training step
    optimizer.zero_grad()

    # --- TIMESTEP LOGIC ---
    with torch.no_grad():
        noise_scheduler.set_timesteps(config.inner_config.train.max_denoising_steps, device=device)
        generator = torch.Generator(device=device).manual_seed(seed)
        
        # Generate a random timestep for each item in the batch
        timesteps_to = torch.randint(
            1, config.train.inner_config.max_denoising_steps, (latents.shape[0],), device=device, generator=generator
        ).long()

        # Add noise to latents using the generated timesteps
        noise = torch.randn(latents.shape, device=latents.device, generator=generator)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps_to)
    
    # --- END TIMESTEP LOGIC ---


    # Predict noise
    with network:
        # The timesteps for the UNet need to be the actual timesteps from the scheduler
        # corresponding to the `timesteps_to` indices.
        unet_timesteps = noise_scheduler.timesteps[timesteps_to].to(weight_dtype)
        
        # Duplicate inputs for classifier-free guidance
        # The batch_train_util.batched_predict_noise_xl_modular expects inputs to be duplicated for CFG
        unet_timesteps_cfg = torch.cat([unet_timesteps, unet_timesteps], dim=0)
        noisy_latents_cfg = torch.cat([noisy_latents, noisy_latents], dim=0)
        text_embeddings_cfg = torch.cat([text_embeddings, text_embeddings], dim=0)
        pooled_embeds_cfg = torch.cat([pooled_embeds, pooled_embeds], dim=0)
        add_time_ids_cfg = torch.cat([add_time_ids, add_time_ids], dim=0)
        # Set LoRA scales, which must match the doubled cfg axis.
        batched_scales = scales.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        batched_scales_cfg = torch.cat([batched_scales,batched_scales], dim=0)
        network.set_lora_scales(batched_scales)

        print(f"Shape of unet_timesteps_cfg: {unet_timesteps_cfg.shape}")
        print(f"Shape of noisy_latents_cfg: {noisy_latents_cfg.shape}")
        print(f"Shape of text_embeddings_cfg: {text_embeddings_cfg.shape}")
        print(f"Shape of pooled_embeds_cfg: {pooled_embeds_cfg.shape}")
        print(f"Shape of add_time_ids_cfg: {add_time_ids_cfg.shape}")
        
        predicted_noise = batch_train_util.batched_predict_noise_xl_modular(
            unet,
            noise_scheduler,
            unet_timesteps_cfg, # This should now have the correct batch size (2N)
            noisy_latents_cfg,
            text_embeddings_cfg,
            pooled_embeds_cfg,
            add_time_ids_cfg,
            guidance_scale=1,
        )

    # Calculate loss per element
    loss_per_element = (predicted_noise - noise).pow(2).to(torch.float32)
    
    # Reduce the loss to a per-item scalar value by taking the mean over non-batch dimensions
    loss_per_item = loss_per_element.mean(dim=list(range(1, loss_per_element.ndim)))

    # Dynamically pair up high and low scale losses
    # ATTNTION! REWRITE PAIR SELECTION TO START OF BATCH STEP! ATTENTION! 
    unique_scales = torch.unique(scales)
    if len(unique_scales) < 2:
        # Cannot form pairs, fall back to mean loss for the entire batch
        loss = loss_per_item.mean()
    else:
        # For simplicity, assume the lowest scale is "low" and highest is "high"
        low_scale = unique_scales.min()
        high_scale = unique_scales.max()

        low_indices = torch.where(scales == low_scale)[0]
        high_indices = torch.where(scales == high_scale)[0]

        num_pairs = min(len(low_indices), len(high_indices))

        if num_pairs > 0:
            # Identify indices for paired items
            low_paired_indices = low_indices[:num_pairs]
            high_paired_indices = high_indices[:num_pairs]

            # Calculate summed losses for pairs
            paired_losses = loss_per_item[low_paired_indices] + loss_per_item[high_paired_indices]

            # Identify indices of all items used in pairs
            used_indices = torch.cat([low_paired_indices, high_paired_indices])
            
            # Create a mask to find leftover items
            mask = torch.ones(len(loss_per_item), dtype=torch.bool, device=loss_per_item.device)
            mask[used_indices] = False
            
            # Get losses for leftover items
            leftover_losses = loss_per_item[mask]

            # Combine paired losses and leftover losses
            all_losses_to_average = torch.cat([paired_losses, leftover_losses])
            
            # Final loss is the mean of the combined list
            loss = all_losses_to_average.mean()
        else:
            # No pairs could be formed, so just average all item losses
            loss = loss_per_item.mean()

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
    config = AttrDict(config_io())

    # Set up the training environment using the local envsetup
    environment = envsetup(config)

    # Create the dataset and dataloader
    # We use use_latents=False to get paths, which are then loaded in the list comprehension
    #must pass inner config or something 
    dataloader = dataset_constructor(config, environment, use_latents=False)

    # Pre-generate all batches to create a static dataset for the training loop
    print("Pre-generating and caching all batches...")
    static_batches = [batch for batch in tqdm(dataloader, desc="Caching batches")]
    print(f"Cached {len(static_batches)} batches.")

    # Clean up VRAM before starting training
    # The VAE and text_encoders are no longer needed after caching the batches
    del environment["vae"]
    del environment["text_encoders"]
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
