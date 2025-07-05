import torch
import numpy as np
from PIL import Image
import os
import random
import gc
import torch.cuda
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
import yaml
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

# Assuming the following files are in the same directory
from trainscripts.imagesliders import train_util, model_util, config_util, prompt_util
from trainscripts.imagesliders import batch_lora as lora # Use the new batched lora
from trainscripts.imagesliders import train_util, batch_train_util # Keep train_util for other functions not yet moved
from trainscripts.imagesliders.prompt_util import PromptEmbedsXL, PromptEmbedsPair
#our new functions live here instead of being rewritten inside of train_util, etc.
#add new batch_... imports as we need to deviate from imagesliders imports.
from trainscripts.imagesliders import batch_train_util
from trainscripts.imagesliders import map_data_to_latents
from trainscripts.imagesliders import batch_config_util

def log_vram_usage(step_name):
    if torch.cuda.is_available():
        print(f"VRAM usage after {step_name}: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
        print(f"Max VRAM usage after {step_name}: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
    else:
        print(f"VRAM usage after {step_name}: CUDA not available.")

def superfunctional_train_step(
    environment: dict,
    batch: dict,
):
    """
    Performs a single training step for a batch of images with multiple network scales.
    """
    unet = environment['unet']
    vae = environment['vae']
    noise_scheduler = environment['noise_scheduler']
    network = environment['network']
    criteria = environment['criteria']
    device = environment['device']
    weight_dtype = environment['weight_dtype']
    config = environment['config']

    latents = batch['latents']
    scales = batch['scales']
    text_embeddings = batch['text_embeddings']
    pooled_embeds = batch['pooled_embeds']
    add_time_ids = batch['add_time_ids']

    with torch.no_grad():
        noise_scheduler.set_timesteps(
            config.train.max_denoising_steps, device=device
        )
        timesteps_to = torch.randint(
            1, config.train.max_denoising_steps, (1,)
        ).item()
        seed = random.randint(0, 2**15)
        generator = torch.manual_seed(seed)

    start_timesteps = 0
    noisy_latents, noise = batch_train_util.get_batched_noisy_images(
        latents,
        vae,
        generator,
        noise_scheduler,
        config,
        start_timesteps,
        timesteps_to,
        device,
        weight_dtype,
    )

    latent_model_input = torch.cat([noisy_latents] * 2)
    batched_scales = scales.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    network.set_lora_scales(batched_scales)

    with network:
        batch_timesteps = noise_scheduler.timesteps[
            int(timesteps_to * 1000 / config.train.max_denoising_steps)
        ]
        predict_latents = batch_train_util.batched_predict_noise_xl_modular(
            unet,
            noise_scheduler,
            batch_timesteps,
            latent_model_input.to(dtype=weight_dtype),
            text_embeddings.to(dtype=weight_dtype),
            pooled_embeds.to(dtype=weight_dtype),
            add_time_ids.to(dtype=weight_dtype),
            guidance_scale=1,
        )

    predict_latents = predict_latents * batched_scales
    loss_per_element = (predict_latents - noise).pow(2).to(torch.float32)
    return loss_per_element.mean()


def config_io():
    import argparse
    from trainscripts.imagesliders import config_util

    parser = argparse.ArgumentParser()
    parser.add_argument("--batchtrainconfig", "--bconfig", "-c", 
                        type=str, 
                        help="Path to the batch training config file.",
                        default="trainscripts/imagesliders/data/batch_config.yaml")
    args = parser.parse_args()

    print(f"Loading batch config from: {args.batchtrainconfig}")
    config = batch_config_util.load_config_from_yaml(args.batchtrainconfig)
    
    inner_config_path = config['obsolete_config']['refpath']
    print(f"Loading and merging inner config from: {inner_config_path}")
    inner_config = config_util.load_config_from_yaml(inner_config_path)
    config.update(inner_config)

    dset_config_path = config['dset_config']['refpath']
    print(f"Loading dataset config from: {dset_config_path}")
    config['dataset_config'] = batch_config_util.load_config_from_yaml(dset_config_path)

    return config

def envsetup(config):
    from trainscripts.imagesliders import batch_lora as lora
    from trainscripts.imagesliders import config_util
    from trainscripts.imagesliders import batch_model_util
    import torch.optim as optim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = config_util.parse_precision(config.train.precision)

    vae, unet, tokenizers, text_encoders, noise_scheduler = batch_model_util.load_models(config, device, weight_dtype)
    
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

    criteria = torch.nn.MSELoss(reduction="none")

    environment = {
        "unet": unet,
        "vae": vae,
        "noise_scheduler": noise_scheduler,
        "tokenizers": tokenizers,
        "text_encoders": text_encoders,
        "network": network,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "criteria": criteria,
        "device": device,
        "weight_dtype": weight_dtype,
        "config": config,
    }
    return environment

def training_loop(
    environment,
    dataset
):  
    optimizer = environment["optimizer"]
    lr_scheduler = environment["lr_scheduler"]

    for i, batch in enumerate(dataset):
        optimizer.zero_grad()
        loss = superfunctional_train_step(environment, batch)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        if i % 10 == 0:
            print(f"Step {i}, Loss: {loss.item()}")

    return environment

def graceful_shutdown(environment):
    print("Training finished. Saving model...")
    network = environment['network']
    network.save_weights(os.path.join(environment['config'].save.path, f"{environment['config'].save.name}.safetensors"))
    print("Model saved.")

def main():
    args = config_io()
    # This is a hack to get the obsolete inner config to work with the new structure
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self
    
    config_dict = {}
    for key, value in args.items():
        if isinstance(value, dict):
            config_dict[key] = AttrDict(value)
        else:
            config_dict[key] = value
    
    config = AttrDict(config_dict)
    
    environment = envsetup(config)
    dataloader = batch_config_util.dataset_constructor(config, environment)
    environment = training_loop(environment, dataloader)
    graceful_shutdown(environment)

import datetime
import sys

def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(log_dir, f"batched_training_loop_{timestamp}.log")
    # sys.stdout = open(log_filename, "w")
    # sys.stderr = sys.stdout
    print(f"Logging output to {log_filename}")
    return log_filename

if __name__ == "__main__":
    log_file_path = setup_logging()
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        # sys.stdout.close()
        # sys.stdout = sys.__stdout__
        # sys.stderr = sys.__stderr__
        print(f"Script finished. Log saved to {log_file_path}")