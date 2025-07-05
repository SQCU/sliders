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

def log_vram_usage(step_name):
    if torch.cuda.is_available():
        print(f"VRAM usage after {step_name}: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
        print(f"Max VRAM usage after {step_name}: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
    else:
        print(f"VRAM usage after {step_name}: CUDA not available.")

#a ton of this is canonical and absolutely must be written exactly this way
#e.g. timestep stuff
def superfunctional_train_step(
    unet: torch.nn.Module,
    vae: torch.nn.Module,
    noise_scheduler,
    img_batches: tuple[torch.Tensor, torch.Tensor],
    scales: tuple[float, float],
    text_embeddings: torch.Tensor,
    pooled_embeds: torch.Tensor,
    config: config_util.RootConfig,
    network: lora.BatchedLoRANetwork,
    criteria: torch.nn.Module,
    device: torch.device,
    weight_dtype: torch.dtype,
    add_time_ids: torch.Tensor,
    seed: int,
):
    """
    Performs a single training step for a batch of images with multiple network scales.

    This function encapsulates the core logic for:
    1. Generating noisy latents for both high and low images.
    2. Preparing and concatenating prompt embeddings for classifier-free guidance.
    3. Applying LoRA scales using the BatchedLoRANetwork.
    4. Performing a batched UNet forward pass to predict noise.
    5. Calculating the loss for both high and low predictions.

        returns torch.Tensor
            A tensor containing the loss at each batch element. 
    """
    #funky undocumented stuff:
    #counterintuitively, set_timesteps does distillation training,
    # the argument is choosing how many sampling steps are used to generate a 'full image'.
    # lower value = fewer sampling steps = bigger prediction distance in the signal-to-noise-ratio domain
    with torch.no_grad():
        noise_scheduler.set_timesteps(
            config.train.max_denoising_steps, device=device
        )

    # 1 ~ 49 からランダム
    #fix this for batching so that each high/low tuple,
    # stemming from a matching image,
    # shares a timestep sampled from this distribution
    # stub code picks an item() from this instead of sampling from it.
        timesteps_to = torch.randint(
            1, config.train.max_denoising_steps, (1,)
        ).item()

        seed = random.randint(0,2*15)
        generator = torch.manual_seed(seed)

    # this is misnamed 'denoised_latents' in upstream. 
    # it retrieves noisy latents in the upstream's functions so we use a less deranged name here.
        start_timesteps = 0
        img_batch = torch.cat(img_batches, dim=0)
        combined_noisy_latents, noise = batch_train_util.get_batched_noisy_images(
            img_batch, # This is a tuple of (high_batch, low_batch)
            vae,
            generator,
            noise_scheduler,
            config,
            start_timesteps,
            timesteps_to,
            device,
            weight_dtype,
        )
    
    # Concatenate text_embeds for high and low cases
    # Create the text_embeddings batch: [positive_uncond, positive_cond, neutral_uncond, neutral_cond]

    # Set LoRA slider for the combined batch.
    # The unsqueeze operations are to make the tensor broadcastable to the expected shape
        batched_scales = torch.tensor(scales, device=device, dtype=weight_dtype).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # Shape (2, 1, 1, 1) for broadcasting

    # Concatenate the latents for classifier-free guidance
    latent_model_input = torch.cat([combined_noisy_latents] * 2)

    network.set_lora_scales(batched_scales)

    with network: # This context manager ensures LoRA weights are applied during the UNet forward pass

        # .timesteps is necessary to predict right noise level for each potentially different timestep in a batch.
        #should form a scalar integer tensor of batch dim size.
        #don't try to understand the calculation here, but it's mandatory.
        batch_timesteps = noise_scheduler.timesteps[
            int(timesteps_to * 1000 / config.train.max_denoising_steps)
        ]

        predict_latents = batch_train_util.batched_predict_noise_xl_modular(
            unet,
            noise_scheduler,
            batch_timesteps,    # tensor of timesteps used in training batch (indexes of noise levels)
            latent_model_input.to(dtype=weight_dtype), # Combined high and low latents
            text_embeddings.to(dtype=weight_dtype),
            pooled_embeds.to(dtype=weight_dtype),
            add_time_ids.to(dtype=weight_dtype),
            guidance_scale=1, # Classifier-free guidance is handled internally by predict_noise_xl_modular
        )

    # OBSOLETE USE CRITERION DEFINED ELSEWHERE IN TRAINING LOOP
    loss_per_element = (predict_latents - noise).pow(2).to(torch.float32)

    return loss_per_element


#HANDWRITTEN AT USER'S EXTREME DISPLEASURE:
def config_io():
    #if you think that this file is getting a bit too big, 
    # think about how you could factor out a stateless config input handler
    # which takes a config argument as input, and returns a training config as output.
    import argparse
    #if batch_config_util doesn't exist yet, think about making it ;)
    from . import batch_config_util
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchtrainconfig", "--bconfig", "-c", 
    type=str, help="path to the batchtrainconfig, which is a config for our revised trainer",
    default=None)
    args = parser.parse_args()
    if args.batchtrainconfig ==  None:
        print("failing over to default config")
        #if this file isn't real and throws an error, create that file!
        bcfgdefault = "trainscripts/imagesliders/data/batch_config.yaml"
        args.batchtrainconfig = bcfgdefault
    #our upstreams used multiple different heterogenous config files scattered throughout project
    #we will fix this patiently and exactingly, by first wrapping their configs
    #we can later refactor if we need to. but we likely won't.
    args.outerconfig = batch_config_util.load_config_from_yaml(args.batchtrainconfig)

    #our default batch_config.yaml contains a reference to "trainscripts/imagesliders/data/config-xl-dilora.yaml". we load that.

    args.obsolete_inner_config = config_util.load_config_from_yaml(args.outerconfig['obsolete_config']['refpath'])
    args.dset_config = batch_config_util.load_config_from_yaml(args.outerconfig['dset_config']['refpath'])
    #if you're confused about the contents of these values...
    #you can add a print(outerconfig.dset_config.refpath), then look at what's inside!
    return args

class BracketDataset(Dataset):
    def __init__(self, config, prompts, transform):
        self.transform = transform
        self.prompts = prompts
        self.image_paths = []
        self.scales = []

        root_folder = Path(config['dataset']['folder_main'])
        subfolder_names = [f.strip() for f in config['dataset']['folders'].split(',')]
        scale_values = [float(s.strip()) for s in config['dataset']['scales'].split(',')]
        
        for i, folder_name in enumerate(subfolder_names):
            subfolder_path = root_folder / folder_name
            scale = scale_values[i]
            for image_path in subfolder_path.glob("*"):
                self.image_paths.append(image_path)
                self.scales.append(scale)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        scale = self.scales[idx]
        return image_tensor, scale, self.prompts

#HANDWRITTEN AT USER'S EXTREME DISPLEASURE:
def dataset_constructor(config):
    with open(config['prompts_file'], 'r') as f:
        prompts = yaml.safe_load(f)

    # Image transformations
    transform = T.Compose([
        T.Resize(1024, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(1024),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ])

    dataset = BracketDataset(config, prompts, transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    return dataloader


#HANDWRITTEN AT USER'S EXTREME DISPLEASURE:
def envsetup(config):
    from trainscripts.imagesliders import batch_lora as lora
    from trainscripts.imagesliders import config_util
    from trainscripts.imagesliders import batch_model_util # New import
    import torch.optim as optim

    # Determine device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = config_util.parse_precision(config.train.precision)

    # Load models using the new utility function
    vae, unet, tokenizers, text_encoders, noise_scheduler =         batch_model_util.load_models(config, device, weight_dtype)
    
    # Initialize LoRA Network
    network = lora.BatchedLoRANetwork(
        unet=unet,
        rank=config.network.rank,
        
        alpha=config.network.alpha,
        train_method=config.network.training_method,
    ).to(device, dtype=weight_dtype)
    network.prepare_optimizer_params() # Prepare parameters for optimizer

    # Optimizer
    optimizer_name = config.train.optimizer.lower()
    if optimizer_name == "adamw":
        optimizer = optim.AdamW(network.parameters(), lr=config.train.lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Learning Rate Scheduler
    lr_scheduler = get_scheduler(
        name=config.train.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=config.train.iterations,
    )

    # Loss function
    criteria = torch.nn.MSELoss(reduction="none") # Use reduction="none" for element-wise loss

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
        "config": config, # Pass the config to the environment
    }
    return environment



#HANDWRITTEN AT USER'S EXTREME DISPLEASURE:
def training_step(
    environment,
    batch,
):
    unet = environment["unet"]
    vae = environment["vae"]
    noise_scheduler = environment["noise_scheduler"]
    tokenizers = environment["tokenizers"]
    text_encoders = environment["text_encoders"]
    network = environment["network"]
    optimizer = environment["optimizer"]
    lr_scheduler = environment["lr_scheduler"]
    criteria = environment["criteria"]
    device = environment["device"]
    weight_dtype = environment["weight_dtype"]
    config = environment["config"]

    img_tensor, scale, prompts = batch
    img_tensor = img_tensor.to(device, dtype=weight_dtype)
    
    # Create prompt embeddings using the new utility function
    text_embeddings, pooled_embeds = batch_train_util.create_batched_prompt_embeddings(
        tokenizers,
        text_encoders,
        prompts[0],
        num_images_per_prompt=1,
    )

    add_time_ids = batch_train_util.get_add_time_ids(
        1024, 1024, False, dtype=weight_dtype
    ).to(device)
    
    # Call superfunctional_train_step
    loss_per_batch_element = superfunctional_train_step(
        unet=unet,
        vae=vae,
        noise_scheduler=noise_scheduler,
        img_batches=(img_tensor, img_tensor), # Using same image for high and low for now
        scales=(scale.item(), scale.item()),
        text_embeddings=text_embeddings,
        pooled_embeds=pooled_embeds,
        config=config,
        network=network,
        criteria=criteria,
        device=device,
        weight_dtype=weight_dtype,
        add_time_ids=add_time_ids,
        seed=random.randint(0, 2**32 - 1),
    )

    # Calculate total loss (e.g., sum or mean of high and low losses)
    total_loss = loss_per_batch_element.sum()

    return total_loss

#HANDWRITTEN AT USER'S EXTREME DISPLEASURE:
def training_loop(
    training_step,
    environment,
    dataset
):  
    def loss_preconditioning(environment, loss_tensor):
        #want to do something like a fancy loss weighting as suggested by kingma and karras et al?
        #you need to implement that *here*.
        
        #stub pass
        pass
    def gradient_cleanup(environment):
        #read configuration from environment on extra stuff to do while training.
        #e.g. constraining maximum gradient norm
        #e.g. some kind of funky jacobian
        #e.g. mutate an auxiliary loss's gradient by sign agreement with primary loss

        #stub pass
        pass
    def intra_loop_logging(environment):
        #printing... 
        # writing to logfiles...
        # progress bars and stuff...
        # all of that is handled here...

        #stub pass
        pass
    def stopping_condition(environment):
        #check for early stopping constraints set in environment
        #if one is triggered you might `break`` or something
        #e.g. there were two all NaN predictions logged in the last 100 steps, time to cut off training.

        #stub pass
        pass

    #non-stub real functions you better be using for real.
    optimizer = environment["optimizer"]
    lr_scheduler = environment["lr_scheduler"]

    for batch in dataset:
        optimizer.zero_grad()
        loss_tensor = training_step(environment,batch)
        loss_tensor.backward()
        gradient_cleanup(environment)
        optimizer.step()
        lr_scheduler.step()
        intra_loop_logging(environment)
        stopping_condition(environment)

    return environment

#HANDWRITTEN AT USER'S EXTREME DISPLEASURE:
def graceful_shutdown(environment):
    def traindone_logging(environment):
        #stub function
        pass
    def model_eval(environment):
        #stub function
        pass
    def save_function(environment):
        #stub function, read appropriate metadata to save trained weights and so on.
        #by default save trained model even without a full config, loudly complaining with prints
        pass
    #this calls at the conclusion of a normal training run where everything is going okay
    traindone_logging(environment)
    save_function(environment)
    model_eval(environment)


#HANDWRITTEN AT USER'S EXTREME DISPLEASURE:
def main():
    #main execution point for all of our major functions.
    #don't write anything here that could be an independent function instead.
    args = config_io()
    environment = envsetup(args.obsolete_inner_config)
    dataset = dataset_constructor(args.dset_config)
    environment = training_loop(training_step, environment, dataset)
    graceful_shutdown(environment)
    #stub pass
    pass

import datetime
import sys

def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(log_dir, f"batched_training_loop_{timestamp}.log")
    sys.stdout = open(log_filename, "w")
    sys.stderr = sys.stdout # Redirect stderr to the same log file
    print(f"Logging output to {log_filename}")
    return log_filename

if __name__ == "__main__":
    # Standard behavior: All script outputs are piped to a timestamped log file.
    # Use the `tail_log.py` helper script to view the end of the log file.
    log_file_path = setup_logging()
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        sys.stdout.close()
        sys.stdout = sys.__stdout__ # Restore original stdout
        sys.stderr = sys.__stderr__ # Restore original stderr
        print(f"Script finished. Log saved to {log_file_path}")