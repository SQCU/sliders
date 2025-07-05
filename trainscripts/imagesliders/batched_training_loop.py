import torch
import numpy as np
from PIL import Image
import os
import random
import gc
import torch.cuda
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler

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


#this function is suspicious, it's a reference case but you should make it better, simpler.
#prefer structs or kv dicts for non-ml data and metadata
#use tensors for absolutely everything interacting with an accelerator.
#even if older code passes dumb tuples around for no reason.
def superfunctional_train_step(
    unet: torch.nn.Module,
    vae: torch.nn.Module,
    noise_scheduler,
    img_batches: tuple[torch.Tensor, torch.Tensor],
    scales: tuple[float, float],
    prompt_embeds: tuple[prompt_util.PromptEmbedsXL, prompt_util.PromptEmbedsXL],
    prompt_pair: prompt_util.PromptEmbedsPair,
    config: config_util.RootConfig,
    network: lora.BatchedLoRANetwork,
    criteria: torch.nn.Module,
    device: torch.device,
    weight_dtype: torch.dtype,
    add_time_ids: torch.Tensor,
    seed: int,
):
    """
    Performs a single training step for a high/low image pair, leveraging batched operations
    for internal efficiency.

    This function encapsulates the core logic for:
    1. Generating noisy latents for both high and low images.
    2. Preparing and concatenating prompt embeddings for classifier-free guidance.
    3. Applying LoRA scales using the BatchedLoRANetwork.
    4. Performing a batched UNet forward pass to predict noise.
    5. Calculating the loss for both high and low predictions.

    Args:
        unet (torch.nn.Module): The UNet model for noise prediction.
        vae (torch.nn.Module): The VAE model for image encoding/decoding.
        noise_scheduler: The noise scheduler (e.g., DDPMScheduler).
        img_batches (tuple[torch.Tensor, torch.Tensor]): A tuple containing two tensors,
            where the first tensor is the batch of 'high' images and the second is
            the batch of 'low' images. Each tensor is expected to have shape (batch_size, C, H, W).
            For this function, batch_size is typically 1, representing a single high/low pair.
        scales (tuple[float, float]): A tuple containing the LoRA scale for the 'high' image
            and the 'low' image, respectively.
        prompt_embeds (tuple[prompt_util.PromptEmbedsXL, prompt_util.PromptEmbedsXL]):
            A tuple where the first element contains prompt embeddings for the 'high' image
            (positive prompt) and the second for the 'low' image (neutral prompt).
        prompt_pair (prompt_util.PromptEmbedsPair): A container object holding the
            positive and neutral prompt embeddings. Used for clarity in some parts.
        config (config_util.RootConfig): The overall training configuration.
        network (lora.BatchedLoRANetwork): The batched LoRA network responsible for
            applying LoRA weights based on the provided scales.
        criteria (torch.nn.Module): The loss function (e.g., MSELoss).
        device (torch.device): The device (e.g., 'cuda:0' or 'cpu') to perform computations on.
        weight_dtype (torch.dtype): The data type for model weights (e.g., torch.float16, torch.bfloat16).
        add_time_ids (torch.Tensor): Additional time IDs for SDXL models, typically
            representing original image dimensions.
        seed (int): The random seed for noise generation.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            A tuple containing:
            - loss_high_per_element (torch.Tensor): Element-wise loss for the 'high' image.
            - loss_low_per_element (torch.Tensor): Element-wise loss for the 'low' image.
            - denoised_latents_high (torch.Tensor): Denoised latents for the 'high' image.
            - high_noise (torch.Tensor): The noise added to the 'high' image latents.
            - target_latents_high (torch.Tensor): Predicted noise for the 'high' image.
            - denoised_latents_low (torch.Tensor): Denoised latents for the 'low' image.
            - low_noise (torch.Tensor): The noise added to the 'low' image latents.
            - target_latents_low (torch.Tensor): Predicted noise for the 'low' image.
    """
    noise_scheduler.set_timesteps(1000)
    current_timestep = noise_scheduler.timesteps[
        int((config.train.max_denoising_steps - 1) * 1000 / config.train.max_denoising_steps)
    ]

    generator = torch.manual_seed(seed)
    
    # Call graph hop 1: get_batched_noisy_images from batch_train_util
    denoised_latents_high, high_noise, denoised_latents_low, low_noise = batch_train_util.get_batched_noisy_images(
        img_batches, # This is a tuple of (high_batch, low_batch)
        vae,
        generator,
        unet,
        noise_scheduler,
        config,
        device,
        weight_dtype,
    )
    denoised_latents_high = denoised_latents_high.to(device, dtype=weight_dtype)
    high_noise = high_noise.to(device, dtype=weight_dtype)
    denoised_latents_low = denoised_latents_low.to(device, dtype=weight_dtype)
    low_noise = low_noise.to(device, dtype=weight_dtype)

    # Prepare for batched predict_noise calls for SD1.5
    # Concatenate text_embeds for high and low cases
    # Each needs to be duplicated for classifier-free guidance within predict_noise
    # So, for a batch of 2 (high and low), we need 4 entries (high_uncond, high_cond, low_uncond, low_cond)

    # For high scale: prompt_pair.positive
    # For low scale: prompt_pair.neutral

    # Create the text_embeddings batch: [positive_uncond, positive_cond, neutral_uncond, neutral_cond]
    text_embeddings_for_noise_pred = torch.cat([
        prompt_pair.positive.text_embeds, # positive uncond
        prompt_pair.positive.text_embeds, # positive cond
        prompt_pair.neutral.text_embeds,  # neutral uncond
        prompt_pair.neutral.text_embeds   # neutral cond
    ], dim=0)

    # For SD1.5, pooled_embeds and add_time_ids are not used in the same way as SDXL.
    # We will pass None for these or adjust the predict_noise function call.
    # For now, let's ensure they are not used in the concatenation.
    # The `batched_predict_noise_xl_modular` will need to be adjusted or replaced with an SD1.5 equivalent.

    # Set LoRA slider for the combined batch.
    # Convert scales to a tensor and set for the batched LoRA network
    # The unsqueeze operations are to make the tensor broadcastable to the expected shape
    # by BatchedLoRANetwork, which expects a (batch_size, 1, 1, 1) tensor.
    # Here, batch_size is 2 (for high and low).
    batched_scales = torch.tensor(scales, device=device, dtype=weight_dtype).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # Shape (2, 1, 1, 1) for broadcasting
    
    # Call graph hop 2: set_lora_scales from BatchedLoRANetwork
    network.set_lora_scales(batched_scales)

    with network: # This context manager ensures LoRA weights are applied during the UNet forward pass
        with torch.no_grad(): # No gradient calculation needed for noise prediction
            # Perform batched predict_noise_xl call for both high and low scales
            # The BatchedLoRANetwork will now apply the correct scale to each item in the batch
            
            # Concatenate denoised latents for high and low cases for the UNet input
            combined_denoised_latents_for_noise_pred = torch.cat([denoised_latents_high, denoised_latents_low], dim=0)

            # Call graph hop 3: batched_predict_noise_xl_modular from batch_train_util
            target_latents_high, target_latents_low = batch_train_util.batched_predict_noise_xl_modular(
                unet,
                noise_scheduler,
                current_timestep,
                combined_denoised_latents_for_noise_pred, # Combined high and low latents
                text_embeddings_for_noise_pred,
                pooled_embeds_for_noise_pred,
                add_time_ids_for_noise_pred,
                guidance_scale=1, # Classifier-free guidance is handled internally by predict_noise_xl_modular
            )

    # Call graph hop 4: Loss calculation (criteria is typically MSELoss)
    loss_high_per_element = (target_latents_high - high_noise).pow(2).to(torch.float32)
    loss_low_per_element = (target_latents_low - low_noise).pow(2).to(torch.float32)

    return loss_high_per_element, loss_low_per_element, denoised_latents_high, high_noise, target_latents_high, denoised_latents_low, low_noise, target_latents_low


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

#HANDWRITTEN AT USER'S EXTREME DISPLEASURE:
def dataset_constructor(config):
    #stubbed out:
    #load a normal debug dataset through normal mechanisms.
    #construct visual 'prompt pairs' 
    #(make sure that each dataset sampled from has at least two different magnitudes in batchsize * gradient accumulation)
    #cast batches to an iterator (hint: a huggingface dataset) for sampling during training 
    pass

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
    # Extract components from environment
    unet = environment["unet"]
    vae = environment["vae"]
    noise_scheduler = environment["noise_scheduler"]
    network = environment["network"]
    criteria = environment["criteria"]
    device = environment["device"]
    weight_dtype = environment["weight_dtype"]
    text_encoder = environment["text_encoder"]
    text_encoder_2 = environment["text_encoder_2"]
    tokenizer = environment["tokenizer"]
    tokenizer_2 = environment["tokenizer_2"]

    # Extract data from batch
    img_batches = batch["img_batches"]
    scales = batch["scales"]
    prompt_pair = batch["prompt_pair"]
    add_time_ids = batch["add_time_ids"]
    seed = batch["seed"]

    # Move data to device and set dtype
    img_batches_high = img_batches[0].to(device, dtype=weight_dtype)
    img_batches_low = img_batches[1].to(device, dtype=weight_dtype)
    img_batches = (img_batches_high, img_batches_low)

    # Ensure prompt embeddings are on the correct device and dtype
    prompt_pair.positive.text_embeds = prompt_pair.positive.text_embeds.to(device, dtype=weight_dtype)
    prompt_pair.positive.pooled_embeds = prompt_pair.positive.pooled_embeds.to(device, dtype=weight_dtype)
    prompt_pair.neutral.text_embeds = prompt_pair.neutral.text_embeds.to(device, dtype=weight_dtype)
    prompt_pair.neutral.pooled_embeds = prompt_pair.neutral.pooled_embeds.to(device, dtype=weight_dtype)
    add_time_ids = add_time_ids.to(device, dtype=weight_dtype)

    # Call superfunctional_train_step
    loss_high_per_element, loss_low_per_element, _, _, _, _, _, _ = superfunctional_train_step(
        unet=unet,
        vae=vae,
        noise_scheduler=noise_scheduler,
        img_batches=img_batches,
        scales=scales,
        prompt_embeds=(prompt_pair.positive, prompt_pair.neutral), # This argument is redundant given prompt_pair
        prompt_pair=prompt_pair,
        config=environment["config"], # Need to pass config from environment
        network=network,
        criteria=criteria,
        device=device,
        weight_dtype=weight_dtype,
        add_time_ids=add_time_ids,
        seed=seed,
    )

    # Calculate total loss (e.g., sum or mean of high and low losses)
    loss_high = loss_high_per_element.mean()
    loss_low = loss_low_per_element.mean()
    total_loss = loss_high + loss_low # Simple sum for now

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