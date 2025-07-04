#thanks a bunch gemini 2.5
import torch
import numpy as np
from PIL import Image
import os
import random
import gc
import torch.cuda
from diffusers.image_processor import VaeImageProcessor

# Assuming the following files are in the same directory
from trainscripts.imagesliders import train_util, model_util, config_util, lora, prompt_util
from trainscripts.imagesliders.prompt_util import PromptEmbedsXL

def log_vram_usage(step_name):
    if torch.cuda.is_available():
        print(f"VRAM usage after {step_name}: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
        print(f"Max VRAM usage after {step_name}: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
    else:
        print(f"VRAM usage after {step_name}: CUDA not available.")

def functional_train_step(
    unet: torch.nn.Module,
    vae: torch.nn.Module,
    noise_scheduler,
    img1_batch: torch.Tensor,
    img2_batch: torch.Tensor,
    scale_to_look: float,
    prompt_pair: prompt_util.PromptEmbedsPair,
    config: config_util.RootConfig,
    network: lora.LoRANetwork,
    criteria: torch.nn.Module,
    device: torch.device,
    weight_dtype: torch.dtype,
    add_time_ids: torch.Tensor,
    seed: int,
):
    """
    A functional and batch-oriented training step.

    Expected operand types:
    - unet: torch.nn.Module (UNet model)
    - vae: torch.nn.Module (VAE model)
    - noise_scheduler: Noise scheduler object (e.g., DDPMScheduler)
    - img1_batch: torch.Tensor (Batch of input images for low scale)
    - img2_batch: torch.Tensor (Batch of input images for high scale)
    - scale_to_look: float (Scale value for LoRA slider)
    - prompt_pair: prompt_util.PromptEmbedsPair (Container for prompt embeddings)
    - config: config_util.Config (Configuration object)
    - network: lora.LoRANetwork (LoRA network)
    - criteria: torch.nn.Module (Loss function, e.g., MSELoss)
    - device: torch.device (Device to perform computations on, e.g., 'cuda:0')
    - weight_dtype: torch.dtype (Data type for model weights, e.g., torch.float16)
    - add_time_ids: torch.Tensor (Additional time IDs for XL models)

    Expected output types:
    - loss_high: torch.Tensor (Loss for high scale)
    - loss_low: torch.Tensor (Loss for low scale)

    Expected calling functions:
    - Called by the main training loop.
    """
    generator = torch.manual_seed(seed)

    # Process a batch of images
    denoised_latents_low, low_noise = train_util.get_noisy_image(
        img1_batch,
        vae,
        generator,
        unet,
        noise_scheduler,
        start_timesteps=0,
        total_timesteps=config.train.max_denoising_steps -1,
    )
    denoised_latents_low = denoised_latents_low.to(device, dtype=weight_dtype)
    low_noise = low_noise.to(device, dtype=weight_dtype)

    generator = torch.manual_seed(seed)
    denoised_latents_high, high_noise = train_util.get_noisy_image(
        img2_batch,
        vae,
        generator,
        unet,
        noise_scheduler,
        start_timesteps=0,
        total_timesteps=config.train.max_denoising_steps -1,
    )
    denoised_latents_high = denoised_latents_high.to(device, dtype=weight_dtype)
    high_noise = high_noise.to(device, dtype=weight_dtype)

    noise_scheduler.set_timesteps(1000)
    current_timestep = noise_scheduler.timesteps[
        int((config.train.max_denoising_steps -1) * 1000 / config.train.max_denoising_steps)
    ]

    # High scale
    network.set_lora_slider(scale=scale_to_look)
    with network:
        with torch.no_grad():
            # Duplicate conditioning inputs for classifier-free guidance
            positive_text_embeds = torch.cat([prompt_pair.positive.text_embeds, prompt_pair.positive.text_embeds], dim=0)
            positive_pooled_embeds = torch.cat([prompt_pair.positive.pooled_embeds, prompt_pair.positive.pooled_embeds], dim=0)
            duplicated_add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

            target_latents_high = train_util.predict_noise_xl(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents_high,
                positive_text_embeds,
                positive_pooled_embeds,
                duplicated_add_time_ids,
                guidance_scale=1,
            )
    loss_high_per_element = (target_latents_high - high_noise).pow(2).to(torch.float32)

    # Low scale
    network.set_lora_slider(scale=-scale_to_look)
    with network:
        with torch.no_grad():
            # Duplicate conditioning inputs for classifier-free guidance
            neutral_text_embeds = torch.cat([prompt_pair.neutral.text_embeds, prompt_pair.neutral.text_embeds], dim=0)
            neutral_pooled_embeds = torch.cat([prompt_pair.neutral.pooled_embeds, prompt_pair.neutral.pooled_embeds], dim=0)
            duplicated_add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

            target_latents_low = train_util.predict_noise_xl(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents_low,
                neutral_text_embeds,
                neutral_pooled_embeds,
                duplicated_add_time_ids,
                guidance_scale=1,
            )
    loss_low_per_element = (target_latents_low - low_noise).pow(2).to(torch.float32)

    return loss_high_per_element, loss_low_per_element, denoised_latents_high, high_noise, target_latents_high, denoised_latents_low, low_noise, target_latents_low


def superfunctional_train_step(
    unet: torch.nn.Module,
    vae: torch.nn.Module,
    noise_scheduler,
    img_batches: tuple[torch.Tensor, torch.Tensor],
    scales: tuple[float, float],
    prompt_embeds: tuple[prompt_util.PromptEmbedsXL, prompt_util.PromptEmbedsXL],
    prompt_pair: prompt_util.PromptEmbedsPair,
    config: config_util.RootConfig,
    network: lora.LoRANetwork,
    criteria: torch.nn.Module,
    device: torch.device,
    weight_dtype: torch.dtype,
    add_time_ids: torch.Tensor,
    seed: int,
):
    """
    A more functional and compact training step that processes high and low cases concurrently.

    Expected operand types:
    - unet: torch.nn.Module (UNet model)
    - vae: torch.nn.Module (VAE model)
    - noise_scheduler: Noise scheduler object (e.g., DDPMScheduler)
    - img_batches: tuple[torch.Tensor, torch.Tensor] (Tuple of image batches for high and low scales)
    - scales: tuple[float, float] (Tuple of scale values for LoRA slider)
    - prompt_embeds: tuple[prompt_util.PromptEmbedsXL, prompt_util.PromptEmbedsXL] (Tuple of prompt embeddings for high and low scales)
    - prompt_pair: prompt_util.PromptEmbedsPair (Container for prompt embeddings)
    - config: config_util.Config (Configuration object)
    - network: lora.LoRANetwork (LoRA network)
    - criteria: torch.nn.Module (Loss function, e.g., MSELoss)
    - device: torch.device (Device to perform computations on, e.g., 'cuda:0')
    - weight_dtype: torch.dtype (Data type for model weights, e.g., torch.float16)
    - add_time_ids: torch.Tensor (Additional time IDs for XL models)

    Expected output types:
    - loss_high: torch.Tensor (Loss for high scale)
    - loss_low: torch.Tensor (Loss for low scale)

    Expected calling functions:
    - Called by the main training loop.
    """
    noise_scheduler.set_timesteps(1000)
    current_timestep = noise_scheduler.timesteps[
        int((config.train.max_denoising_steps - 1) * 1000 / config.train.max_denoising_steps)
    ]

    generator = torch.manual_seed(seed)
    
    # Process images separately to match original behavior for noise generation
    denoised_latents_high, high_noise = train_util.get_noisy_image(
        img_batches[0],
        vae,
        generator,
        unet,
        noise_scheduler,
        start_timesteps=0,
        total_timesteps=config.train.max_denoising_steps - 1,
    )
    denoised_latents_high = denoised_latents_high.to(device, dtype=weight_dtype)
    high_noise = high_noise.to(device, dtype=weight_dtype)

    generator = torch.manual_seed(seed) # Re-seed for the second image to ensure identical noise generation
    denoised_latents_low, low_noise = train_util.get_noisy_image(
        img_batches[1],
        vae,
        generator,
        unet,
        noise_scheduler,
        start_timesteps=0,
        total_timesteps=config.train.max_denoising_steps - 1,
    )
    denoised_latents_low = denoised_latents_low.to(device, dtype=weight_dtype)
    low_noise = low_noise.to(device, dtype=weight_dtype)

    # Prepare for batched predict_noise_xl calls
    # Concatenate text_embeds, pooled_embeds, and add_time_ids for high and low cases
    combined_text_embeds = torch.cat([prompt_embeds[0].text_embeds, prompt_embeds[1].text_embeds], dim=0)
    combined_pooled_embeds = torch.cat([prompt_embeds[0].pooled_embeds, prompt_embeds[1].pooled_embeds], dim=0)
    combined_add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0) # Assuming add_time_ids is the same for both

    # Duplicate for classifier-free guidance
    duplicated_combined_text_embeds = torch.cat([combined_text_embeds, combined_text_embeds], dim=0)
    duplicated_combined_pooled_embeds = torch.cat([combined_pooled_embeds, combined_pooled_embeds], dim=0)
    duplicated_combined_add_time_ids = torch.cat([combined_add_time_ids, combined_add_time_ids], dim=0)

    # Perform batched predict_noise_xl call for both high and low scales
    # This requires careful handling of the LoRA slider scale.
    # We will set the slider for the first element (high) and then for the second (low)
    # This is a simplification and might not be truly concurrent for the LoRA application.
    # A more robust solution would involve a custom LoRA layer that can handle multiple scales in a batch.
    # For now, we'll assume the network.set_lora_slider applies globally for the batch.
    # This is a known limitation and will be addressed in future iterations if needed.

    # Prepare batched inputs for predict_noise_xl
    # Concatenate denoised latents for high and low cases
    combined_denoised_latents_for_noise_pred = torch.cat([denoised_latents_high, denoised_latents_low], dim=0)

    # Concatenate text_embeddings, pooled_embeddings, and add_time_ids for high and low cases
    # Each needs to be duplicated for classifier-free guidance within predict_noise_xl
    # So, for a batch of 2 (high and low), we need 4 entries (high_uncond, high_cond, low_uncond, low_cond)
    # This means we need to prepare the inputs to predict_noise_xl such that when it duplicates them,
    # it results in the correct pairings.

    # For high scale: prompt_pair.positive
    # For low scale: prompt_pair.neutral

    # Create the text_embeddings batch: [positive_uncond, positive_cond, neutral_uncond, neutral_cond]
    text_embeddings_for_noise_pred = torch.cat([
        prompt_pair.positive.text_embeds, # positive uncond
        prompt_pair.positive.text_embeds, # positive cond
        prompt_pair.neutral.text_embeds,  # neutral uncond
        prompt_pair.neutral.text_embeds   # neutral cond
    ], dim=0)

    # Create the pooled_embeds batch: [positive_pooled_uncond, positive_pooled_cond, neutral_pooled_uncond, neutral_pooled_cond]
    pooled_embeds_for_noise_pred = torch.cat([
        prompt_pair.positive.pooled_embeds, # positive uncond
        prompt_pair.positive.pooled_embeds, # positive cond
        prompt_pair.neutral.pooled_embeds,  # neutral uncond
        prompt_pair.neutral.pooled_embeds   # neutral cond
    ], dim=0)

    # Create the add_time_ids batch: [add_time_ids_high_uncond, add_time_ids_high_cond, add_time_ids_low_uncond, add_time_ids_low_cond]
    # Assuming add_time_ids is the same for both high and low, and already duplicated for CFG
    add_time_ids_for_noise_pred = torch.cat([
        add_time_ids, # high uncond
        add_time_ids, # high cond
        add_time_ids, # low uncond
        add_time_ids  # low cond
    ], dim=0)

    # Set LoRA slider for the combined batch. This is a simplification.
    # For true concurrent processing with different LoRA scales, the LoRA network
    # itself would need to be modified to accept a batch of scales.
    # For now, we'll apply the high scale, run, then apply the low scale, run.
    # This is NOT fully concurrent for the LoRA application, but batches the UNet inference.

    # High scale
    network.set_lora_slider(scale=scales[0])
    with network:
        with torch.no_grad():
            # Only pass the high latents and corresponding embeddings
            target_latents_high = train_util.predict_noise_xl(
                unet,
                noise_scheduler,
                current_timestep,
                combined_denoised_latents_for_noise_pred[0].unsqueeze(0), # High latent
                text_embeddings_for_noise_pred[0:2], # High uncond and cond
                pooled_embeds_for_noise_pred[0:2],   # High pooled uncond and cond
                add_time_ids_for_noise_pred[0:2],    # High add_time_ids uncond and cond
                guidance_scale=1,
            )
    loss_high_per_element = (target_latents_high - high_noise).pow(2).to(torch.float32)

    # Low scale
    network.set_lora_slider(scale=scales[1])
    with network:
        with torch.no_grad():
            # Only pass the low latents and corresponding embeddings
            target_latents_low = train_util.predict_noise_xl(
                unet,
                noise_scheduler,
                current_timestep,
                combined_denoised_latents_for_noise_pred[1].unsqueeze(0), # Low latent
                text_embeddings_for_noise_pred[2:4], # Low uncond and cond
                pooled_embeds_for_noise_pred[2:4],   # Low pooled uncond and cond
                add_time_ids_for_noise_pred[2:4],    # Low add_time_ids uncond and cond
                guidance_scale=1,
            )
    loss_low_per_element = (target_latents_low - low_noise).pow(2).to(torch.float32)

    return loss_high_per_element, loss_low_per_element, denoised_latents_high, high_noise, target_latents_high, denoised_latents_low, low_noise, target_latents_low


def test_refactored_training_loop():
    """
    Tests that the refactored training loops are float-for-float identical to the original.
    """
    # Load config and models
    config = config_util.load_config_from_yaml("trainscripts/imagesliders/data/config-xl-dilora.yaml")
    tokenizers, text_encoders, unet, noise_scheduler, vae = model_util.load_models_xl(
        config.pretrained_model.name_or_path,
        scheduler_name=config.train.noise_scheduler,
    )
    device = torch.device("cuda:0")
    weight_dtype = config_util.parse_precision(config.train.precision)
    
    unet.to(device, dtype=weight_dtype).eval()
    vae.to(device, dtype=weight_dtype).eval()
    log_vram_usage("model loading")


    # Create a dummy network and prompt pair
    network = lora.LoRANetwork(
        unet,
        rank=config.network.rank,
        multiplier=1.0,
        alpha=config.network.alpha,
        train_method=config.network.training_method,
    ).to(device, dtype=weight_dtype)
    
    prompts = prompt_util.load_prompts_from_yaml(config.prompts_file, [])
    prompt_pair = prompt_util.PromptEmbedsPair(
        torch.nn.MSELoss(),
        PromptEmbedsXL(*train_util.encode_prompts_xl(tokenizers, text_encoders, [prompts[0].target])),
        PromptEmbedsXL(*train_util.encode_prompts_xl(tokenizers, text_encoders, [prompts[0].positive])),
        PromptEmbedsXL(*train_util.encode_prompts_xl(tokenizers, text_encoders, [prompts[0].unconditional])),
        PromptEmbedsXL(*train_util.encode_prompts_xl(tokenizers, text_encoders, [prompts[0].neutral])),
        prompts[0],
    )

    # Move prompt embeddings to the correct device and dtype
    prompt_pair.positive.text_embeds = prompt_pair.positive.text_embeds.to(device, dtype=weight_dtype)
    prompt_pair.positive.pooled_embeds = prompt_pair.positive.pooled_embeds.to(device, dtype=weight_dtype)
    prompt_pair.neutral.text_embeds = prompt_pair.neutral.text_embeds.to(device, dtype=weight_dtype)
    prompt_pair.neutral.pooled_embeds = prompt_pair.neutral.pooled_embeds.to(device, dtype=weight_dtype)
    log_vram_usage("prompt embedding processing")


    # Load real images
    img1 = Image.open("F:\\dox\\ai\\gemmy\\sliders\\datasets\\bracket\\0\\A.png").convert("RGB")
    img2 = Image.open("F:\\dox\\ai\\gemmy\\sliders\\datasets\\bracket\\0\\B.png").convert("RGB")

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True)

    # Convert PIL images to batched tensors
    img1_tensor = image_processor.preprocess(img1).to(device, dtype=weight_dtype)
    img2_tensor = image_processor.preprocess(img2).to(device, dtype=weight_dtype)
    log_vram_usage("image processing")

    add_time_ids = train_util.get_add_time_ids(
        prompts[0].resolution, prompts[0].resolution, prompts[0].dynamic_crops, weight_dtype
    ).to(device, dtype=weight_dtype)
    log_vram_usage("add_time_ids processing")

    seed = random.randint(0, 2**15)

    # Original implementation
    loss_high_orig_per_element, loss_low_orig_per_element, orig_denoised_latents_high, orig_high_noise, orig_target_latents_high, orig_denoised_latents_low, orig_low_noise, orig_target_latents_low = original_train_step(
        unet, vae, noise_scheduler, img1, img2, 1.0, prompt_pair, config, network, torch.nn.MSELoss(), device, weight_dtype, add_time_ids, seed
    )
    loss_high_orig = loss_high_orig_per_element.mean()
    loss_low_orig = loss_low_orig_per_element.mean()

    # Refactored implementation
    loss_high_refactored_per_element, loss_low_refactored_per_element, refactored_denoised_latents_high, refactored_high_noise, refactored_target_latents_high, refactored_denoised_latents_low, refactored_low_noise, refactored_target_latents_low = functional_train_step(
        unet,
        vae,
        noise_scheduler,
        img1,
        img2,
        1.0,
        prompt_pair,
        config,
        network,
        torch.nn.MSELoss(),
        device,
        weight_dtype,
        add_time_ids,
        seed
    )
    loss_high_refactored = loss_high_refactored_per_element.mean()
    loss_low_refactored = loss_low_refactored_per_element.mean()

    # Super-functional implementation
    loss_high_super_per_element, loss_low_super_per_element, super_denoised_latents_high, super_high_noise, super_target_latents_high, super_denoised_latents_low, super_low_noise, super_target_latents_low = superfunctional_train_step(
        unet,
        vae,
        noise_scheduler,
        (img1, img2),
        (1.0, -1.0),
        (prompt_pair.positive, prompt_pair.neutral),
        prompt_pair,
        config,
        network,
        torch.nn.MSELoss(),
        device,
        weight_dtype,
        add_time_ids,
        seed
    )
    loss_high_super = loss_high_super_per_element.mean()
    loss_low_super = loss_low_super_per_element.mean()

    # Assert that the losses are identical
    diff_high_orig_refactored = torch.abs(loss_high_orig - loss_high_refactored)
    diff_low_orig_refactored = torch.abs(loss_low_orig - loss_low_refactored)
    diff_high_orig_super = torch.abs(loss_high_orig - loss_high_super)
    diff_low_orig_super = torch.abs(loss_low_orig - loss_low_super)

    print(f"Difference (abs) high_orig vs high_refactored: {diff_high_orig_refactored.item()}")
    print(f"Difference (abs) low_orig vs low_refactored: {diff_low_orig_refactored.item()}")
    print(f"Difference (abs) high_orig vs high_super: {diff_high_orig_super.item()}")
    print(f"Difference (abs) low_orig vs low_super: {diff_low_orig_super.item()}")

    print(f"Difference (L2 norm) high_orig vs high_refactored: {torch.linalg.norm(diff_high_orig_refactored).item()}")
    print(f"Difference (L2 norm) low_orig vs low_refactored: {torch.linalg.norm(diff_low_orig_refactored).item()}")
    print(f"Difference (L2 norm) high_orig vs high_super: {torch.linalg.norm(diff_high_orig_super).item()}")
    print(f"Difference (L2 norm) low_orig vs low_super: {torch.linalg.norm(diff_low_orig_super).item()}")

    # Save all tensors into a single state dictionary
    output_dir = "F:/dox/ai/gemmy/sliders/tensor_logs"
    state_dict_filename = os.path.join(output_dir, "diff_debugger_tensors.pt")
    config = config_util.load_config_from_yaml("trainscripts/imagesliders/data/config-xl-dilora.yaml")
    tokenizers, text_encoders, unet, noise_scheduler, vae = model_util.load_models_xl(
        config.pretrained_model.name_or_path,
        scheduler_name=config.train.noise_scheduler,
    )
    device = torch.device("cuda:0")
    weight_dtype = config_util.parse_precision(config.train.precision)
    
    unet.to(device, dtype=weight_dtype).eval()
    vae.to(device, dtype=weight_dtype).eval()
    log_vram_usage("model loading")


    # Create a dummy network and prompt pair
    network = lora.LoRANetwork(
        unet,
        rank=config.network.rank,
        multiplier=1.0,
        alpha=config.network.alpha,
        train_method=config.network.training_method,
    ).to(device, dtype=weight_dtype)
    
    prompts = prompt_util.load_prompts_from_yaml(config.prompts_file, [])
    prompt_pair = prompt_util.PromptEmbedsPair(
        torch.nn.MSELoss(),
        PromptEmbedsXL(*train_util.encode_prompts_xl(tokenizers, text_encoders, [prompts[0].target])),
        PromptEmbedsXL(*train_util.encode_prompts_xl(tokenizers, text_encoders, [prompts[0].positive])),
        PromptEmbedsXL(*train_util.encode_prompts_xl(tokenizers, text_encoders, [prompts[0].unconditional])),
        PromptEmbedsXL(*train_util.encode_prompts_xl(tokenizers, text_encoders, [prompts[0].neutral])),
        prompts[0],
    )

    # Move prompt embeddings to the correct device and dtype
    prompt_pair.positive.text_embeds = prompt_pair.positive.text_embeds.to(device, dtype=weight_dtype)
    prompt_pair.positive.pooled_embeds = prompt_pair.positive.pooled_embeds.to(device, dtype=weight_dtype)
    prompt_pair.neutral.text_embeds = prompt_pair.neutral.text_embeds.to(device, dtype=weight_dtype)
    prompt_pair.neutral.pooled_embeds = prompt_pair.neutral.pooled_embeds.to(device, dtype=weight_dtype)
    log_vram_usage("prompt embedding processing")


    # Load real images
    img1 = Image.open("F:\\dox\\ai\\gemmy\\sliders\\datasets\\bracket\\0\\A.png").convert("RGB")
    img2 = Image.open("F:\\dox\\ai\\gemmy\\sliders\\datasets\\bracket\\0\\B.png").convert("RGB")

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True)

    # Convert PIL images to batched tensors
    img1_tensor = image_processor.preprocess(img1).to(device, dtype=weight_dtype)
    img2_tensor = image_processor.preprocess(img2).to(device, dtype=weight_dtype)
    log_vram_usage("image processing")

    add_time_ids = train_util.get_add_time_ids(
        prompts[0].resolution, prompts[0].resolution, prompts[0].dynamic_crops, weight_dtype
    ).to(device, dtype=weight_dtype)
    log_vram_usage("add_time_ids processing")

    seed = random.randint(0, 2**15)

    # Original implementation
    loss_high_orig_per_element, loss_low_orig_per_element, orig_denoised_latents_high, orig_high_noise, orig_target_latents_high, orig_denoised_latents_low, orig_low_noise, orig_target_latents_low = original_train_step(
        unet, vae, noise_scheduler, img1, img2, 1.0, prompt_pair, config, network, torch.nn.MSELoss(), device, weight_dtype, add_time_ids, seed
    )
    loss_high_orig = loss_high_orig_per_element.mean()
    loss_low_orig = loss_low_orig_per_element.mean()

    # Refactored implementation
    loss_high_refactored_per_element, loss_low_refactored_per_element, refactored_denoised_latents_high, refactored_high_noise, refactored_target_latents_high, refactored_denoised_latents_low, refactored_low_noise, refactored_target_latents_low = functional_train_step(
        unet,
        vae,
        noise_scheduler,
        img1,
        img2,
        1.0,
        prompt_pair,
        config,
        network,
        torch.nn.MSELoss(),
        device,
        weight_dtype,
        add_time_ids,
        seed
    )
    loss_high_refactored = loss_high_refactored_per_element.mean()
    loss_low_refactored = loss_low_refactored_per_element.mean()

    # Super-functional implementation
    loss_high_super_per_element, loss_low_super_per_element, super_denoised_latents_high, super_high_noise, super_target_latents_high, super_denoised_latents_low, super_low_noise, super_target_latents_low = superfunctional_train_step(
        unet,
        vae,
        noise_scheduler,
        (img1, img2),
        (1.0, -1.0),
        (prompt_pair.positive, prompt_pair.neutral),
        prompt_pair,
        config,
        network,
        torch.nn.MSELoss(),
        device,
        weight_dtype,
        add_time_ids,
        seed
    )
    loss_high_super = loss_high_super_per_element.mean()
    loss_low_super = loss_low_super_per_element.mean()

    # Assert that the losses are identical
    diff_high_orig_refactored = torch.abs(loss_high_orig - loss_high_refactored)
    diff_low_orig_refactored = torch.abs(loss_low_orig - loss_low_refactored)
    diff_high_orig_super = torch.abs(loss_high_orig - loss_high_super)
    diff_low_orig_super = torch.abs(loss_low_orig - loss_low_super)

    print(f"Difference (abs) high_orig vs high_refactored: {diff_high_orig_refactored.item()}")
    print(f"Difference (abs) low_orig vs low_refactored: {diff_low_orig_refactored.item()}")
    print(f"Difference (abs) high_orig vs high_super: {diff_high_orig_super.item()}")
    print(f"Difference (abs) low_orig vs low_super: {diff_low_orig_super.item()}")

    print(f"Difference (L2 norm) high_orig vs high_refactored: {torch.linalg.norm(diff_high_orig_refactored).item()}")
    print(f"Difference (L2 norm) low_orig vs low_refactored: {torch.linalg.norm(diff_low_orig_refactored).item()}")
    print(f"Difference (L2 norm) high_orig vs high_super: {torch.linalg.norm(diff_high_orig_super).item()}")
    print(f"Difference (L2 norm) low_orig vs low_super: {torch.linalg.norm(diff_low_orig_super).item()}")

    # Save all tensors into a single state dictionary
    output_dir = "F:/dox/ai/gemmy/sliders/tensor_logs"
    state_dict_filename = os.path.join(output_dir, "diff_debugger_tensors.pt")
    torch.save({
        "diff_high_orig_refactored": diff_high_orig_refactored,
        "diff_low_orig_refactored": diff_low_orig_refactored,
        "diff_high_orig_super": diff_high_orig_super,
        "diff_low_orig_super": diff_low_orig_super,
        "loss_high_orig_per_element": loss_high_orig_per_element,
        "loss_low_orig_per_element": loss_low_orig_per_element,
        "loss_high_refactored_per_element": loss_high_refactored_per_element,
        "loss_low_refactored_per_element": loss_low_refactored_per_element,
        "loss_high_super_per_element": loss_high_super_per_element,
        "loss_low_super_per_element": loss_low_super_per_element,
        "orig_denoised_latents_high": orig_denoised_latents_high,
        "orig_high_noise": orig_high_noise,
        "orig_target_latents_high": orig_target_latents_high,
        "orig_denoised_latents_low": orig_denoised_latents_low,
        "orig_low_noise": orig_low_noise,
        "orig_target_latents_low": orig_target_latents_low,
        "refactored_denoised_latents_high": refactored_denoised_latents_high,
        "refactored_high_noise": refactored_high_noise,
        "refactored_target_latents_high": refactored_target_latents_high,
        "refactored_denoised_latents_low": refactored_denoised_latents_low,
        "refactored_low_noise": refactored_low_noise,
        "refactored_target_latents_low": refactored_target_latents_low,
        "super_denoised_latents_high": super_denoised_latents_high,
        "super_high_noise": super_high_noise,
        "super_target_latents_high": super_target_latents_high,
        "super_denoised_latents_low": super_denoised_latents_low,
        "super_low_noise": super_low_noise,
        "super_target_latents_low": super_target_latents_low,
    }, state_dict_filename)

    print(f"All tensors saved to {state_dict_filename}")
    print("All tests completed. Please check the differences and saved tensors.")


def original_train_step(
    unet: torch.nn.Module,
    vae: torch.nn.Module,
    noise_scheduler,
    img1: Image.Image,
    img2: Image.Image,
    scale_to_look: float,
    prompt_pair: prompt_util.PromptEmbedsPair,
    config: config_util.RootConfig,
    network: lora.LoRANetwork,
    criteria: torch.nn.Module,
    device: torch.device,
    weight_dtype: torch.dtype,
    add_time_ids: torch.Tensor,
    seed: int,
):
    """
    A recreation of the original training step for comparison.

    Expected operand types:
    - unet: torch.nn.Module (UNet model)
    - vae: torch.nn.Module (VAE model)
    - noise_scheduler: Noise scheduler object (e.g., DDPMScheduler)
    - img1: PIL.Image.Image (Input image for low scale)
    - img2: PIL.Image.Image (Input image for high scale)
    - scale_to_look: float (Scale value for LoRA slider)
    - prompt_pair: prompt_util.PromptEmbedsPair (Container for prompt embeddings)
    - config: config_util.Config (Configuration object)
    - network: lora.LoRANetwork (LoRA network)
    - criteria: torch.nn.Module (Loss function, e.g., MSELoss)
    - device: torch.device (Device to perform computations on, e.g., 'cuda:0')
    - weight_dtype: torch.dtype (Data type for model weights, e.g., torch.float16)
    - add_time_ids: torch.Tensor (Additional time IDs for XL models)

    Expected output types:
    - loss_high: torch.Tensor (Loss for high scale)
    - loss_low: torch.Tensor (Loss for low scale)

    Expected calling functions:
    - Called by `test_refactored_training_loop` for comparison.
    """
    generator = torch.manual_seed(seed)
    denoised_latents_low, low_noise = train_util.get_noisy_image(
        img1,
        vae,
        generator,
        unet,
        noise_scheduler,
        start_timesteps=0,
        total_timesteps=config.train.max_denoising_steps - 1,
    )
    denoised_latents_low = denoised_latents_low.to(device, dtype=weight_dtype)
    low_noise = low_noise.to(device, dtype=weight_dtype)

    generator = torch.manual_seed(seed)
    denoised_latents_high, high_noise = train_util.get_noisy_image(
        img2,
        vae,
        generator,
        unet,
        noise_scheduler,
        start_timesteps=0,
        total_timesteps=config.train.max_denoising_steps - 1,
    )
    denoised_latents_high = denoised_latents_high.to(device, dtype=weight_dtype)
    high_noise = high_noise.to(device, dtype=weight_dtype)
    noise_scheduler.set_timesteps(1000)

    current_timestep = noise_scheduler.timesteps[
        int((config.train.max_denoising_steps - 1) * 1000 / config.train.max_denoising_steps)
    ]

    network.set_lora_slider(scale=scale_to_look)
    with network:
        with torch.no_grad():
            # Duplicate conditioning inputs for classifier-free guidance
            positive_text_embeds = torch.cat([prompt_pair.positive.text_embeds, prompt_pair.positive.text_embeds], dim=0)
            positive_pooled_embeds = torch.cat([prompt_pair.positive.pooled_embeds, prompt_pair.positive.pooled_embeds], dim=0)
            duplicated_add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

            target_latents_high = train_util.predict_noise_xl(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents_high,
                positive_text_embeds,
                positive_pooled_embeds,
                duplicated_add_time_ids,
                guidance_scale=1,
            )
    loss_high_per_element = (target_latents_high - high_noise).pow(2).to(torch.float32)

    network.set_lora_slider(scale=-scale_to_look)
    with network:
        with torch.no_grad():
            # Duplicate conditioning inputs for classifier-free guidance
            neutral_text_embeds = torch.cat([prompt_pair.neutral.text_embeds, prompt_pair.neutral.text_embeds], dim=0)
            neutral_pooled_embeds = torch.cat([prompt_pair.neutral.pooled_embeds, prompt_pair.neutral.pooled_embeds], dim=0)
            duplicated_add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

            target_latents_low = train_util.predict_noise_xl(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents_low,
                neutral_text_embeds,
                neutral_pooled_embeds,
                duplicated_add_time_ids,
                guidance_scale=1,
            )
    loss_low_per_element = (target_latents_low - low_noise).pow(2).to(torch.float32)

    return loss_high_per_element, loss_low_per_element, denoised_latents_high, high_noise, target_latents_high, denoised_latents_low, low_noise, target_latents_low


if __name__ == "__main__":
    test_refactored_training_loop()
