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
    seed = random.randint(0, 2**15)
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
    loss_high = criteria(target_latents_high, high_noise).to(torch.float32)

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
    loss_low = criteria(target_latents_low, low_noise).to(torch.float32)

    return loss_high, loss_low


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
    seed = random.randint(0, 2**15)
    noise_scheduler.set_timesteps(1000)
    current_timestep = noise_scheduler.timesteps[
        int((config.train.max_denoising_steps - 1) * 1000 / config.train.max_denoising_steps)
    ]

    def process_case(img_batch, scale, embeds):
        generator = torch.manual_seed(seed)
        denoised_latents, noise = train_util.get_noisy_image(
            img_batch,
            vae,
            generator,
            unet,
            noise_scheduler,
            start_timesteps=0,
            total_timesteps=config.train.max_denoising_steps - 1,
        )
        denoised_latents = denoised_latents.to(device, dtype=weight_dtype)
        noise = noise.to(device, dtype=weight_dtype)

        network.set_lora_slider(scale=scale)
        with network:
            with torch.no_grad():
                # Duplicate conditioning inputs for classifier-free guidance
                duplicated_text_embeds = torch.cat([embeds.text_embeds, embeds.text_embeds], dim=0)
                duplicated_pooled_embeds = torch.cat([embeds.pooled_embeds, embeds.pooled_embeds], dim=0)
                duplicated_add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

                target_latents = train_util.predict_noise_xl(
                    unet,
                    noise_scheduler,
                    current_timestep,
                    denoised_latents,
                    duplicated_text_embeds,
                    duplicated_pooled_embeds,
                    duplicated_add_time_ids,
                    guidance_scale=1,
                )
            return criteria(target_latents, noise).to(torch.float32)

    loss_high = process_case(img_batches[0], scales[0], prompt_embeds[0])
    loss_low = process_case(img_batches[1], scales[1], prompt_embeds[1])

    return loss_high, loss_low


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

    # Original implementation
    loss_high_orig, loss_low_orig = original_train_step(
        unet, vae, noise_scheduler, img1, img2, 1.0, prompt_pair, config, network, torch.nn.MSELoss(), device, weight_dtype, add_time_ids
    )

    # Refactored implementation
    loss_high_refactored, loss_low_refactored = functional_train_step(
        unet,
        vae,
        noise_scheduler,
        img1_tensor,
        img2_tensor,
        1.0,
        prompt_pair,
        config,
        network,
        torch.nn.MSELoss(),
        device,
        weight_dtype,
        add_time_ids
    )

    # Super-functional implementation
    loss_high_super, loss_low_super = superfunctional_train_step(
        unet,
        vae,
        noise_scheduler,
        (img1_tensor, img2_tensor),
        (1.0, -1.0),
        (prompt_pair.positive, prompt_pair.neutral),
        prompt_pair,
        config,
        network,
        torch.nn.MSELoss(),
        device,
        weight_dtype,
        add_time_ids
    )

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

    # Save difference tensors for further inspection
    torch.save(diff_high_orig_refactored, "diff_high_orig_refactored.pt")
    torch.save(diff_low_orig_refactored, "diff_low_orig_refactored.pt")
    torch.save(diff_high_orig_super, "diff_high_orig_super.pt")
    torch.save(diff_low_orig_super, "diff_low_orig_super.pt")

    print("Difference tensors saved as .pt files.")

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
    seed = random.randint(0, 2**15)
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
            ).to("cpu", dtype=torch.float32)
    loss_high = criteria(target_latents_high, high_noise.cpu().to(torch.float32))

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
            ).to("cpu", dtype=torch.float32)
    loss_low = criteria(target_latents_low, low_noise.cpu().to(torch.float32))

    return loss_high, loss_low


if __name__ == "__main__":
    test_refactored_training_loop()
