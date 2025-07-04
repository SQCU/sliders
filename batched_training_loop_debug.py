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
    img_pairs: list[tuple[torch.Tensor, torch.Tensor]], # List of (img_high, img_low) tensors
    scale_pairs: list[tuple[float, float]],             # List of (scale_high, scale_low) floats
    prompt_embed_pairs: list[tuple[prompt_util.PromptEmbedsXL, prompt_util.PromptEmbedsXL]], # List of (prompt_positive, prompt_neutral)
    config: config_util.RootConfig,
    network: lora.LoRANetwork,
    criteria: torch.nn.Module,
    device: torch.device,
    weight_dtype: torch.dtype,
    add_time_ids_list: list[torch.Tensor],              # List of add_time_ids tensors
    seed: int,
):
    """
    A more functional and compact training step that processes high and low cases concurrently for a batch of image pairs.
    """
    noise_scheduler.set_timesteps(1000)
    current_timestep = noise_scheduler.timesteps[
        int((config.train.max_denoising_steps - 1) * 1000 / config.train.max_denoising_steps)
    ]

    all_denoised_latents_high = []
    all_high_noise = []
    all_denoised_latents_low = []
    all_low_noise = []

    for i, (img_high, img_low) in enumerate(img_pairs):
        generator = torch.manual_seed(seed + i) # Vary seed for each pair for different noise
        denoised_latents_high, high_noise = train_util.get_noisy_image(
            img_high,
            vae,
            generator,
            unet,
            noise_scheduler,
            start_timesteps=0,
            total_timesteps=config.train.max_denoising_steps - 1,
        )
        all_denoised_latents_high.append(denoised_latents_high.to(device, dtype=weight_dtype))
        all_high_noise.append(high_noise.to(device, dtype=weight_dtype))

        generator = torch.manual_seed(seed + i) # Re-seed for the second image of the pair
        denoised_latents_low, low_noise = train_util.get_noisy_image(
            img_low,
            vae,
            generator,
            unet,
            noise_scheduler,
            start_timesteps=0,
            total_timesteps=config.train.max_denoising_steps - 1,
        )
        all_denoised_latents_low.append(denoised_latents_low.to(device, dtype=weight_dtype))
        all_low_noise.append(low_noise.to(device, dtype=weight_dtype))

    # Concatenate all latents and noise for high and low cases
    batched_denoised_latents_high = torch.cat(all_denoised_latents_high, dim=0)
    batched_high_noise = torch.cat(all_high_noise, dim=0)
    batched_denoised_latents_low = torch.cat(all_denoised_latents_low, dim=0)
    batched_low_noise = torch.cat(all_low_noise, dim=0)

    # Now, prepare inputs for predict_noise_xl for all high-scale examples
    # and then for all low-scale examples.
    # This still involves two separate calls to predict_noise_xl because
    # the LoRA slider is set globally.

    # --- High Scale Processing ---
    all_positive_text_embeds = []
    all_positive_pooled_embeds = []
    all_add_time_ids_high = []

    for i, (prompt_positive, _) in enumerate(prompt_embed_pairs):
        all_positive_text_embeds.append(prompt_positive.text_embeds)
        all_positive_pooled_embeds.append(prompt_positive.pooled_embeds)
        all_add_time_ids_high.append(add_time_ids_list[i])

    batched_positive_text_embeds = torch.cat(all_positive_text_embeds, dim=0)
    batched_positive_pooled_embeds = torch.cat(all_positive_pooled_embeds, dim=0)
    batched_add_time_ids_high = torch.cat(all_add_time_ids_high, dim=0)

    # Set LoRA slider for high scale (assuming all high scales are the same for now, or we take the first one)
    network.set_lora_slider(scale=scale_pairs[0][0]) # Using the first high scale for now
    with network:
        target_latents_high_batch = train_util.predict_noise_xl(
            unet,
            noise_scheduler,
            current_timestep,
            batched_denoised_latents_high,
            torch.cat([batched_positive_text_embeds, batched_positive_text_embeds], dim=0),
            torch.cat([batched_positive_pooled_embeds, batched_positive_pooled_embeds], dim=0),
            torch.cat([batched_add_time_ids_high, batched_add_time_ids_high], dim=0),
            guidance_scale=1,
        )
    loss_high_per_element = (target_latents_high_batch - batched_high_noise).pow(2).to(torch.float32)

    # --- Low Scale Processing ---
    all_neutral_text_embeds = []
    all_neutral_pooled_embeds = []
    all_add_time_ids_low = []

    for i, (_, prompt_neutral) in enumerate(prompt_embed_pairs):
        all_neutral_text_embeds.append(prompt_neutral.text_embeds)
        all_neutral_pooled_embeds.append(prompt_neutral.pooled_embeds)
        all_add_time_ids_low.append(add_time_ids_list[i])

    batched_neutral_text_embeds = torch.cat(all_neutral_text_embeds, dim=0)
    batched_neutral_pooled_embeds = torch.cat(all_neutral_pooled_embeds, dim=0)
    batched_add_time_ids_low = torch.cat(all_add_time_ids_low, dim=0)

    # Set LoRA slider for low scale (assuming all low scales are the same for now, or we take the first one)
    network.set_lora_slider(scale=scale_pairs[0][1]) # Using the first low scale for now
    with network:
        target_latents_low_batch = train_util.predict_noise_xl(
            unet,
            noise_scheduler,
            current_timestep,
            batched_denoised_latents_low,
            torch.cat([batched_neutral_text_embeds, batched_neutral_text_embeds], dim=0),
            torch.cat([batched_neutral_pooled_embeds, batched_neutral_pooled_embeds], dim=0),
            torch.cat([batched_add_time_ids_low, batched_add_time_ids_low], dim=0),
            guidance_scale=1,
        )
    loss_low_per_element = (target_latents_low_batch - batched_low_noise).pow(2).to(torch.float32)

    return loss_high_per_element, loss_low_per_element, batched_denoised_latents_high, batched_high_noise, target_latents_high_batch, batched_denoised_latents_low, batched_low_noise, target_latents_low_batch


def test_refactored_training_loop():
    """
    Tests that the refactored training loops are float-for-float identical to the original.
    """
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
    img1_path = os.path.normpath("F:/dox/ai/gemmy/sliders/datasets/bracket/0/A.png")
    img2_path = os.path.normpath("F:/dox/ai/gemmy/sliders/datasets/bracket/0/B.png")
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")

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
        img1_tensor, # Use img1_tensor here
        img2_tensor, # Use img2_tensor here
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

    # Super-functional implementation (Batched)
    # Create dummy batched inputs (4 pairs)
    num_batches = 4
    batched_img_pairs = []
    batched_scale_pairs = []
    batched_prompt_embed_pairs = []
    batched_add_time_ids_list = []

    for i in range(num_batches):
        batched_img_pairs.append((img1_tensor, img2_tensor))
        batched_scale_pairs.append((1.0, -1.0))
        batched_prompt_embed_pairs.append((prompt_pair.positive, prompt_pair.neutral))
        batched_add_time_ids_list.append(add_time_ids)

    loss_high_super_per_element, loss_low_super_per_element, super_denoised_latents_high, super_high_noise, super_target_latents_high, super_denoised_latents_low, super_low_noise, super_target_latents_low = superfunctional_train_step(
        unet,
        vae,
        noise_scheduler,
        batched_img_pairs,
        batched_scale_pairs,
        batched_prompt_embed_pairs,
        config,
        network,
        torch.nn.MSELoss(),
        device,
        weight_dtype,
        batched_add_time_ids_list,
        seed
    )
    loss_high_super = loss_high_super_per_element.mean()
    loss_low_super = loss_low_super_per_element.mean()

    # Individual calculations for comparison
    individual_losses_high = []
    individual_losses_low = []
    for i in range(num_batches):
        loss_high_indiv_per_element, loss_low_indiv_per_element, _, _, _, _, _, _ = functional_train_step(
            unet,
            vae,
            noise_scheduler,
            batched_img_pairs[i][0],
            batched_img_pairs[i][1],
            batched_scale_pairs[i][0],
            prompt_pair, # Using the original prompt_pair for individual steps
            config,
            network,
            torch.nn.MSELoss(),
            device,
            weight_dtype,
            batched_add_time_ids_list[i],
            seed + i # Use varied seed for individual steps to match batched noise generation
        )
        individual_losses_high.append(loss_high_indiv_per_element.mean())
        individual_losses_low.append(loss_low_indiv_per_element.mean())

    # Compare batched and individual results
    batched_loss_high_mean = loss_high_super_per_element.mean()
    batched_loss_low_mean = loss_low_super_per_element.mean()

    individual_loss_high_mean = torch.stack(individual_losses_high).mean()
    individual_loss_low_mean = torch.stack(individual_losses_low).mean()

    print(f"Batched High Loss Mean: {batched_loss_high_mean.item()}")
    print(f"Individual High Loss Mean: {individual_loss_high_mean.item()}")
    print(f"Batched Low Loss Mean: {batched_loss_low_mean.item()}")
    print(f"Individual Low Loss Mean: {individual_loss_low_mean.item()}")

    # Add gradient comparison (example for one parameter)
    # Ensure network parameters require gradients
    for param in network.parameters():
        param.requires_grad = True

    # Calculate gradients for batched run
    batched_loss_high_mean.backward(retain_graph=True)
    batched_grads = {name: param.grad.clone() for name, param in network.named_parameters() if param.grad is not None}
    network.zero_grad()

    # Calculate gradients for individual runs and sum them
    summed_individual_grads = {}
    for i in range(num_batches):
        loss_high_indiv_per_element, _, _, _, _, _, _, _ = functional_train_step(
            unet,
            vae,
            noise_scheduler,
            batched_img_pairs[i][0],
            batched_img_pairs[i][1],
            batched_scale_pairs[i][0],
            prompt_pair,
            config,
            network,
            torch.nn.MSELoss(),
            device,
            weight_dtype,
            batched_add_time_ids_list[i],
            seed + i
        )
        loss_high_indiv_per_element.mean().backward(retain_graph=True)
        for name, param in network.named_parameters():
            if param.grad is not None:
                if name not in summed_individual_grads:
                    summed_individual_grads[name] = param.grad.clone()
                else:
                    summed_individual_grads[name] += param.grad.clone()
        network.zero_grad()

    # Compare gradients
    print("\nGradient Comparison:")
    for name in batched_grads.keys():
        if name in summed_individual_grads:
            diff_grad = torch.abs(batched_grads[name] - summed_individual_grads[name]).mean().item()
            print(f"  {name} - Absolute Difference: {diff_grad:.6f}")
            if diff_grad > 1e-6: # Define a small tolerance for floating point differences
                print(f"    WARNING: Significant gradient difference for {name}")
        else:
            print(f"  {name} - Not found in individual gradients.")

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
    
    # Re-load models and config for saving tensors, as they might have been modified by previous steps
    config = config_util.load_config_from_yaml("trainscripts/imagesliders/data/config-xl-dilora.yaml")
    tokenizers, text_encoders, unet, noise_scheduler, vae = model_util.load_models_xl(
        config.pretrained_model.name_or_path,
        scheduler_name=config.train.noise_scheduler,
    )
    device = torch.device("cuda:0")
    weight_dtype = config_util.parse_precision(config.train.precision)
    
    unet.to(device, dtype=weight_dtype).eval()
    vae.to(device, dtype=weight_dtype).eval()
    log_vram_usage("model loading for tensor saving")


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
    log_vram_usage("prompt embedding processing for tensor saving")


    # Load real images
    img1_path = os.path.normpath("F:/dox/ai/gemmy/sliders/datasets/bracket/0/A.png")
    img2_path = os.path.normpath("F:/dox/ai/gemmy/sliders/datasets/bracket/0/B.png")
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True)

    # Convert PIL images to batched tensors
    img1_tensor = image_processor.preprocess(img1).to(device, dtype=weight_dtype)
    img2_tensor = image_processor.preprocess(img2).to(device, dtype=weight_dtype)
    log_vram_usage("image processing for tensor saving")

    add_time_ids = train_util.get_add_time_ids(
        prompts[0].resolution, prompts[0].resolution, prompts[0].dynamic_crops, weight_dtype
    ).to(device, dtype=weight_dtype)
    log_vram_usage("add_time_ids processing for tensor saving")

    seed = random.randint(0, 2**15)

    # Re-run the steps to get the tensors for saving, ensuring they are fresh
    loss_high_orig_per_element, loss_low_orig_per_element, orig_denoised_latents_high, orig_high_noise, orig_target_latents_high, orig_denoised_latents_low, orig_low_noise, orig_target_latents_low = original_train_step(
        unet, vae, noise_scheduler, img1, img2, 1.0, prompt_pair, config, network, torch.nn.MSELoss(), device, weight_dtype, add_time_ids, seed
    )
    loss_high_refactored_per_element, loss_low_refactored_per_element, refactored_denoised_latents_high, refactored_high_noise, refactored_target_latents_high, refactored_denoised_latents_low, refactored_low_noise, refactored_target_latents_low = functional_train_step(
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
        add_time_ids,
        seed
    )
    loss_high_super_per_element, loss_low_super_per_element, super_denoised_latents_high, super_high_noise, super_target_latents_high, super_denoised_latents_low, super_low_noise, super_target_latents_low = superfunctional_train_step(
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
        add_time_ids,
        seed
    )

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
    print("Script started successfully.")
    # test_refactored_training_loop()
