# This file contains documentation and code for batching compatibility of training functions.
# It is designed to be modular, especially for guidance rules, allowing for easy swapping of CFG implementations.

import torch
from diffusers.utils.torch_utils import randn_tensor
from typing import Tuple

# --- Analysis of train_util.get_noisy_image ---
# Function Operand: imgs (single Image.Image or list of Image.Image), vae, generator, unet, scheduler, total_timesteps, start_timesteps.
# Can be batched? Yes. The function already handles a list of Image.Image and concatenates them into image_batch.
# vae.encode can handle batched inputs. randn_tensor can generate batched noise. scheduler.add_noise can handle batched latents and noise.
# Needs to be batched? Yes. In superfunctional_train_step, get_noisy_image is called twice. To truly batch the training step, these two calls should ideally be combined into a single call that processes both high and low images in a single batch.

def get_batched_noisy_images(
    img_batches: Tuple[torch.Tensor, torch.Tensor],
    vae: torch.nn.Module,
    generator: torch.Generator,
    unet: torch.nn.Module,
    noise_scheduler,
    config,
    device: torch.device,
    weight_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combines the process of getting noisy images for high and low scales into a single batched operation.
    """
    # Combine high and low latents for batched noise generation
    combined_latents_for_noise_addition = torch.cat([img_batches[0], img_batches[1]], dim=0)
    
    # Generate noise for the combined batch
    noise_shape = combined_latents_for_noise_addition.shape
    combined_noise = randn_tensor(noise_shape, generator=generator, device=device)

    # Determine the timestep for adding noise (same for both high and low)
    noise_timestep = noise_scheduler.timesteps[config.train.max_denoising_steps - 1]

    # Add noise to the combined latents
    combined_denoised_latents = noise_scheduler.add_noise(combined_latents_for_noise_addition, combined_noise, noise_timestep)

    # Split back into high and low components
    denoised_latents_high = combined_denoised_latents[0].unsqueeze(0)
    high_noise = combined_noise[0].unsqueeze(0)

    denoised_latents_low = combined_denoised_latents[1].unsqueeze(0)
    low_noise = combined_noise[1].unsqueeze(0)

    return denoised_latents_high, high_noise, denoised_latents_low, low_noise

# --- Modular functions for predict_noise_xl and guidance ---

def prepare_unet_input_for_cfg(
    latents: torch.FloatTensor,
    scheduler,
    timestep: int,
) -> torch.FloatTensor:
    """
    Prepares the latent input for the UNet by expanding for classifier-free guidance
    and scaling according to the scheduler.
    """
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)
    return latent_model_input

def run_unet_inference(
    unet: torch.nn.Module,
    latent_model_input: torch.FloatTensor,
    timestep: int,
    text_embeddings: torch.FloatTensor,
    add_text_embeddings: torch.FloatTensor,
    add_time_ids: torch.FloatTensor,
) -> torch.FloatTensor:
    """
    Performs the UNet forward pass to predict the noise residual.
    """
    added_cond_kwargs = {
        "text_embeds": add_text_embeddings,
        "time_ids": add_time_ids,
    }
    noise_pred = unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=text_embeddings,
        added_cond_kwargs=added_cond_kwargs,
    ).sample
    return noise_pred

def apply_cfg_guidance(
    noise_pred: torch.FloatTensor,
    guidance_scale: float,
) -> torch.FloatTensor:
    """
    Applies classifier-free guidance to the predicted noise.
    """
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    guided_target = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )
    return guided_target

def apply_rescale_noise_cfg(
    noise_cfg: torch.FloatTensor,
    noise_pred_text: torch.FloatTensor,
    guidance_rescale: float,
) -> torch.FloatTensor:
    """
    Rescales `noise_cfg` according to `guidance_rescale`.
    Based on findings of [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
    """
    std_text = noise_pred_text.std(
        dim=list(range(1, noise_pred_text.ndim)), keepdim=True
    )
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = (
        guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    )
    return noise_cfg

def batched_predict_noise_xl_modular(
    unet: torch.nn.Module,
    scheduler,
    timestep: int,
    latents: torch.FloatTensor,
    text_embeddings: torch.FloatTensor,
    add_text_embeddings: torch.FloatTensor,
    add_time_ids: torch.FloatTensor,
    guidance_scale: float = 7.5,
    guidance_rescale: float = 0.7,
) -> torch.FloatTensor:
    """
    A modular version of predict_noise_xl, orchestrating the sub-components.
    This function can be easily modified to swap out different guidance strategies.
    """
    latent_model_input = prepare_unet_input_for_cfg(latents, scheduler, timestep)
    
    noise_pred = run_unet_inference(
        unet,
        latent_model_input,
        timestep,
        text_embeddings,
        add_text_embeddings,
        add_time_ids,
    )
    
    guided_target = apply_cfg_guidance(noise_pred, guidance_scale)
    
    # noise_pred_text is needed for rescale_noise_cfg, so we need to re-chunk noise_pred
    _, noise_pred_text = noise_pred.chunk(2) 
    guided_target = apply_rescale_noise_cfg(guided_target, noise_pred_text, guidance_rescale)
    
    return guided_target
