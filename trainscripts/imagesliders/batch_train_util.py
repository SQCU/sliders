# This file contains documentation and code for batching compatibility of training functions.
# It is designed to be modular, especially for guidance rules, allowing for easy swapping of CFG implementations.

import torch
from diffusers.utils.torch_utils import randn_tensor
from typing import Tuple, Union
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from diffusers import UNet2DConditionModel, SchedulerMixin

SDXL_TEXT_ENCODER_TYPE = Union[CLIPTextModel, CLIPTextModelWithProjection]

# --- Analysis of train_util.get_noisy_image ---
# Function Operand: imgs (single Image.Image or list of Image.Image), vae, generator, unet, scheduler, total_timesteps, start_timesteps.
# Can be batched? Yes. The function already handles a list of Image.Image and concatenates them into image_batch.
# vae.encode can handle batched inputs. randn_tensor can generate batched noise. scheduler.add_noise can handle batched latents and noise.
# Needs to be batched? Yes. In superfunctional_train_step, get_noisy_image is called twice. To truly batch the training step, these two calls should ideally be combined into a single call that processes both high and low images in a single batch.

def get_batched_noisy_images(
    img_batch: torch.Tensor,
    generator: torch.Generator,
    noise_scheduler,
    config,
    total_timesteps:int, # = 1000,
    start_timesteps:int, #=0,
    device: torch.device,
    weight_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combines the process of getting noisy images for high and low scales into a single batched operation.
    """
   
    # Generate noise for the combined batch
    noise_shape = img_batch.shape
    combined_noise = randn_tensor(noise_shape, generator=generator, device=device)

    #timestep interval logic from upstream
    time_ = total_timesteps
    timestep = scheduler.timesteps[time_:time_+1]

    # Add noise to the combined latents
    noisy_latents = noise_scheduler.add_noise(img_batch, combined_noise, timestep)

    #return without splitting and unsplitting several times for no reason!
    return noisy_latents, noise

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
    
    return guided_target

def diffusion(
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    latents: torch.FloatTensor,  # ただのノイズだけのlatents
    text_embeddings: torch.FloatTensor,
    total_timesteps: int = 1000,
    start_timesteps=0,
    **kwargs,
):
    # latents_steps = []

    for timestep in tqdm(scheduler.timesteps[start_timesteps:total_timesteps]):
        noise_pred = run_unet_inference(
            unet, scheduler, timestep, latents, text_embeddings, **kwargs
        )

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    # return latents_steps
    return latents

def text_tokenize(
    tokenizer: CLIPTokenizer,  # 普通ならひとつ、XLならふたつ！
    prompts: list[str],
):
    token_ids = [
        tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        for prompt in prompts
    ]
    return torch.cat(token_ids)


def text_encode_xl(
    text_encoder: SDXL_TEXT_ENCODER_TYPE,
    tokens: torch.FloatTensor,
    num_images_per_prompt: int = 1,
):
    prompt_embeds = text_encoder(
        tokens.to(text_encoder.device), output_hidden_states=True
    )
    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]  # always penultimate layer

    bs_embed, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds

def encode_prompts_xl(
    tokenizers: list[CLIPTokenizer],
    text_encoders: list[SDXL_TEXT_ENCODER_TYPE],
    prompts: list[str],
    num_images_per_prompt: int = 1,
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    # text_encoder and text_encoder_2's penuultimate layer's output
    text_embeds_list = []
    pooled_text_embeds = None  # always text_encoder_2's pool

    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        text_tokens_input_ids = text_tokenize(tokenizer, prompts)
        text_embeds, pooled_text_embeds = text_encode_xl(
            text_encoder, text_tokens_input_ids, num_images_per_prompt
        )

        text_embeds_list.append(text_embeds)

    bs_embed = pooled_text_embeds.shape[0]
    pooled_text_embeds = pooled_text_embeds.repeat(1, num_images_per_prompt).view(
        bs_embed * num_images_per_prompt, -1
    )

    return torch.concat(text_embeds_list, dim=-1), pooled_text_embeds


def create_batched_prompt_embeddings(
    tokenizers: list[CLIPTokenizer],
    text_encoders: list[SDXL_TEXT_ENCODER_TYPE],
    prompts: dict,
    num_images_per_prompt: int = 1,
):
    """
    Creates a batched prompt embedding tensor for use in the training loop.
    """
    positive_prompts = [prompts['positive']]
    unconditional_prompts = [prompts['unconditional']]
    neutral_prompts = [prompts['neutral']]

    # Encode all prompts
    positive_text_embeds, positive_pooled_embeds = encode_prompts_xl(
        tokenizers, text_encoders, positive_prompts, num_images_per_prompt
    )
    unconditional_text_embeds, unconditional_pooled_embeds = encode_prompts_xl(
        tokenizers, text_encoders, unconditional_prompts, num_images_per_prompt
    )
    neutral_text_embeds, neutral_pooled_embeds = encode_prompts_xl(
        tokenizers, text_encoders, neutral_prompts, num_images_per_prompt
    )

    # Concatenate for the UNet
    text_embeddings_for_noise_pred = torch.cat([
        positive_text_embeds,
        unconditional_text_embeds,
        neutral_text_embeds,
    ], dim=0)

    # Pooled embeds are also needed for XL
    pooled_embeds_for_noise_pred = torch.cat([
        positive_pooled_embeds,
        unconditional_pooled_embeds,
        neutral_pooled_embeds,
    ], dim=0)

    return text_embeddings_for_noise_pred, pooled_embeds_for_noise_pred