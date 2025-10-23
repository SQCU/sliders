from typing import Optional, Union

import torch
import torch.nn as nn  # <--- ADD THIS LINE

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, SchedulerMixin
from diffusers.image_processor import VaeImageProcessor
from model_util import SDXL_TEXT_ENCODER_TYPE
from diffusers.utils.torch_utils import randn_tensor

from tqdm import tqdm

UNET_IN_CHANNELS = 4  # Stable Diffusion の in_channels は 4 で固定。XLも同じ。
VAE_SCALE_FACTOR = 8  # 2 ** (len(vae.config.block_out_channels) - 1) = 8

UNET_ATTENTION_TIME_EMBED_DIM = 256  # XL
TEXT_ENCODER_2_PROJECTION_DIM = 1280
UNET_PROJECTION_CLASS_EMBEDDING_INPUT_DIM = 2816


def get_random_noise(
    batch_size: int, height: int, width: int, generator: torch.Generator = None
) -> torch.Tensor:
    return torch.randn(
        (
            batch_size,
            UNET_IN_CHANNELS,
            height // VAE_SCALE_FACTOR,  # 縦と横これであってるのかわからないけど、どっちにしろ大きな問題は発生しないのでこれでいいや
            width // VAE_SCALE_FACTOR,
        ),
        generator=generator,
        device="cpu",
    )


# https://www.crosslabs.org/blog/diffusion-with-offset-noise
def apply_noise_offset(latents: torch.FloatTensor, noise_offset: float):
    latents = latents + noise_offset * torch.randn(
        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
    )
    return latents


def get_initial_latents(
    scheduler: SchedulerMixin,
    n_imgs: int,
    height: int,
    width: int,
    n_prompts: int,
    generator=None,
) -> torch.Tensor:
    noise = get_random_noise(n_imgs, height, width, generator=generator).repeat(
        n_prompts, 1, 1, 1
    )

    latents = noise * scheduler.init_noise_sigma

    return latents


def text_tokenize(
    tokenizer: CLIPTokenizer,  # 普通ならひとつ、XLならふたつ！
    prompts: list[str],
):
    return tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids


def text_encode(text_encoder: CLIPTextModel, tokens):
    return text_encoder(tokens.to(text_encoder.device))[0]


def encode_prompts(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTokenizer,
    prompts: list[str],
):

    text_tokens = text_tokenize(tokenizer, prompts)
    text_embeddings = text_encode(text_encoder, text_tokens)
    
    

    return text_embeddings


# https://github.com/huggingface/diffusers/blob/78922ed7c7e66c20aa95159c7b7a6057ba7d590d/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L334-L348
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


def concat_embeddings(
    unconditional: torch.FloatTensor,
    conditional: torch.FloatTensor,
    n_imgs: int,
):
    return torch.cat([unconditional, conditional]).repeat_interleave(n_imgs, dim=0)


# ref: https://github.com/huggingface/diffusers/blob/0bab447670f47c28df60fbd2f6a0f833f75a16f5/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L721
def predict_noise(
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    timestep: int,  # 現在のタイムステップ
    latents: torch.FloatTensor,
    text_embeddings: torch.FloatTensor,  # uncond な text embed と cond な text embed を結合したもの
    guidance_scale=7.5,
) -> torch.FloatTensor:
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

    # predict the noise residual
    noise_pred = unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=text_embeddings,
    ).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    guided_target = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )

    return guided_target



# ref: https://github.com/huggingface/diffusers/blob/0bab447670f47c28df60fbd2f6a0f833f75a16f5/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L746
@torch.no_grad()
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
        noise_pred = predict_noise(
            unet, scheduler, timestep, latents, text_embeddings, **kwargs
        )

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    # return latents_steps
    return latents

@torch.no_grad()
def get_noisy_image(
    img,
    vae,
    generator,
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    total_timesteps: int = 1000,
    start_timesteps=0,
    
    **kwargs,
):
    # latents_steps = []
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor,do_convert_rgb=True)

    image = img#.convert('RGB')
    im_orig = image
    device = vae.device
    image = image_processor.preprocess(image).to(device)

    init_latents = vae.encode(image).latent_dist.sample(None)
    init_latents = vae.config.scaling_factor * init_latents

    init_latents = torch.cat([init_latents], dim=0)

    shape = init_latents.shape

    noise = randn_tensor(shape, generator=generator, device=device)

    time_ = total_timesteps
    timestep = scheduler.timesteps[time_:time_+1]
    # get latents
    init_latents = scheduler.add_noise(init_latents, noise, timestep)
    
    return init_latents, noise

@torch.no_grad()
def get_noisy_image_and_init_image(
    img,
    vae,
    generator,
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    total_timesteps: int = 1000,
    start_timesteps=0,
    
    **kwargs,
):
    # latents_steps = []
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor,do_convert_rgb=True)

    image = img#.convert('RGB')
    im_orig = image
    device = vae.device
    image = image_processor.preprocess(image).to(device)

    init_latents = vae.encode(image).latent_dist.sample(None)
    init_latents = vae.config.scaling_factor * init_latents

    init_latents = torch.cat([init_latents], dim=0)

    shape = init_latents.shape

    noise = randn_tensor(shape, generator=generator, device=device)

    time_ = total_timesteps
    timestep = scheduler.timesteps[time_:time_+1]
    # get latents
    noisy_latents = scheduler.add_noise(init_latents, noise, timestep)
    
    return noisy_latents, noise, init_latents

@torch.no_grad()
def prepare_correlated_noisy_latents(
    batch_packet: dict,
    vae: nn.Module,
    # nope we don't pass the processor in, don't recreate it
    scheduler: SchedulerMixin,
    timesteps_to: int, # This is an INDEX (e.g., 20)
    dtype: torch.dtype
):
    """
    Handles batch image processing, VAE encoding, and correlated noise generation.
    Takes a structured batch packet and returns the core tensors for the UNet.
    """
    # --- Unpack data and get shapes ---
    images = batch_packet["images"]
    tuple_prng_seeds = batch_packet["tuple_prng_seeds"]
    batch_size, n_tuple = tuple_prng_seeds.shape

    # --- Image Preprocessing & Tensorization ---
    # Flatten the (B, N) list of lists into a single list of (B * N) images
    flat_images = [img for tpl in images for img in tpl]
    
    # Preprocess the entire batch of images. The VaeImageProcessor handles this.
    # The result is a single tensor of shape (B*N, 3, H, W)
    vae_preprocess_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_preprocess_scale_factor,do_convert_rgb=True)
    image_tensor = image_processor.preprocess(flat_images).to(device=vae.device, dtype=dtype)

    # Encode the whole batch in one go for maximum efficiency.
    x0_latents_flat = vae.encode(image_tensor).latent_dist.sample(None)

    # Apply the required VAE scaling factor.
    x0_latents_flat = vae.config.scaling_factor * x0_latents_flat

    # --- Correlated Noise Generation & Application ---
    # Reshape the clean latents into their logical tuple structure
    C, H, W = x0_latents_flat.shape[1:]
    x0_latents = x0_latents_flat.view(batch_size, n_tuple, C, H, W)

    # We only need one unique seed per tuple from our broadcasted tensor
    unique_tuple_seeds = tuple_prng_seeds[:, 0] # Shape: (B,)
    
    # Generate one unique noise mask for each tuple in the batch
    noise_masks = torch.stack([
        torch.randn(
            (C, H, W), 
            generator=torch.Generator(device=vae.device).manual_seed(seed.item()), 
            device=vae.device,
            dtype=dtype
        ) for seed in unique_tuple_seeds
    ]) # Final shape: (B, C, H, W)

    expanded_noise = noise_masks.unsqueeze(1).expand(-1, n_tuple, -1, -1, -1)
    # The `expanded_noise` tensor now has shape (B, N, C, H, W), exactly matching `x0_latents`.
    # Get the per-tuple timestep indices from the packet
    timesteps_indices = batch_packet["timesteps_to"] # Shape: (B, N)
    # We only need one index per tuple for the lookup
    unique_indices = timesteps_indices[:, 0] # Shape: (B,)
    # Look up the actual timestep VALUES from the scheduler's array
    timestep_values = scheduler.timesteps[unique_indices] # Shape: (B,)
    # The scheduler.add_noise function expects one timestep per batch item.
    # We need to broadcast our per-tuple values to the per-item N dimension.
    # Shape: (B,) -> (B, 1) -> (B, N)
    broadcast_timestep_values = timestep_values.unsqueeze(1).expand(-1, n_tuple)

    # Apply the SAME noise mask to all N items within each tuple via broadcasting.
    # The scheduler's add_noise handles this if we give it latents of (B, N, ...)
    # and noise of (B, 1, ...). We unsqueeze the noise to enable this.
    noisy_latents = scheduler.add_noise(
        x0_latents, 
        expanded_noise,
        broadcast_timestep_values.view(-1) # Shape: (B*N,) 
    )

    # --- 5. Return Flattened Tensors ready for the U-Net ---
    total_batch_size = batch_size * n_tuple     # B * N
    
    return {
        "noisy_latents": noisy_latents.view(total_batch_size, C, H, W),
        "ground_truth_noise": expanded_noise.view(total_batch_size, C, H, W), # Use the same expanded tensor
        "x0_latents": x0_latents_flat
    }

def get_x0_from_xt_eps(
    xt_latents: torch.Tensor,
    eps_theta: torch.Tensor,
    timestep_t: int,
    scheduler: SchedulerMixin,
) -> torch.Tensor:
    """
    Calculates the predicted original sample (x_hat_0) from the noisy sample (xt)
    and the predicted noise (eps_theta).
    
    Formula: x_hat_0 = (xt - sqrt(1-alpha_t) * eps_theta) / sqrt(alpha_t)
    """
    # Get the scheduler constants for the current timestep
    alpha_prod_t = scheduler.alphas_cumprod[timestep_t]
    
    # Perform the calculation
    predicted_x0 = (xt_latents - (1 - alpha_prod_t).sqrt() * eps_theta) / alpha_prod_t.sqrt()
    return predicted_x0

def statistical_matching_loss(eps_pred: torch.Tensor) -> torch.Tensor:
    """
    Calculates a loss based on how well the statistics of the predicted noise
    match the target statistics of a standard normal distribution (mean=0, std=1).
    """
    # To calculate the statistic for each item, we reduce over the
    # channel, height, and width dimensions (1, 2, and 3).
    mean_pred_per_item = eps_pred.mean(dim=(1, 2, 3))
    std_pred_per_item = eps_pred.std(dim=(1, 2, 3))
    
    # Calculate the squared error from the target values (0 for mean, 1 for std)
    loss_mean = mean_pred_per_item**2
    loss_std = (std_pred_per_item - 1)**2
    
    # Return two tensors of shape (B,)
    return loss_mean, loss_std

def get_latent_statistics(tensor: torch.Tensor) -> dict:
    """
    Calculates a detailed statistical fingerprint of a latent tensor.
    This is used to diagnose deviations from the target N(0, I) distribution.
    """
    # Ensure tensor is on CPU for some calculations and in float32 for precision
    tensor_fp32 = tensor.detach().to(torch.float32)
    
    stats = {}
    
    # --- Level 1: The Basics ---
    stats['mean'] = tensor_fp32.mean().item()
    stats['std'] = tensor_fp32.std().item()

    # --- Level 2: The "Rainbow" Check (Per-Channel Variance) ---
    # Are some channels consistently higher-energy than others?
    # Dims are (batch, channel, height, width), so we reduce over batch, h, w
    per_channel_std = torch.std(tensor_fp32, dim=[0, 2, 3])
    for i, std in enumerate(per_channel_std):
        stats[f'std_ch{i}'] = std.item()

    # --- Level 3: The "Crunchy" Check (Kurtosis) ---
    # Kurtosis measures the "tailedness" or "peakedness" of the distribution.
    # A standard normal distribution has a kurtosis of 3.0.
    # Higher values indicate a "spiky" or "crunchy" distribution with many outliers.
    n = tensor_fp32.numel()
    mean = stats['mean']
    std = stats['std']
    # Manual Kurtosis calculation: E[(X - μ)⁴] / σ⁴
    kurt = (torch.sum((tensor_fp32 - mean)**4) / n) / (std**4)
    stats['kurtosis'] = kurt.item()

    # --- Level 4: The "Sharpness" Check (Outlier Magnitude) ---
    # How extreme are the brightest and darkest "pixels" in the latent?
    # For N(0,I), values are rarely outside [-4, 4].
    stats['max_val'] = tensor_fp32.max().item()
    stats['min_val'] = tensor_fp32.min().item()
    
    return stats

def rescale_noise_cfg(
    noise_cfg: torch.FloatTensor, noise_pred_text, guidance_rescale=0.0
):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
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

# =========================================================================
# VVVVVVVVVVVVVVVVVV NEW, CLEAN NON-CFG HELPER FUNCTIONS VVVVVVVVVVVVVVVVVV
# =========================================================================

def prepare_non_cfg_embeddings(
    positive_text_embeds: torch.FloatTensor,
    positive_pooled_embeds: torch.FloatTensor,
    neutral_text_embeds: torch.FloatTensor,
    neutral_pooled_embeds: torch.FloatTensor,
    batch_size_high: int,
    batch_size_low: int
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    """
    Prepares separate, non-CFG text and pooled embeddings for a mixed batch.
    This version is decoupled from PromptEmbedsPair and accepts raw tensors.
    """
    # High pass uses the 'positive' prompt
    text_embeds_high = positive_text_embeds.repeat(batch_size_high, 1, 1)
    pooled_embeds_high = positive_pooled_embeds.repeat(batch_size_high, 1)

    # Low pass uses the 'neutral' prompt
    text_embeds_low = neutral_text_embeds.repeat(batch_size_low, 1, 1)
    pooled_embeds_low = neutral_pooled_embeds.repeat(batch_size_low, 1)
    
    # Combine into a single batch
    combined_text_embeds = torch.cat([text_embeds_high, text_embeds_low], dim=0)
    combined_pooled_embeds = torch.cat([pooled_embeds_high, pooled_embeds_low], dim=0)
    
    return combined_text_embeds, combined_pooled_embeds

# To be placed in train_util.py

def sdxl_condnet_batchjoin(
    noisy_latents: torch.Tensor,
    text_embeddings: torch.Tensor,
    pooled_embeddings: torch.Tensor,
    time_ids: torch.Tensor,
    scheduler: "SchedulerMixin",
    timestep: int
) -> tuple[list, dict]:
    """
    Assembles the exact *args and **kwargs required for the SDXL UNet's
    forward pass in a non-CFG, batched training context.
    
    This function is the single point of truth for the UNet's API contract.
    """
    total_batch_size = noisy_latents.shape[0]
    
    # --- 1. Prepare Positional Args for `unet.forward()` ---
    scaled_latents = scheduler.scale_model_input(noisy_latents, timestep)
    timestep_tensor = torch.tensor([timestep] * total_batch_size, device=noisy_latents.device)
    unet_args = [scaled_latents, timestep_tensor]
    
    # --- 2. Prepare Keyword Args for `unet.forward()` ---
    added_cond_kwargs = {
        "text_embeds": pooled_embeddings,
        "time_ids": time_ids
    }
    unet_kwargs = {
        "encoder_hidden_states": text_embeddings,
        "added_cond_kwargs": added_cond_kwargs
    }
    # --- 3. Return the final structure for splatting ---
    return (unet_args, unet_kwargs)

def predict_noise_non_cfg(
    unet: nn.Module,
    scheduler: nn.Module,
    timestep: int,
    latents: torch.FloatTensor,
    text_embeddings: torch.FloatTensor,
    pooled_embeddings: torch.FloatTensor,
    time_ids: torch.FloatTensor,
) -> torch.FloatTensor:
    """
    Performs a single, direct U-Net forward pass without any CFG logic.
    Expects all input tensors to have the same batch dimension.
    """
    latent_model_input = scheduler.scale_model_input(latents, timestep)

    added_cond_kwargs = {
        "text_embeds": pooled_embeddings,
        "time_ids": time_ids,
    }

    noise_pred = unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=text_embeddings,
        added_cond_kwargs=added_cond_kwargs,
    ).sample
    
    return noise_pred
 
def predict_noise_xl(
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    timestep: int,  # 現在のタイムステップ
    latents: torch.FloatTensor,
    text_embeddings: torch.FloatTensor,  # uncond な text embed と cond な text embed を結合したもの
    add_text_embeddings: torch.FloatTensor,  # pooled なやつ
    add_time_ids: torch.FloatTensor,
    guidance_scale=7.5,
    guidance_rescale=0.7,
) -> torch.FloatTensor:
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)
    #print(f"concatenated latents shape:{latent_model_input.shape}")

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

    added_cond_kwargs = {
        "text_embeds": add_text_embeddings,
        "time_ids": add_time_ids,
    }

    # predict the noise residual
    noise_pred = unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=text_embeddings,
        added_cond_kwargs=added_cond_kwargs,
    ).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    guided_target = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )

    # https://github.com/huggingface/diffusers/blob/7a91ea6c2b53f94da930a61ed571364022b21044/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L775
    #noise_pred = rescale_noise_cfg(
    #    noise_pred, noise_pred_text, guidance_rescale=guidance_rescale
    #)

    return guided_target


@torch.no_grad()
def diffusion_xl(
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    latents: torch.FloatTensor,  # ただのノイズだけのlatents
    text_embeddings: tuple[torch.FloatTensor, torch.FloatTensor],
    add_text_embeddings: torch.FloatTensor,  # pooled なやつ
    add_time_ids: torch.FloatTensor,
    guidance_scale: float = 1.0,
    total_timesteps: int = 1000,
    start_timesteps=0,
):
    # latents_steps = []

    for timestep in tqdm(scheduler.timesteps[start_timesteps:total_timesteps]):
        noise_pred = predict_noise_xl(
            unet,
            scheduler,
            timestep,
            latents,
            text_embeddings,
            add_text_embeddings,
            add_time_ids,
            guidance_scale=guidance_scale,
            guidance_rescale=0.7,
        )

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    # return latents_steps
    return latents


# for XL
def get_add_time_ids(
    height: int,
    width: int,
    dynamic_crops: bool = False,
    dtype: torch.dtype = torch.float32,
):
    if dynamic_crops:
        # random float scale between 1 and 3
        random_scale = torch.rand(1).item() * 2 + 1
        original_size = (int(height * random_scale), int(width * random_scale))
        # random position
        crops_coords_top_left = (
            torch.randint(0, original_size[0] - height, (1,)).item(),
            torch.randint(0, original_size[1] - width, (1,)).item(),
        )
        target_size = (height, width)
    else:
        original_size = (height, width)
        crops_coords_top_left = (0, 0)
        target_size = (height, width)

    # this is expected as 6
    add_time_ids = list(original_size + crops_coords_top_left + target_size)

    # this is expected as 2816
    passed_add_embed_dim = (
        UNET_ATTENTION_TIME_EMBED_DIM * len(add_time_ids)  # 256 * 6
        + TEXT_ENCODER_2_PROJECTION_DIM  # + 1280
    )
    if passed_add_embed_dim != UNET_PROJECTION_CLASS_EMBEDDING_INPUT_DIM:
        raise ValueError(
            f"Model expects an added time embedding vector of length {UNET_PROJECTION_CLASS_EMBEDDING_INPUT_DIM}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
        )

    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    return add_time_ids

def batch_add_time_ids(
    original_sizes: torch.Tensor, # Shape: (B, N, 2)
    crop_coords: torch.Tensor,    # Shape: (B, N, 4)
    target_sizes: torch.Tensor,   # Shape: (B, N, 2)
    dtype: torch.dtype,
    device: torch.device
) -> torch.Tensor:
    """
    Creates the SDXL time conditioning vectors for an entire batch.
    Vectorized for efficiency.
    """
    batch_size, n_tuple = original_sizes.shape[:2]
    total_batch_size = batch_size * n_tuple

    # The crop_coords tensor is (left, top, right, bottom). We need (top, left).
    crop_coords_top_left = crop_coords[:, :, [1, 0]] # Re-order to (top, left)

    # Concatenate all components along a new final dimension
    # Shapes: (B,N,2) + (B,N,2) + (B,N,2) -> (B,N,6)
    combined_ids = torch.cat([original_sizes, crop_coords_top_left, target_sizes], dim=2)
    
    # Flatten to the final shape required by the U-Net
    # Shape: (B,N,6) -> (B*N, 6)
    flat_time_ids = combined_ids.view(total_batch_size, 6)
    
    return flat_time_ids.to(device, dtype=dtype)

def get_optimizer(name: str):
    name = name.lower()

    if name.startswith("dadapt"):
        import dadaptation

        if name == "dadaptadam":
            return dadaptation.DAdaptAdam
        elif name == "dadaptlion":
            return dadaptation.DAdaptLion
        else:
            raise ValueError("DAdapt optimizer must be dadaptadam or dadaptlion")

    elif name.endswith("8bit"):  # 検証してない
        import bitsandbytes as bnb

        if name == "adam8bit":
            return bnb.optim.Adam8bit
        elif name == "lion8bit":
            return bnb.optim.Lion8bit
        else:
            raise ValueError("8bit optimizer must be adam8bit or lion8bit")

    else:
        if name == "adam":
            return torch.optim.Adam
        elif name == "adamw":
            return torch.optim.AdamW
        elif name == "lion":
            from lion_pytorch import Lion

            return Lion
        elif name == "prodigy":
            import prodigyopt
            
            return prodigyopt.Prodigy
        elif name == "gluondist":
            from gluon_experiment import GlazyGloptimizer
            return GlazyGloptimizer
        else:
            raise ValueError("Optimizer must be adam, adamw, lion or Prodigy")


def get_lr_scheduler(
    name: Optional[str],
    optimizer: torch.optim.Optimizer,
    max_iterations: Optional[int],
    lr_min: Optional[float],
    **kwargs,
):
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_iterations, eta_min=lr_min, **kwargs
        )
    elif name == "cosine_with_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max_iterations // 10, T_mult=2, eta_min=lr_min, **kwargs
        )
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max_iterations // 100, gamma=0.999, **kwargs
        )
    elif name == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, **kwargs)
    elif name == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer, factor=0.5, total_iters=max_iterations // 100, **kwargs
        )
    else:
        raise ValueError(
            "Scheduler must be cosine, cosine_with_restarts, step, linear or constant"
        )


def get_random_resolution_in_bucket(bucket_resolution: int = 512) -> tuple[int, int]:
    max_resolution = bucket_resolution
    min_resolution = bucket_resolution // 2

    step = 64

    min_step = min_resolution // step
    max_step = max_resolution // step

    height = torch.randint(min_step, max_step, (1,)).item() * step
    width = torch.randint(min_step, max_step, (1,)).item() * step

    return height, width

def broadcast_prompts_to_n_tuple(
    positive_text_embeds: torch.FloatTensor,
    positive_pooled_embeds: torch.FloatTensor,
    neutral_text_embeds: torch.FloatTensor,
    neutral_pooled_embeds: torch.FloatTensor,
    scales: torch.Tensor
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    """
    Broadcasts a binary (high/low) prompt pair to an N-tuple of scales
    based on a set of logical rules.

    Args:
        positive_text_embeds (torch.FloatTensor): The 'high' text embeddings, shape (1, 77, D).
        positive_pooled_embeds (torch.FloatTensor): The 'high' pooled embeddings, shape (1, P).
        neutral_text_embeds (torch.FloatTensor): The 'low' text embeddings, shape (1, 77, D).
        neutral_pooled_embeds (torch.FloatTensor): The 'low' pooled embeddings, shape (1, P).
        scales (torch.Tensor): The tensor of scales for the batch, shape (BATCH_SIZE, N_tuple).

    Returns:
        A tuple of (combined_text_embeds, combined_pooled_embeds) ready for the U-Net.
    """
    batch_size, n_tuple = scales.shape
    
    # --- 1. Determine the prompt labels for each scale in the batch ---
    
    # Find min/max values and indices for each tuple in the batch
    min_scales, min_indices = torch.min(scales, dim=1)
    max_scales, max_indices = torch.max(scales, dim=1)
    
    # Calculate the mean of the edges for each tuple
    # Unsqueeze to make it broadcastable: shape (BATCH_SIZE, 1)
    cardinal_edge_mean = ((min_scales + max_scales) / 2).unsqueeze(1)
    
    # Apply Rule 3: All scales >= mean are 'high' (1), others are 'low' (0)
    # This creates a boolean mask of shape (BATCH_SIZE, N_tuple)
    labels = (scales >= cardinal_edge_mean).long() # Convert boolean to long (0s and 1s)

    # Apply Rules 1 & 2: Explicitly set the lowest to 'low' and highest to 'high'
    # This ensures the absolute edges always get the correct label.
    # We use a one-hot encoding trick to create indices for scatter_
    row_indices = torch.arange(batch_size).unsqueeze(1)
    labels[row_indices, min_indices.unsqueeze(1)] = 0 # Rule 1: Set min to 'low'
    labels[row_indices, max_indices.unsqueeze(1)] = 1 # Rule 2: Set max to 'high'
    
    # --- 2. Assemble the embedding tensors using the labels ---

    # Stack the base high/low embeddings for easy indexing
    # Shape: (2, 77, D) and (2, P)
    base_text_embeds = torch.cat([neutral_text_embeds, positive_text_embeds], dim=0)
    base_pooled_embeds = torch.cat([neutral_pooled_embeds, positive_pooled_embeds], dim=0)

    # Flatten the labels to create a 1D index tensor
    # Shape: (BATCH_SIZE * N_tuple)
    flat_labels = labels.view(-1)
    
    # Use the labels to gather the correct embeddings
    # This is a highly efficient way to build the final tensors
    combined_text_embeds = base_text_embeds[flat_labels]
    combined_pooled_embeds = base_pooled_embeds[flat_labels]
    
    return combined_text_embeds, combined_pooled_embeds