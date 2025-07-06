# This file contains documentation and code for batching compatibility of training functions.
# It is designed to be modular, especially for guidance rules, allowing for easy swapping of CFG implementations.

import torch
from diffusers.utils.torch_utils import randn_tensor
from typing import Tuple, Union
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from diffusers import UNet2DConditionModel, SchedulerMixin

UNET_ATTENTION_TIME_EMBED_DIM = 256  # XL
TEXT_ENCODER_2_PROJECTION_DIM = 1280
UNET_PROJECTION_CLASS_EMBEDDING_INPUT_DIM = 2816

SDXL_TEXT_ENCODER_TYPE = Union[CLIPTextModel, CLIPTextModelWithProjection]

# --- Analysis of train_util.get_noisy_image ---
# Function Operand: imgs (single Image.Image or list of Image.Image), vae, generator, unet, scheduler, total_timesteps, start_timesteps.
# Can be batched? Yes. The function already handles a list of Image.Image and concatenates them into image_batch.
# vae.encode can handle batched inputs. randn_tensor can generate batched noise. scheduler.add_noise can handle batched latents and noise.
# Needs to be batched? Yes. In superfunctional_train_step, get_noisy_image is called twice. To truly batch the training step, these two calls should ideally be combined into a single call that processes both high and low images in a single batch.

def get_batched_noisy_images(
    img_batch: torch.Tensor,
    vae: torch.nn.Module,
    generator: torch.Generator,
    noise_scheduler: SchedulerMixin,
    config,
    total_timesteps:int, # = 1000,
    start_timesteps:int, #=0,
    device: torch.device,
    weight_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combines the process of getting noisy images for high and low scales into a single batched operation.
    """
    with torch.no_grad():
        latents = vae.encode(img_batch.to(dtype=weight_dtype)).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
   
    # Generate noise for the combined batch
    noise_shape = latents.shape
    noise = randn_tensor(noise_shape, generator=generator, device=device)

    #timestep interval logic from upstream
    time_ = total_timesteps
    timestep = noise_scheduler.timesteps[time_:time_+1]

    # Add noise to the combined latents
    noisy_latents = noise_scheduler.add_noise(latents, noise, timestep)

    #return without splitting and unsplitting several times for no reason!
    return noisy_latents, noise

# --- Modular functions for predict_noise_xl and guidance ---

def batched_predict_noise_xl_modular(
    unet: torch.nn.Module,
    scheduler,
    timestep: torch.Tensor, # Changed to torch.Tensor
    latents: torch.FloatTensor,
    text_embeddings: torch.FloatTensor, # uncond な text embed と cond な text embed を結合したもの
    add_text_embeddings: torch.FloatTensor, # pooled なやつ
    add_time_ids: torch.FloatTensor,
    guidance_scale: float = 7.5,
) -> torch.FloatTensor:
    """
    A modular version of predict_noise_xl, orchestrating the sub-components.
    This function can be easily modified to swap out different guidance strategies.
    """

    device = unet.device
    latent_model_input = latents
    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

    added_cond_kwargs = {
        "text_embeds": add_text_embeddings,
        "time_ids": add_time_ids,
    }

    #debugging logging block:
    print(f"latent_model_input shape: {latent_model_input.shape}")
    print(f"timestep shape: {timestep.shape}")
    print(f"text_embeddings shape: {text_embeddings.shape}")
    print(f"add_text_embeddings shape: {add_text_embeddings.shape}")
    print(f"add_time_ids shape: {add_time_ids.shape}")


    noise_pred = unet(
        latent_model_input.to(unet.dtype),
        timestep.to(unet.dtype),
        text_embeddings.to(unet.dtype),
        added_cond_kwargs
    ).sample
    
    #expects latents to be pre-doubled, 
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    guided_target = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )
    guided_target = apply_cfg_guidance(noise_pred, guidance_scale)
    
    return guided_target

def simple_predict_noise_xl(
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    timestep: torch.Tensor,
    latents: torch.FloatTensor,
    text_embeddings: torch.FloatTensor,
    add_text_embeddings: torch.FloatTensor,
    add_time_ids: torch.FloatTensor,
    guidance_scale: float = 7.5,
) -> torch.FloatTensor:
    """
    Reproduces the core noise prediction logic of train_util.predict_noise_xl
    without the guidance_rescale part.
    """
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = latents

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

    #line950 of unet_2d_condition.py. read it and weep.
    added_cond_kwargs = {
        "text_embeds": add_text_embeddings,
        "time_ids": add_time_ids,
    }

    # predict the noise residual
    noise_pred = unet(
        latent_model_input.to(unet.dtype),
        torch.tensor([timestep], device=unet.device, dtype=unet.dtype).expand(latent_model_input.shape[0]),
        encoder_hidden_states=text_embeddings.to(unet.dtype),
        added_cond_kwargs=added_cond_kwargs,
    ).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    guided_target = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )

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
    #num_images_per_prompt: int = 1,
    #get rid of that horrible operand
):
    prompt_embeds = text_encoder(
        tokens.to(text_encoder.device), output_hidden_states=True
    )
    pooled_prompt_embeds = prompt_embeds[0]
    print(F"pre concatenation pooled prompt embeds of shape:{pooled_prompt_embeds.shape}")
    prompt_embeds = prompt_embeds.hidden_states[-2]  # always penultimate layer
    print(F"1/3pre concatenation prompt hidden states of shape:{prompt_embeds.shape}")
    #something weird is happening here.
    #i think it's part of the upstream batch code they *couldn't get working*.
    #bs_embed, seq_len, _ = prompt_embeds.shape
    #prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    #print(F"2/3pre concatenation prompt hidden states of shape:{prompt_embeds.shape}")
    #prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
    #print(F"3/3pre concatenation prompt hidden states of shape:{prompt_embeds.shape}")

    return prompt_embeds, pooled_prompt_embeds

def encode_prompts_xl(
    tokenizers: list[CLIPTokenizer],
    text_encoders: list[SDXL_TEXT_ENCODER_TYPE],
    prompts: list[str],
    #num_images_per_prompt: int = 1,
    #get rid of horrible thing
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    # text_encoder and text_encoder_2's penuultimate layer's output
    text_embeds_list = []
    pooled_text_embeds = None  # always text_encoder_2's pool

    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        text_tokens_input_ids = text_tokenize(tokenizer, prompts)
        text_embeds, pooled_text_embeds = text_encode_xl(
            text_encoder, text_tokens_input_ids, #num_images_per_prompt
        )

        text_embeds_list.append(text_embeds)

    print(F"pre susblock prompt hidden states of shape:{pooled_text_embeds.shape}")
    #suspicious block
    #something weird is happening here.
    #i think it's part of the upstream batch code they *couldn't get working*.
    #bs_embed = pooled_text_embeds.shape[0]
    #pooled_text_embeds = pooled_text_embeds.repeat(1, num_images_per_prompt).view(
    #    bs_embed * num_images_per_prompt, -1
    #)
    print(F"post susblock prompt hidden states of shape:{pooled_text_embeds.shape}")

    return torch.concat(text_embeds_list, dim=-1), pooled_text_embeds


def create_batched_prompt_embeddings(
    tokenizers: list[CLIPTokenizer],
    text_encoders: list[SDXL_TEXT_ENCODER_TYPE],
    prompts: dict,
    num_images_per_prompt: int = 1,
    #deprecated but must still accept it for now.
    #get rid of that horrible thing
):
    """
    Creates a batched prompt embedding tensor for use in the training loop.
    """
    positive_prompts = [prompts[0]['positive']]
    unconditional_prompts = [prompts[0]['unconditional']]
    neutral_prompts = [prompts[0]['neutral']]

    # Encode all prompts
    positive_text_embeds, positive_pooled_embeds = encode_prompts_xl(
        tokenizers, text_encoders, positive_prompts, #num_images_per_prompt
    )
    unconditional_text_embeds, unconditional_pooled_embeds = encode_prompts_xl(
        tokenizers, text_encoders, unconditional_prompts, #num_images_per_prompt
    )
    neutral_text_embeds, neutral_pooled_embeds = encode_prompts_xl(
        tokenizers, text_encoders, neutral_prompts, #num_images_per_prompt
    )

    # Concatenate for the UNet
    text_embeddings_for_noise_pred = torch.cat([
        positive_text_embeds,
        unconditional_text_embeds,
        neutral_text_embeds,
    ], dim=0)
    print(F"from batched_prompt_embeddings post concatenation prompt hidden states of shape:{text_embeddings_for_noise_pred.shape}")
    # Pooled embeds are also needed for XL
    pooled_embeds_for_noise_pred = torch.cat([
        positive_pooled_embeds,
        unconditional_pooled_embeds,
        neutral_pooled_embeds,
    ], dim=0)
    print(F"from batched_prompt_embeddings post concatenation pooled prompt embeds of shape:{pooled_embeds_for_noise_pred.shape}")

    return text_embeddings_for_noise_pred, pooled_embeds_for_noise_pred

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