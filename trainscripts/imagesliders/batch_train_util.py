#batch_train_util.py
import torch
from typing import Tuple, Union, Literal, List, Dict, Any
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from diffusers import UNet2DConditionModel, SchedulerMixin

UNET_IN_CHANNELS = 4  # Stable Diffusion の in_channels は 4 で固定。XLも同じ。
VAE_SCALE_FACTOR = 8  # 2 ** (len(vae.config.block_out_channels) - 1) = 8

UNET_ATTENTION_TIME_EMBED_DIM = 256  # XL
TEXT_ENCODER_2_PROJECTION_DIM = 1280
UNET_PROJECTION_CLASS_EMBEDDING_INPUT_DIM = 2816

SDXL_TEXT_ENCODER_TYPE = Union[CLIPTextModel, CLIPTextModelWithProjection]

def text_tokenize(
    tokenizer: CLIPTokenizer,
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
):
    prompt_embeds = text_encoder(
        tokens.to(text_encoder.device), output_hidden_states=True
    )
    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]  # always penultimate layer

    return prompt_embeds, pooled_prompt_embeds

def encode_prompts_xl(
    tokenizers: list[CLIPTokenizer],
    text_encoders: list[SDXL_TEXT_ENCODER_TYPE],
    prompts: list[str],
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    text_embeds_list = []
    pooled_text_embeds = None

    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        text_tokens_input_ids = text_tokenize(tokenizer, prompts)
        text_embeds, pooled_text_embeds = text_encode_xl(
            text_encoder, text_tokens_input_ids,
        )

        text_embeds_list.append(text_embeds)

    return torch.concat(text_embeds_list, dim=-1), pooled_text_embeds

def create_batched_prompt_embeddings(
    tokenizers: list[CLIPTokenizer],
    text_encoders: list[SDXL_TEXT_ENCODER_TYPE],
    prompts: dict,
):
    positive_prompts = [prompts['positive']]
    unconditional_prompts = [prompts['unconditional']]
    neutral_prompts = [prompts['neutral']]

    positive_text_embeds, positive_pooled_embeds = encode_prompts_xl(
        tokenizers, text_encoders, positive_prompts,
    )
    unconditional_text_embeds, unconditional_pooled_embeds = encode_prompts_xl(
        tokenizers, text_encoders, unconditional_prompts,
    )
    neutral_text_embeds, neutral_pooled_embeds = encode_prompts_xl(
        tokenizers, text_encoders, neutral_prompts,
    )

    text_embeddings_for_noise_pred = torch.cat([
        positive_text_embeds,
        unconditional_text_embeds,
        neutral_text_embeds,
    ], dim=0)
    pooled_embeds_for_noise_pred = torch.cat([
        positive_pooled_embeds,
        unconditional_pooled_embeds,
        neutral_pooled_embeds,
    ], dim=0)

    return text_embeddings_for_noise_pred, pooled_embeds_for_noise_pred

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
    #debug print no longer needed.
    #print(f"Shape of add_time_ids in get_add_time_ids: {add_time_ids.shape}")
    return add_time_ids

def get_random_noise(
    batch_size: int, height: int, width: int, generator: torch.Generator = None
) -> torch.Tensor:
    return torch.randn(
        (
            batch_size,
            UNET_IN_CHANNELS,
            height // VAE_SCALE_FACTOR,  # 縦と横これであってるのかわからないけど、どっちにしろ大きな問題は発生しないのでこれでいいや
            width // VAE_SCALE_FACTOR,   # 更新：長さと幅は間違いなく正しいです。ああ、これは大変な問題でした。
        ),
        generator=generator,
        device="cpu",
    )

def nocfg_predict_noise_xl(
    unet: torch.nn.Module,
    scheduler,
    timestep: torch.Tensor,
    latents: torch.FloatTensor,
    text_embeddings: torch.FloatTensor,
    add_text_embeddings: torch.FloatTensor,
    add_time_ids: torch.FloatTensor,
    guidance_scale = None,
) -> torch.FloatTensor:
    #device = unet.device
    latent_model_input = latents
    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)
    if guidance_scale is not None:
        print("you seem to have passed a guidance scale:{guidance_scale}. this function deprecates that!\ncontinuing...\n")
    added_cond_kwargs = {
        "text_embeds": add_text_embeddings,
        "time_ids": add_time_ids.to(torch.float32),
    }

    noise_pred = unet(
        latent_model_input.to(unet.dtype),
        timestep,
        encoder_hidden_states=text_embeddings.to(unet.dtype),
        added_cond_kwargs=added_cond_kwargs
    ).sample
    return noise_pred

def batched_predict_noise_xl(
    unet: torch.nn.Module,
    scheduler,
    timestep: torch.Tensor,
    latents: torch.FloatTensor,
    text_embeddings: torch.FloatTensor,
    add_text_embeddings: torch.FloatTensor,
    add_time_ids: torch.FloatTensor,
    guidance_scale: float = 7.5,
) -> torch.FloatTensor:
    # Call the new no-CFG prediction function
    noise_pred = nocfg_predict_noise_xl(
        unet,
        scheduler,
        timestep,
        latents,
        text_embeddings,
        add_text_embeddings,
        add_time_ids,
    )

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