import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DDPMScheduler, SchedulerMixin, DDIMScheduler, LMSDiscreteScheduler, EulerAncestralDiscreteScheduler, UNet2DConditionModel, AutoencoderKL
import torch
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from typing import Literal, Union

AVAILABLE_SCHEDULERS = Literal["ddim", "ddpm", "lms", "euler_a"]

SDXL_TEXT_ENCODER_TYPE = Union[CLIPTextModel, CLIPTextModelWithProjection]

DIFFUSERS_CACHE_DIR = None # if you want to change the cache dir, change this

def load_models(config, device, weight_dtype):
    print(f"Loading models from {config.pretrained_model.name_or_path} to device: {device} with dtype: {weight_dtype}")

    # Load the pipeline from the local .safetensors file
    pipe = StableDiffusionXLPipeline.from_single_file(
        config.pretrained_model.name_or_path,
        torch_dtype=weight_dtype,
        cache_dir=DIFFUSERS_CACHE_DIR,
    )

    unet = pipe.unet.to(device, weight_dtype)
    print(f"UNet time_embedding_dim: {unet.config.time_embedding_dim}")
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
    text_encoders = [pipe.text_encoder.to(device, weight_dtype), pipe.text_encoder_2.to(device, weight_dtype)]
    if len(text_encoders) == 2:
        text_encoders[1].pad_token_id = 0
    vae = pipe.vae.to(device, weight_dtype)
    del pipe
    #GET RID OF PIPE!!! 
    #you HAVE TO GET RID OF THE PIPE EVERY TIME!!!
    #if you do a 'blah = pipe.blah.to(device, dtype)' you DOUBLE LOAD THE MODEL,

    return vae, unet, tokenizers, text_encoders


def create_noise_scheduler(
    scheduler_name: AVAILABLE_SCHEDULERS = "ddpm",
    prediction_type: Literal["epsilon", "v_prediction"] = "epsilon",
) -> SchedulerMixin:
    # 正直、どれがいいのかわからない。元の実装だとDDIMとDDPMとLMSを選べたのだけど、どれがいいのかわからぬ。

    name = scheduler_name.lower().replace(" ", "_")
    if name == "ddim":
        # https://huggingface.co/docs/diffusers/v0.17.1/en/api/schedulers/ddim
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
            prediction_type=prediction_type,  # これでいいの？
        )
    elif name == "ddpm":
        # https://huggingface.co/docs/diffusers/v0.17.1/en/api/schedulers/ddpm
        scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
            prediction_type=prediction_type,
        )
    elif name == "lms":
        # https://huggingface.co/docs/diffusers/v0.17.1/en/api/schedulers/lms_discrete
        scheduler = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            prediction_type=prediction_type,
        )
    elif name == "euler_a":
        # https://huggingface.co/docs/diffusers/v0.17.1/en/api/schedulers/euler_ancestral
        scheduler = EulerAncestralDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            prediction_type=prediction_type,
        )
    else:
        raise ValueError(f"Unknown scheduler name: {name}")

    return scheduler

def load_diffusers_model_xl(
    pretrained_model_name_or_path: str,
    weight_dtype: torch.dtype = torch.float32,
) -> tuple[list[CLIPTokenizer], list[SDXL_TEXT_ENCODER_TYPE], UNet2DConditionModel,]:
    # returns tokenizer, tokenizer_2, text_encoder, text_encoder_2, unet

    tokenizers = [
        CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer",
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
        ),
        CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
            pad_token_id=0,  # same as open clip
        ),
    ]

    text_encoders = [
        CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
        ),
        CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
        ),
    ]

    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
        torch_dtype=weight_dtype,
        cache_dir=DIFFUSERS_CACHE_DIR,
    )
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
    return tokenizers, text_encoders, unet, vae