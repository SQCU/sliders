import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DDPMScheduler
import torch
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection

DIFFUSERS_CACHE_DIR = None # if you want to change the cache dir, change this

def load_models(config, device, weight_dtype):
    print(f"Loading models from {config.pretrained_model.name_or_path} to device: {device} with dtype: {weight_dtype}")

    # Load the pipeline from the local .safetensors file
    if "XL" in config.pretrained_model.name_or_path.upper():
        print("Detected SDXL model.")
        pipe = StableDiffusionXLPipeline.from_single_file(
            config.pretrained_model.name_or_path,
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
        )
        unet = pipe.unet.to(device)
        vae = pipe.vae.to(device)
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder.to(device)
        tokenizer_2 = pipe.tokenizer_2
        text_encoder_2 = pipe.text_encoder_2.to(device)
        noise_scheduler = DDPMScheduler() # Initialize directly for single file
    else:
        print("Detected SD1.5 model.")
        pipe = StableDiffusionPipeline.from_single_file(
            config.pretrained_model.name_or_path,
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
        )
        unet = pipe.unet.to(device)
        vae = pipe.vae.to(device)
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder.to(device)
        tokenizer_2 = None
        text_encoder_2 = None
        noise_scheduler = DDPMScheduler() # Initialize directly for single file

    # Set requires_grad to False and eval mode for inference components
    unet.requires_grad_(False).eval()
    vae.requires_grad_(False).eval()
    text_encoder.requires_grad_(False).eval()
    text_encoders = [text_encoder]
    tokenizers = [tokenizer]
    if text_encoder_2 is not None:
        text_encoders.append(text_encoder_2)
        tokenizers.append(tokenizer_2)
        if len(text_encoders) == 2: text_encoders[1].pad_token_id = 0

    return vae, unet, tokenizers, text_encoders, noise_scheduler
