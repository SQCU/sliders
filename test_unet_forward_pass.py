import torch
from pathlib import Path
import sys
import os
import argparse
import yaml
import datetime
from typing import Dict, Any, Tuple

# Add the parent directory of trainscripts to sys.path for imports
sys.path.append(str(Path(__file__).parent))

# Minimal necessary components from batch_config_util and batch_model_util
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = AttrDict(value)

def load_config_from_yaml(filepath):
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

def parse_precision(precision: str) -> torch.dtype:
    if precision == "fp32" or precision == "float32":
        return torch.float32
    elif precision == "fp16" or precision == "float16":
        return torch.float16
    elif precision == "bf16" or precision == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Invalid precision type: {precision}")

from diffusers import DDPMScheduler, UNet2DConditionModel, AutoencoderKL, StableDiffusionXLPipeline
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from diffusers import DDIMScheduler, LMSDiscreteScheduler, EulerAncestralDiscreteScheduler, SchedulerMixin

# Re-implement load_models and create_noise_scheduler to load actual models
def load_models(config, device, weight_dtype):
    print(f"Loading models from {config.pretrained_model.name_or_path} to device: {device} with dtype: {weight_dtype}")

    pipe = StableDiffusionXLPipeline.from_single_file(
        config.pretrained_model.name_or_path,
        torch_dtype=weight_dtype,
    )

    unet = pipe.unet.to(device, dtype=weight_dtype)
    unet.requires_grad_(False).eval()

    vae = pipe.vae.to(device, dtype=weight_dtype)
    vae.requires_grad_(False).eval()

    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
    text_encoder_one = pipe.text_encoder.to(device, dtype=weight_dtype)
    text_encoder_one.requires_grad_(False).eval()
    text_encoder_two = pipe.text_encoder_2.to(device, dtype=weight_dtype)
    text_encoder_two.requires_grad_(False).eval()
    text_encoders = [text_encoder_one, text_encoder_two]

    del pipe # Crucially, delete the pipe to free up memory
    torch.cuda.empty_cache()

    return vae, unet, tokenizers, text_encoders

def create_noise_scheduler(scheduler_name: str) -> SchedulerMixin:
    name = scheduler_name.lower().replace(" ", "_")
    if name == "ddim":
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
            prediction_type="epsilon",
        )
    elif name == "ddpm":
        scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
            prediction_type="epsilon",
        )
    elif name == "lms":
        scheduler = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            prediction_type="epsilon",
        )
    elif name == "euler_a":
        scheduler = EulerAncestralDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            prediction_type="epsilon",
        )
    else:
        raise ValueError(f"Unknown scheduler name: {name}")
    return scheduler

def get_optimizer(name: str):
    return None

def envsetup(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = parse_precision(config.train.precision)
    vae, unet, tokenizers, text_encoders = load_models(config, device, weight_dtype)
    noise_scheduler = create_noise_scheduler(config.train.noise_scheduler)
    
    environment = {
        "unet": unet,
        "vae": vae,
        "noise_scheduler": noise_scheduler,
        "tokenizers": tokenizers,
        "text_encoders": text_encoders,
        "device": device,
        "weight_dtype": weight_dtype,
        "config": config,
        "optimizer": get_optimizer(config.train.optimizer),
    }
    return environment

def setup_logging(runname="test_unet_forward_pass"):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_filename = os.path.join(log_dir, runname+f"_{timestamp}.log")

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    log_file = open(log_filename, "w")
    sys.stdout = log_file
    sys.stderr = log_file

    print(f"Logging output to {log_filename}")

    return log_filename, original_stdout, original_stderr, log_file

# --- Copied from batch_train_util.py for self-containment ---
UNET_ATTENTION_TIME_EMBED_DIM = 256  # XL
TEXT_ENCODER_2_PROJECTION_DIM = 1280
UNET_PROJECTION_CLASS_EMBEDDING_INPUT_DIM = 2816

def batched_predict_noise_xl(
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
        "time_ids": add_time_ids.to(torch.float32),
    }

    noise_pred = unet(
        latent_model_input.to(unet.dtype),
        timestep,   #don't cast to network dtype, these need to be torch.long
        encoder_hidden_states=text_embeddings.to(unet.dtype),
        added_cond_kwargs=added_cond_kwargs
    ).sample
    
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    guided_target = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )
    return guided_target
# --- End copied functions ---

def test_unet_forward_pass():
    # i am writing this documentation to understand the flow of data and flow of control in the executed program.
    # i am not editing control flow or data flow. i am here to understand and judge without controlling what i read.
    # This test specifically tracks VRAM usage during the UNet forward pass.

    # Simulate command-line arguments for config_io
    original_argv = sys.argv
    sys.argv = ['test_script.py', '--batchtrainconfig', 'trainscripts/imagesliders/data/batch_config.yaml']
    
    # Load main config
    main_config = AttrDict(load_config_from_yaml(sys.argv[2]))
    
    # Load and merge model config
    model_config_path = main_config.model_config.refpath
    model_config = AttrDict(load_config_from_yaml(model_config_path))
    
    # Merge model_config into main_config (mimicking original script's behavior)
    main_config.update(model_config)
    config = main_config

    sys.argv = original_argv # Restore original argv

    environment = envsetup(config)

    unet = environment["unet"]
    noise_scheduler = environment["noise_scheduler"]
    device = environment["device"]
    weight_dtype = environment["weight_dtype"]
    
    # Extract seed from the captured directory name
    capture_dir_name = "train_step_377077264765000" # Example captured directory
    seed_str = capture_dir_name.split('_')[-1]
    seed = int(seed_str)

    capture_dir = Path(f"F:/dox/ai/gemmy/sliders/state_capture/{capture_dir_name}")

    # Clear CUDA cache and log initial memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"VRAM (initial): {torch.cuda.memory_allocated() / (1024**2):.2f} MB")

    # Load captured inputs for UNet forward pass
    latents_cfg = torch.load(capture_dir / "02_latents_cfg.pt").to(device, dtype=weight_dtype)
    text_embeddings_cfg = torch.load(capture_dir / "03_text_embeddings_cfg.pt").to(device, dtype=weight_dtype)
    pooled_embeds_cfg = torch.load(capture_dir / "04_pooled_embeds_cfg.pt").to(device, dtype=weight_dtype)
    add_time_ids_cfg = torch.load(capture_dir / "05_add_time_ids_cfg.pt").to(device, dtype=torch.float32) # Keep float32
    unet_timesteps_cfg = torch.load(capture_dir / "06_unet_timesteps_cfg.pt").to(device) # Timesteps are long

    if torch.cuda.is_available():
        print(f"VRAM (after loading and moving captured data): {torch.cuda.memory_allocated() / (1024**2):.2f} MB")

    # Get guidance_scale from the initial batch (assuming it's consistent)
    initial_batch = torch.load(capture_dir / "00_initial_batch.pt")
    guidance_scale = initial_batch["guidance_scale"]

    if torch.cuda.is_available():
        print(f"VRAM (before UNet forward pass): {torch.cuda.memory_allocated() / (1024**2):.2f} MB")

    # Perform UNet forward pass
    with torch.no_grad(): # UNet forward pass is typically done without grad for inference/evaluation
        predicted_noise = batched_predict_noise_xl(
            unet,
            noise_scheduler,
            unet_timesteps_cfg,
            latents_cfg,
            text_embeddings_cfg,
            pooled_embeds_cfg,
            add_time_ids_cfg,
            guidance_scale=guidance_scale,
        )

    if torch.cuda.is_available():
        print(f"VRAM (after UNet forward pass): {torch.cuda.memory_allocated() / (1024**2):.2f} MB")

    # Assertions for output shape and type (we don't have a captured predicted_noise)
    print(f"Predicted noise shape: {predicted_noise.shape}")
    print(f"Latents CFG shape: {latents_cfg.shape}")
    assert predicted_noise.shape == latents_cfg.shape
    assert predicted_noise.dtype == weight_dtype
    assert predicted_noise.device == device

    print("Successfully simulated UNet forward pass. Review VRAM logs for memory behavior.")

if __name__ == "__main__":
    log_file_path, orig_stdout, orig_stderr, log_file = None, None, None, None
    try:
        log_file_path, orig_stdout, orig_stderr, log_file = setup_logging()
        print(f"--- Starting UNet Forward Pass Test ---")
        print(f"All output will be redirected to: {log_file_path}")
        test_unet_forward_pass()
    except Exception as e:
        import traceback
        print("--- EXCEPTION OCCURRED ---", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        if orig_stderr:
            print(f"\n--- EXCEPTION OCCURRED ---", file=orig_stderr)
            print(f"An error occurred. Check the log file for details: {log_file_path}", file=orig_stderr)
            traceback.print_exc(file=orig_stderr)
        raise
    finally:
        if log_file:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            log_file.close()
            print(f"--- Script finished. Log saved to {log_file_path} ---")
