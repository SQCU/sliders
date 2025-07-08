import torch
from pathlib import Path
import sys
import os
import argparse
import yaml
import datetime
from typing import Dict, Any, Tuple
import functools


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = AttrDict(value)

def log_step(message):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print(f"STEP: {message}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

def log_vram(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_vram = torch.cuda.memory_allocated() / (1024**3)
            result = func(*args, **kwargs)
            final_vram = torch.cuda.memory_allocated() / (1024**3)
            print(f"VRAM before {func.__name__}: {initial_vram:.2f} GB, after: {final_vram:.2f} GB")
            return result
        else:
            return func(*args, **kwargs)
    return wrapper

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
from trainscripts.imagesliders.batch_train_util import nocfg_predict_noise_xl as batched_predict_noise_xl # Renamed for clarity
from trainscripts.imagesliders import batch_lora

# NEW: scale_n_tuple_loss function
def scale_n_tuple_loss(
    predicted_noise: torch.Tensor,
    target_noise: torch.Tensor,
    group_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Calculates a loss for scale-n-tuple groups.
    Assumes predicted_noise and target_noise are already CFG-processed and N_eff sized.
    """
    # Calculate MSE loss per-element, then reduce to a per-item scalar
    # This will have shape (N_eff,)
    loss_per_item = (predicted_noise - target_noise).pow(2).mean(dim=[-3, -2, -1])

    # Group losses by group_indices and sum them
    unique_groups = torch.unique(group_indices)
    summed_group_losses = []
    for group_id in unique_groups:
        group_mask = (group_indices == group_id)
        summed_group_losses.append(loss_per_item[group_mask].sum()) # Sum within each group

    if not summed_group_losses:
        return torch.tensor(0.0, device=predicted_noise.device, dtype=predicted_noise.dtype)

    # Average the summed group losses
    return torch.stack(summed_group_losses).mean()

@log_step("Loading models...")
@log_vram
def load_models(config, device, weight_dtype):
    print(f"Loading models from {config.pretrained_model.name_or_path} to device: {device} with dtype: {weight_dtype}")

    pipe = StableDiffusionXLPipeline.from_single_file(
        config.pretrained_model.name_or_path,
        torch_dtype=weight_dtype,
    )

    unet = pipe.unet.to(device, dtype=weight_dtype)
    unet.requires_grad_(False).eval()

    # Ensure config.other exists and enable gradient checkpointing
    if not hasattr(config, 'other'):
        config.other = AttrDict()
    config.other.gradient_checkpointing = True

    # Enable gradient checkpointing if configured
    if hasattr(config, 'other') and hasattr(config.other, 'gradient_checkpointing') and config.other.gradient_checkpointing:
        print("Enabling gradient checkpointing for UNet.")
        unet.enable_gradient_checkpointing()

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

@log_step("Creating noise scheduler...")
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

@log_step("Setting up environment...")
@log_vram
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

def setup_logging(runname="test_backpropagation"):
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

@log_step("Loading and preparing config...")
def load_and_prepare_config():
    original_argv = sys.argv
    sys.argv = ['test_script.py', '--batchtrainconfig', 'trainscripts/imagesliders/data/batch_config.yaml']
    main_config = AttrDict(load_config_from_yaml(sys.argv[2]))
    model_config_path = main_config.model_config.refpath
    model_config = AttrDict(load_config_from_yaml(model_config_path))
    main_config.update(model_config)
    config = main_config
    sys.argv = original_argv
    return config

def get_seed_from_capture_dir(capture_dir_name: str) -> int:
    return int(capture_dir_name.split('_')[-1])

@log_step("Initializing LoRA network...")
@log_vram
def initialize_lora_network(unet, config, device, weight_dtype):
    return batch_lora.BatchedLoRANetwork(
        unet=unet,
        rank=config.network.rank,
        alpha=config.network.alpha,
        train_method=config.network.training_method,
    ).to(device, dtype=weight_dtype)

@log_step("Loading initial data...")
@log_vram
def load_initial_data(capture_dir, device, weight_dtype):
    data = AttrDict()
    data.text_embeddings_cfg = torch.load(capture_dir / "03_text_embeddings_cfg.pt").to(device, dtype=weight_dtype)
    data.pooled_embeds_cfg = torch.load(capture_dir / "04_pooled_embeds_cfg.pt").to(device, dtype=weight_dtype)
    data.add_time_ids_cfg = torch.load(capture_dir / "05_add_time_ids_cfg.pt").to(device, dtype=torch.float32)

    initial_batch = torch.load(capture_dir / "00_initial_batch.pt")
    data.latents = initial_batch["latents"].to(device, dtype=weight_dtype)
    data.group_indices = initial_batch["pair_indices"].to(device)
    data.scales = initial_batch["scales"].to(device, dtype=weight_dtype)
    data.guidance_scale = initial_batch["guidance_scale"]
    return data

@log_step("Generating noise and preparing CFG...")
@log_vram
def generate_noise_and_prepare_cfg(noise_scheduler, config, device, seed, latents):
    with torch.no_grad():
        noise_scheduler.set_timesteps(config.train.max_denoising_steps, device=device)
        generator = torch.Generator(device=device).manual_seed(seed)
        
        timesteps_to = torch.randint(
            1, config.train.max_denoising_steps, (latents.shape[0],), device=device, generator=generator
        ).long()

        target_noise = torch.randn(latents.shape, device=latents.device, generator=generator)
        noisy_latents = noise_scheduler.add_noise(latents, target_noise, timesteps_to)

        latents_cfg = torch.cat([noisy_latents] * 2, dim=0)
        unet_timesteps_cfg = torch.cat([timesteps_to] * 2, dim=0)
    return AttrDict(target_noise=target_noise, latents_cfg=latents_cfg, unet_timesteps_cfg=unet_timesteps_cfg)

@log_step("Performing UNet forward pass with LoRA...")
@log_vram
def perform_forward_pass(unet, noise_scheduler, unet_timesteps_cfg, latents_cfg, text_embeddings_cfg, pooled_embeds_cfg, add_time_ids_cfg, guidance_scale):
    predicted_noise_raw = batched_predict_noise_xl(
        unet,
        noise_scheduler,
        unet_timesteps_cfg,
        latents_cfg,
        text_embeddings_cfg,
        pooled_embeds_cfg,
        add_time_ids_cfg,
        guidance_scale=guidance_scale,
    )
    predicted_noise_uncond, predicted_noise_text = predicted_noise_raw.chunk(2)
    predicted_noise = (predicted_noise_uncond + guidance_scale * (
        predicted_noise_text - predicted_noise_uncond
    )).to(unet.device)
    return predicted_noise

@log_step("Calculating loss...")
@log_vram
def calculate_loss(predicted_noise, target_noise, group_indices):
    return scale_n_tuple_loss(predicted_noise, target_noise, group_indices)

@log_step("Performing backpropagation...")
@log_vram
def perform_backpropagation(loss):
    loss.backward()

def test_backpropagation():
    config = load_and_prepare_config()
    environment = envsetup(config)

    unet = environment["unet"]
    noise_scheduler = environment["noise_scheduler"]
    device = environment["device"]
    weight_dtype = environment["weight_dtype"]
    
    capture_dir_name = "train_step_377077264765000"
    seed = get_seed_from_capture_dir(capture_dir_name)
    capture_dir = Path(f"F:/dox/ai/gemmy/sliders/state_capture/{capture_dir_name}")

    network = initialize_lora_network(unet, config, device, weight_dtype)

    data = load_initial_data(capture_dir, device, weight_dtype)

    print(f"Shape of latents: {data.latents.shape}, group_indices: {data.group_indices.shape}, scales: {data.scales.shape}")
    assert data.latents.shape[0] == data.group_indices.shape[0], f"Batch size mismatch: latents.shape[0]={data.latents.shape[0]}, group_indices.shape[0]={data.group_indices.shape[0]}"
    assert data.latents.shape[0] == data.scales.shape[0], f"Batch size mismatch: latents.shape[0]={data.latents.shape[0]}, scales.shape[0]={data.scales.shape[0]}"

    noise_data = generate_noise_and_prepare_cfg(noise_scheduler, config, device, seed, data.latents)

    batched_scales_cfg = torch.cat([data.scales] * 2, dim=0).unsqueeze(-1)
    network.set_lora_scales(batched_scales_cfg)

    with network:
        predicted_noise = perform_forward_pass(
            unet,
            noise_scheduler,
            noise_data.unet_timesteps_cfg,
            noise_data.latents_cfg,
            data.text_embeddings_cfg,
            data.pooled_embeds_cfg,
            data.add_time_ids_cfg,
            data.guidance_scale
        )
        loss = calculate_loss(predicted_noise, noise_data.target_noise, data.group_indices)
        perform_backpropagation(loss)

    assert isinstance(loss, torch.Tensor), "Loss should be a torch.Tensor"
    assert loss.ndim == 0, "Loss should be a scalar tensor"
    assert loss.device.type == device.type, f"Loss device type mismatch: expected {device.type}, got {loss.device.type}"
    assert loss.dtype == weight_dtype or (weight_dtype == torch.bfloat16 and loss.dtype == torch.float32), f"Loss dtype mismatch: expected {weight_dtype} or float32, got {loss.dtype}"
    
    print(f"Calculated loss: {loss.item()}")
    print("Successfully performed backpropagation and verified output.")

if __name__ == "__main__":
    log_file_path, orig_stdout, orig_stderr, log_file = None, None, None, None
    try:
        log_file_path, orig_stdout, orig_stderr, log_file = setup_logging()
        print(f"--- Starting Backpropagation Test ---")
        print(f"All output will be redirected to: {log_file_path}")
        test_backpropagation()
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
