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
from trainscripts.imagesliders.batch_train_util import nocfg_predict_noise_xl as batched_predict_noise_xl # Renamed for clarity
from trainscripts.imagesliders import batch_lora as lora # Import batch_lora

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
    
    # Initialize LoRA network
    network = lora.BatchedLoRANetwork(
        unet=unet,
        rank=config.network.rank,
        alpha=config.network.alpha,
        train_method=config.network.training_method,
    ).to(device, dtype=weight_dtype)

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
        "network": network, # Add network to environment
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

def test_backpropagation():
    # This test specifically tracks VRAM usage during backpropagation with LoRA.

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
    network = environment["network"] # Get network from environment
    
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
    # We will re-generate noisy_latents and noise, but use captured text embeddings etc.
    text_embeddings_cfg = torch.load(capture_dir / "03_text_embeddings_cfg.pt").to(device, dtype=weight_dtype)
    pooled_embeds_cfg = torch.load(capture_dir / "04_pooled_embeds_cfg.pt").to(device, dtype=weight_dtype)
    add_time_ids_cfg = torch.load(capture_dir / "05_add_time_ids_cfg.pt").to(device, dtype=torch.float32) # Keep float32
    # unet_timesteps_cfg will be generated based on timesteps_to

    # Get guidance_scale from the initial batch (assuming it's consistent)
    initial_batch = torch.load(capture_dir / "00_initial_batch.pt")
    latents = initial_batch["latents"].to(device, dtype=weight_dtype) # Original noiseless latents
    group_indices = initial_batch["pair_indices"].to(device) # Using pair_indices as group_indices for now
    scales = initial_batch["scales"].to(device, dtype=weight_dtype) # Get scales for LoRA
    guidance_scale = initial_batch["guidance_scale"]

    # --- Batch validation logic ---
    print(f"Shape of latents from initial_batch: {latents.shape}")
    print(f"Shape of group_indices (pair_indices) from initial_batch: {group_indices.shape}")
    print(f"Shape of scales from initial_batch: {scales.shape}")
    # Assert that the batch dimensions are consistent
    assert latents.shape[0] == group_indices.shape[0], \
        f"Batch size mismatch: latents.shape[0]={latents.shape[0]}, group_indices.shape[0]={group_indices.shape[0]}"
    assert latents.shape[0] == scales.shape[0], \
        f"Batch size mismatch: latents.shape[0]={latents.shape[0]}, scales.shape[0]={scales.shape[0]}"

    if torch.cuda.is_available():
        print(f"VRAM (after loading initial data): {torch.cuda.memory_allocated() / (1024**2):.2f} MB")

    # --- Replicate TIMESTEP LOGIC from train_step ---
    with torch.no_grad():
        noise_scheduler.set_timesteps(config.train.max_denoising_steps, device=device)
        generator = torch.Generator(device=device).manual_seed(seed)
        
        # Generate a random timestep for each item in the batch
        timesteps_to = torch.randint(
            1, config.train.max_denoising_steps, (latents.shape[0],), device=device, generator=generator
        ).long()

        # Generate noise (this will be our target_noise)
        target_noise = torch.randn(latents.shape, device=latents.device, generator=generator)
        
        # Add noise to latents to get noisy_latents (input to UNet)
        noisy_latents = noise_scheduler.add_noise(latents, target_noise, timesteps_to)

        # Prepare CFG-ready latents and timesteps for UNet
        latents_cfg = torch.cat([noisy_latents, noisy_latents], dim=0)
        unet_timesteps_cfg = torch.cat([timesteps_to, timesteps_to], dim=0)

    if torch.cuda.is_available():
        print(f"VRAM (after noise generation and CFG prep): {torch.cuda.memory_allocated() / (1024**2):.2f} MB")

    # --- Set LoRA scales and perform UNet forward pass ---
    # Prepare batched_scales_cfg for LoRA network
    batched_scales_cfg = torch.cat([scales, scales], dim=0).unsqueeze(-1)
    network.set_lora_scales(batched_scales_cfg)

    with network: # Activates LoRA modules
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
        # Apply CFG logic explicitly to get N_eff sized predicted_noise
        predicted_noise_uncond, predicted_noise_text = predicted_noise_raw.chunk(2)
        predicted_noise = (predicted_noise_uncond + guidance_scale * (
            predicted_noise_text - predicted_noise_uncond
        )).to(device)

    if torch.cuda.is_available():
        print(f"VRAM (after UNet forward pass with LoRA): {torch.cuda.memory_allocated() / (1024**2):.2f} MB")

    # --- Perform loss calculation ---
    # Ensure predicted_noise and target_noise are on the correct dtype for loss calculation if needed
    # (Relaxed assertion allows float32 loss even with bfloat16 inputs)
    loss = scale_n_tuple_loss(predicted_noise, target_noise, group_indices)

    if torch.cuda.is_available():
        print(f"VRAM (after loss calculation): {torch.cuda.memory_allocated() / (1024**2):.2f} MB")

    # --- Perform backpropagation ---
    print("Performing backpropagation...")
    if torch.cuda.is_available():
        print(f"VRAM (before backpropagation): {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
    
    loss.backward()

    if torch.cuda.is_available():
        print(f"VRAM (after backpropagation): {torch.cuda.memory_allocated() / (1024**2):.2f} MB")

    # --- Assertions ---
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