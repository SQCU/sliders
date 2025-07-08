import torch
from pathlib import Path
import sys
import os
import argparse
import yaml
import datetime
from typing import Dict, Any

# Add the parent directory of trainscripts to sys.path for imports
# This assumes the test script is placed in F:/dox/ai/gemmy/sliders/
sys.path.append(str(Path(__file__).parent))

# Minimal necessary components from batch_config_util and batch_model_util
# to set up the environment for this specific test.
# In a real scenario, these would ideally be imported directly if possible,
# but for self-containment and avoiding complex import paths,
# we're including the core logic here.

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

# Dummy load_models and create_noise_scheduler for envsetup
# In a real test, you might mock these or load minimal versions
# For this test, we only need noise_scheduler, so we'll create a dummy
# for the other return values of load_models.
from diffusers import DDPMScheduler # Assuming this is the scheduler type

def load_models(config, device, weight_dtype):
    # This is a dummy function for the test.
    # In a real scenario, you'd load actual models.
    # For this test, we don't need the actual models, just the environment setup.
    return None, None, None, None # vae, unet, tokenizers, text_encoders

def create_noise_scheduler(name: str):
    # This is a dummy function for the test.
    # In a real scenario, you'd create the actual scheduler.
    # For this test, we assume DDPMScheduler based on common diffusers usage.
    return DDPMScheduler()

def get_optimizer(name: str):
    # Dummy optimizer getter for envsetup
    return None

def envsetup(config):
    # Simplified envsetup for testing purposes
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
        "optimizer": get_optimizer(config.train.optimizer), # Dummy optimizer
    }
    return environment

def setup_logging(runname="test_noise_and_timestep_generation"):
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

def test_noise_and_timestep_generation():
    # i am writing this documentation to understand the flow of data and flow of control in the executed program.
    # i am not editing control flow or data flow. i am here to understand and judge without controlling what i read.
    # This test specifically tracks VRAM usage during noise and timestep generation.

    # Simulate command-line arguments for config_io
    original_argv = sys.argv
    sys.argv = ['test_script.py', '--batchtrainconfig', 'trainscripts/imagesliders/data/batch_config.yaml']
    
    config = AttrDict(load_config_from_yaml(sys.argv[2])) # Load config directly
    sys.argv = original_argv # Restore original argv

    environment = envsetup(config)

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

    # Load captured initial batch and noisy latents
    initial_batch = torch.load(capture_dir / "00_initial_batch.pt")
    captured_noisy_latents = torch.load(capture_dir / "01_noisy_latents.pt")

    if torch.cuda.is_available():
        print(f"VRAM (after loading captured data): {torch.cuda.memory_allocated() / (1024**2):.2f} MB")

    # Move initial latents to device
    latents = initial_batch["latents"].to(device, dtype=weight_dtype)

    if torch.cuda.is_available():
        print(f"VRAM (after moving latents to device): {torch.cuda.memory_allocated() / (1024**2):.2f} MB")

    # Replicate TIMESTEP LOGIC from train_step
    with torch.no_grad():
        noise_scheduler.set_timesteps(config.train.max_denoising_steps, device=device)
        generator = torch.Generator(device=device).manual_seed(seed)
        
        timesteps_to = torch.randint(
            1, config.train.max_denoising_steps, (latents.shape[0],), device=device, generator=generator
        ).long()

        noise = torch.randn(latents.shape, device=latents.device, generator=generator)
        
        if torch.cuda.is_available():
            print(f"VRAM (before add_noise): {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
        
        simulated_noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps_to)
        
        if torch.cuda.is_available():
            print(f"VRAM (after add_noise): {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
    
    # Log properties of simulated and captured noisy latents
    print(f"Simulated noisy latents - Mean: {simulated_noisy_latents.mean().item():.6f}, Variance: {simulated_noisy_latents.var().item():.6f}")
    print(f"Captured noisy latents - Mean: {captured_noisy_latents.mean().item():.6f}, Variance: {captured_noisy_latents.var().item():.6f}")

    # We are not asserting bit-for-bit equality of noise, but rather tracking memory behavior.
    # The goal is to ensure the process runs and we can observe VRAM changes.

    print("Successfully simulated noise and timestep generation. Review VRAM logs for memory behavior.")

if __name__ == "__main__":
    log_file_path, orig_stdout, orig_stderr, log_file = None, None, None, None
    try:
        log_file_path, orig_stdout, orig_stderr, log_file = setup_logging()
        print(f"--- Starting Noise and Timestep Generation Test ---")
        print(f"All output will be redirected to: {log_file_path}")
        test_noise_and_timestep_generation()
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
