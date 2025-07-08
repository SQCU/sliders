import torch
from pathlib import Path
import sys
import os
import argparse
import yaml
import datetime
from typing import Dict, Any, Tuple

# Add the parent directory of trainscripts to sys.path for imports
# This assumes the test script is placed in F:/dox/ai/gemmy/sliders/
sys.path.append(str(Path(__file__).parent))

# Minimal necessary components from batch_config_util and batch_model_util
# to set up the environment for this specific test.

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
from diffusers import DDPMScheduler

def load_models(config, device, weight_dtype):
    return None, None, None, None

def create_noise_scheduler(name: str):
    return DDPMScheduler()

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

def setup_logging(runname="test_cfg_batch_preparation"):
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

# --- Copied from batched_training_loop.py for self-containment ---
def prepare_cfg_batch(
    batch: dict,
    noisy_latents: torch.Tensor,
    timesteps_to: torch.Tensor,
    noise_scheduler,
    weight_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepares a CFG-ready batch by concatenating unconditional and conditional embeddings.
    """
    cond_text_embeddings = batch['cond_text_embeddings']
    cond_pooled_embeds = batch['cond_pooled_embeds']
    uncond_text_embeddings = batch['uncond_text_embeddings']
    uncond_pooled_embeds = batch['uncond_pooled_embeds']
    add_time_ids = batch['add_time_ids']

    latents_cfg = torch.cat([noisy_latents, noisy_latents], dim=0)
    text_embeddings_cfg = torch.cat([uncond_text_embeddings, cond_text_embeddings], dim=0)
    pooled_embeds_cfg = torch.cat([uncond_pooled_embeds, cond_pooled_embeds], dim=0)
    add_time_ids_cfg = torch.cat([add_time_ids, add_time_ids], dim=0)

    #not to unet.dtype! these need to be a torch.long
    unet_timesteps = noise_scheduler.timesteps[timesteps_to]#.to(weight_dtype)
    unet_timesteps_cfg = torch.cat([unet_timesteps, unet_timesteps], dim=0)

    return latents_cfg, text_embeddings_cfg, pooled_embeds_cfg, add_time_ids_cfg, unet_timesteps_cfg
# --- End copied functions ---

def test_cfg_batch_preparation():
    # i am writing this documentation to understand the flow of data and flow of control in the executed program.
    # i am not editing control flow or data flow. i am here to understand and judge without controlling what i read.
    # This test specifically tracks VRAM usage during CFG batch preparation.

    # Simulate command-line arguments for config_io
    original_argv = sys.argv
    sys.argv = ['test_script.py', '--batchtrainconfig', 'trainscripts/imagesliders/data/batch_config.yaml']
    
    config = AttrDict(load_config_from_yaml(sys.argv[2])) # Load config directly
    sys.argv = original_argv # Restore original argv

    environment = envsetup(config);

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

    # Load captured inputs for prepare_cfg_batch
    initial_batch = torch.load(capture_dir / "00_initial_batch.pt")
    noisy_latents = torch.load(capture_dir / "01_noisy_latents.pt")
    
    # Re-create timesteps_to (as it's not directly saved, but derived from seed and config)
    with torch.no_grad():
        noise_scheduler.set_timesteps(config.train.max_denoising_steps, device=device)
        generator = torch.Generator(device=device).manual_seed(seed)
        timesteps_to = torch.randint(
            1, config.train.max_denoising_steps, (noisy_latents.shape[0],), device=device, generator=generator
        ).long()

    # Load captured outputs for comparison
    captured_latents_cfg = torch.load(capture_dir / "02_latents_cfg.pt")
    captured_text_embeddings_cfg = torch.load(capture_dir / "03_text_embeddings_cfg.pt")
    captured_pooled_embeds_cfg = torch.load(capture_dir / "04_pooled_embeds_cfg.pt")
    captured_add_time_ids_cfg = torch.load(capture_dir / "05_add_time_ids_cfg.pt")
    captured_unet_timesteps_cfg = torch.load(capture_dir / "06_unet_timesteps_cfg.pt")

    if torch.cuda.is_available():
        print(f"VRAM (after loading captured data): {torch.cuda.memory_allocated() / (1024**2):.2f} MB")

    # Move initial_batch components to device for prepare_cfg_batch
    # These were already moved in train_step before prepare_cfg_batch was called
    initial_batch['cond_text_embeddings'] = initial_batch['cond_text_embeddings'].to(device, dtype=weight_dtype)
    initial_batch['cond_pooled_embeds'] = initial_batch['cond_pooled_embeds'].to(device, dtype=weight_dtype)
    initial_batch['uncond_text_embeddings'] = initial_batch['uncond_text_embeddings'].to(device, dtype=weight_dtype)
    initial_batch['uncond_pooled_embeds'] = initial_batch['uncond_pooled_embeds'].to(device, dtype=weight_dtype)
    initial_batch['add_time_ids'] = initial_batch['add_time_ids'].to(device, dtype=torch.float32)

    if torch.cuda.is_available():
        print(f"VRAM (before prepare_cfg_batch): {torch.cuda.memory_allocated() / (1024**2):.2f} MB")

    # Call the function under test
    simulated_latents_cfg, simulated_text_embeddings_cfg, simulated_pooled_embeds_cfg, simulated_add_time_ids_cfg, simulated_unet_timesteps_cfg = prepare_cfg_batch(
        initial_batch,
        noisy_latents,
        timesteps_to,
        noise_scheduler,
        weight_dtype,
    )

    if torch.cuda.is_available():
        print(f"VRAM (after prepare_cfg_batch): {torch.cuda.memory_allocated() / (1024**2):.2f} MB")

    # Compare simulated outputs with captured outputs
    assert torch.allclose(simulated_latents_cfg, captured_latents_cfg, atol=1e-4, rtol=1e-3), "latents_cfg mismatch!"
    assert torch.allclose(simulated_text_embeddings_cfg, captured_text_embeddings_cfg, atol=1e-4, rtol=1e-3), "text_embeddings_cfg mismatch!"
    assert torch.allclose(simulated_pooled_embeds_cfg, captured_pooled_embeds_cfg, atol=1e-4, rtol=1e-3), "pooled_embeds_cfg mismatch!"
    assert torch.allclose(simulated_add_time_ids_cfg, captured_add_time_ids_cfg, atol=1e-4, rtol=1e-3), "add_time_ids_cfg mismatch!"
    assert torch.allclose(simulated_unet_timesteps_cfg, captured_unet_timesteps_cfg, atol=1e-4, rtol=1e-3), "unet_timesteps_cfg mismatch!"

    print("Successfully simulated CFG batch preparation and verified outputs. Review VRAM logs for memory behavior.")

if __name__ == "__main__":
    log_file_path, orig_stdout, orig_stderr, log_file = None, None, None, None
    try:
        log_file_path, orig_stdout, orig_stderr, log_file = setup_logging()
        print(f"--- Starting CFG Batch Preparation Test ---")
        print(f"All output will be redirected to: {log_file_path}")
        test_cfg_batch_preparation()
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
