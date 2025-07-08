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

from diffusers import DDPMScheduler

def load_models(config, device, weight_dtype):
    return None, None, None, None

def create_noise_scheduler(name: str):
    return DDPMScheduler()

def get_optimizer(name: str):
    return None

# Mock BatchedLoRANetwork for testing purposes
class MockBatchedLoRANetwork(torch.nn.Module):
    def __init__(self, unet, rank, alpha, train_method):
        super().__init__()
        self.unet = unet
        self.rank = rank
        self.alpha = alpha
        self.train_method = train_method
        self.lora_scales = None

    def set_lora_scales(self, scales):
        self.lora_scales = scales
        print(f"MockBatchedLoRANetwork: LoRA scales set. Shape: {scales.shape}, Dtype: {scales.dtype}, Device: {scales.device}")

    def prepare_optimizer_params(self):
        return [] # Dummy for optimizer setup

    def save_weights(self, path):
        pass # Dummy save

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

def envsetup(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = parse_precision(config.train.precision)
    vae, unet, tokenizers, text_encoders = load_models(config, device, weight_dtype)
    noise_scheduler = create_noise_scheduler(config.train.noise_scheduler)
    
    # Create a mock network instead of the real one
    network = MockBatchedLoRANetwork(
        unet=unet, # Still pass dummy unet
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
        "network": network, # Add the mock network to environment
    }
    return environment

def setup_logging(runname="test_lora_scale_setting"):
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

def test_lora_scale_setting():
    # i am writing this documentation to understand the flow of data and flow of control in the executed program.
    # i am not editing control flow or data flow. i am here to understand and judge without controlling what i read.
    # This test specifically tracks VRAM usage during LoRA scale setting.

    # Simulate command-line arguments for config_io
    original_argv = sys.argv
    sys.argv = ['test_script.py', '--batchtrainconfig', 'trainscripts/imagesliders/data/batch_config.yaml']
    
    config = AttrDict(load_config_from_yaml(sys.argv[2])) # Load config directly
    sys.argv = original_argv # Restore original argv

    environment = envsetup(config)

    network = environment["network"]
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

    # Load captured initial batch
    initial_batch = torch.load(capture_dir / "00_initial_batch.pt")

    if torch.cuda.is_available():
        print(f"VRAM (after loading captured data): {torch.cuda.memory_allocated() / (1024**2):.2f} MB")

    # Move scales to device
    scales = initial_batch["scales"].to(device, dtype=weight_dtype)

    if torch.cuda.is_available():
        print(f"VRAM (after moving scales to device): {torch.cuda.memory_allocated() / (1024**2):.2f} MB")

    # Replicate LoRA scale setting logic
    batched_scales_cfg = torch.cat([scales, scales], dim=0).unsqueeze(-1)

    if torch.cuda.is_available():
        print(f"VRAM (before set_lora_scales): {torch.cuda.memory_allocated() / (1024**2):.2f} MB")

    network.set_lora_scales(batched_scales_cfg)

    if torch.cuda.is_available():
        print(f"VRAM (after set_lora_scales): {torch.cuda.memory_allocated() / (1024**2):.2f} MB")

    # Verify that scales were set on the mock network
    assert network.lora_scales is not None
    assert torch.allclose(network.lora_scales, batched_scales_cfg), "LoRA scales mismatch!"

    print("Successfully simulated LoRA scale setting and verified outputs. Review VRAM logs for memory behavior.")

if __name__ == "__main__":
    log_file_path, orig_stdout, orig_stderr, log_file = None, None, None, None
    try:
        log_file_path, orig_stdout, orig_stderr, log_file = setup_logging()
        print(f"--- Starting LoRA Scale Setting Test ---")
        print(f"All output will be redirected to: {log_file_path}")
        test_lora_scale_setting()
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
