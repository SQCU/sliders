import yaml
import os
import torch
import argparse
import datetime
import sys

# --- Copied from batch_dataset_encoding.py ---
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

def config_io():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchtrainconfig", "--bconfig", "-c",
                        type=str,
                        help="Path to the batch training config file.",
                        default="trainscripts/imagesliders/data/batch_config.yaml")
    args = parser.parse_args()

    print(f"Loading batch config from: {args.batchtrainconfig}")
    config = AttrDict(load_config_from_yaml(args.batchtrainconfig))

    # Load and merge model config if specified
    if 'model_config' in config and 'refpath' in config.model_config:
        model_config_path = config.model_config.refpath
        print(f"Loading and merging model config from: {model_config_path}")
        model_config = AttrDict(load_config_from_yaml(model_config_path))
        # Merge model_config into the main config, overwriting existing keys
        config.update(model_config)

    return config

# envsetup will require load_models and create_noise_scheduler from batch_model_util.py
# I will add the import for batch_model_util here, assuming it will be created/updated.
from .batch_model_util import load_models, create_noise_scheduler

def get_optimizer(name: str):
    name = name.lower()

    if name.startswith("dadapt"):
        import dadaptation

        if name == "dadaptadam":
            return dadaptation.DAdaptAdam
        elif name == "dadaptlion":
            return dadaptation.DAdaptLion
        else:
            raise ValueError("DAdapt optimizer must be dadaptadam or dadaptlion")

    elif name.endswith("8bit"):  # 検証してない
        import bitsandbytes as bnb

        if name == "adam8bit":
            return bnb.optim.Adam8bit
        elif name == "lion8bit":
            return bnb.optim.Lion8bit
        else:
            raise ValueError("8bit optimizer must be adam8bit or lion8bit")

    else:
        if name == "adam":
            return torch.optim.Adam
        elif name == "adamw":
            return torch.optim.AdamW
        elif name == "lion":
            from lion_pytorch import Lion

            return Lion
        elif name == "prodigy":
            import prodigyopt
            
            return prodigyopt.Prodigy
        else:
            raise ValueError("Optimizer must be adam, adamw, lion or Prodigy")

def envsetup(config):
    from torch.backends.cuda import enable_flash_sdp
    enable_flash_sdp(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = parse_precision(config.train.precision)
    vae, unet, tokenizers, text_encoders = load_models(config, device, weight_dtype)
    unet.requires_grad_(False).eval()
    vae.requires_grad_(False).eval()
    for text_encoder in text_encoders:
        text_encoder.requires_grad_(False).eval()

    optimizer_name = config.train.optimizer.lower()
    optimizer = get_optimizer(optimizer_name)

    noise_scheduler = create_noise_scheduler(config.train.noise_scheduler) # Initialize directly for single file
    environment = {
        "unet": unet,
        "vae": vae,
        "noise_scheduler": noise_scheduler,
        "tokenizers": tokenizers,
        "text_encoders": text_encoders,
        "device": device,
        "weight_dtype": weight_dtype,
        "config": config,
        "optimizer": optimizer,
    }
    return environment

def setup_logging(runname="batch_dataset_encoding_test"):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_filename = os.path.join(log_dir, runname+f"_{timestamp}.log") # Using the same log name convention

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    log_file = open(log_filename, "w")
    sys.stdout = log_file
    sys.stderr = log_file

    print(f"Logging output to {log_filename}")

    return log_filename, original_stdout, original_stderr
