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

def deep_merge_attrdict(d1, d2):
    for k, v in d2.items():
        if k in d1 and isinstance(d1[k], AttrDict) and isinstance(v, AttrDict):
            d1[k] = deep_merge_attrdict(d1[k], v)
        else:
            d1[k] = v
    return d1

def config_io():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchtrainconfig", "--bconfig", "-c",
                        type=str,
                        help="Path to the batch training config file.",
                        default="trainscripts/imagesliders/data/batch_config.yaml")
    args = parser.parse_args()

    print(f"Loading batch config from: {args.batchtrainconfig}")
    config = AttrDict(load_config_from_yaml(args.batchtrainconfig))

    if 'dataset' in config and 'prompts_file_path' in config.dataset:
        prompts_file_path = "trainscripts/imagesliders/data/prompts-xl-dilora-bracket.yaml"
        print(f"Loading and merging promptsfile from: {prompts_file_path}")
        prompts_config = AttrDict(load_config_from_yaml(prompts_file_path))
        config.dataset.update(prompts_config)

    # Load and merge model config if specified
    if 'model_config' in config and config.model_config is not None and 'refpath' in config.model_config:
        model_config_path = config.model_config.refpath
        print(f"Loading and merging model config from: {model_config_path}")
        model_config = AttrDict(load_config_from_yaml(model_config_path))
        # Use deep_merge_attrdict for merging model_config
        config = deep_merge_attrdict(config, model_config)

    # Ensure config.other exists and validate key performance features
    if not hasattr(config, 'other'):
        config.other = AttrDict() # Initialize if missing to avoid further errors

    # Validate gradient_checkpointing
    if not hasattr(config.other, 'gradient_checkpointing'):
        error_msg = "CRITICAL ERROR: 'gradient_checkpointing' attribute is missing from 'config.other'. Please specify 'gradient_checkpointing: true' or 'gradient_checkpointing: false' in your config file."
        print(f"!!! {error_msg} !!!", file=sys.stderr)
        raise ValueError(error_msg)
    
    # TODO: Add similar validation for torch_compile and VAE batchsize autocalibration

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
    save_dtype = parse_precision(config.save.precision)
    
    # Load models from checkpoint
    vae, unet, tokenizers, text_encoders = load_models(config, device, weight_dtype)
    #exile unet to cpu we aren't using it yet
    unet.requires_grad_(False).eval().to(torch.device("cpu"))
    
    # nograd vae and text encoders
    vae.requires_grad_(False).eval()
    for i in range(len(text_encoders)):
        text_encoders[i].requires_grad_(False).eval()

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
        "save_dtype": save_dtype,
        "config": config,
        "optimizer": optimizer,
        "generator": torch.Generator(device=device).manual_seed(config.train.seed),
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
