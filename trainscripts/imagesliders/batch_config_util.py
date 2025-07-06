from typing import Literal, Optional
import yaml
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from trainscripts.imagesliders import map_data_to_latents
from trainscripts.imagesliders import batch_train_util
import datetime
import sys
from pydantic import BaseModel
from trainscripts.imagesliders import batch_model_util
import torch.optim as optim
import argparse

def load_config_from_yaml(filepath):
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = AttrDict(value)

#this is RADIOACTIVE FUCKING POISON WHY IS THERE LOGIC IN HERE!!!
#SOMEHOW THIS IS CAUSING THE COLLATE_FN TO REMOVE RANDOM PARTS OF TEXT EMBEDDDING TENSORS WHICH EXIST INSIDE OF OTHER LOGIC!!!!!!

class ImageScaleDataset(Dataset):
    def __init__(self, config, vae, device, weight_dtype, use_latents=True):
        self.config = config
        self.vae = vae
        self.device = device
        self.weight_dtype = weight_dtype
        self.use_latents = use_latents
        
        self.image_paths = []
        self.scales = []
        self.latents = []

        print(config.keys())
        #print(config.keys().keys())

        self.latent_cache_dir = Path(self.config.dataset_config.dataset.folder_main) / "latents"
        os.makedirs(self.latent_cache_dir, exist_ok=True)

        print("Mapping dataset and caching latents...")
        subfolder_names = [f.strip() for f in self.config.dataset_config.dataset.folders.split(',')]
        scale_values = [float(s.strip()) for s in self.config.dataset_config.dataset.scales.split(',')]
        
        for i, folder_name in enumerate(subfolder_names):
            subfolder_path = Path(self.config.dataset_config.dataset.folder_main) / folder_name
            scale = scale_values[i]
            for image_path in subfolder_path.glob("*"):
                if image_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    self.image_paths.append(str(image_path))
                    self.scales.append(scale)
                    
                    # Always ensure latent is cached on disk
                    latent = map_data_to_latents.get_latent_for_image(
                        str(image_path), self.vae, self.device, self.weight_dtype, self.latent_cache_dir, self.vae.state_dict()
                    )
                    if self.use_latents:
                        self.latents.append(latent)


    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        if self.use_latents:
            return self.latents[idx], self.scales[idx]
        else:
            latent_path = self.latent_cache_dir / (Path(self.image_paths[idx]).stem + ".pt")
            return str(latent_path.resolve()), self.scales[idx]

def collate_fn(batch, tokenizers, text_encoders, config, device, weight_dtype, use_latents=True):
    items, scales = zip(*batch)
    
    if use_latents:
        latents = torch.cat(items, dim=0).to(device, dtype=weight_dtype)
    else:
        latents = torch.cat([torch.load(p) for p in items], dim=0).to(device, dtype=weight_dtype)

    scales = torch.tensor(scales, dtype=weight_dtype, device=device)
    
    with open(config.dataset_config.prompts_file, 'r') as f:
        prompts = yaml.safe_load(f)

    text_embeddings, pooled_embeds = batch_train_util.create_batched_prompt_embeddings(
        tokenizers,
        text_encoders,
        prompts,
        num_images_per_prompt=len(latents),
    )

    print(f"from collate_fn: te{text_embeddings.shape},pe{pooled_embeds.shape}")

    add_time_ids = batch_train_util.get_add_time_ids(
        1024, 1024, False, dtype=latents.dtype
    ).repeat(len(latents), 1).to(device, dtype=weight_dtype)

    return {
        "latents": latents,
        "scales": scales,
        "text_embeddings": text_embeddings.to(device, dtype=weight_dtype),
        "pooled_embeds": pooled_embeds.to(device, dtype=weight_dtype),
        "add_time_ids": add_time_ids,
    }

def dataset_constructor(config, environment, use_latents=True):
    """
    THIS IS RADIOACTIVE FUCKING POISON!!!!
    IT BROKE EVERYTHING WE'RE WOKRING WITH AND FOR SOME REASON HJIDES / ABSTRACTS THE CREATION OF THE ACTUAL DATA!
    """
    dataset = ImageScaleDataset(config, environment['vae'], environment['device'], environment['weight_dtype'], use_latents=use_latents)
    
    collate_wrapper = lambda b: collate_fn(b, environment['tokenizers'], environment['text_encoders'], config, environment['device'], environment['weight_dtype'], use_latents=use_latents)
    
    dataloader = DataLoader(dataset, batch_size=config.train.batch_size, shuffle=True, collate_fn=collate_wrapper)
    return dataloader

def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(log_dir, f"batch_config_util_test_{timestamp}.log")
    sys.stdout = open(log_filename, "w")
    sys.stderr = sys.stdout # Redirect stderr to the same log file
    print(f"Logging output to {log_filename}")
    return log_filename

PRECISION_TYPES = Literal["fp32", "fp16", "bf16", "float32", "float16", "bfloat16"]
NETWORK_TYPES = Literal["lierla", "c3lier"]


class PretrainedModelConfig(BaseModel):
    name_or_path: str
    v2: bool = False
    v_pred: bool = False

    clip_skip: Optional[int] = None


class NetworkConfig(BaseModel):
    type: NETWORK_TYPES = "lierla"
    rank: int = 4
    alpha: float = 1.0

    training_method: str = "full"


class TrainConfig(BaseModel):
    batch_size: int = 4
    precision: PRECISION_TYPES = "bfloat16"
    noise_scheduler: Literal["ddim", "ddpm", "lms", "euler_a"] = "ddim"

    iterations: int = 500
    lr: float = 1e-4
    optimizer: str = "adamw"
    optimizer_args: str = ""
    lr_scheduler: str = "constant"

    max_denoising_steps: int = 50
    batch_size: int = 1 #now must be assigned here


class SaveConfig(BaseModel):
    name: str = "untitled"
    path: str = "./output"
    per_steps: int = 200
    precision: PRECISION_TYPES = "float32"


class LoggingConfig(BaseModel):
    use_wandb: bool = False

    verbose: bool = False


class OtherConfig(BaseModel):
    use_xformers: bool = False
    use_pytorch_SDPA: bool = True
    torch_compile: bool = False
    lycorisize:bool = False
    gradient_checkpointing: bool = False
    torch_amp: bool = False


class RootConfig(BaseModel):
    prompts_file: str
    pretrained_model: PretrainedModelConfig
    latent_cache_dir: Optional[str] = None

    network: NetworkConfig

    train: Optional[TrainConfig]

    save: Optional[SaveConfig]

    logging: Optional[LoggingConfig]

    other: Optional[OtherConfig]


def parse_precision(precision: str) -> torch.dtype:
    if precision == "fp32" or precision == "float32":
        return torch.float32
    elif precision == "fp16" or precision == "float16":
        return torch.float16
    elif precision == "bf16" or precision == "bfloat16":
        return torch.bfloat16

    raise ValueError(f"Invalid precision type: {precision}")


#WHY DOES THIS EXIST WHY ARE THERE TWO WAYS TO LOAD A CONFIG THIS HAS BROKEN SO MANY THIGNS
def load_config_from_yaml_and_merge(config_path: str) -> RootConfig:
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    root = RootConfig(**config)

    if root.train is None:
        root.train = TrainConfig()

    if root.save is None:
        root.save = SaveConfig()

    if root.logging is None:
        root.logging = LoggingConfig()

    if root.other is None:
        root.other = OtherConfig()

    return root

def config_io():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchtrainconfig", "--bconfig", "-c", 
                        type=str, 
                        help="Path to the batch training config file.",
                        default="trainscripts/imagesliders/data/batch_config.yaml")
    args = parser.parse_args()

    print(f"Loading batch config from: {args.batchtrainconfig}")
    config = load_config_from_yaml(args.batchtrainconfig)
    
    inner_config_path = config['obsolete_config']['refpath']
    print(f"Loading and merging inner config from: {inner_config_path}")
    inner_config = load_config_from_yaml_and_merge(inner_config_path)
    config.update(inner_config)

    dset_config_path = config['dset_config']['refpath']
    print(f"Loading dataset config from: {dset_config_path}")
    config['dataset_config'] = load_config_from_yaml(dset_config_path)

    return config

def envsetup(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = parse_precision(config.train.precision)

    vae, unet, tokenizers, text_encoders, noise_scheduler = batch_model_util.load_models(config, device, weight_dtype)

    environment = {
        "unet": unet,
        "vae": vae,
        "noise_scheduler": noise_scheduler,
        "tokenizers": tokenizers,
        "text_encoders": text_encoders,
        "device": device,
        "weight_dtype": weight_dtype,
        "config": config,
    }
    return environment

if __name__ == '__main__':
    def main():
        args = config_io()
        config = AttrDict(args)
        
        environment = envsetup(config)
        
        # Test with use_latents=False (references to latent paths)
        print("Testing with latent paths (use_latents=False)...")
        dataloader_paths = dataset_constructor(config, environment, use_latents=False)
        
        for i, batch in enumerate(dataloader_paths):
            if i >= 240:
                break
            print(f"Batch {i+1}/240 (paths):")
            print(f"  Latents shape: {batch['latents'].shape}")
            print(f"  Scales: {batch['scales']}")

        # Test with use_latents=True (in-memory latents)
        print("\nTesting with in-memory latents (use_latents=True)...")
        dataloader_latents = dataset_constructor(config, environment, use_latents=True)

        for i, batch in enumerate(dataloader_latents):
            if i >= 240:
                break
            print(f"Batch {i+1}/240 (in-memory):")
            print(f"  Latents shape: {batch['latents'].shape}")
            print(f"  Scales: {batch['scales']}")

        print("\nDataset and Dataloader test finished.")

    log_file_path = setup_logging()
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        sys.stdout.close()
        sys.stdout = sys.__stdout__ # Restore original stdout
        sys.stderr = sys.__stderr__ # Restore original stderr
        print(f"Script finished. Log saved to {log_file_path}")