import yaml
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from trainscripts.imagesliders import map_data_to_latents
from trainscripts.imagesliders import batch_train_util
import datetime
import sys
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

class ImageScaleDataset(Dataset):
    def __init__(self, config):
        self.config = config
        
        self.image_paths = []
        self.scales = []

        self.latent_cache_dir = Path(self.config.dataset_config.dataset.folder_main) / "latents"
        os.makedirs(self.latent_cache_dir, exist_ok=True)

        print("Collecting image paths and scales...")
        subfolder_names = [f.strip() for f in self.config.dataset_config.dataset.folders.split(',')]
        scale_values = [float(s.strip()) for s in self.config.dataset_config.dataset.scales.split(',')]
        
        for i, folder_name in enumerate(subfolder_names):
            subfolder_path = Path(self.config.dataset_config.dataset.folder_main) / folder_name
            scale = scale_values[i]
            for image_path in subfolder_path.glob("*"):
                if image_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    self.image_paths.append(str(image_path))
                    self.scales.append(scale)
        
        # Load prompts
        with open(self.config.dataset_config.prompts_file, 'r') as f:
            self.prompts_data = yaml.safe_load(f)

        self.total_dataset_size = self.config.train.iterations * self.config.train.batch_size

    def __len__(self):
        return self.total_dataset_size

    def __getitem__(self, idx):
        image_idx = idx % len(self.image_paths)
        prompt_idx = idx % len(self.prompts_data)
        return self.image_paths[image_idx], self.scales[image_idx], self.prompts_data[prompt_idx]

def collate_fn(batch, tokenizers, text_encoders, config, vae, device, weight_dtype):
    image_paths, scales, prompts_data = zip(*batch)
    
    latents = []
    for img_path in image_paths:
        latent = map_data_to_latents.get_latent_for_image(
            img_path, vae, device, weight_dtype, Path(config.dataset_config.dataset.folder_main) / "latents", vae.state_dict()
        )
        latents.append(latent)
    latents = torch.cat(latents, dim=0).to(device, dtype=weight_dtype)

    scales = torch.tensor(scales, dtype=weight_dtype, device=device)
    
    # prompts_data is already a list of dictionaries, each containing 'positive', 'unconditional', 'neutral'
    # We need to process each item in the batch individually for prompt embeddings
    # and then concatenate them.
    all_text_embeddings = []
    all_pooled_embeds = []

    for prompt_dict in prompts_data:
        text_embeddings, pooled_embeds = batch_train_util.create_batched_prompt_embeddings(
            tokenizers,
            text_encoders,
            prompt_dict,
        )
        all_text_embeddings.append(text_embeddings)
        all_pooled_embeds.append(pooled_embeds)

    text_embeddings_batch = torch.cat(all_text_embeddings, dim=0).to(device, dtype=weight_dtype)
    pooled_embeds_batch = torch.cat(all_pooled_embeds, dim=0).to(device, dtype=weight_dtype)

    add_time_ids = batch_train_util.get_add_time_ids(
        1024, 1024, False, dtype=latents.dtype
    ).repeat(len(latents), 1).to(device, dtype=weight_dtype)

    return {
        "latents": latents,
        "scales": scales,
        "text_embeddings": text_embeddings_batch,
        "pooled_embeds": pooled_embeds_batch,
        "add_time_ids": add_time_ids,
    }

def dataset_constructor(config, environment):
    dataset = ImageScaleDataset(config)
    
    collate_wrapper = lambda b: collate_fn(b, environment['tokenizers'], environment['text_encoders'], config, environment['vae'], environment['device'], environment['weight_dtype'])
    
    print(f"Batch size from config in batch_config_util.py: {config.train.batch_size}")
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
    
    inner_config_path = config.obsolete_config.refpath
    print(f"Loading and merging inner config from: {inner_config_path}")
    inner_config = AttrDict(load_config_from_yaml(inner_config_path))
    config.update(inner_config)

    dset_config_path = config.dset_config.refpath
    print(f"Loading dataset config from: {dset_config_path}")
    config.dataset_config = AttrDict(load_config_from_yaml(dset_config_path))

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
        config = config_io()
        
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
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        print(f"Script finished. Log saved to {log_file_path}")