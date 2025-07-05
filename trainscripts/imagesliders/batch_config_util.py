import yaml
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from trainscripts.imagesliders import map_data_to_latents
from trainscripts.imagesliders import batch_train_util

def load_config_from_yaml(filepath):
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

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

        self.latent_cache_dir = Path(self.config['dataset_config']['dataset']['folder_main']) / "latents"
        os.makedirs(self.latent_cache_dir, exist_ok=True)

        print("Mapping dataset...")
        subfolder_names = [f.strip() for f in self.config['dataset_config']['dataset']['folders'].split(',')]
        scale_values = [float(s.strip()) for s in self.config['dataset_config']['dataset']['scales'].split(',')]
        
        for i, folder_name in enumerate(subfolder_names):
            subfolder_path = Path(self.config['dataset_config']['dataset']['folder_main']) / folder_name
            scale = scale_values[i]
            for image_path in subfolder_path.glob("*"):
                if image_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    self.image_paths.append(str(image_path))
                    self.scales.append(scale)
                    if self.use_latents:
                        latent = map_data_to_latents.get_latent_for_image(
                            str(image_path), self.vae, self.device, self.weight_dtype, self.latent_cache_dir, self.vae.state_dict()
                        )
                        self.latents.append(latent)


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.use_latents:
            return self.latents[idx], self.scales[idx]
        else:
            return self.image_paths[idx], self.scales[idx]

def collate_fn(batch, tokenizers, text_encoders, config, device, weight_dtype):
    latents, scales = zip(*batch)
    
    latents = torch.cat(latents, dim=0).to(device, dtype=weight_dtype)
    scales = torch.tensor(scales, dtype=weight_dtype, device=device)
    
    with open(config['dataset_config']['prompts_file'], 'r') as f:
        prompts = yaml.safe_load(f)

    text_embeddings, pooled_embeds = batch_train_util.create_batched_prompt_embeddings(
        tokenizers,
        text_encoders,
        prompts,
        num_images_per_prompt=len(latents),
    )

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
    dataset = ImageScaleDataset(config, environment['vae'], environment['device'], environment['weight_dtype'], use_latents=use_latents)
    
    collate_wrapper = lambda b: collate_fn(b, environment['tokenizers'], environment['text_encoders'], config, environment['device'], environment['weight_dtype'])
    
    dataloader = DataLoader(dataset, batch_size=config['train']['batch_size'], shuffle=True, collate_fn=collate_wrapper)
    return dataloader

if __name__ == '__main__':
    from trainscripts.imagesliders import config_util
    from trainscripts.imagesliders import batch_model_util
    import torch.optim as optim

    def config_io():
        import argparse

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
        inner_config = config_util.load_config_from_yaml(inner_config_path)
        config.update(inner_config)

        dset_config_path = config['dset_config']['refpath']
        print(f"Loading dataset config from: {dset_config_path}")
        config['dataset_config'] = load_config_from_yaml(dset_config_path)

        return config

    def envsetup(config):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weight_dtype = config_util.parse_precision(config.train.precision)

        vae, unet, tokenizers, text_encoders, noise_scheduler = batch_model_util.load_models(config, device, weight_dtype)

        environment = {
            "unet": unet,
            "vae": vae,
            "noise_scheduler": noise_scheduler,
            "tokenizers": tokenizers,
            "text_encoders": text_encoders,
            "network": network,
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "criteria": criteria,
            "device": device,
            "weight_dtype": weight_dtype,
            "config": config,
        }
        return environment

    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self

    args = config_io()
    config_dict = {}
    for key, value in args.items():
        if isinstance(value, dict):
            config_dict[key] = AttrDict(value)
        else:
            config_dict[key] = value
    
    config = AttrDict(config_dict)
    
    environment = envsetup(config)
    
    # Test with use_latents=False (references to latent paths)
    print("Testing with latent paths (use_latents=False)...")
    dataloader_paths = dataset_constructor(config, environment, use_latents=False)
    
    for i, batch in enumerate(dataloader_paths):
        if i >= 240:
            break
        print(f"Batch {i+1}/240 (paths):")
        # In a real scenario, you would load the latents here before passing to the model
        # For this test, we just check that the batching works
        print(f"  Latent paths: {batch['latents']}")
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