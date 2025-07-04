# Written by Gemini 2.5, under review

from typing import List, Optional
import argparse
import ast
from pathlib import Path
import gc, os
import numpy as np

import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from .config import TrainingConfig
from .data_preprocessing import create_dataloader
import train_util
import random
import model_util
import prompt_util
from prompt_util import (
    PromptEmbedsCache,
    PromptEmbedsPair,
    PromptSettings,
    PromptEmbedsXL,
)
import debug_util
import config_util
from config_util import RootConfig

import wandb

NUM_IMAGES_PER_PROMPT = 1
from lora import LoRANetwork, DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV

class EMA:
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}

    def update(self, name, x):
        if name not in self.shadow:
            self.shadow[name] = x.clone().detach()
        else:
            self.shadow[name].mul_(self.decay).add_(x, alpha=1 - self.decay)
        return self.shadow[name]

    def get(self, name):
        return self.shadow.get(name, None)


def flush():
    torch.cuda.empty_cache()
    gc.collect()


def train(
    config: RootConfig,
    training_config: TrainingConfig,
    prompts: list[PromptSettings],
    device,
    folder_main: str,
    folders,
    scales,
    use_latents: bool = False,
):
    # Create the dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataloader = create_dataloader(folder_main, folders, scales, training_config, use_latents, vae_checksum, transform)

    metadata = {
        "prompts": ",".join([prompt.json() for prompt in prompts]),
        "config": config.json(),
    }
    save_path = Path(config.save.path)

    modules = DEFAULT_TARGET_REPLACE
    if config.network.type == "c3lier":
        modules += UNET_TARGET_REPLACE_MODULE_CONV

    if config.logging.verbose:
        print(metadata)

    if config.logging.use_wandb:
        wandb.init(project=f"LECO_{config.save.name}", config=metadata)

    weight_dtype = config_util.parse_precision(config.train.precision)
    save_weight_dtype = config_util.parse_precision(config.train.precision)

    (
        tokenizers,
        text_encoders,
        unet,
        noise_scheduler,
        vae
    ) = model_util.load_models_xl(
        config.pretrained_model.name_or_path,
        scheduler_name=config.train.noise_scheduler,
    )

    for text_encoder in text_encoders:
        text_encoder.to(device, dtype=weight_dtype)
        text_encoder.requires_grad_(False)
        text_encoder.eval()

    unet.to(device, dtype=weight_dtype)
    if config.other.use_pytorch_SDPA:
        from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
        enable_cudnn_sdp(True)
        enable_flash_sdp(True)
    elif config.other.use_xformers:
        unet.enable_xformers_memory_efficient_attention()

    unet.requires_grad_(False)
    unet.eval()
    
    vae.to(device, dtype=weight_dtype)
    vae.requires_grad_(False)
    vae.eval()

    vae_checksum = None
    if config.train.use_latents:
        vae_checksum = hashlib.sha256(str(vae.state_dict()).encode('utf-8')).hexdigest()

    if config.other.torch_compile:
        unet = torch.compile(unet)
    
    lycorisized = False
    if config.other.lycorisize:
        from lycoris import create_lycoris, LycorisNetwork
        print("Total params:", sum(p.numel() for p in unet.parameters()))
        network = create_lycoris(unet, 
        1.0, #initial multiplier i think?
        config.network.rank, #dim -> linear dim -> (linear dim, conv dim)
        config.network.alpha, #alpha -> linear alpha -> (linear alpha, conv alpha)
        algo="glora")
        network.to(device, dtype=weight_dtype)
        network.apply_to() # mandatory lycoris boilerplate
        
        print("lyc Params:", sum(p.numel() for p in network.parameters()))

        lycorisized = True
    else:
        #lora branch
        network = LoRANetwork(
            unet,
            rank=config.network.rank,
            multiplier=1.0,
            alpha=config.network.alpha,
            train_method=config.network.training_method,
            target_replace=modules, #unstub code from upstream
        ).to(device, dtype=weight_dtype)
    
    optimizer_module = train_util.get_optimizer(config.train.optimizer)
    optimizer_kwargs = {}
    if config.train.optimizer_args is not None and len(config.train.optimizer_args) > 0:
        for arg in config.train.optimizer_args.split(" "):
            key, value = arg.split("=")
            value = ast.literal_eval(value)
            optimizer_kwargs[key] = value
    
    optimizer = optimizer_module(
        network.prepare_optimizer_params(lr=config.train.lr),  #... lycoris... weird compatibility...
        lr=config.train.lr, 
        **optimizer_kwargs)

    lr_scheduler = train_util.get_lr_scheduler(
        config.train.lr_scheduler,
        optimizer,
        max_iterations=config.train.iterations,
        lr_min=config.train.lr / 100,
    )
    criteria = torch.nn.MSELoss()

    print("Prompts")
    for settings in prompts:
        print(settings)

    debug_util.check_requires_grad(network)
    debug_util.check_training_mode(network)

    cache = PromptEmbedsCache()
    prompt_pairs: list[PromptEmbedsPair] = []

    loss_ema = EMA(decay=0.999) # EMA for loss tracking

    with torch.no_grad():
        for settings in prompts:
            print(settings)
            for prompt in [
                settings.target,
                settings.positive,
                settings.neutral,
                settings.unconditional,
            ]:
                if cache[prompt] == None:
                    tex_embs, pool_embs = train_util.encode_prompts_xl(
                            tokenizers,
                            text_encoders,
                            [prompt],
                            num_images_per_prompt=NUM_IMAGES_PER_PROMPT,
                        )
                    cache[prompt] = PromptEmbedsXL(
                        tex_embs,
                        pool_embs
                    )

            prompt_pairs.append(
                PromptEmbedsPair(
                    criteria,
                    cache[settings.target],
                    cache[settings.positive],
                    cache[settings.unconditional],
                    cache[settings.neutral],
                    settings,
                )
            )

    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        del tokenizer, text_encoder

    flush()

    pbar = tqdm(range(config.train.iterations))

    loss = None

    for i in pbar:
        optimizer.zero_grad()

        for (img1_batch, scale1_batch), (img2_batch, scale2_batch) in dataloader:
            img1_batch = img1_batch.to(device, dtype=weight_dtype)
            img2_batch = img2_batch.to(device, dtype=weight_dtype)

            prompt_pair: PromptEmbedsPair = prompt_pairs[
                torch.randint(0, len(prompt_pairs), (1,)).item()
            ]

            loss_high, loss_low = superfunctional_train_step(
                unet,
                vae,
                noise_scheduler,
                (img1_batch, img2_batch),
                (scale1_batch, scale2_batch),
                (prompt_pair.positive, prompt_pair.neutral),
                prompt_pair,
                config,
                network,
                criteria,
                device,
                weight_dtype,
            )
            
            loss = loss_high + loss_low
            
            pbar.set_description(f"Loss*1k: {loss.item()*1000:.4f}")
            
            # Update and log EMA of loss
            ema_loss = loss_ema.update("loss", loss)
            if config.logging.use_wandb:
                wandb.log(
                    {"loss": loss, "ema_loss": ema_loss, "iteration": i, "lr": lr_scheduler.get_last_lr()[0]}
                )

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        if (
            i % config.save.per_steps == 0
            and i != 0
            and i != config.train.iterations - 1
        ):
            print("Saving...")
            save_path.mkdir(parents=True, exist_ok=True)
            network.save_weights(
                save_path / f"{config.save.name}_{i}steps.safetensors",
                dtype=save_weight_dtype,
                metadata=None,
            )

    print("Saving...")
    save_path.mkdir(parents=True, exist_ok=True)
    network.save_weights(
        save_path / f"{config.save.name}_last.safetensors",
        dtype=save_weight_dtype,
        metadata=None,
    )

    del (
        unet,
        noise_scheduler,
        loss,
        optimizer,
        network,
    )

    flush()

    print("Done.")


def main(args):
    config_file = args.config_file

    config = config_util.load_config_from_yaml(config_file)
    training_config = TrainingConfig()

    if args.name is not None:
        config.save.name = args.name
    attributes = []
    if args.attributes is not None:
        attributes = args.attributes.split(',')
        attributes = [a.strip() for a in attributes]
    
    config.network.alpha = args.alpha
    config.network.rank = args.rank
    config.save.name += f'_alpha{args.alpha}'
    config.save.name += f'_rank{config.network.rank }'
    config.save.name += f'_{config.network.training_method}'
    config.save.path += f'/{config.save.name}'
    
    prompts = prompt_util.load_prompts_from_yaml(config.prompts_file, attributes)
    
    device = torch.device(f"cuda:{args.device}")
    
    folders = args.folders.split(',')
    folders = [f.strip() for f in folders]
    scales = args.scales
    
    print(folders, scales)
    if len(scales) != len(folders):
        raise Exception('the number of folders need to match the number of scales')
    
    if args.stylecheck is not None:
        check = args.stylecheck.split('-')
        
        for i in range(int(check[0]), int(check[1])):
            folder_main = args.folder_main+ f'{i}'
            config.save.name = f'{os.path.basename(folder_main)}'
            config.save.name += f'_alpha{args.alpha}'
            config.save.name += f'_rank{config.network.rank }'
            config.save.path = f'models/{config.save.name}'
            train(config=config, training_config=training_config, prompts=prompts, device=device, folder_main = folder_main, folders = folders, scales = scales, use_latents=args.use_latents)
    else:
        train(config=config, training_config=training_config, prompts=prompts, device=device, folder_main = args.folder_main, folders = folders, scales = scales, use_latents=args.use_latents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        required=True,
        help="Config file for training.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="LoRA weight.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        required=False,
        help="Rank of LoRA.",
        default=4,
    )
    parser.add_argument(
        "--device",
        type=int,
        required=False,
        default=0,
        help="Device to train on.",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=False,
        default=None,
        help="Device to train on.",
    )
    parser.add_argument(
        "--attributes",
        type=str,
        required=False,
        default=None,
        help="attritbutes to disentangle (comma seperated string)",
    )
    parser.add_argument(
        "--folder_main",
        type=str,
        required=True,
        help="The folder to check",
    )
    
    parser.add_argument(
        "--stylecheck",
        type=str,
        required=False,
        default = None,
        help="The folder to check",
    )
    
    parser.add_argument(
        "--folders",
        type=str,
        required=False,
        default = 'verylow, low, high, veryhigh',
        help="folders with different attribute-scaled images",
    )
    parser.add_argument(
        "--scales",
        type=float, 
        required=False,
        nargs='*',
        default = [-2, -1, 1, 2],
        help="scales for different attribute-scaled images",
    )
    
    
    parser.add_argument(
        "--use_latents",
        action="store_true",
        help="Use cached latents instead of images.",
    )
    
    args = parser.parse_args()

    main(args)

