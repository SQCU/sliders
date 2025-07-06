import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
import hashlib
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import AutoencoderKL, DDPMScheduler, SchedulerMixin, DDIMScheduler, LMSDiscreteScheduler, EulerAncestralDiscreteScheduler, StableDiffusionXLPipeline
from diffusers.image_processor import VaeImageProcessor
import json
import yaml
from pathlib import Path
import time
from tqdm import tqdm
from typing import Tuple, Union, Literal, List, Dict, Any
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
import datetime
import sys
import argparse

# --- Training Data Flow in Diffusion Models ---
# In diffusion model training, the objective is to learn a transition kernel that transforms a noisy example (x_t)
# back to a cleaner version (x_t-1), eventually leading to the original, clean data (x_0).
#
# x_0: This is the invariant "clean latent" or "clean image embedding" of our training data.
#      It's the ground truth that the model aims to recover from noise.
# eps: This is a literal noise signal, typically sampled from a Gaussian distribution, which is added to x_0
#      to create x_t. The amount of noise added depends on the timestep 't'.
# x_t: A noisy version of x_0 at a specific timestep 't'.
# x_T: The training example with the maximum amount of noise signal added, as defined by the training protocol.
#
# Our training protocol involves caching and persisting *only* the x_0 values (clean latents) for our training batches.
# This is because generating noise (eps) is computationally inexpensive compared to the bandwidth to/from tensor cores
# for VAE encoding/decoding. By pre-caching x_0, we avoid redundant VAE encoding during training.
# We also want to persist the pseudorandom number generator (PRNG) seed used for 'random' calls for each batch element,
# though this is not yet implemented.
#
# The current issue is that the latent caching process is slow, often re-computing latents for unchanged images.
# This suggests issues with the cache validation or the VAE encoding process itself.


#DO NOT DELETE THESE HYPERPARAMETERS. WHY WOULD YOU EVEN THINK OF SUCH A THING.
UNET_ATTENTION_TIME_EMBED_DIM = 256  # XL
TEXT_ENCODER_2_PROJECTION_DIM = 1280
UNET_PROJECTION_CLASS_EMBEDDING_INPUT_DIM = 2816

SDXL_TEXT_ENCODER_TYPE = Union[CLIPTextModel, CLIPTextModelWithProjection]
DIFFUSERS_CACHE_DIR = None # if you want to change the cache dir, change this

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

def get_sha256_checksum(file_path):
    start_time = time.time()
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256.update(byte_block)
    end_time = time.time()
    return sha256.hexdigest(), (end_time - start_time)

def encode_images_to_latents(images, vae, device, weight_dtype):
    start_time = time.time()
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True)
    image_tensors = [image_processor.preprocess(image).to(device, dtype=weight_dtype) for image in images]
    image_batch = torch.cat(image_tensors, dim=0)
    latents = vae.encode(image_batch).latent_dist.sample(None)
    end_time = time.time()
    return latents, (end_time - start_time)

def save_latents_to_disk(latents, output_dir, image_path, vae_state_dict):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    latent_filename = os.path.splitext(os.path.basename(image_path))[0] + ".pt"
    latent_path = os.path.join(output_dir, latent_filename)
    torch.save(latents, latent_path)

    vae_checksum_hasher = hashlib.sha256()
    for k, v in vae_state_dict.items():
        vae_checksum_hasher.update(k.encode('utf-8'))
        vae_checksum_hasher.update(v.cpu().to(torch.float32).numpy().tobytes())
    vae_checksum = vae_checksum_hasher.hexdigest()

    latent_checksum, _ = get_sha256_checksum(latent_path) # Get checksum without timing here
    metadata = {
        "image_checksum": get_sha256_checksum(image_path)[0], # Get checksum without timing here
        "vae_checksum": vae_checksum,
        "latent_checksum": latent_checksum,
    }

    metadata_filename = os.path.splitext(os.path.basename(image_path))[0] + ".json"
    metadata_path = os.path.join(output_dir, metadata_filename)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

def check_and_encode_latent(image_path, vae, device, weight_dtype, output_dir, vae_state_dict, force_reencode=False):
    latent_filename = os.path.splitext(os.path.basename(image_path))[0] + ".pt"
    latent_path = os.path.join(output_dir, latent_filename)

    latent_encoding_time = 0

    if not force_reencode and os.path.exists(latent_path):
        # If not forcing re-encode and latent exists, assume it's valid for now
        # (checksumming logic removed as per instruction)
        return True, latent_encoding_time
    else:
        if force_reencode:
            print(f"Force re-encoding latents for {image_path}.")
        else:
            print(f"Latents for {image_path} not found. Encoding.")

    image = Image.open(image_path).convert("RGB")
    latents, encoding_time = encode_images_to_latents([image], vae, device, weight_dtype)
    latent_encoding_time += encoding_time
    save_latents_to_disk(latents, output_dir, image_path, vae_state_dict)
    return False, latent_encoding_time

def get_latent_for_image(image_path, vae, device, weight_dtype, output_dir, vae_state_dict, force_reencode=False):
    is_cached, encoding_time = check_and_encode_latent(image_path, vae, device, weight_dtype, output_dir, vae_state_dict, force_reencode=force_reencode)
    latent_filename = os.path.splitext(os.path.basename(image_path))[0] + ".pt"
    latent_path = os.path.join(output_dir, latent_filename)
    
    load_start_time = time.time()
    loaded_latent = torch.load(latent_path, weights_only=True)
    load_end_time = time.time()
    latent_load_time = load_end_time - load_start_time

    return loaded_latent, encoding_time, latent_load_time

# --- Copied from batch_train_util.py for self-containment ---

def text_tokenize(
    tokenizer: CLIPTokenizer,
    prompts: list[str],
):
    token_ids = [
        tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        for prompt in prompts
    ]
    return torch.cat(token_ids)


def text_encode_xl(
    text_encoder: SDXL_TEXT_ENCODER_TYPE,
    tokens: torch.FloatTensor,
):
    prompt_embeds = text_encoder(
        tokens.to(text_encoder.device), output_hidden_states=True
    )
    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]  # always penultimate layer

    return prompt_embeds, pooled_prompt_embeds

def encode_prompts_xl(
    tokenizers: list[CLIPTokenizer],
    text_encoders: list[SDXL_TEXT_ENCODER_TYPE],
    prompts: list[str],
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    text_embeds_list = []
    pooled_text_embeds = None

    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        text_tokens_input_ids = text_tokenize(tokenizer, prompts)
        text_embeds, pooled_text_embeds = text_encode_xl(
            text_encoder, text_tokens_input_ids,
        )

        text_embeds_list.append(text_embeds)

    return torch.concat(text_embeds_list, dim=-1), pooled_text_embeds


def create_batched_prompt_embeddings(
    tokenizers: list[CLIPTokenizer],
    text_encoders: list[SDXL_TEXT_ENCODER_TYPE],
    prompts: dict,
):
    positive_prompts = [prompts['positive']]
    unconditional_prompts = [prompts['unconditional']]
    neutral_prompts = [prompts['neutral']]

    positive_text_embeds, positive_pooled_embeds = encode_prompts_xl(
        tokenizers, text_encoders, positive_prompts,
    )
    unconditional_text_embeds, unconditional_pooled_embeds = encode_prompts_xl(
        tokenizers, text_encoders, unconditional_prompts,
    )
    neutral_text_embeds, neutral_pooled_embeds = encode_prompts_xl(
        tokenizers, text_encoders, neutral_prompts,
    )

    text_embeddings_for_noise_pred = torch.cat([
        positive_text_embeds,
        unconditional_text_embeds,
        neutral_text_embeds,
    ], dim=0)
    pooled_embeds_for_noise_pred = torch.cat([
        positive_pooled_embeds,
        unconditional_pooled_embeds,
        neutral_pooled_embeds,
    ], dim=0)

    return text_embeddings_for_noise_pred, pooled_embeds_for_noise_pred

def get_add_time_ids(
    height: int,
    width: int,
    dynamic_crops: bool = False,
    dtype: torch.dtype = torch.float32,
):
    if dynamic_crops:
        random_scale = torch.rand(1).item() * 2 + 1
        original_size = (int(height * random_scale), int(width * random_scale))
        crops_coords_top_left = (
            torch.randint(0, original_size[0] - height, (1,)).item(),
            torch.randint(0, original_size[1] - width, (1,)).item(),
        )
        target_size = (height, width)
    else:
        original_size = (height, width)
        crops_coords_top_left = (0, 0)
        target_size = (height, width)

    add_time_ids = list(original_size + crops_coords_top_left + target_size)

    passed_add_embed_dim = (
        UNET_ATTENTION_TIME_EMBED_DIM * len(add_time_ids)
        + TEXT_ENCODER_2_PROJECTION_DIM
    )
    if passed_add_embed_dim != UNET_PROJECTION_CLASS_EMBEDDING_INPUT_DIM:
        raise ValueError(
            f"Model expects an added time embedding vector of length {UNET_PROJECTION_CLASS_EMBEDDING_INPUT_DIM}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
        )

    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    return add_time_ids




def initialize_latent_cache(config, environment):
    dataset = ImageScaleDataset(config)
    unique_image_paths = sorted(list(set(dataset.image_paths))) # sorted for determinism
    print(f"Found {len(unique_image_paths)} unique images to process for latent cache initialization.")

    vae = environment['vae']
    device = environment['device']
    weight_dtype = environment['weight_dtype']
    output_dir = Path(config.dataset_config.dataset.folder_main) / "latents"
    
    # Let's use a batch size from the config, or a default.
    # The main training batch size might be too large for VAE encoding depending on image size.
    # Let's add a specific config for this.
    vae_batch_size = config.train.get("vae_encoding_batch_size", 4)
    print(f"Using VAE encoding batch size: {vae_batch_size}")

    total_comparison_time = 0
    mismatched_files = 0

    for i in tqdm(range(0, len(unique_image_paths), vae_batch_size), desc="Initializing latent cache"):
        batch_paths = unique_image_paths[i:i+vae_batch_size]
        
        images_to_encode = [Image.open(p).convert("RGB") for p in batch_paths]
        
        # Encode the batch of images
        new_latents_gpu, encoding_time = encode_images_to_latents(images_to_encode, vae, device, weight_dtype)
        new_latents_cpu = new_latents_gpu.cpu()

        # Compare and save for each image in the batch
        for j, img_path in enumerate(batch_paths):
            new_latent_single = new_latents_cpu[j].unsqueeze(0)
            
            latent_filename = os.path.splitext(os.path.basename(img_path))[0] + ".pt"
            latent_path = os.path.join(output_dir, latent_filename)

            if os.path.exists(latent_path):
                try:
                    old_latent = torch.load(latent_path, map_location='cpu')
                    
                    comparison_start_time = time.time()
                    are_close = torch.allclose(old_latent, new_latent_single, atol=1e-4, rtol=1e-3) # Use tolerances
                    comparison_time = time.time() - comparison_start_time
                    total_comparison_time += comparison_time
                    
                    if not are_close:
                        mismatched_files += 1
                        diff = torch.mean(torch.abs(old_latent - new_latent_single))
                        print(f"Latent mismatch for {img_path}. Mean absolute difference: {diff.item()}. Comparison took {comparison_time:.4f}s.")
                except Exception as e:
                    print(f"Warning: Could not load or compare existing latent for {img_path}: {e}")

            # Save the new latent, overwriting if it exists
            torch.save(new_latent_single, latent_path)
            
    print(f"Latent cache initialization finished.")
    print(f"Total time spent on parity checks: {total_comparison_time:.4f} seconds.")
    if mismatched_files > 0:
        print(f"Found {mismatched_files} mismatched latents that were updated.")

from .data_schedule import TrainingSchedule

def initialize_text_embedding_cache(config, environment, training_schedule):
    print("Initializing text embedding cache...")
    tokenizers = environment['tokenizers']
    text_encoders = environment['text_encoders']
    device = environment['device']
    weight_dtype = environment['weight_dtype']

    unique_prompts = training_schedule.get_unique_prompts()
    text_embedding_cache = {}

    for prompt_dict in tqdm(unique_prompts, desc="Encoding unique prompts"):
        text_embeddings, pooled_embeds = create_batched_prompt_embeddings(
            tokenizers,
            text_encoders,
            prompt_dict,
        )
        # Store the embeddings in CPU memory to save VRAM, move to GPU when needed
        text_embedding_cache[frozenset(prompt_dict.items())] = (
            text_embeddings.cpu(), pooled_embeds.cpu()
        )
    print(f"Cached {len(unique_prompts)} unique text embeddings.")
    return text_embedding_cache

def prepare_cached_batches(config, environment):
    """
    Pre-generates and caches all batches for the training loop using the TrainingSchedule.
    """
    if config.train.get("force_init_latentcache", False):
        initialize_latent_cache(config, environment)

    training_schedule = TrainingSchedule(config)
    text_embedding_cache = initialize_text_embedding_cache(config, environment, training_schedule)

    print("Pre-generating and caching all batches from schedule...")
    static_batches = []
    total_latent_encoding_time = 0
    total_latent_load_time = 0
    total_cat_latents_time = 0
    total_cat_text_embeds_time = 0
    total_cat_pooled_embeds_time = 0

    vae = environment['vae']
    device = environment['device']
    weight_dtype = environment['weight_dtype']
    tokenizers = environment['tokenizers']
    text_encoders = environment['text_encoders']

    for batch_items in tqdm(training_schedule, desc="Caching batches"):
        latents = []
        scales = []
        all_text_embeddings = []
        all_pooled_embeds = []

        pair_indices = []
        is_low_cases = []
        for item in batch_items:
            latent, encoding_time, load_time = get_latent_for_image(
                item.image_path, vae, device, weight_dtype, 
                Path(config.dataset_config.dataset.folder_main) / "latents", 
                vae.state_dict(),
                force_reencode=config.train.get("force_reencode_latents", False)
            )
            latents.append(latent)
            scales.append(item.scale)
            pair_indices.append(item.pair_index)
            is_low_cases.append(item.is_low_case)
            total_latent_encoding_time += encoding_time
            total_latent_load_time += load_time

            # Retrieve text embeddings from cache
            cached_text_embeds, cached_pooled_embeds = text_embedding_cache[frozenset(item.prompt.items())]
            all_text_embeddings.append(cached_text_embeds.to(device, dtype=weight_dtype))
            all_pooled_embeds.append(cached_pooled_embeds.to(device, dtype=weight_dtype))

        cat_latents_start_time = time.time()
        latents_batch = torch.stack(latents).to(device, dtype=weight_dtype)
        cat_latents_time = time.time() - cat_latents_start_time

        scales_batch = torch.tensor(scales, dtype=weight_dtype, device=device)

        cat_text_embeds_start_time = time.time()
        text_embeddings_batch = torch.cat(all_text_embeddings, dim=0).to(device, dtype=weight_dtype)
        cat_text_embeds_time = time.time() - cat_text_embeds_start_time

        cat_pooled_embeds_start_time = time.time()
        pooled_embeds_batch = torch.cat(all_pooled_embeds, dim=0).to(device, dtype=weight_dtype)
        cat_pooled_embeds_time = time.time() - cat_pooled_embeds_start_time

        add_time_ids = get_add_time_ids(
            1024, 1024, False, dtype=latents_batch.dtype
        ).repeat(len(latents_batch), 1).to(device, dtype=weight_dtype)

        batch = {
            "latents": latents_batch,
            "scales": scales_batch,
            "text_embeddings": text_embeddings_batch,
            "pooled_embeds": pooled_embeds_batch,
            "add_time_ids": add_time_ids,
            "pair_indices": torch.tensor(pair_indices, dtype=torch.long, device=device),
            "is_low_cases": torch.tensor(is_low_cases, dtype=torch.bool, device=device),
        }
        static_batches.append(batch)

        # Debug print for the first batch
        if len(static_batches) == 1:
            print("\n--- Example Training Batch (First Batch) ---")
            print("Tensor Shapes:")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
                else:
                    print(f"  {key}: {type(value)}")
            
            print("\nMetadata for Cached Items:")
            for i, item in enumerate(batch_items):
                print(f"  Item {i}:")
                print(f"    Image Path: {item.image_path}")
                print(f"    Prompt: {item.prompt}")
                print(f"    Scale: {item.scale}")
                print(f"    Pair Index: {item.pair_index}")
                print(f"    Is Low Case: {item.is_low_case}")
            print("--------------------------------------------")

        total_cat_latents_time += cat_latents_time
        total_cat_text_embeds_time += cat_text_embeds_time
        total_cat_pooled_embeds_time += cat_pooled_embeds_time

    print(f"Cached {len(static_batches)} batches.")
    print(f"Total time spent in latent encoding: {total_latent_encoding_time:.4f} seconds")
    print(f"Total time spent in latent loading: {total_latent_load_time:.4f} seconds")
    print(f"Total time spent concatenating latents: {total_cat_latents_time:.4f} seconds")
    print(f"Total time spent concatenating text embeddings: {total_cat_text_embeds_time:.4f} seconds")
    print(f"Total time spent concatenating pooled embeddings: {total_cat_pooled_embeds_time:.4f} seconds")

    return static_batches

# --- Copied from batch_model_util.py for self-containment ---
AVAILABLE_SCHEDULERS = Literal["ddim", "ddpm", "lms", "euler_a"]

def load_models(config, device, weight_dtype):
    print(f"Loading models from {config.pretrained_model.name_or_path} to device: {device} with dtype: {weight_dtype}")

    # Load the pipeline from the local .safetensors file
    pipe = StableDiffusionXLPipeline.from_single_file(
        config.pretrained_model.name_or_path,
        torch_dtype=weight_dtype,
        cache_dir=DIFFUSERS_CACHE_DIR,
    )

    unet = pipe.unet
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
    text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
    if len(text_encoders) == 2:
        text_encoders[1].pad_token_id = 0
    vae = pipe.vae
    del pipe
    torch.cuda.empty_cache()
    #GET RID OF PIPE!!! 
    #you HAVE TO GET RID OF THE PIPE EVERY TIME!!!
    #if you do a 'blah = pipe.blah.to(device, dtype)' you DOUBLE LOAD THE MODEL,

    unet.requires_grad_(False).eval()
    vae.requires_grad_(False).eval()
    for text_encoder in text_encoders:
        text_encoder.requires_grad_(False).eval()

    unet = unet.to(device, weight_dtype)
    #tokenizers = [tokenizer.to(device, weight_dtype) for tokenizer in tokenizers]
    text_encoders = [text_encoder.to(device, weight_dtype) for text_encoder in text_encoders]
    vae = vae.to(device, weight_dtype)
    #i HATE huggingface wwrappers

    return vae, unet, tokenizers, text_encoders

def create_noise_scheduler(
    scheduler_name: AVAILABLE_SCHEDULERS = "ddpm",
    prediction_type: Literal["epsilon", "v_prediction"] = "epsilon",
) -> SchedulerMixin:
    name = scheduler_name.lower().replace(" ", "_")
    if name == "ddim":
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
            prediction_type=prediction_type,
        )
    elif name == "ddpm":
        scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
            prediction_type=prediction_type,
        )
    elif name == "lms":
        scheduler = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            prediction_type=prediction_type,
        )
    elif name == "euler_a":
        scheduler = EulerAncestralDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            prediction_type=prediction_type,
        )
    else:
        raise ValueError(f"Unknown scheduler name: {name}")

    return scheduler

# --- Copied from batch_config_util.py for self-containment ---
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

def envsetup(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = parse_precision(config.train.precision)
    vae, unet, tokenizers, text_encoders = load_models(config, device, weight_dtype)

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
    }
    return environment

def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(log_dir, f"batch_dataset_encoding_test_{timestamp}.log")
    
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    log_file = open(log_filename, "w")
    sys.stdout = log_file
    sys.stderr = log_file
    
    print(f"Logging output to {log_filename}")
    
    return log_filename, original_stdout, original_stderr


import torch.cuda as cuda

def get_memory_usage():
    # Return memory usage in MB
    return {
        'allocated': cuda.memory_allocated() / 1024**2,
        'reserved': cuda.memory_reserved() / 1024**2,
        'peak': cuda.max_memory_allocated() / 1024**2
    }

import random
from PIL import Image, ImageOps

import math
import random
from PIL import Image, ImageOps


def find_optimal_vae_batch_size(config, environment, vram_max_threshold=0.85, max_test_batch_size=128, slowdown_threshold=0.5):
    print("--- Starting Dynamic VAE Batch Size Bounds Test ---")
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping VRAM tests.")
        return

    try:
        backend = torch.cuda.get_allocator_backend()
        print(f"PyTorch CUDA Allocator Backend: {backend}")
    except Exception as e:
        print(f"Could not determine CUDA allocator backend: {e}")

    training_schedule = TrainingSchedule(config)
    unique_image_paths = sorted(list(set([item.image_path for batch in training_schedule for item in batch])))
    if not unique_image_paths:
        print("No unique images found to perform bounds test.")
        return

    vae = environment['vae']
    device = environment['device']
    weight_dtype = environment['weight_dtype']
    
    num_base_images = min(len(unique_image_paths), 16)
    base_test_images = [Image.open(p).convert("RGB") for p in unique_image_paths[:num_base_images]]

    total_vram = cuda.get_device_properties(0).total_memory
    vram_max_bytes = total_vram * vram_max_threshold
    
    torch.cuda.synchronize()
    cuda.empty_cache()
    base_mem = get_memory_usage()['allocated']
    print(f"\nBase VRAM usage (models loaded): {base_mem:.2f} MB")
    print(f"Total VRAM: {total_vram / 1024**2:.2f} MB, Targeting <= {vram_max_threshold*100:.0f}%: {vram_max_bytes / 1024**2:.2f} MB")

    results = []
    peak_throughput = 0.0
    print("\n--- Running Incremental Benchmark ---")
    
    batch_size = 1
    while batch_size <= max_test_batch_size:
        print(f"\nTesting VAE encoding with batch size: {batch_size}")
        test_batch = [base_test_images[i % len(base_test_images)] for i in range(batch_size)]

        torch.cuda.synchronize()
        cuda.empty_cache()
        cuda.reset_peak_memory_stats()
        
        pre_test_mem = get_memory_usage()['allocated']
        print(f"  VRAM allocated before encoding: {pre_test_mem:.2f} MB")

        try:
            torch.cuda.synchronize()
            start_time = time.time()
            
            _ = encode_images_to_latents(test_batch, vae, device, weight_dtype)
            
            torch.cuda.synchronize()
            end_time = time.time()

            peak_mem_bytes = cuda.max_memory_allocated()
            peak_mem_mb = peak_mem_bytes / 1024**2
            total_time = end_time - start_time
            images_per_second = batch_size / total_time if total_time > 0 else float('inf')

            print(f"  Success! Throughput: {images_per_second:.2f} images/sec, Peak VRAM: {peak_mem_mb:.2f} MB")

            current_result = {
                'batch_size': batch_size,
                'images_per_second': images_per_second,
                'peak_vram_mb': peak_mem_mb,
                'status': 'Success'
            }
            results.append(current_result)

            if peak_mem_bytes > vram_max_bytes:
                print(f"  Stopping: Peak VRAM ({peak_mem_mb:.2f} MB) exceeded target threshold ({vram_max_bytes / 1024**2:.2f} MB).")
                current_result['status'] = 'VRAM Exceeded'
                break

            if peak_throughput > 0 and images_per_second < peak_throughput * slowdown_threshold:
                print(f"  Stopping: Throughput ({images_per_second:.2f} img/s) dropped significantly from peak ({peak_throughput:.2f} img/s).")
                print(f"  This likely indicates VRAM spillover to shared/paged memory.")
                current_result['status'] = 'Slowdown'
                break

            peak_throughput = max(peak_throughput, images_per_second)

            if batch_size < 8:
                batch_size += 1
            elif batch_size < 32:
                batch_size += 4
            else:
                batch_size += 8

        except torch.cuda.OutOfMemoryError:
            print(f"  Failed with OOM at batch size {batch_size}.")
            peak_mem_mb = cuda.max_memory_allocated() / 1024**2
            results.append({'batch_size': batch_size, 'images_per_second': 0, 'peak_vram_mb': peak_mem_mb, 'status': 'OOM'})
            break
        except Exception as e:
            print(f"  Failed with an unexpected error at batch size {batch_size}: {e}")
            break

    print("\n--- VAE Batch Size Bounds Test Summary ---")
    if not results:
        print("No successful runs to summarize.")
        return

    successful_results = [r for r in results if r['status'] == 'Success']

    if not successful_results:
        print("No runs completed successfully within the VRAM or performance limits.")
        if results:
             res = results[0]
             print(f"First attempt failed at Batch Size {res['batch_size']} with status: {res['status']}")
        return

    for res in results:
        print(f"Batch Size: {res['batch_size']:<4} | Status: {res['status']:<15} | Throughput: {res['images_per_second']:<8.2f} images/sec | Peak VRAM: {res['peak_vram_mb']:<8.2f} MB")

    best_throughput_result = max(successful_results, key=lambda x: x['images_per_second'])
    largest_successful_batch = max(successful_results, key=lambda x: x['batch_size'])

    print(f"\nLargest successful batch size: {largest_successful_batch['batch_size']}")
    print(f"Batch size with best throughput: {best_throughput_result['batch_size']} ({best_throughput_result['images_per_second']:.2f} images/sec)")
    print(f"\nRecommendation:")
    print(f"Setting 'vae_encoding_batch_size' in config to {largest_successful_batch['batch_size']} for a balance of speed and stability.")
    config.train.vae_encoding_batch_size = largest_successful_batch['batch_size']

def main():
    log_file_path, orig_stdout, orig_stderr = setup_logging()
    try:
        print(f"--- Starting Batch Dataset Encoding Test ---", file=orig_stdout)
        print(f"All output will be redirected to: {log_file_path}", file=orig_stdout)
        
        config = config_io()
        environment = envsetup(config)

        if config.train.get("bounds_test_vae_batch_size", False):
            print("Preparing for VAE bounds test by removing UNet and cleaning memory...")
            import gc
            if 'unet' in environment:
                del environment['unet']
            gc.collect()
            torch.cuda.empty_cache()
            find_optimal_vae_batch_size(config, environment)
        else:
            static_batches = prepare_cached_batches(config, environment)
            
            print(f"Successfully prepared {len(static_batches)} batches.", file=orig_stdout)
        
    except Exception as e:
        import traceback
        print("--- EXCEPTION OCCURRED ---", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print(f"\n--- EXCEPTION OCCURRED ---", file=orig_stderr)
        print(f"An error occurred. Check the log file for details: {log_file_path}", file=orig_stderr)
        traceback.print_exc(file=orig_stderr)
        raise
        
    finally:
        sys.stdout.close()
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        print(f"--- Script finished. Log saved to {log_file_path} ---", file=orig_stdout)


if __name__ == "__main__":
    main()