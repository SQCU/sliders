import os
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

# --- Checklist for Diagnosing Latent Caching and Batching Slowdown ---
# 1.  **Verify Cache Invalidation Logic:**
#     *   Ensure the VAE checksum calculation is stable and accurately reflects VAE changes, not just string representations.
#     *   Confirm that image checksums are correctly generated and compared.
# 2.  **Profile Latent Encoding Time:**
#     *   Measure the time taken for `vae.encode()` calls with varying batch sizes.
#     *   Compare the time for individual image encoding vs. batched encoding.
# 3.  **Optimize VAE Input Batching:**
#     *   Determine the largest VAE input image batch that can be processed concurrently without OOM errors.
#     *   Adjust the `collate_fn` to create batches of images (not just paths) for VAE encoding, up to the optimal batch size.
# 4.  **Profile Parity Check Mechanism:**
#     *   Measure the time cost of `get_sha256_checksum` for both image files and latent files.
#     *   Evaluate if the overhead of checksumming is significant compared to re-encoding.
# 5.  **Analyze Dataset Repetition:**
#     *   Confirm that the `ImageScaleDataset.__getitem__` correctly cycles through `image_paths` and `prompts_data` to avoid redundant computations for the same physical image/prompt combination within a single epoch.

UNET_ATTENTION_TIME_EMBED_DIM = 256  # XL
TEXT_ENCODER_2_PROJECTION_DIM = 1280
UNET_PROJECTION_CLASS_EMBEDDING_INPUT_DIM = 2816

SDXL_TEXT_ENCODER_TYPE = Union[CLIPTextModel, CLIPTextModelWithProjection]

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

def check_and_encode_latent(image_path, vae, device, weight_dtype, output_dir, vae_state_dict):
    latent_filename = os.path.splitext(os.path.basename(image_path))[0] + ".pt"
    latent_path = os.path.join(output_dir, latent_filename)
    metadata_filename = os.path.splitext(os.path.basename(image_path))[0] + ".json"
    metadata_path = os.path.join(output_dir, metadata_filename)

    current_image_checksum, image_checksum_time = get_sha256_checksum(image_path)
    
    current_vae_checksum_hasher = hashlib.sha256()
    for k, v in vae_state_dict.items():
        current_vae_checksum_hasher.update(k.encode('utf-8'))
        current_vae_checksum_hasher.update(v.cpu().to(torch.float32).numpy().tobytes())
    current_vae_checksum = current_vae_checksum_hasher.hexdigest()

    total_checksum_time = image_checksum_time
    latent_encoding_time = 0

    if os.path.exists(latent_path) and os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            latent_file_checksum, latent_checksum_time = get_sha256_checksum(latent_path)
            total_checksum_time += latent_checksum_time

            if (metadata.get("image_checksum") == current_image_checksum and
                metadata.get("vae_checksum") == current_vae_checksum and
                metadata.get("latent_checksum") == latent_file_checksum):
                # print(f"Latents for {image_path} are already cached and valid. Skipping.")
                return True, latent_encoding_time, total_checksum_time
            else:
                print(f"Latents for {image_path} are outdated or corrupted. Re-encoding.")
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"Metadata for {image_path} is corrupted or missing. Re-encoding.")
    else:
        print(f"Latents for {image_path} not found. Encoding.")

    image = Image.open(image_path).convert("RGB")
    latents, encoding_time = encode_images_to_latents([image], vae, device, weight_dtype)
    latent_encoding_time += encoding_time
    save_latents_to_disk(latents, output_dir, image_path, vae_state_dict)
    return False, latent_encoding_time, total_checksum_time

def get_latent_for_image(image_path, vae, device, weight_dtype, output_dir, vae_state_dict):
    _, encoding_time, checksum_time = check_and_encode_latent(image_path, vae, device, weight_dtype, output_dir, vae_state_dict)
    latent_filename = os.path.splitext(os.path.basename(image_path))[0] + ".pt"
    latent_path = os.path.join(output_dir, latent_filename)
    return torch.load(latent_path), encoding_time, checksum_time

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
    total_encoding_time_batch = 0
    total_checksum_time_batch = 0

    for img_path in image_paths:
        latent, encoding_time, checksum_time = get_latent_for_image(
            img_path, vae, device, weight_dtype, Path(config.dataset_config.dataset.folder_main) / "latents", vae.state_dict()
        )
        latents.append(latent)
        total_encoding_time_batch += encoding_time
        total_checksum_time_batch += checksum_time

    latents = torch.cat(latents, dim=0).to(device, dtype=weight_dtype)

    scales = torch.tensor(scales, dtype=weight_dtype, device=device)
    
    all_text_embeddings = []
    all_pooled_embeds = []

    for prompt_dict in prompts_data:
        text_embeddings, pooled_embeds = create_batched_prompt_embeddings(
            tokenizers,
            text_encoders,
            prompt_dict,
        )
        all_text_embeddings.append(text_embeddings)
        all_pooled_embeds.append(pooled_embeds)

    text_embeddings_batch = torch.cat(all_text_embeddings, dim=0).to(device, dtype=weight_dtype)
    pooled_embeds_batch = torch.cat(all_pooled_embeds, dim=0).to(device, dtype=weight_dtype)

    add_time_ids = get_add_time_ids(
        1024, 1024, False, dtype=latents.dtype
    ).repeat(len(latents), 1).to(device, dtype=weight_dtype)

    return {
        "latents": latents,
        "scales": scales,
        "text_embeddings": text_embeddings_batch,
        "pooled_embeds": pooled_embeds_batch,
        "add_time_ids": add_time_ids,
        "profiling_encoding_time": total_encoding_time_batch,
        "profiling_checksum_time": total_checksum_time_batch,
    }

def dataset_constructor(config, environment):
    dataset = ImageScaleDataset(config)
    
    collate_wrapper = lambda b: collate_fn(b, environment['tokenizers'], environment['text_encoders'], config, environment['vae'], environment['device'], environment['weight_dtype'])
    
    print(f"Batch size from config in batch_dataset_encoding.py: {config.train.batch_size}")
    dataloader = DataLoader(dataset, batch_size=config.train.batch_size, shuffle=True, collate_fn=collate_wrapper)
    return dataloader

def prepare_cached_batches(config, environment):
    """
    Pre-generates and caches all batches for the training loop.
    Includes profiling for latent encoding and parity checks.
    """
    dataloader = dataset_constructor(config, environment)

    print("Pre-generating and caching all batches...")
    static_batches = []
    total_latent_encoding_time = 0
    total_checksum_time = 0

    for i, batch in enumerate(tqdm(dataloader, desc="Caching batches")):
        static_batches.append(batch)
        total_latent_encoding_time += batch["profiling_encoding_time"]
        total_checksum_time += batch["profiling_checksum_time"]
    
    print(f"Cached {len(static_batches)} batches.")
    print(f"Total time spent in latent encoding: {total_latent_encoding_time:.4f} seconds")
    print(f"Total time spent in checksums: {total_checksum_time:.4f} seconds")

    return static_batches

# --- Copied from batch_model_util.py for self-containment ---
AVAILABLE_SCHEDULERS = Literal["ddim", "ddpm", "lms", "euler_a"]

def load_models(config, device, weight_dtype):
    print(f"Loading models from {config.pretrained_model.name_or_path} to device: {device} with dtype: {weight_dtype}")

    pipe = StableDiffusionXLPipeline.from_single_file(
        config.pretrained_model.name_or_path,
        torch_dtype=weight_dtype,
        cache_dir=None,
    )
    unet = pipe.unet.to(device, dtype=weight_dtype)
    vae = pipe.vae.to(device, dtype=weight_dtype)
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
    text_encoders = [pipe.text_encoder, pipe.text_encoder_2]

    if len(text_encoders) == 2:
        text_encoders[1].pad_token_id = 0

    noise_scheduler = create_noise_scheduler(config.train.noise_scheduler)

    unet.requires_grad_(False).eval()
    vae.requires_grad_(False).eval()
    for text_encoder in text_encoders:
        text_encoder.requires_grad_(False).eval()

    return vae, unet, tokenizers, text_encoders, noise_scheduler

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

    vae, unet, tokenizers, text_encoders, noise_scheduler = load_models(config, device, weight_dtype)

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


def main():
    log_file_path, orig_stdout, orig_stderr = setup_logging()
    try:
        print(f"--- Starting Batch Dataset Encoding Test ---", file=orig_stdout)
        print(f"All output will be redirected to: {log_file_path}", file=orig_stdout)
        
        config = config_io()
        environment = envsetup(config)
        
        # We only need VAE, tokenizers, text_encoders for batch caching
        # Unet and noise_scheduler are not used in prepare_cached_batches
        # but are part of the environment returned by envsetup.
        # We can clean them up if memory is an issue, but for now, keep for simplicity.

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