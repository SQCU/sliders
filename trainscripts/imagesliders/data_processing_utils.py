#data_processing_utils.py
import torch
import os
import hashlib
import time
import json
from PIL import Image
from diffusers.image_processor import VaeImageProcessor
from typing import Tuple

def get_sha256_checksum(file_path):
    start_time = time.time()
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256.update(byte_block)
    end_time = time.time()
    return sha256.hexdigest(), (end_time - start_time)

def resize_image_if_needed(image: Image.Image, max_resolution: Tuple[int, int]) -> Image.Image:
    """
    Resizes a PIL image to fit within a maximum resolution while preserving aspect ratio.
    
    Args:
        image (Image.Image): The input image.
        max_resolution (Tuple[int, int]): A tuple of (max_width, max_height).

    Returns:
        Image.Image: The resized image, or the original if it was already small enough.
    """
    max_w, max_h = max_resolution
    if image.width <= max_w and image.height <= max_h:
        return image

    # Calculate the new size
    width_ratio = max_w / image.width
    height_ratio = max_h / image.height
    ratio = min(width_ratio, height_ratio) # Use the smaller ratio to ensure both dims fit

    new_width = int(image.width * ratio)
    new_height = int(image.height * ratio)

    # Use LANCZOS for high-quality downsampling
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return resized_image

def encode_images_to_latents(images, vae, device, weight_dtype):
    start_time = time.time()
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True)
    image_tensors = [image_processor.preprocess(image).to(dtype=weight_dtype) for image in images]
    image_batch = torch.cat(image_tensors, dim=0)
    latents = vae.encode(image_batch).latent_dist.sample(None)
    #...huh?
    latents = vae.config.scaling_factor * latents
    end_time = time.time()
    return latents.to(device=torch.device("cpu")), (end_time - start_time)

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

def check_and_encode_latent(image_path, vae, device, weight_dtype, output_dir, vae_state_dict, config, force_reencode=False):
    latent_filename = os.path.splitext(os.path.basename(image_path))[0] + ".pt"
    latent_path = os.path.join(output_dir, latent_filename)

    latent_encoding_time = 0

    if not force_reencode and os.path.exists(latent_path):
        return True, latent_encoding_time
    else:
        if force_reencode:
            print(f"Force re-encoding latents for {image_path}.")
        else:
            print(f"Latents for {image_path} not found. Encoding.")

    max_resolution = tuple(config.train.get("resolution", (512, 512)))
    image = Image.open(image_path).convert("RGB")
    image = resize_image_if_needed(image, max_resolution)

    latents_gpu, encoding_time = encode_images_to_latents([image], vae, device, weight_dtype)
    latent_encoding_time += encoding_time
    save_latents_to_disk(latents_gpu.cpu(), output_dir, image_path, vae_state_dict)
    return False, latent_encoding_time

def get_latent_for_image(image_path, vae, device, weight_dtype, output_dir, vae_state_dict, config, force_reencode=False):
     # This function now has an internal 'device' argument for the VAE call,
    # but the returned latent will be on the CPU.
    is_cached, encoding_time = check_and_encode_latent(image_path, vae, 
    device, weight_dtype, output_dir, vae_state_dict, config, 
    force_reencode=force_reencode)
    latent_filename = os.path.splitext(os.path.basename(image_path))[0] + ".pt"
    latent_path = os.path.join(output_dir, latent_filename)
    
    load_start_time = time.time()
    # Load the latent tensor. It's already on the CPU since we save it that way.
    # The map_location is a good safeguard.
    loaded_latent = torch.load(latent_path, map_location="cpu",  weights_only=True)
    load_end_time = time.time()
    latent_load_time = load_end_time - load_start_time

    return loaded_latent, encoding_time, latent_load_time
