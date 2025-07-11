import os
import hashlib
from PIL import Image
import torch
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor

def get_sha256_checksum(file_path):
    """
    Calculates the SHA256 checksum of a file.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The SHA256 checksum of the file.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256.update(byte_block)
    return sha256.hexdigest()

def load_images_from_directory(directory):
    """
    Loads all images from a directory.

    Args:
        directory (str): The path to the directory.

    Returns:
        list: A list of PIL.Image.Image objects.
    """
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img_path = os.path.join(directory, filename)
            images.append(Image.open(img_path).convert("RGB"))
    return images

def encode_images_to_latents(images, vae, device, weight_dtype):
    """
    Encodes a list of images to latents using the VAE.

    Args:
        images (list): A list of PIL.Image.Image objects.
        vae (AutoencoderKL): The VAE model.
        device (torch.device): The device to perform computations on.
        weight_dtype (torch.dtype): The data type for model weights.

    Returns:
        torch.Tensor: A tensor of latents.
    """
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True)
    image_tensors = [image_processor.preprocess(image).to(device, dtype=weight_dtype) for image in images]
    image_batch = torch.cat(image_tensors, dim=0)
    latents = vae.encode(image_batch).latent_dist.sample(None)
    latents = vae.config.scaling_factor * latents
    return latents

import json

def save_latents_to_disk(latents, output_dir, image_path, vae_state_dict):
    """
    Saves a tensor of latents to disk, along with a metadata file containing checksums.

    Args:
        latents (torch.Tensor): A tensor of latents.
        output_dir (str): The directory to save the latents to.
        image_path (str): The path to the original image.
        vae_state_dict (dict): The state dict of the VAE model.
    """
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

    metadata = {
        "image_checksum": get_sha256_checksum(image_path),
        "vae_checksum": vae_checksum,
        "latent_checksum": get_sha256_checksum(latent_path),
    }

    metadata_filename = os.path.splitext(os.path.basename(image_path))[0] + ".json"
    metadata_path = os.path.join(output_dir, metadata_filename)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)


def load_latents_from_disk(directory):
    """
    Loads all latent tensors from a directory.

    Args:
        directory (str): The directory containing the latents.

    Returns:
        list[torch.Tensor]: A list of latent tensors.
    """
    all_latents = []
    for filename in os.listdir(directory):
        if filename.endswith(".pt"):
            latent_path = os.path.join(directory, filename)
            all_latents.append(torch.load(latent_path))
    return all_latents

def check_and_encode_latent(image_path, vae, device, weight_dtype, output_dir, vae_state_dict):
    latent_filename = os.path.splitext(os.path.basename(image_path))[0] + ".pt"
    latent_path = os.path.join(output_dir, latent_filename)
    metadata_filename = os.path.splitext(os.path.basename(image_path))[0] + ".json"
    metadata_path = os.path.join(output_dir, metadata_filename)

    current_image_checksum = get_sha256_checksum(image_path)
    current_vae_checksum_hasher = hashlib.sha256()
    for k, v in vae_state_dict.items():
        current_vae_checksum_hasher.update(k.encode('utf-8'))
        current_vae_checksum_hasher.update(v.cpu().to(torch.float32).numpy().tobytes())
    current_vae_checksum = current_vae_checksum_hasher.hexdigest()

    if os.path.exists(latent_path) and os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            # Verify checksums
            if (metadata.get("image_checksum") == current_image_checksum and
                metadata.get("vae_checksum") == current_vae_checksum and
                metadata.get("latent_checksum") == get_sha256_checksum(latent_path)):
                print(f"Latents for {image_path} are already cached and valid. Skipping.")
                return True
            else:
                print(f"Latents for {image_path} are outdated or corrupted. Re-encoding.")
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"Metadata for {image_path} is corrupted or missing. Re-encoding.")
    else:
        print(f"Latents for {image_path} not found. Encoding.")

    # If we reach here, latents need to be encoded
    image = Image.open(image_path).convert("RGB")
    latents = encode_images_to_latents([image], vae, device, weight_dtype) # encode_images_to_latents expects a list
    save_latents_to_disk(latents, output_dir, image_path, vae_state_dict)
    return False

def get_latent_for_image(image_path, vae, device, weight_dtype, output_dir, vae_state_dict):
    check_and_encode_latent(image_path, vae, device, weight_dtype, output_dir, vae_state_dict)
    latent_filename = os.path.splitext(os.path.basename(image_path))[0] + ".pt"
    latent_path = os.path.join(output_dir, latent_filename)
    return torch.load(latent_path)

if __name__ == "__main__":
    #scope imports where possible
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, help="The directory containing the images to cache.")
    parser.add_argument("--model_name", type=str, help="The name of the VAE model to use.")
    parser.add_argument("--output_dir", type=str, help="The directory to save the latents to.")
    parser.add_argument("--load_latents", action="store_true", help="Load latents from disk instead of encoding images.")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16

    vae = AutoencoderKL.from_pretrained(args.model_name, subfolder="vae").to(device, dtype=weight_dtype)
    vae_state_dict = vae.state_dict()

    image_paths_to_process = []
    for filename in os.listdir(args.image_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_paths_to_process.append(os.path.join(args.image_dir, filename))

    print(f"Checking/encoding latents for {len(image_paths_to_process)} images...")
    for image_path in image_paths_to_process:
        check_and_encode_latent(image_path, vae, device, weight_dtype, args.output_dir, vae_state_dict)

    if args.load_latents:
        print(f"All latents checked/encoded. You can now load them from {args.output_dir}")
    else:
        print(f"All latents checked/encoded. Proceeding with main script logic.")
