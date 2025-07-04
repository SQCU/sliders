# Written by Gemini 2.5, under review

# Written by Gemini 2.5, under review

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import random
from torchvision import transforms

from .config import TrainingConfig

import json

import hashlib

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

class PairedImageDataset(Dataset):
    """
    A PyTorch Dataset for loading paired images from folders with different scales.
    """
    def __init__(self, folder_main, folders, scales, config: TrainingConfig, use_latents=False, vae_checksum=None, transform=None):
        self.folder_main = folder_main
        self.folders = folders
        self.scales = scales
        self.config = config
        self.use_latents = use_latents
        self.vae_checksum = vae_checksum
        self.transform = transform
        self.image_pairs = self._create_image_pairs()

    def _create_image_pairs(self):
        image_pairs = []
        for i in range(len(self.folders)):
            for j in range(i + 1, len(self.folders)):
                folder1 = self.folders[i]
                folder2 = self.folders[j]
                scale1 = self.scales[i]
                scale2 = self.scales[j]

                if self.use_latents:
                    ims1 = [f for f in os.listdir(os.path.join(self.folder_main, folder1)) if f.endswith(".pt")]
                    ims2 = [f for f in os.listdir(os.path.join(self.folder_main, folder2)) if f.endswith(".pt")]
                else:
                    ims1 = os.listdir(os.path.join(self.folder_main, folder1))
                    ims2 = os.listdir(os.path.join(self.folder_main, folder2))

                # Find common images
                common_ims = list(set(ims1) & set(ims2))

                for im_name in common_ims:
                    img1_path = os.path.join(self.folder_main, folder1, im_name)
                    img2_path = os.path.join(self.folder_main, folder2, im_name)
                    image_pairs.append(((img1_path, scale1), (img2_path, scale2)))
        return image_pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        (img1_path, scale1), (img2_path, scale2) = self.image_pairs[idx]

        if self.use_latents:
            latent1_path = img1_path
            latent2_path = img2_path

            metadata1_path = os.path.splitext(latent1_path)[0] + ".json"
            metadata2_path = os.path.splitext(latent2_path)[0] + ".json"

            with open(metadata1_path, "r") as f:
                metadata1 = json.load(f)
            with open(metadata2_path, "r") as f:
                metadata2 = json.load(f)

            # Verify checksums
            if self.vae_checksum is not None:
                if metadata1["vae_checksum"] != self.vae_checksum or metadata2["vae_checksum"] != self.vae_checksum:
                    raise ValueError("VAE checksum mismatch")
            
            if metadata1["latent_checksum"] != get_sha256_checksum(latent1_path) or metadata2["latent_checksum"] != get_sha256_checksum(latent2_path):
                raise ValueError("Latent checksum mismatch")

            img1 = torch.load(latent1_path)
            img2 = torch.load(latent2_path)
        else:
            img1 = Image.open(img1_path).convert("RGB")
            img2 = Image.open(img2_path).convert("RGB")

            # Aspect ratio preserving resize
            img1.thumbnail(self.config.image_resolution)
            img2.thumbnail(self.config.image_resolution)

            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

        return (img1, scale1), (img2, scale2)

def create_dataloader(folder_main, folders, scales, config: TrainingConfig, use_latents=False, vae_checksum=None, transform=None):
    """
    Creates a DataLoader for the PairedImageDataset.
    """
    dataset = PairedImageDataset(folder_main, folders, scales, config, use_latents, vae_checksum, transform)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

