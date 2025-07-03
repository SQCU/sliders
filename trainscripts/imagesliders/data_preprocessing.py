# Written by Gemini 2.5, under review

# Written by Gemini 2.5, under review

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import random
from torchvision import transforms

from .config import TrainingConfig

class PairedImageDataset(Dataset):
    """
    A PyTorch Dataset for loading paired images from folders with different scales.
    """
    def __init__(self, folder_main, folders, scales, config: TrainingConfig, transform=None):
        self.folder_main = folder_main
        self.folders = folders
        self.scales = scales
        self.config = config
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

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        # Aspect ratio preserving resize
        img1.thumbnail(self.config.image_resolution)
        img2.thumbnail(self.config.image_resolution)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return (img1, scale1), (img2, scale2)

def create_dataloader(folder_main, folders, scales, config: TrainingConfig, transform=None):
    """
    Creates a DataLoader for the PairedImageDataset.
    """
    dataset = PairedImageDataset(folder_main, folders, scales, config, transform)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

