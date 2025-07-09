import torch
import yaml
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
import itertools
from collections import defaultdict
from tqdm import tqdm # New import
from .embedding_cache_util import build_unified_embedding_cache # New import
from .batch_train_util import get_add_time_ids # New import

class TrainingItem:
    def __init__(self, image_path: str, scale: float, prompt: Dict[str, Any], pair_index: int, is_low_case: bool):
        self.image_path = image_path
        self.scale = scale
        self.prompt = prompt
        self.pair_index = pair_index
        self.is_low_case = is_low_case

    def __repr__(self):
        return f"TrainingItem(image={Path(self.image_path).name}, scale={self.scale}, pair={self.pair_index}, low_case={self.is_low_case})"

class TrainingSchedule:
    def __init__(self, config):
        self.config = config
        self.schedule: List[List[TrainingItem]] = []
        self._build_schedule()

    def _get_data_pool(self) -> Dict[str, Dict[float, str]]:
        """
        Creates a master dictionary of image paths, grouped by filename and scale.
        Returns:
            Dict[str, Dict[float, str]]: A dictionary where keys are image filenames (e.g., "image001.png")
                                         and values are dictionaries mapping scales to their full image paths.
        """
        print("Collecting all possible training data combinations...")
        
        image_data_by_filename: Dict[str, Dict[float, str]] = {}
        subfolder_names = [f.strip() for f in self.config.dataset.folders.split(',')]
        scale_values = [float(s.strip()) for s in self.config.dataset.scales.split(',')]
        
        if len(subfolder_names) != len(scale_values):
            raise ValueError("Number of folders must match number of scales in dataset configuration.")

        for i, folder_name in enumerate(subfolder_names):
            subfolder_path = Path(self.config.dataset.folder_main) / folder_name
            current_scale = scale_values[i]
            
            for image_path in subfolder_path.glob("*"):
                if image_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
                    filename = image_path.name
                    if filename not in image_data_by_filename:
                        image_data_by_filename[filename] = {}
                    image_data_by_filename[filename][current_scale] = str(image_path)

        print(f"Created data pool with {len(image_data_by_filename)} unique image filenames across scales.")
        return image_data_by_filename

    def _build_schedule(self):
        print("Building pseudo-randomized training schedule...")
        
        seed = self.config.train.get('seed', 42)
        rng = random.Random(seed)
        print(f"Using random seed: {seed}")

        data_pool_by_filename = self._get_data_pool()
        if not data_pool_by_filename:
            raise ValueError("Data pool is empty. Check your dataset configuration.")

        all_image_filenames = list(data_pool_by_filename.keys())
        all_scales = sorted(list(set(scale for filename_data in data_pool_by_filename.values() for scale in filename_data.keys())))

        with open(self.config.dataset.prompts_file, 'r') as f:
            prompts_data = yaml.safe_load(f)
        if not prompts_data:
            raise ValueError("Prompts data is empty. Check your prompts file.")

        total_training_steps = self.config.train.iterations
        batch_size = self.config.train.batch_size
        
        if batch_size % 2 != 0:
            raise ValueError("Batch size must be an even number to support paired training.")

        for i in range(total_training_steps):
            batch_items = []
            for j in range(batch_size // 2): # Iterate for pairs
                # 1. Randomly select a prompt
                selected_prompt = rng.choice(prompts_data)

                # 2. Randomly select an image filename that exists across all relevant scale folders
                #    (i.e., has entries for all scales in data_pool_by_filename)
                valid_filenames = [f for f, scales_map in data_pool_by_filename.items() if all(s in scales_map for s in all_scales)]
                if not valid_filenames:
                    raise ValueError("No image filenames found that exist across all configured scales. Ensure your dataset is complete.")
                selected_filename = rng.choice(valid_filenames)
                
                # 3. Randomly select two distinct scales and sort them
                if len(all_scales) < 2:
                    raise ValueError("Need at least two distinct scales configured for paired training.")
                
                # Ensure we pick two distinct scales
                scale_choices = rng.sample(all_scales, 2)
                low_scale, high_scale = sorted(scale_choices)

                # Get image paths for the selected scales and filename
                image_path_low = data_pool_by_filename[selected_filename][low_scale]
                image_path_high = data_pool_by_filename[selected_filename][high_scale]

                # Create TrainingItem for high_scale (positive/target)
                item_high = TrainingItem(
                    image_path=image_path_high,
                    scale=high_scale,
                    prompt=selected_prompt,
                    pair_index=j,
                    is_low_case=False # This is the 'high' or 'positive' case
                )
                batch_items.append(item_high)

                # Create TrainingItem for low_scale (neutral/negative)
                item_low = TrainingItem(
                    image_path=image_path_low,
                    scale=low_scale,
                    prompt=selected_prompt,
                    pair_index=j,
                    is_low_case=True # This is the 'low' or 'neutral' case
                )
                batch_items.append(item_low)
            # The pair_index groups items that form a 'scale-tuple', which is the smallest
            # semantically valid unit for our training objective. Losses for items
            # within the same pair_index will be summed before further reduction.
            
            # Shuffle the batch items to mix high and low cases within the batch
            rng.shuffle(batch_items)
            self.schedule.append(batch_items)
            
        print(f"Built schedule with {len(self.schedule)} batches of size {batch_size}.")
        
        # Verify total sampled items
        actual_total_sampled_items = len(self.schedule) * batch_size
        expected_total_sampled_items_naive = total_training_steps * batch_size
        if actual_total_sampled_items < expected_total_sampled_items_naive:
            print(f"WARNING: Dataset sampled fewer items ({actual_total_sampled_items}) than the naive product of iterations * batch_size ({expected_total_sampled_items_naive}). This might indicate an issue with data availability or scheduling logic.")

    def __len__(self):
        return len(self.schedule)

    def __getitem__(self, idx):
        return self.schedule[idx]

    def get_unique_prompts(self) -> List[Dict[str, Any]]:
        unique_prompts = set()
        for batch_items in self.schedule:
            for item in batch_items:
                # Convert the prompt dictionary to a hashable type (e.g., a frozenset of items)
                # This assumes prompt dictionaries are simple and don't contain mutable objects
                unique_prompts.add(frozenset(item.prompt.items()))
        
        # Convert back to list of dictionaries
        return [dict(p) for p in unique_prompts]

def prepare_cached_batches(config, environment):
    """
    Pre-generates and caches all batches for the training loop using the TrainingSchedule.
    Refactored to use more efficient tensor operations and list comprehensions.
    """
    training_schedule = TrainingSchedule(config)
    unified_cache = build_unified_embedding_cache(config, environment, training_schedule)
    image_latents_cache = unified_cache["image_latents"]
    text_embeddings_cache = unified_cache["text_embeddings"]

    print("Pre-generating and caching all batches from schedule...")
    static_batches = []

    device = environment['device']
    weight_dtype = environment['weight_dtype']

    for batch_items in tqdm(training_schedule, desc="Caching batches"):
        latents = []
        scales = []
        all_cfg_text_embeddings = []
        all_cfg_pooled_embeds = []
        all_add_time_ids = [] # New list for add_time_ids
        original_image_sizes = [] # New list for original_image_sizes

        pair_indices = []
        is_low_cases = []

        target_resolution = tuple(config.train.get("resolution", (512, 512)))

        for item_idx, item in enumerate(batch_items):
            # Unpack the cached data
            latents_by_resolution_dict, original_image_size = image_latents_cache[item.image_path]

            latent = latents_by_resolution_dict[target_resolution]
            latents.append(latent)
            scales.append(item.scale)
            pair_indices.append(item.pair_index)
            is_low_cases.append(item.is_low_case)
            original_image_sizes.append(original_image_size) # Store original image size

            # Generate add_time_ids for the current item using original_image_size
            add_time_ids_for_item = get_add_time_ids(
                original_image_size[0], original_image_size[1], False, dtype=torch.float32
            )
            all_add_time_ids.append(add_time_ids_for_item)

            # ... (rest of the existing loop for text embeddings) ...
            (
                positive_text_embeds,
                positive_pooled_embeds,
                uncond_text_embeds,
                uncond_pooled_embeds,
                neutral_text_embeds,
                neutral_pooled_embeds,
            ) = text_embeddings_cache[frozenset(item.prompt.items())]

            selected_cond_text_embeds = neutral_text_embeds if item.is_low_case else positive_text_embeds
            selected_cond_pooled_embeds = neutral_pooled_embeds if item.is_low_case else positive_pooled_embeds
            
            cfg_text_embeds_for_item = torch.cat([uncond_text_embeds, selected_cond_text_embeds], dim=0)
            cfg_pooled_embeds_for_item = torch.cat([uncond_pooled_embeds, selected_cond_pooled_embeds], dim=0)

            all_cfg_text_embeddings.append(cfg_text_embeds_for_item)
            all_cfg_pooled_embeds.append(cfg_pooled_embeds_for_item)

        # Concatenate all collected tensors into batch tensors
        latents_batch = torch.cat(latents).to(dtype=weight_dtype)
        scales_batch = torch.tensor(scales, dtype=weight_dtype)
        
        cfg_text_embeddings_batch = torch.cat(all_cfg_text_embeddings, dim=0)
        cfg_pooled_embeds_batch = torch.cat(all_cfg_pooled_embeds, dim=0)
        add_time_ids_batch = torch.cat(all_add_time_ids, dim=0) # Concatenate all add_time_ids

        # Determine guidance_scale from the first item in the batch (assuming it's consistent within a batch)
        guidance_scale = batch_items[0].prompt.get("guidance_scale", 1.0)

        batch = {
            "latents": latents_batch,
            "scales": scales_batch,
            "cfg_text_embeddings": cfg_text_embeddings_batch,
            "cfg_pooled_embeds": cfg_pooled_embeds_batch,
            "add_time_ids": add_time_ids_batch, # Use the concatenated batch
            "pair_indices": torch.tensor(pair_indices, dtype=torch.long, device=device),
            "is_low_cases": torch.tensor(is_low_cases, dtype=torch.bool, device=device),
            "guidance_scale": guidance_scale,
            "original_image_sizes": original_image_sizes, # Add original_image_sizes to the batch
        }
        static_batches.append(batch)
        
    print(f"Cached {len(static_batches)} batches.")

    return static_batches
