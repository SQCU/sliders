import yaml
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
import itertools

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
            
            # Shuffle the batch items to mix high and low cases within the batch
            rng.shuffle(batch_items)
            self.schedule.append(batch_items)
            
        print(f"Built schedule with {len(self.schedule)} batches of size {batch_size}.")

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