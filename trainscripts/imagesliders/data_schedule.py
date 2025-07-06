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

    def _get_data_pool(self) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Creates a master list of all possible data combinations."""
        print("Collecting all possible training data combinations...")
        
        image_paths_and_scales: List[Tuple[str, float]] = []
        subfolder_names = [f.strip() for f in self.config.dataset_config.dataset.folders.split(',')]
        scale_values = [float(s.strip()) for s in self.config.dataset_config.dataset.scales.split(',')]
        
        for i, folder_name in enumerate(subfolder_names):
            subfolder_path = Path(self.config.dataset_config.dataset.folder_main) / folder_name
            for image_path in subfolder_path.glob("*"):
                if image_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    image_paths_and_scales.append((str(image_path), scale_values[i]))

        with open(self.config.dataset_config.prompts_file, 'r') as f:
            prompts_data = yaml.safe_load(f)

        # Create a Cartesian product of all images, scales, and prompts
        # This creates the full pool of potential training items.
        data_pool = list(itertools.product(image_paths_and_scales, prompts_data))
        
        # Unpack the (image_path, scale) tuple
        unpacked_pool = [
            (img_path, scale, prompt) 
            for ((img_path, scale), prompt) in data_pool
        ]
        
        print(f"Created data pool with {len(unpacked_pool)} unique combinations.")
        return unpacked_pool

    def _build_schedule(self):
        print("Building pseudo-randomized training schedule...")
        
        seed = self.config.train.get('seed', 42)
        rng = random.Random(seed)
        print(f"Using random seed: {seed}")

        data_pool = self._get_data_pool()
        if not data_pool:
            raise ValueError("Data pool is empty. Check your dataset configuration.")

        total_training_steps = self.config.train.iterations
        batch_size = self.config.train.batch_size
        
        if batch_size % 2 != 0:
            raise ValueError("Batch size must be an even number to support pairing.")

        for i in range(total_training_steps):
            batch_items = []
            for j in range(batch_size):
                # Sample a random item from the entire data pool
                image_path, scale, prompt = rng.choice(data_pool)
                
                # The current pairing logic is simple: adjacent items form a pair.
                # The 'low_case' is assigned to the second item in each pair.
                # This can be replaced with more complex pairing logic,
                # e.g., ensuring pairs have the same prompt but different images,
                # or using a predefined pairing map.
                pair_index = j // 2
                is_low_case = (j % 2 == 1)

                item = TrainingItem(
                    image_path=image_path,
                    scale=scale,
                    prompt=prompt,
                    pair_index=pair_index,
                    is_low_case=is_low_case
                )
                batch_items.append(item)
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