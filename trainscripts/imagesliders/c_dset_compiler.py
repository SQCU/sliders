# The key insight: Everything flows through serializable bottlenecks!
# Config -> TrainingPlan -> CachedAssets -> BatchSpec -> Tensors

from dataclasses import dataclass
from typing import Iterator, Dict, Any
import json
from pathlib import Path

@dataclass
class TrainingUnit:
    """Serializable training unit specification"""
    unit_id: str
    high_path: str
    low_path: str
    high_scale: float
    low_scale: float
    high_prompt_key: str
    low_prompt_key: str
    seed: int  # For reproducible noise

@dataclass
class TrainingPlan:
    """Complete serializable training plan"""
    units: list[TrainingUnit]
    prompt_recipes: dict[str, dict]  # prompt_key -> recipe
    
    def save(self, path: Path):
        """Save as JSON - the key bottleneck!"""
        data = {
            'units': [unit.__dict__ for unit in self.units],
            'prompt_recipes': self.prompt_recipes
        }
        path.write_text(json.dumps(data, indent=2))
    
    @classmethod
    def load(cls, path: Path):
        """Load from JSON"""
        data = json.loads(path.read_text())
        return cls(
            units=[TrainingUnit(**u) for u in data['units']],
            prompt_recipes=data['prompt_recipes']
        )

@dataclass
class AssetManifest:
    """Serializable asset inventory"""
    latent_paths: dict[str, dict]  # path -> {tensor_key, original_size}
    embedding_keys: dict[str, dict]  # prompt_key -> {tensor_key}
    
    def save(self, path: Path):
        path.write_text(json.dumps(self.__dict__, indent=2))
    
    @classmethod
    def load(cls, path: Path):
        data = json.loads(path.read_text())
        return cls(**data)

# ============================================================================
# PURE FUNCTIONAL PIPELINE
# ============================================================================

def generate_training_plan(config: dict) -> TrainingPlan:
    """Config -> TrainingPlan (pure function, no side effects)"""
    rng = random.Random(config['seed'])
    
    # Load metadata once
    with open(config['prompt_sources']['root']['file']) as f:
        root_prompts = yaml.safe_load(f)
    with open(config['prompt_sources']['metadata']['file']) as f:
        metadata = json.load(f)
    
    # Discovery phase (pure)
    data_pool = discover_data_pool(config)
    valid_filenames = filter_valid_filenames(data_pool, config)
    
    # Generation phase (pure)
    units = []
    used_recipes = {}
    
    for step in range(config['iterations']):
        for batch_idx in range(config['batch_size'] // 2):
            filename = rng.choice(valid_filenames)
            scales = sorted(rng.sample(list(data_pool[filename].keys()), 2))
            
            # Generate unit
            unit = TrainingUnit(
                unit_id=f"unit_{step}_{batch_idx}_{filename}",
                high_path=data_pool[filename][scales[1]],
                low_path=data_pool[filename][scales[0]],
                high_scale=scales[1],
                low_scale=scales[0],
                high_prompt_key=get_prompt_key(scales[1], filename, root_prompts, metadata, rng),
                low_prompt_key=get_prompt_key(scales[0], filename, root_prompts, metadata, rng),
                seed=rng.randint(0, 2**31-1)
            )
            units.append(unit)
            
            # Collect unique recipes
            for key in [unit.high_prompt_key, unit.low_prompt_key]:
                if key not in used_recipes:
                    used_recipes[key] = get_recipe_for_key(key, root_prompts, metadata)
    
    return TrainingPlan(units=units, prompt_recipes=used_recipes)

def materialize_assets(plan: TrainingPlan, config: dict, environment: dict) -> AssetManifest:
    """TrainingPlan -> AssetManifest (caching side effects contained here)"""
    
    # Extract requirements from plan
    required_paths = set()
    for unit in plan.units:
        required_paths.update([unit.high_path, unit.low_path])
    
    required_recipes = set(plan.prompt_recipes.keys())
    
    # Materialize latents (idempotent caching)
    latent_manifest = ensure_latents_cached(required_paths, config, environment)
    
    # Materialize embeddings (idempotent caching)
    embedding_manifest = ensure_embeddings_cached(required_recipes, plan.prompt_recipes, config, environment)
    
    return AssetManifest(
        latent_paths=latent_manifest,
        embedding_keys=embedding_manifest
    )

def batch_generator(plan: TrainingPlan, manifest: AssetManifest, config: dict) -> Iterator[dict]:
    """TrainingPlan + AssetManifest -> Tensor Batches (generator)"""
    
    # Load cached tensors once
    latent_cache = load_tensor_cache(config['latent_cache_dir'])
    embedding_cache = load_tensor_cache(config['embedding_cache_dir'])
    
    # Group units into batches
    batch_size = config['batch_size']
    for i in range(0, len(plan.units), batch_size):
        batch_units = plan.units[i:i + batch_size]
        
        # Transform each unit to tensor data
        batch_data = [
            unit_to_tensors(unit, latent_cache, embedding_cache, manifest, config)
            for unit in batch_units
        ]
        
        # Collate into final batch
        yield collate_batch(batch_data, config)

def unit_to_tensors(unit: TrainingUnit, latent_cache: dict, embedding_cache: dict, 
                   manifest: AssetManifest, config: dict) -> dict:
    """Pure function: TrainingUnit -> tensor data"""
    
    # Get cached assets
    high_latent = latent_cache[manifest.latent_paths[unit.high_path]['tensor_key']]
    low_latent = latent_cache[manifest.latent_paths[unit.low_path]['tensor_key']]
    
    high_embed = embedding_cache[manifest.embedding_keys[unit.high_prompt_key]['tensor_key']]
    low_embed = embedding_cache[manifest.embedding_keys[unit.low_prompt_key]['tensor_key']]
    
    # Generate reproducible noise
    noise_gen = torch.Generator().manual_seed(unit.seed)
    high_noise = torch.randn_like(high_latent, generator=noise_gen)
    low_noise = torch.randn_like(low_latent, generator=noise_gen)
    
    # Generate timesteps
    ts_gen = torch.Generator().manual_seed(unit.seed)
    timesteps = torch.randint(1, config['max_denoising_steps'], (1,), generator=ts_gen)
    
    return {
        'high': {
            'latent': high_latent,
            'noise': high_noise,
            'embedding': high_embed,
            'scale': unit.high_scale,
            'timestep': timesteps
        },
        'low': {
            'latent': low_latent,
            'noise': low_noise,
            'embedding': low_embed,
            'scale': unit.low_scale,
            'timestep': timesteps
        }
    }

# ============================================================================
# USAGE: The entire pipeline becomes a simple composition
# ============================================================================

def run_training_pipeline(config: dict, environment: dict):
    """The complete pipeline as a simple composition"""
    
    # 1. Generate plan (pure)
    plan_path = Path(config['cache_dir']) / 'training_plan.json'
    if not plan_path.exists():
        plan = generate_training_plan(config)
        plan.save(plan_path)
    else:
        plan = TrainingPlan.load(plan_path)
    
    # 2. Materialize assets (idempotent)
    manifest_path = Path(config['cache_dir']) / 'asset_manifest.json'
    if not manifest_path.exists():
        manifest = materialize_assets(plan, config, environment)
        manifest.save(manifest_path)
    else:
        manifest = AssetManifest.load(manifest_path)
    
    # 3. Generate batches (streaming)
    for batch in batch_generator(plan, manifest, config):
        yield prepare_training_batch(batch, environment['scheduler'], 
                                   environment['device'], environment['weight_dtype'])

# ============================================================================
# HELPER FUNCTIONS (implementation details)
# ============================================================================

def discover_data_pool(config: dict) -> dict:
    """Pure function: config -> data pool mapping"""
    data_pool = defaultdict(dict)
    for folder_name, scale in zip(config['folders'], config['scales']):
        folder_path = Path(config['folder_main']) / folder_name
        if folder_path.exists():
            for img_path in folder_path.glob("*"):
                if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
                    data_pool[img_path.name][float(scale)] = str(img_path)
    return dict(data_pool)

def filter_valid_filenames(data_pool: dict, config: dict) -> list:
    """Pure function: data pool -> valid filenames"""
    if config['pairing_strategy'] == 'strict':
        return [f for f, s_map in data_pool.items() 
                if len(s_map) == len(config['scales'])]
    elif config['pairing_strategy'] == 'relaxed':
        return [f for f, s_map in data_pool.items() 
                if len(s_map) >= config['min_scales_required']]
    else:
        raise ValueError(f"Unknown pairing_strategy: {config['pairing_strategy']}")

def ensure_latents_cached(paths: set, config: dict, environment: dict) -> dict:
    """Idempotent caching function"""
    # Similar to original but focused on the specific paths needed
    # Returns mapping: path -> {tensor_key, original_size}
    pass

def ensure_embeddings_cached(keys: set, recipes: dict, config: dict, environment: dict) -> dict:
    """Idempotent caching function"""
    # Similar to original but focused on the specific recipes needed
    # Returns mapping: prompt_key -> {tensor_key}
    pass