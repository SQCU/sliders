#dataset_strategizer.py
import yaml
from pathlib import Path
from collections import defaultdict
import os
import random
import hashlib
from typing import Iterator, Callable
from discovery_scanner import call_from_path as discover_data_pool_from_manifest

# --- Helper Functions ---

def _filter_valid_filenames(data_pool: dict, config: dict) -> list:
    """Pure function: data pool -> valid filenames based on pairing strategy"""
    all_scales = [float(s) for s in config['scales']]
    if config['pairing_strategy'] == 'strict':
        return [f for f, s_map in data_pool.items() if len(s_map) == len(all_scales)]
    elif config['pairing_strategy'] == 'relaxed':
        return [f for f, s_map in data_pool.items() if len(s_map) >= config['min_scales_required']]
    else:
        raise ValueError(f"Unknown pairing_strategy: {config['pairing_strategy']}")

def _get_recipe_key(recipe: dict) -> str:
    """Generates a stable hash for a prompt recipe dictionary."""
    return hashlib.sha1(yaml.dump(recipe, sort_keys=True).encode()).hexdigest()

def _get_composed_prompt(item_path: Path, prompt_sources_config: dict, root_prompts: dict, metadata: dict, rng: random.Random) -> dict:
    """
    Selects a final prompt configuration for an item based on a hierarchy and sampling importance.
    This is a simplified version for demonstration.
    """
    item_filename = item_path.name
    folder_name = item_path.parent.name

    sources = []
    weights = []

    # Root prompt (always available as a fallback)
    sources.append(root_prompts['default_prompt'])
    weights.append(prompt_sources_config['root']['importance'])

    # Folder-level prompt
    if folder_name in metadata.get('folders', {}):
        folder_prompt_key = metadata['folders'][folder_name]['prompt_key']
        if folder_prompt_key in root_prompts:
            sources.append(root_prompts[folder_prompt_key])
            weights.append(prompt_sources_config['metadata']['folder_importance'])
    
    # Item-level prompt
    if item_filename in metadata.get('items', {}):
        item_prompt_key = metadata['items'][item_filename]['prompt_key']
        if item_prompt_key in root_prompts:
            sources.append(root_prompts[item_prompt_key])
            weights.append(prompt_sources_config['metadata']['item_importance'])

    # Perform weighted random sampling to choose one prompt source
    chosen_prompt_recipe = rng.choices(sources, weights=weights, k=1)[0]

    return chosen_prompt_recipe

# --- Core Generator Function (now standalone) ---

def _generate_training_units(rng: random.Random, data_pool: dict, valid_filenames: list, 
                            data_config: dict, root_prompts: dict, metadata: dict) -> Iterator[dict]:
    """
    Generator function that yields training unit dictionaries indefinitely.
    This function is agnostic to batch size and total iterations; the consumer controls consumption.
    """
    unit_counter = 0
    while True:
        # Deterministic sampling of filename
        selected_filename = rng.choice(valid_filenames)
        available_scales = sorted(data_pool[selected_filename].keys())
        
        if len(available_scales) < 2:
            # This should ideally not happen if _filter_valid_filenames is correct
            # but as a safeguard.
            continue 

        low_scale, high_scale = sorted(rng.sample(available_scales, 2))

        high_path = Path(data_pool[selected_filename][high_scale])
        low_path = Path(data_pool[selected_filename][low_scale])
        
        # Deterministic prompt composition
        high_prompt_recipe = _get_composed_prompt(high_path, data_config['prompt_sources'], root_prompts, metadata, rng)
        low_prompt_recipe = _get_composed_prompt(low_path, data_config['prompt_sources'], root_prompts, metadata, rng)

        high_recipe_key = _get_recipe_key(high_prompt_recipe)
        low_recipe_key = _get_recipe_key(low_prompt_recipe)
        
        unit_seed = rng.randint(0, 2**31 - 1) # Unit-specific seed for noise

        training_unit = {
            "unit_id": f"unit_{unit_counter}_{selected_filename.replace('.', '_')}",
            "high_path": str(high_path.resolve()),
            "low_path": str(low_path.resolve()),
            "high_scale": high_scale,
            "low_scale": low_scale,
            "high_prompt_recipe": high_prompt_recipe,
            "low_prompt_recipe": low_prompt_recipe,
            "high_recipe_key": high_recipe_key,
            "low_recipe_key": low_recipe_key,
            "seed": unit_seed
        }
        yield training_unit
        unit_counter += 1

# --- Strategizer (Generator Factory) ---

def dataset_strategizer(manifest_path: Path) -> Callable[[], Iterator[dict]]:
    """
    Synthesizes a generator function that deterministically samples training units
    from a data pool based on the experiment manifest.

    Args:
        manifest_path (Path): The path to the experiment manifest YAML file.

    Returns:
        Callable[[], Iterator[dict]]: A generator factory function. When called,
        this factory will return a new iterator that yields training unit dictionaries
        indefinitely, based on the configuration loaded from the manifest.
    """
    # 1. Load Manifest and Extract Configuration
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    try:
        with open(manifest_path, 'r') as f:
            manifest_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML manifest {manifest_path}: {e}")

    data_config = manifest_config.get('data_setup', {}).get('config', {})
    if not data_config:
        raise ValueError("'data_setup.config' not found in the manifest.")
    
    train_config = manifest_config.get('data_setup', {}).get('config', {})
    if not train_config:
        raise ValueError("'data_setup.config' (for train config) not found in the manifest.")

    # 2. Get Data Pool (using discover_data_pool_from_manifest)
    data_pool = discover_data_pool_from_manifest(manifest_path)
    if not data_pool:
        raise ValueError("Data pool discovery failed or returned empty.")

    # Load prompt sources
    _project_root = Path(os.getcwd())
    root_prompts_path = _project_root / data_config['prompt_sources']['root']['file']
    metadata_path = _project_root / data_config['prompt_sources']['metadata']['file']

    with open(root_prompts_path, 'r') as f:
        root_prompts = yaml.safe_load(f)
    with open(metadata_path, 'r') as f:
        metadata = yaml.safe_load(f) # Assuming metadata is also YAML for simplicity

    # Filter valid filenames based on pairing strategy
    valid_filenames = _filter_valid_filenames(data_pool, data_config)
    if not valid_filenames:
        raise ValueError("No valid filenames found after applying pairing strategy.")

    # Prepare initial seed for deterministic sampling
    initial_seed = train_config.get('seed', 42) # Default seed if not specified

    # Return a lambda that, when called, creates a new generator iterator
    # and a NEW random.Random instance with the initial_seed
    return lambda: _generate_training_units(random.Random(initial_seed), data_pool, valid_filenames, data_config, root_prompts, metadata)

# --- Composable Entry Point ---

def call_from_path(manifest_path: Path) -> Callable[[], Iterator[dict]]:
    """
    Primary entry point for other modules to get a dataset generator factory.
    Loads the manifest and returns a factory function that produces dataset iterators.
    """
    return dataset_strategizer(manifest_path)

# --- Debug Printing Helper ---

def debug_print_generator(generator_factory: Callable[[], Iterator[dict]], print_limit: int = 5):
    """
    Helper to print a limited number of items from a generator and demonstrate determinism.
    """
    print(f"--- Sample Training Units (first {print_limit}) ---")
    schedule_generator = generator_factory()
    for i, unit in enumerate(schedule_generator):
        if i >= print_limit:
            break
        print(f"Unit {i+1}: {unit}")
        print("---")
    
    print(f"\n--- Demonstrating Determinism (first unit again from a new iterator) ---")
    re_run_generator = generator_factory()
    first_unit_again = next(re_run_generator)
    print(f"First unit on re-run: {first_unit_again}")

# --- Shallow Main for Self-Testing ---

def main():
    project_root = Path(os.getcwd())
    default_manifest_path = project_root / "run_artifacts" / "yamlzoo" / "experiment_manifest_xl.yaml"

    print(f"Synthesizing dataset strategizer from: {default_manifest_path}")
    try:
        # Get the generator factory
        dataset_generator_factory = call_from_path(default_manifest_path)
        
        # Use the debug printing helper
        debug_print_generator(dataset_generator_factory, print_limit=5)

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
