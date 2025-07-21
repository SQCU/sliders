# dataset_strategizer.py
# python dataset_strategizer_dagv5.3.py >> dset_strat_dagv5.3.txt 2>&1
import yaml
from pathlib import Path
from collections import defaultdict
import os
import random
import hashlib
from typing import Iterator, Callable, Dict, Any

# CORRECTED IMPORT: We now import the core function directly, not the path-based helper.
from discovery_scanner import discover_data_pool

# --- Collection of Pure Strategizer Functions ---

def derive_required_assets_from_model(config: Dict, **kwargs) -> Dict:
    """
    HONEST STUB: This function's responsibility is to inspect a model and return
    a configuration describing the assets needed to call it.
    """
    print("  (STUB: This is where we would inspect the model signature for 'base_model_name')")
    return {'reqassets_config': {'assets': []}}

def create_imageslider_pool_iterator(config: Dict, **kwargs) -> Dict:
    """
    The authoritative data discovery stage. This is a direct and correct
    recomposition of the logic from the original, working dataset_strategizer.py.
    It performs REAL file discovery and REAL prompt composition.
    """
    # --- Recomposed logic from the original file ---
    def _filter_valid_filenames(data_pool: dict, config: dict) -> list:
        if config['pairing_strategy'] == 'relaxed':
            return [f for f, s_map in data_pool.items() if len(s_map) >= config['min_scales_required']]
        else: # strict
            all_scales = [float(s) for s in config['scales']]
            return [f for f, s_map in data_pool.items() if len(s_map) == len(all_scales)]

    def _get_composed_prompt(item_path: Path, prompt_sources_config: dict, root_prompts: dict, metadata: dict, rng: random.Random) -> dict:
        item_filename = item_path.name
        folder_name = item_path.parent.name
        sources, weights = [], []

        sources.append(root_prompts['default_prompt'])
        weights.append(prompt_sources_config['root']['importance'])

        if folder_name in metadata.get('folders', {}):
            key = metadata['folders'][folder_name]['prompt_key']
            if key in root_prompts:
                sources.append(root_prompts[key])
                weights.append(prompt_sources_config['metadata']['folder_importance'])

        if item_filename in metadata.get('items', {}):
            key = metadata['items'][item_filename]['prompt_key']
            if key in root_prompts:
                sources.append(root_prompts[key])
                weights.append(prompt_sources_config['metadata']['item_importance'])
        
        return rng.choices(sources, weights=weights, k=1)[0]

    # --- Main function logic using the REAL, working code ---
    print("  (INFO: Executing REAL logic to discover files and compose prompts)")

    # CORRECTED CALL: The 'config' dict received by this function is now passed
    # directly to the imported 'discover_data_pool' function. No more fake paths.
    data_pool = discover_data_pool(config)

    valid_filenames = _filter_valid_filenames(data_pool, config)

    # Load prompt source files - this is real I/O as required.
    _project_root = Path(os.getcwd())
    root_prompts_path = _project_root / config['prompt_sources']['root']['file']
    metadata_path = _project_root / config['prompt_sources']['metadata']['file']
    with open(root_prompts_path, 'r') as f:
        root_prompts = yaml.safe_load(f)
    with open(metadata_path, 'r') as f:
        metadata = yaml.safe_load(f)

    if not valid_filenames:
        raise ValueError("No valid filenames found in create_imageslider_pool_iterator")

    def generator() -> Iterator[Dict]:
        rng = random.Random(42)
        while True:
            selected_filename = rng.choice(valid_filenames)
            available_scales = sorted(data_pool[selected_filename].keys())
            if len(available_scales) < 2: continue
            
            low_scale, high_scale = sorted(rng.sample(available_scales, 2))
            high_path = Path(data_pool[selected_filename][high_scale])
            low_path = Path(data_pool[selected_filename][low_scale])
            
            high_prompt_recipe = _get_composed_prompt(high_path, config['prompt_sources'], root_prompts, metadata, rng)
            low_prompt_recipe = _get_composed_prompt(low_path, config['prompt_sources'], root_prompts, metadata, rng)

            yield {
                "source_identifier": selected_filename,
                "high_scale_info": {"scale": high_scale, "path": str(high_path), "prompt_recipe": high_prompt_recipe},
                "low_scale_info": {"scale": low_scale, "path": str(low_path), "prompt_recipe": low_prompt_recipe},
            }

    return {
        'imagepool_config': config,
        'sliderimage_pool': generator()
    }

def create_unified_asset_worklist(config: Dict, **dependencies) -> Dict:
    """HONEST STUB: Consumes configs and data to plan all encoding work."""
    print("  (STUB: This is where we would iterate through the image_pool and assets_config)")
    print("  (STUB: to create a unified list of all files to encode and assets to generate.)")
    def empty_generator():
        yield from ()
    return {'unified_worklist': empty_generator()}

def execute_caching_iterator(config: Dict, **dependencies) -> Dict:
    """HONEST STUB: Consumes a worklist and produces a cache manifest."""
    print("  (STUB: This is where we would consume the 'unified_worklist', run encoders,)")
    print("  (STUB: and save tensors to the cache_location_root.)")
    return {'cache_manifest': {'description': 'Stubbed cache manifest.', 'assets': {}}}

def fold_cache_manifest_into_dataset(config: Dict, **dependencies) -> Dict:
    """
    HONEST STUB: Consumes manifests to re-assemble a final dataset.
    This stub now correctly returns ALL keys promised by the manifest.
    """
    print("  (STUB: This is where we would iterate through 'slider_dataset', using)")
    print("  (STUB: 'cache_manifest' and 'assets_config' to look up cached asset paths)")
    print("  (STUB: and pair them for CFG and high/low scale batches.)")

    def empty_generator():
        yield from ()
    # --- THIS IS THE FIX ---
    # The function now returns a dictionary containing BOTH the dataset key and the
    # config key, as specified in the manifest. The config is an empty dict,
    # which is a valid and honest placeholder.
    return {
        'final_training_dataset': empty_generator(),
        'sliders_batch_resampling_config': {} # Fulfills the contract
    }

def create_sampling_iterator(config: Dict, **dependencies) -> Dict:
    """Consumes a final dataset and yields items for training indefinitely."""
    print("  (INFO: Creating final sampling iterator.)")
    source_dataset = list(dependencies['source_dataset'])
    if not source_dataset:
        print("  (INFO: Source dataset is empty, likely from an upstream stub. This iterator will be empty.)")
    def generator():
        if not source_dataset: return
        rng = random.Random(config.get('seed', 42))
        for _ in range(config.get('iterations', 10)):
             yield rng.choice(source_dataset)
    return {'training_iterator': generator()}

# --- Generic DAG Runner (unchanged) ---

STRATEGIZER_FUNCTION_MAP = {
    'derive_required_assets_from_model': derive_required_assets_from_model,
    'create_imageslider_pool_iterator': create_imageslider_pool_iterator,
    'create_unified_asset_worklist': create_unified_asset_worklist,
    'execute_caching_iterator': execute_caching_iterator,
    'fold_cache_manifest_into_dataset': fold_cache_manifest_into_dataset,
    'create_sampling_iterator': create_sampling_iterator,
}

def call_from_path(manifest_path: Path) -> Dict[str, Any]:
    with open(manifest_path, 'r') as f: manifest = yaml.safe_load(f)
    graph_stages = manifest['execution_graph']

    execution_results = {}
    processed_stages = set()
    
    # --- FIX: Pre-process the graph to map output keys to the stage that produces them. ---
    key_producer_map = {}
    for stage_outer in graph_stages:
        stage = stage_outer['dataset']
        stage_name = stage['name']
        if stage.get('return_config_key'):
            key_producer_map[stage['return_config_key']] = stage_name
        if stage.get('return_dataset_key'):
            key_producer_map[stage['return_dataset_key']] = stage_name
    # --- END FIX ---

    max_passes = len(graph_stages) + 1
    passes = 0
    while len(processed_stages) < len(graph_stages) and passes < max_passes:
        for stage_outer in graph_stages:
            stage = stage_outer['dataset']
            stage_name = stage['name']
            if stage_name in processed_stages: continue

            # --- LOGGING: Announce which stage is being evaluated ---
            print(f"\n[DAG RUNNER] Evaluating stage: '{stage_name}'")

            scheduling_deps = stage.get('requires', [])
            argument_global_keys = stage.get('arguments', {}).values()
            data_producing_deps = [key_producer_map[key] for key in argument_global_keys]
            all_dependencies = set(scheduling_deps + data_producing_deps)
            
            print(f"  - Calculated Dependencies: {all_dependencies or 'None'}")

            # 4. Check if all prerequisite stages have been processed.
            deps_met = all(dep_name in processed_stages for dep_name in all_dependencies)
            # --- END FIX ---
            if deps_met:
                print(f"  - Status: [READY]")
                newly_processed_this_pass = True
                fn = STRATEGIZER_FUNCTION_MAP[stage['function']]
                kwargs = {local_name: execution_results[global_name] for local_name, global_name in stage.get('arguments', {}).items()}

                print(f"  - Executing function: '{stage['function']}'...")
                stage_outputs = fn(stage['config'], **kwargs)
                
                if stage.get('return_config_key'):
                    key = stage['return_config_key']
                    execution_results[key] = stage_outputs[key]
                if stage.get('return_dataset_key'):
                    key = stage['return_dataset_key']
                    execution_results[key] = stage_outputs[key]
                processed_stages.add(stage_name)
            else:
                # --- LOGGING: Explicitly state which dependencies are missing ---
                missing_deps = all_dependencies - processed_stages
                print(f"  - Status: [WAITING]")
                print(f"  - Missing Dependencies: {missing_deps}")
        if not newly_processed_this_pass and len(processed_stages) < len(graph_stages):
            break
        passes += 1
    if len(processed_stages) < len(graph_stages):
        unprocessed = set(s['dataset']['name'] for s in graph_stages) - processed_stages
        # The informative error message is still a good idea, as it's the final output of the observation.
        raise RuntimeError(f"DAG execution failed. Deadlock detected. Unprocessed stages: {unprocessed}")
    return execution_results

# --- Main for Self-Testing ---
def main():
    project_root = Path(os.getcwd())
    default_manifest_path = project_root / "run_artifacts" / "yamlzoo" / "experiment_manifest_dag.yaml"
    print(f"--- Executing DAG from: {default_manifest_path} ---")
    try:
        final_results = call_from_path(default_manifest_path)
        print("\n--- DAG Execution Complete. Final results available: ---")
        for key, value in final_results.items():
            value_type = "iterator" if hasattr(value, '__next__') else type(value).__name__
            print(f"  - '{key}' (type: {value_type})")
        print("\n--- Demonstrating a REAL iterator ('sliderimage_pool') ---")
        real_iterator = final_results['sliderimage_pool']
        for i, item in enumerate(real_iterator):
            if i >= 3: break
            print(f"  Item {i+1}: {item}")
        print("\n--- Demonstrating the FINAL (stubbed) iterator ('training_iterator') ---")
        final_iterator = final_results.get('training_iterator')
        if final_iterator:
            items = list(final_iterator)
            print(f"  Final iterator yielded {len(items)} items. (Correctly 0 due to upstream stubs)")
    except Exception as e:
        import traceback
        print(f"\n[ERROR] An error occurred during DAG execution: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()