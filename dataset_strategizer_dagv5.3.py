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

# In dataset_strategizer.py

# --- NEW: Model Asset Registry ---
# This declarative registry maps a model name to its required "bill of materials".
# This is the "configuration" that drives the rest of the generic pipeline.
# It is the single source of truth for what assets a model needs.

MODEL_ASSET_REGISTRY = {
    'StableDiffusionXL_UNet': {
        'assets': [
            # The UNet's 'sample' input is a latent that has had noise applied.
            # The primitive asset we must cache is the 'latent' itself.
            {
                "asset_type": "latent",
                "consumer_signature": "sdxl_vae_encoder_v1", # The tool to make this asset
                "source_type_required": "image" # The source material needed
            },
            # The UNet's 'encoder_hidden_states'
            {
                "asset_type": "text_embedding",
                "consumer_signature": "openai_clip_vit_l_14_text_encoder",
                "source_type_required": "prompt"
            },
            # The 'text_embeds' from 'added_cond_kwargs'
            {
                "asset_type": "pooled_text_embedding",
                "consumer_signature": "openai_clip_vit_l_14_text_encoder", # Same consumer, different output
                "source_type_required": "prompt"
            },
            # The 'time_ids' from 'added_cond_kwargs'
            {
                "asset_type": "time_embedding",
                "consumer_signature": "sdxl_time_id_synthesizer_v1",
                "source_type_required": "image" # Depends on original image resolution
            },
            # The 'timestep' is algorithmically generated, not cached from a source file.
            # For now, we can represent it as requiring an 'algorithmic' source,
            # which signals to the planner that no file-based work is needed for this.
            {
                "asset_type": "timestep",
                "consumer_signature": "noise_scheduler_sampler_v1",
                "source_type_required": "algorithmic"
            }
        ]
    },
    'SDXL_VAE_Decoder': {
        'assets': [
            # The VAE Decoder's primary input is a pre-existing latent.
            # An experiment training this model would need a cache of these.
            {
                "asset_type": "latent",
                "consumer_signature": "sdxl_vae_encoder_v1",
                "source_type_required": "image"
            }
        ]
    },
    'SDXL_VAE_Encoder': {
        # The VAE Encoder's primary input ('x') is a raw image.
        # It is a primitive source, not a pre-computed and cached asset.
        # Therefore, for an experiment where this is the final consumer, the list
        # of *required pre-cached assets* is correctly empty.
        'assets': []
    }
    # To support a new model (e.g., ControlNet), one would simply add a new entry here.
}

def derive_required_assets_from_model(config: Dict, **kwargs) -> Dict:
    """
    Looks up the required "bill of materials" for a given model from a
    central registry. This function is now fully unstubbed and acts as a
    configuration dispatcher.
    """
    print("  (INFO: Unstubbed logic is executing.)")
    print("  (INFO: Deriving asset requirements from model registry...)")

    model_name = config.get('base_model_name')
    if not model_name:
        raise ValueError("Config for 'derive_required_assets_from_model' must contain 'base_model_name'.")

    # Look up the model's requirements in the registry.
    asset_requirements = MODEL_ASSET_REGISTRY.get(model_name)

    if asset_requirements is None:
        raise ValueError(f"Unknown 'base_model_name': '{model_name}'. No entry found in MODEL_ASSET_REGISTRY.")

    print(f"  (INFO: Found {len(asset_requirements.get('assets',[]))} asset requirements for '{model_name}'.)")

    # Return the requirements in the standard 'reqassets_config' format.
    return {'reqassets_config': asset_requirements}

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
        #why was this hardcoded instead of passed from config? fix in next pass.
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

            # REVISION 1.1: Yield a self-describing data package with a generic 'primitives' list.
            yield {
                "source_identifier": selected_filename,
                "primitives": [
                    {"qualifier": "high_scale", "type": "image", "data": str(high_path)},
                    {"qualifier": "high_scale", "type": "prompt", "data": high_prompt_recipe},
                    {"qualifier": "low_scale", "type": "image", "data": str(low_path)},
                    {"qualifier": "low_scale", "type": "prompt", "data": low_prompt_recipe},
                ]
            }
    
    # REVISION 1.2: Add the explicit processing rules to the returned config.
    # This config now contains the 'how-to' guide for the planner.
    output_config = config.copy() # Start with the original config
    output_config['processing_rules'] = [
        {"source_qualifier": "high_scale", "source_type": "image", "consumer_signature": "sdxl_vae_encoder_v1"},
        {"source_qualifier": "low_scale",  "source_type": "image", "consumer_signature": "sdxl_vae_encoder_v1"},
        {"source_qualifier": "high_scale", "source_type": "image", "consumer_signature": "sdxl_time_id_synthesizer_v1"},
        {"source_qualifier": "low_scale",  "source_type": "image", "consumer_signature": "sdxl_time_id_synthesizer_v1"},
        {"source_qualifier": "high_scale", "source_type": "prompt", "consumer_signature": "openai_clip_vit_l_14_text_encoder"},
        {"source_qualifier": "low_scale",  "source_type": "prompt", "consumer_signature": "openai_clip_vit_l_14_text_encoder"}
    ]

    return {
        'imagepool_config': output_config,
        'data_package': generator()
    }


import itertools
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, Any, List, Set, Tuple

def _generate_stable_hash(s: str) -> str:
    """Helper to create a stable hash for cache keys."""
    return hashlib.sha1(s.encode()).hexdigest()

def create_unified_asset_worklist(config: Dict, **dependencies) -> Dict:
    """
    A truly generic function that plans all encoding work. It consumes a configured
    number of items from a data source iterator, deduplicates the required encoding
    tasks, organizes them into per-consumer queues, and returns both the work queues
    and a resolution map for final assembly.

    This is the definitive Layer 3: The Work Planning Engine.
    """
    print("  (INFO: Executing MASTER asset worklist planner...)")

    # --- 1. Unpack all FOUR required inputs ---
    data_package_iterator = dependencies['data_package']
    assets_config = dependencies['assets_config']
    # The 'how-to' guide from the discovery function (Layer 2)
    processing_rules_config = dependencies['processing_rules_config']
    
    # The new configuration that governs the planner's own behavior
    # This comes from the 'config' block of this stage in the manifest.
    max_iterations = config.get('max_iterations', 100) # Default to 100 if not specified
    cache_root = Path(config.get("cache_location_root", "./cache/"))

    # Extract the necessary data from the configs
    required_consumers = {
        asset['consumer_signature']: asset['asset_type'] 
        for asset in assets_config.get('assets', [])
    }
    processing_rules = processing_rules_config.get('processing_rules', [])

    # --- 2. Initialize Internal State ---
    
    # The final, optimized work queues, keyed by consumer. This is for Layer 4.
    work_queues: Dict[str, List[Dict]] = defaultdict(list)
    
    # A set to track unique work items for deduplication. The value is a hash
    # representing a unique encoding task.
    seen_work_items: Set[str] = set()

    # The Rosetta Stone for Layer 5. A list where each entry corresponds to a
    # training step and maps its abstract needs to a concrete cache path hash.
    work_resolution_map: List[Dict[Tuple[str, str], str]] = []

    print(f"  (INFO: Planning to draw {max_iterations} samples from the data iterator.)")

    # --- 3. Main Loop: Consume, Plan, Deduplicate, and Map ---

    # Use itertools.islice to safely draw N items from a potentially infinite iterator
    for data_item in itertools.islice(data_package_iterator, max_iterations):
        source_id = data_item['source_identifier']
        
        # This dictionary will map the needs for THIS specific training step.
        current_step_resolution: Dict[Tuple[str, str], str] = {}

        # Join the primitives from this data item with the processing rules
        for primitive in data_item.get('primitives', []):
            for rule in processing_rules:
                # Check if this rule applies to this primitive
                if (rule['source_qualifier'] == primitive['qualifier'] and 
                    rule['source_type'] == primitive['type']):
                    
                    consumer_sig = rule['consumer_signature']

                    # Check if the model actually needs the asset this consumer produces
                    if consumer_sig in required_consumers:
                        asset_type = required_consumers[consumer_sig]

                        # A. Generate a unique, deterministic identifier for this specific work task.
                        # This becomes the key for deduplication and the value in the resolution map.
                        unique_work_id = _generate_stable_hash(f"{source_id}_{primitive['qualifier']}_{asset_type}")
                        
                        # B. Deduplication: The core optimization.
                        if unique_work_id not in seen_work_items:
                            # This is the first time we've seen this exact task. Plan it.
                            seen_work_items.add(unique_work_id)
                            
                            output_path = cache_root / asset_type / f"{unique_work_id}.pt"

                            work_item = {
                                "source_primitive": primitive['data'],
                                "asset_type": asset_type,
                                "consumer_signature": consumer_sig,
                                "output_cache_path": str(output_path),
                                # Add the unique ID for the executor to use when building the manifest
                                "work_id": unique_work_id,
                            }
                            work_queues[consumer_sig].append(work_item)

                        # C. Mapping: Always update the map for the current training step.
                        # This maps the abstract need (e.g., "high_scale latent") to the
                        # concrete ID of the work that will produce it.
                        resolution_key = (primitive['qualifier'], asset_type)
                        current_step_resolution[resolution_key] = unique_work_id
        
        work_resolution_map.append(current_step_resolution)

    print(f"  (INFO: Planning complete. Total unique work items: {len(seen_work_items)}.)")
    print(f"  (INFO: Work queues generated for consumers: {list(work_queues.keys())})")

    # --- 4. Return the Complete Master Plan ---
    # The output contract is now two keys, for two different downstream consumers.
    return {
        'unified_worklist': dict(work_queues), # For the Asset Execution Engine (Layer 4)
        'work_resolution_map': work_resolution_map # For the Training Dataset Assembly (Layer 5)
    }

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
        print("\n--- Demonstrating a REAL iterator ('data_package') ---")
        real_iterator = final_results['data_package']
        for i, item in enumerate(real_iterator):
            if i >= 3: break
            print(f"  Item {i+1}: {item}")
        print("\n--- Demonstrating the FINAL (stubbed) iterator ('training_iterator') ---")
        final_iterator = final_results.get('training_iterator')
        if final_iterator:
            items = list(final_iterator)
            print(f"  Final iterator yielded {len(items)} items. (Correctly 0 due to upstream stubs)")
        else:
            print(f"  Final iterator enigmatically, evaluated as falsy. final_results.get('training_iterator'):{final_results.get('training_iterator')}")
    except Exception as e:
        import traceback
        print(f"\n[ERROR] An error occurred during DAG execution: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()