# FFW8: dataset_strategizer_dagv5.4.py
# This version is a complete architectural refactor to enforce strict layer separation
# as per the "Semantic Layer Specifications" and the provided critique.
# python dataset_strategizer_dagv5.4.py >> dset_strat_dagv5.4.txt 2>&1

import yaml
from pathlib import Path
from collections import defaultdict
import os
import random
import hashlib
import itertools
from typing import Iterator, Callable, Dict, Any, List, Tuple, Set

# ARCH-FIX: Import discovery_scanner, assuming it's a clean, Layer-2 compliant module.
from discovery_scanner import discover_data_pool

# --- Layer 1: Model Asset Registry (Interface Contract Definition) ---
# ARCH-FIX: VIOLATION 1 & 3 FIXED.
# The registry is now purely declarative and defines INTERFACES, not IMPLEMENTATIONS.
# It specifies *what* a model needs, not *how* to create it. It contains no
# hardcoded 'consumer_signature' names. This passes the "Litmus Test" because
# the tools used to satisfy these interfaces can be changed in a separate
# configuration without touching this registry.

MODEL_INTERFACE_REGISTRY = {
    'StableDiffusionXL_UNet': {
        'required_assets': [
            {"asset_type": "noisy_latent", "asset_interface": "image_to_denoise_inputs_encoder", "source_types_required": ["image", "prng_seed"]},
            {"asset_type": "timestep_for_unet", "asset_interface": "image_to_denoise_inputs_encoder", "source_types_required": ["image", "prng_seed"]},
            {"asset_type": "text_embedding", "asset_interface": "text_to_embedding_encoder", "source_types_required": ["prompt"]},
            {"asset_type": "pooled_text_embedding", "asset_interface": "text_to_embedding_encoder", "source_types_required": ["prompt"]},
            {"asset_type": "time_embedding", "asset_interface": "image_to_time_id_synthesizer", "source_types_required": ["image"]},
            {"asset_type": "scales_tensor", "asset_interface": "scale_tensor_synthesizer", "source_types_required": ["scale_metadata"]}
        ]
    },
    'SDXL_VAE_Decoder': {
        'required_assets': [
            {
                "asset_type": "latent",
                "asset_interface": "image_to_latent_encoder",
                "source_type_required": "image"
            }
        ]
    }
}

def get_required_asset_spec(own_config: Dict, **kwargs) -> Dict[str, Any]:
    """
    LAYER 1: Looks up the required asset INTERFACES for a model.
    This function is a pure configuration dispatcher. It does not know or care
    how the assets will be made, only what is required.
    """
    print("[LAYER 1] Looking up required asset specification...")
    model_name = own_config.get('base_model_name')
    if not model_name:
        raise ValueError("L1 Config must contain 'base_model_name'.")

    asset_spec = MODEL_INTERFACE_REGISTRY.get(model_name)
    if asset_spec is None:
        raise ValueError(f"Unknown model '{model_name}' in MODEL_INTERFACE_REGISTRY.")

    print(f"  - Found {len(asset_spec['required_assets'])} required assets for '{model_name}'.")
    return {'required_assets_spec': asset_spec}


# --- Layer 2: Data Discovery Engine ---

def discover_source_data_stream(own_config: Dict, **kwargs) -> Dict[str, Any]:
    """
    LAYER 2: Discovers real source data and yields a stream of generic primitive packages.
    
    FIX: This version restores the EXACT original logic for file filtering and prompt
    composition, including the use of metadata for folder- and item-specific prompts.
    The output remains architecturally pure, adhering to the layer separation contract.
    """
    print("[LAYER 2] Discovering source data and composing primitives...")

    # --- FIX: Restored original, authoritative internal logic ---

    # Call the external discovery utility as before.
    data_pool = discover_data_pool(own_config)

    # Helper to filter filenames based on the layer's own configuration (Identical to original)
    def _filter_valid_filenames(pool: dict, cfg: dict) -> list:
        if cfg['pairing_strategy'] == 'relaxed':
            return [f for f, s_map in pool.items() if len(s_map) >= cfg['min_scales_required']]
        else: # strict
            all_scales = [float(s) for s in cfg['scales']]
            return [f for f, s_map in pool.items() if len(s_map) == len(all_scales)]

    # Load prompt source files - this is real I/O as required. (Identical to original)
    _project_root = Path(os.getcwd())
    root_prompts_path = _project_root / own_config['prompt_sources']['root']['file']
    metadata_path = _project_root / own_config['prompt_sources']['metadata']['file']
    with open(root_prompts_path, 'r') as f:
        root_prompts = yaml.safe_load(f)
    with open(metadata_path, 'r') as f:
        metadata = yaml.safe_load(f)
        
    # Helper for prompt composition (Identical to original, now correctly uses all required inputs)
    def _get_composed_prompt(item_path: Path, prompt_cfg: dict, rng: random.Random) -> dict:
        item_filename = item_path.name
        folder_name = item_path.parent.name
        sources, weights = [], []

        sources.append(root_prompts['default_prompt'])
        weights.append(prompt_cfg['root']['importance'])

        # FIX: Restored the crucial metadata lookup for folder-specific prompts.
        if folder_name in metadata.get('folders', {}):
            key = metadata['folders'][folder_name]['prompt_key']
            if key in root_prompts:
                sources.append(root_prompts[key])
                weights.append(prompt_cfg['metadata']['folder_importance'])

        # FIX: Restored the crucial metadata lookup for item-specific prompts.
        if item_filename in metadata.get('items', {}):
            key = metadata['items'][item_filename]['prompt_key']
            if key in root_prompts:
                sources.append(root_prompts[key])
                weights.append(prompt_cfg['metadata']['item_importance'])
        
        return rng.choices(sources, weights=weights, k=1)[0]
        
    valid_filenames = _filter_valid_filenames(data_pool, own_config)
    if not valid_filenames:
        raise ValueError("L2: No valid filenames found after filtering.")

    def generator() -> Iterator[Dict]:
        rng = random.Random(own_config.get('seed', 42))
        while True:
            filename = rng.choice(valid_filenames)
            scales = sorted(data_pool[filename].keys())
            if len(scales) < 2: continue
            
            low_s, high_s = sorted(rng.sample(scales, 2))
            high_path = Path(data_pool[filename][high_s])
            low_path = Path(data_pool[filename][low_s])
            
            # Use the fully-featured prompt composer
            high_prompt_recipe = _get_composed_prompt(high_path, own_config['prompt_sources'], rng)
            low_prompt_recipe = _get_composed_prompt(low_path, own_config['prompt_sources'], rng)

            # --- ARCHITECTURAL FIX: The output is a clean, generic data package ---
            # This 'yield' statement remains the only change from the original function's core loop.
            yield {
                "source_identifier": filename,
                "primitives": [
                    {"qualifier": "high_scale", "type": "image", "data_path": str(high_path)},
                    {"qualifier": "high_scale", "type": "prompt", "data_content": high_prompt_recipe},
                    {"qualifier": "high_scale", "type": "scale_metadata", "data_value": high_s},
                    {"qualifier": "low_scale", "type": "image", "data_path": str(low_path)},
                    {"qualifier": "low_scale", "type": "prompt", "data_content": low_prompt_recipe},
                    {"qualifier": "low_scale", "type": "scale_metadata", "data_value": low_s},
                    {"qualifier": "global", "type": "prng_seed", "data_value": rng.getrandbits(64)}
                ]
            }
    
    print(f"  - Discovered {len(valid_filenames)} valid items. Returning data stream.")
    # The output is PURE DATA, not configuration, with the key expected by the DAG.
    return {'source_data_stream': generator()}

# --- Layer 3: Work Planning Engine ---

def plan_asset_workload(own_config: Dict, **upstream_data) -> Dict[str, Any]:
    print("[LAYER 3] Planning abstract asset workload...")
    asset_spec = upstream_data['required_assets_spec']
    data_iterator = upstream_data['source_data_stream']
    max_items_to_plan = own_config.get('max_training_items', 100)

    work_items_by_interface: Dict[str, List[Dict]] = defaultdict(list)
    seen_work_ids: Set[str] = set()
    assembly_resolution_map: List[Dict[Any, Any]] = []

    reqs_by_interface = defaultdict(list)
    for req in asset_spec.get('required_assets', []):
        reqs_by_interface[req['asset_interface']].append(req)

    for data_item in itertools.islice(data_iterator, max_items_to_plan):
        source_id = data_item['source_identifier']
        current_item_resolution_map: Dict[Any, Any] = {}
        primitives_by_type = defaultdict(list)
        for p in data_item.get('primitives', []): primitives_by_type[p['type']].append(p)

        # BUGFIX: This loop is now beautifully simple. It treats ALL interfaces the same.
        # No more `if interface == 'identity'`
        for interface, req_assets in reqs_by_interface.items():
            all_source_types = sorted(list(set(st for req in req_assets for st in req['source_types_required'])))
            candidate_primitives_per_type = [primitives_by_type[st] for st in all_source_types]
            if not all(candidate_primitives_per_type): continue

            for primitive_bundle in itertools.product(*candidate_primitives_per_type):
                def get_primitive_data_as_string(p):
                    if 'data_path' in p: return str(p['data_path'])
                    if 'data_value' in p: return str(p['data_value'])
                    if 'data_content' in p: return str(p['data_content'])
                    return ""
                input_ids = "".join(get_primitive_data_as_string(p) for p in primitive_bundle)
                work_id = hashlib.sha1(f"{interface}-{input_ids}".encode()).hexdigest()

                if work_id not in seen_work_ids:
                    seen_work_ids.add(work_id)
                    work_item = { "work_id": work_id, "source_primitives": primitive_bundle }
                    work_items_by_interface[interface].append(work_item)
                
                for req in req_assets:
                    req_source_types = set(req['source_types_required'])
                    primary_primitive = next((p for p in primitive_bundle if p['type'] in req_source_types and p['qualifier'] != 'global'), None)
                    if primary_primitive:
                        assembly_key = (primary_primitive['qualifier'], req['asset_type'])
                        # The instruction is ALWAYS a work_id.
                        current_item_resolution_map[assembly_key] = work_id

        assembly_resolution_map.append(current_item_resolution_map)
    
    print(f"  - Planning complete. Unique work items: {len(seen_work_ids)}. Items planned: {len(assembly_resolution_map)}.")
    return {'abstract_work_plan': dict(work_items_by_interface), 'assembly_resolution_map': assembly_resolution_map}



# --- Layer 4: Asset Execution Engine (Corrected) ---
# Define our library of actual consumer functions (stubs for now)
def dummy_denoise_input_encoder(**kwargs) -> dict:
    return {"noisy_latent": f"stub_latent_for_{kwargs['prng_seed']}", "timestep_for_unet": f"stub_timestep_for_{kwargs['prng_seed']}"}

def dummy_text_encoder(**kwargs) -> dict:
    return {"text_embedding": "stub_text_embed", "pooled_text_embedding": "stub_pooled_embed"}

def dummy_time_id_synthesizer(**kwargs) -> dict:
    return {"time_embedding": "stub_time_embed"}

def scale_tensor_synthesizer_v1(**kwargs) -> dict:
    """
    A simple consumer that takes the 'scale_metadata' primitive and returns
    it packaged as the 'scales_tensor' asset.
    """
    return {"scales_tensor": kwargs['scale_metadata']}

ASSET_CONSUMER_FUNCTIONS = {
    "denoise_input_encoder_v1": dummy_denoise_input_encoder,
    "text_encoder_v1": dummy_text_encoder,
    "time_id_synthesizer_v1": dummy_time_id_synthesizer,
    "scale_tensor_synthesizer_v1": scale_tensor_synthesizer_v1
}

def execute_asset_caching(own_config, **upstream_data):
    print("[LAYER 4] Executing asset caching (STUBBED)...")
    work_plan = upstream_data['abstract_work_plan']
    consumer_map = own_config.get("consumer_implementation_map", {})
    cache_files = own_config.get("cache_files", {})
    cache_manifest: Dict[Tuple[str, str], Tuple[str, str]] = {}
    print(f"  - Received work plan for interfaces: {list(work_plan.keys())}")
    for interface, work_items in work_plan.items():
        if interface not in consumer_map: print(f"  - WARNING: No implementation for interface '{interface}'."); continue
        consumer_name = consumer_map[interface]
        consumer_func = ASSET_CONSUMER_FUNCTIONS.get(consumer_name)
        if not consumer_func: print(f"  - WARNING: Consumer function '{consumer_name}' not found."); continue
        print(f"  - Processing {len(work_items)} items for '{interface}' with '{consumer_name}'")
        for item in work_items:
            work_id = item['work_id']
            kwargs = {p['type']: p.get('data_path') or p.get('data_value') or p.get('data_content') for p in item['source_primitives']}
            output_assets = consumer_func(**kwargs)
            for asset_type, tensor_data in output_assets.items():
                target_cache_file = cache_files.get(asset_type, "default_cache.safetensors")
                manifest_key = (work_id, asset_type)
                location_in_file = (target_cache_file, f"{work_id}_{asset_type}")
                cache_manifest[manifest_key] = location_in_file
    print(f"  - Caching complete. Generated manifest with {len(cache_manifest)} asset locations.")
    return {'asset_cache_manifest': cache_manifest}


# --- Layer 5: Training Dataset Assembly (Reverted to simpler, uniform logic) ---
def assemble_training_dataset(own_config: Dict, **upstream_data) -> Dict[str, Any]:
    print("[LAYER 5] Assembling final training dataset (STUBBED)...")
    resolution_map = upstream_data['assembly_resolution_map']
    cache_manifest = upstream_data['asset_cache_manifest']
    print(f"  - Received resolution map for {len(resolution_map)} items and manifest for {len(cache_manifest)} assets.")
    def final_dataset_generator() -> Iterator[Dict]:
        for i, item_map in enumerate(resolution_map):
            training_batch = {"sample_id": i}; all_assets_found = True
            # BUGFIX: The logic is simple again. It only knows how to look up work_ids.
            for (qualifier, asset_type), work_id in item_map.items():
                manifest_key = (work_id, asset_type)
                if manifest_key in cache_manifest:
                    filepath, key_in_file = cache_manifest[manifest_key]
                    training_batch[f"{qualifier}_{asset_type}"] = f"loc:{filepath}|key:{key_in_file}"
                else:
                    print(f"  - WARNING: Missing cached asset for work_id '{work_id}' (asset: {asset_type}).")
                    all_assets_found = False; break
            if all_assets_found: yield training_batch
    print("  - Assembly complete. Returning final dataset iterator.")
    return {'final_training_dataset': final_dataset_generator()}


# --- Layer 6: DAG Orchestration Engine ---
# ARCH-FIX: This engine remains generic. Its correctness depends on the strict
# interface contracts of the layers it calls. It correctly injects the 'own_config'
# for each stage, ensuring configuration doesn't leak between layers.

# BUGFIX: The main map now ONLY contains the layer functions.
STRATEGIZER_FUNCTION_MAP = {
    "get_required_asset_spec": get_required_asset_spec,
    "discover_source_data_stream": discover_source_data_stream,
    "plan_asset_workload": plan_asset_workload,
    "execute_asset_caching": execute_asset_caching,
    "assemble_training_dataset": assemble_training_dataset,
}

def execute_dag_from_manifest(manifest_path: Path) -> Dict[str, Any]:
    """
    LAYER 6: Executes the full pipeline based on a manifest's dependency graph.
    """
    with open(manifest_path, 'r') as f:
        manifest = yaml.safe_load(f)
    
    graph_stages = manifest['execution_graph']
    execution_results: Dict[str, Any] = {}
    processed_stages: Set[str] = set()
    
    # Map all possible output keys to the stage that produces them
    key_producer_map = {
        output_key: stage['stage_name']
        for stage in graph_stages
        for output_key in stage.get('outputs', [])
    }

    for _ in range(len(graph_stages)):
        newly_processed = False
        for stage in graph_stages:
            stage_name = stage['stage_name']
            if stage_name in processed_stages:
                continue

            # Determine dependencies
            input_keys = stage.get('inputs', {})
            dep_stages = {key_producer_map[k] for k in input_keys.values() if k in key_producer_map}
            
            if dep_stages.issubset(processed_stages):
                print(f"\n[DAG RUNNER] Executing stage: '{stage_name}'")
                fn = STRATEGIZER_FUNCTION_MAP[stage['function']]
                
                # Assemble kwargs from previous results
                kwargs = {local_name: execution_results[global_name] for local_name, global_name in input_keys.items()}
                
                # The function gets its OWN config and the data from its dependencies.
                stage_outputs = fn(own_config=stage.get('config', {}), **kwargs)

                # Store all outputs from the stage into the global results dictionary.
                for output_key in stage.get('outputs', []):
                    if output_key in stage_outputs:
                        execution_results[output_key] = stage_outputs[output_key]

                processed_stages.add(stage_name)
                newly_processed = True
        
        if not newly_processed and len(processed_stages) < len(graph_stages):
            unprocessed = {s['stage_name'] for s in graph_stages} - processed_stages
            raise RuntimeError(f"DAG execution failed. Deadlock detected. Unprocessed stages: {unprocessed}")

    return execution_results

# --- Main for Self-Testing ---
def main():
    # ARCH-FIX: Point to the new, architecturally correct manifest.
    project_root = Path(os.getcwd())
    manifest_path = project_root / "run_artifacts" / "yamlzoo" / "experiment_manifest_dagv5.4.yaml"
    print(f"--- Executing Refactored DAG from: {manifest_path} ---")
    try:
        final_results = execute_dag_from_manifest(manifest_path)
        print("\n--- DAG Execution Complete. Final results available: ---")
        for key, value in final_results.items():
            value_type = "iterator" if hasattr(value, '__next__') else type(value).__name__
            print(f"  - '{key}' (type: {value_type})")
        
        print("\n--- Demonstrating the FINAL 'final_training_dataset' iterator ---")
        final_iterator = final_results.get('final_training_dataset')
        if final_iterator:
            for i, item in enumerate(final_iterator):
                if i >= 3: break
                print(f"  Item {i}: {item}")
        else:
            print("  - Final iterator not found in results.")
            
    except Exception as e:
        import traceback
        print(f"\n[ERROR] An error occurred during DAG execution: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # To run this, you will need:
    # 1. The `experiment_manifest_refactored.yaml` file in the same directory.
    # 2. A `discovery_scanner.py` file with a `discover_data_pool` function.
    # 3. The dataset and prompt files referenced in the manifest.
    main() # Commented out to prevent execution without setup.