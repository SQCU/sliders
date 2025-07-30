# FFW13: dataset_strategizer_dagv5.8.py
# This version is a complete architectural refactor to enforce strict layer separation
# as per the "Semantic Layer Specifications" and the provided critique.
# uv run python dataset_strategizer_dagv5.8.py >> dset_strat_dagv5.8.txt 2>&1

import json
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
import multiprocessing as mp # <--- ADD THIS IMPORT AT THE TOP
import safetensors

#disable nuisanceware logprints by library developers who don't know right from wrong
import warnings
import re

# --- Layer 1: Model Asset Registry (Interface Contract Definition) ---
# ARCH-FIX: VIOLATION 1 & 3 FIXED.
# The registry is now purely declarative and defines INTERFACES, not IMPLEMENTATIONS.
# It specifies *what* a model needs, not *how* to create it. It contains no
# hardcoded 'consumer_signature' names. This passes the "Litmus Test" because
# the tools used to satisfy these interfaces can be changed in a separate
# configuration without touching this registry.

# --- Layer 1: Data Consumer Registry (Interface Contract Definition) ---
# ARCH-FIX (v5.5): This version completely decouples the registry from implementation by
# replacing the hardcoded 'asset_interface' name with a declarative 'capability_requirements'
# block. This registry now only describes WHAT a consumer needs and the abstract PROPERTIES
# of the process that creates it, not a named reference to the process itself.
# This design is now truly agnostic to the consumer's domain (e.g., image generation vs. text evaluation).

DATA_CONSUMER_REGISTRY = {
    # --- USE CASE 1: Image Generation Model ---
    'StableDiffusionXL_UNet': {
        'required_assets': [
            # Note: The 'capability_requirements' block is identical for both assets
            # produced by the same underlying capability. This is intentional and explicit.
            {
                "asset_type": "noisy_latent",
                "capability_requirements": {
                    "processing_category": "image_denoising_prep",
                    "input_types": ["image", "prng_seed"],
                    "output_types": ["noisy_latent", "timestep_for_unet"]
                }
            },
            {
                "asset_type": "timestep_for_unet",
                "capability_requirements": {
                    "processing_category": "image_denoising_prep",
                    "input_types": ["image", "prng_seed"],
                    "output_types": ["noisy_latent", "timestep_for_unet"]
                }
            },
            {
                "asset_type": "text_embedding",
                "capability_requirements": {
                    "processing_category": "text_encoding",
                    "input_types": ["prompt"],
                    "output_types": ["text_embedding", "pooled_text_embedding"]
                }
            },
            {
                "asset_type": "pooled_text_embedding",
                "capability_requirements": {
                    "processing_category": "text_encoding",
                    "input_types": ["prompt"],
                    "output_types": ["text_embedding", "pooled_text_embedding"]
                }
            },
            {
                "asset_type": "time_embedding",
                "capability_requirements": {
                    "processing_category": "temporal_encoding",
                    "input_types": ["image"],
                    "output_types": ["time_embedding"]
                }
            },
            {
                "asset_type": "scales_tensor",
                "capability_requirements": {
                    "processing_category": "metadata_synthesis",
                    "input_types": ["scale_metadata"],
                    "output_types": ["scales_tensor"]
                }
            }
        ]
    },
    'SDXL_VAE_Encoder': {
        'required_assets': [
            {
                "asset_type": "latent",
                "capability_requirements": {
                    "processing_category": "image_to_latent_encoding",
                    "input_types": ["image"],
                    "output_types": ["latent"]
                }
            }
        ]
    },
    # --- VAE DECODING TASK (CORRECTLY NAMED) ---
    'SDXL_VAE_Decoder': {
        'required_assets': [
            {
                "asset_type": "decoded_image",
                "capability_requirements": {
                    "processing_category": "latent_to_image_decoding",
                    "input_types": ["latent"],
                    "output_types": ["decoded_image"]
                }
            }
        ]
    },
    # --- USE CASE 2 (Fictional): Text Evaluation Script (Our "The Mask" Contrast Case) ---
    'TheMask_Eval_Consumer': {
        'required_assets': [
            {
                "asset_type": "masked_corpus_chunk",
                "capability_requirements": {
                    "processing_category": "the_mask_language_objective",
                    "input_types": ["raw_text"],
                    "output_types": ["masked_corpus_chunk", "original_text_target"]
                }
            },
            {
                "asset_type": "original_text_target",
                "capability_requirements": {
                    "processing_category": "the_mask_language_objective",
                    "input_types": ["raw_text"],
                    "output_types": ["masked_corpus_chunk", "original_text_target"]
                }
            }
        ]
    },
    'VAE_Validation_Report': {
    'required_assets': [
        {
            "asset_type": "reconstruction_metrics",
            "capability_requirements": {
                "processing_category": "image_reconstruction_evaluation",
                "input_types": ["original_image", "reconstructed_image"],
                "output_types": ["reconstruction_metrics"]
            }
        }
    ]
}   ,
    # To add a new data consumer, a developer would simply add a new entry here.
    # The rest of the DAG is responsible for fulfilling the declared capabilities.
}

def get_required_asset_spec(own_config: Dict, **kwargs) -> Dict[str, Any]:
    """
    LAYER 1: Looks up the required asset CAPABILITIES for a data consumer.
    This function is a pure configuration dispatcher. It does not know or care
    how the assets will be made, only what capabilities are required.
    """
    print("[LAYER 1] Looking up required asset specification...")
    consumer_name = own_config.get('data_consumer_name')
    if not consumer_name:
        raise ValueError("L1 Config must contain 'data_consumer_name'.")

    asset_spec = DATA_CONSUMER_REGISTRY.get(consumer_name)
    if asset_spec is None:
        raise ValueError(f"Unknown data consumer '{consumer_name}' in DATA_CONSUMER_REGISTRY.")

    print(f"  - Found {len(asset_spec['required_assets'])} required assets for consumer '{consumer_name}'.")
    return {'required_assets_spec': asset_spec}


# --- Layer 2: Data Discovery Engine ---

def discover_slider_pairs_from_filesystem(own_config: Dict, **kwargs) -> Dict[str, Any]:
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

def discover_reconstruction_pairs_from_manifests(own_config: Dict, **upstream_data) -> Dict[str, Any]:
    """
    LAYER 2 (CORRECTED): Discovers (original, reconstructed) image pairs by tracing
    the lineage of work_ids through the encoding and decoding plans.
    """
    print("[LAYER 2] Discovering reconstruction pairs from manifests...")
    # Input 1: The original plan to encode images. This links an original image path to an 'encode_work_id'.
    encoding_plan = upstream_data['source_plan']
    # Input 2: The plan that DECODED the latents. This links a 'decode_work_id' back to an 'encode_work_id'.
    decoding_plan = upstream_data['decoding_plan']
    # Input 3: The final manifest. This links a 'decode_work_id' to a reconstructed image location.
    reconstructed_manifest = upstream_data['reconstructed_asset_manifest']

    # --- THE FIX ---
    # 1. Create a lookup map of {encode_work_id -> original_image_path}
    encode_id_to_path = {
        spec['work_id']: spec['input_data']['image']['data_path']
        for spec in encoding_plan
    }

    # 2. Create a lookup map of {decode_work_id -> reconstructed_image_location}
    decode_id_to_location = {
        work_id: loc
        for (work_id, asset_type), loc in reconstructed_manifest.items()
        if asset_type == 'reconstructed_image' # Make sure we only get the right assets
    }

    def generator() -> Iterator[Dict]:
        # 3. Iterate through the DECODING plan, which contains the critical link.
        for decode_spec in decoding_plan:
            decode_work_id = decode_spec['work_id']

            # 4. Extract the original 'encode_work_id' from the latent's key.
            # The key is structured like: 'ENCODE_WORK_ID_latent'
            latent_location = decode_spec['input_data']['latent']['data_location']
            latent_key = {p.split(':')[0]: p.split(':')[1] for p in latent_location.split('|')}['key']
            # Split the key and take the first part, which is the original encode_work_id
            original_encode_id = latent_key.split('_latent')[0]

            # 5. Use the IDs to look up the paths/locations in our maps.
            if original_encode_id in encode_id_to_path and decode_work_id in decode_id_to_location:
                original_path = encode_id_to_path[original_encode_id]
                recon_location = decode_id_to_location[decode_work_id]

                # Yield a standard primitive package for the validation planner.
                yield {
                    "source_identifier": original_encode_id, # Use original ID for consistency
                    "primitives": [
                        {"qualifier": "original", "type": "original_image", "data_path": original_path},
                        {"qualifier": "reconstructed", "type": "reconstructed_image", "data_location": recon_location}
                    ]
                }

    print(f"  - Tracing lineage for {len(decoding_plan)} decoded images.")
    return {'source_data_stream': generator()}

# this is a good example of a layer 2 function!
def discover_assets_from_manifest(own_config: Dict, **upstream_data) -> Dict[str, Any]:
    """
    LAYER 2: A generic discovery function that turns an asset manifest into a
    source_data_stream. It treats each asset in the manifest as a new primitive.
    """
    print("[LAYER 2] Discovering assets from an upstream manifest...")
    asset_manifest = upstream_data['asset_manifest']
    asset_type_to_find = own_config['asset_type_to_find']
    primitive_type_label = own_config['primitive_type_label']

    def generator() -> Iterator[Dict]:
        # We iterate through the manifest, which is the ground truth of what was created.
        for (work_id, asset_type), location_str in asset_manifest.items():
            if asset_type == asset_type_to_find:
                # For each matching asset, we create a new data item for the planner.
                yield {
                    "source_identifier": work_id,
                    "primitives": [
                        {
                           "qualifier": "reconstruction_source",
                           "type": primitive_type_label,
                           # The "data" for this primitive is its location in the cache.
                           "data_location": location_str
                        }
                    ]
                }

    print(f"  - Discovering assets of type '{asset_type_to_find}' from manifest.")
    return {'source_data_stream': generator()}

# --- Layer 3: Work Planning Engine (v5.5 - Final Revision) ---

def plan_asset_workload(own_config: Dict, **upstream_data) -> Dict[str, Any]:
    """
    LAYER 3 (v5.5): Plans a generic workload and produces an explicit set of
    assembly instructions, fully decoupling it from Layer 5.
    """
    print("[LAYER 3] Planning abstract asset workload...")
    asset_spec = upstream_data['required_assets_spec']
    data_iterator = upstream_data['source_data_stream']
    max_items_to_plan = own_config.get('max_training_items', 100)

    work_specifications: List[Dict] = []
    # ARCH-FIX: This is the new, explicit data contract.
    assembly_instructions: List[Dict] = []
    seen_work_ids: Set[str] = set()

    reqs_by_capability = defaultdict(list)
    for req in asset_spec.get('required_assets', []):
        capability = req['capability_requirements']

        # BUGFIX: Recursively make the capability structure hashable by converting
        # internal lists to sorted tuples, ensuring a stable and valid key.
        cap_key = tuple(sorted(
            (k, tuple(sorted(v)) if isinstance(v, list) else v)
            for k, v in capability.items()
        ))
        
        reqs_by_capability[cap_key].append(req)

    for i, data_item in enumerate(itertools.islice(data_iterator, max_items_to_plan)):
        # This temporary map is now a pure implementation detail of L3.
        current_item_resolution_map: Dict[Tuple[str, str], str] = {}
        primitives_by_type = defaultdict(list)
        for p in data_item.get('primitives', []):
            primitives_by_type[p['type']].append(p)

        for cap_key, req_assets in reqs_by_capability.items():
            capability = dict(cap_key)
            all_source_types = sorted(capability['input_types'])
            
            candidate_primitives_per_type = [primitives_by_type[st] for st in all_source_types]
            if not all(candidate_primitives_per_type): continue

            for primitive_bundle in itertools.product(*candidate_primitives_per_type):
                def get_primitive_data_as_string(p):
                    if 'data_location' in p: return str(p['data_location'])
                    if 'data_path' in p: return str(p['data_path'])
                    if 'data_value' in p: return str(p['data_value'])
                    if 'data_content' in p: return str(p['data_content'])
                    return ""

                input_ids = "".join(get_primitive_data_as_string(p) for p in primitive_bundle)
                work_id = hashlib.sha1(f"{capability['processing_category']}-{input_ids}".encode()).hexdigest()

                if work_id not in seen_work_ids:
                    seen_work_ids.add(work_id)
                    work_specifications.append({
                        "work_id": work_id,
                        "required_capabilities": capability,
                        # --- THE MINIMAL, CORRECT FIX ---
                        # Instead of extracting one value, we pass the ENTIRE
                        # primitive dictionary for each type. Layer 3 remains
                        # agnostic to its contents.
                        "input_data": {p['type']: p for p in primitive_bundle},
                        "expected_outputs": capability['output_types']
                    })

                for req in req_assets:
                    primary_primitive = next((p for p in primitive_bundle if p['type'] in req['capability_requirements']['input_types'] and p['qualifier'] != 'global'), None)
                    if primary_primitive:
                        assembly_key = (primary_primitive['qualifier'], req['asset_type'])
                        current_item_resolution_map[assembly_key] = work_id
        
        # ARCH-FIX: Transform the internal resolution map into the public assembly_instructions contract.
        item_assets_to_assemble = []
        for (qualifier, asset_type), work_id in current_item_resolution_map.items():
            item_assets_to_assemble.append({
                "final_asset_key": f"{qualifier}_{asset_type}",
                "lookup_work_id": work_id,
                "lookup_asset_type": asset_type
            })
        assembly_instructions.append({
            "sample_id": i,
            "assets_to_assemble": item_assets_to_assemble
        })


    print(f"  - Planning complete. Unique work specs: {len(work_specifications)}. Items planned: {len(assembly_instructions)}.")
    # ARCH-FIX: The output contract is now clean, explicit, and fully decoupled.
    return {'work_specifications': work_specifications, 'assembly_instructions': assembly_instructions}


# --- Layer 4: Asset Execution Engine (Corrected) ---
# Define our library of actual consumer functions (stubs for now)
# REFACTOR: Import the tools from the dedicated library file.
from d_dset_functions import (
    real_denoise_input_encoder,
    real_sdxl_text_encoder,
    real_time_id_synthesizer,
    scale_tensor_synthesizer_v1,
    real_image_to_latent_encoder,
    real_latent_to_image_decoder,
    compute_validation_metrics,
    
)

ASSET_CONSUMER_FUNCTIONS = {
    "denoise_input_encoder_v1": real_denoise_input_encoder,
    "text_encoder_v1": real_sdxl_text_encoder,
    "time_id_synthesizer_v1": real_time_id_synthesizer,
    "scale_tensor_synthesizer_v1": scale_tensor_synthesizer_v1,
    "real_image_to_latent_encoder": real_image_to_latent_encoder,
    "real_latent_to_image_decoder": real_latent_to_image_decoder,
    "compute_validation_metrics": compute_validation_metrics,
}


# This helper function is the core of the new decoupled logic.
def find_matching_capability(required_caps: Dict, capability_map: Dict) -> Dict:
    """Matches a work spec's requirements to an available implementation."""
    req_category = required_caps.get('processing_category')
    req_inputs = set(required_caps.get('input_types', []))
    req_outputs = set(required_caps.get('output_types', []))

    for impl_spec in capability_map.values():
        if impl_spec.get('processing_category') != req_category:
            continue
        
        # An implementation is valid if it can handle all required inputs
        # and produce all expected outputs. (It can handle more than required).
        available_inputs = set(impl_spec.get('input_types', []))
        available_outputs = set(impl_spec.get('output_types', []))

        if req_inputs.issubset(available_inputs) and req_outputs.issubset(available_outputs):
            return impl_spec
    return None

def execute_asset_caching(own_config: Dict, **upstream_data) -> Dict[str, Any]:
    """
    LAYER 4 (v5.7): Executes asset caching by delegating blocks of work to
    specialized consumer functions, all of which conform to a unified iterator-
    based contract.
    """
    print("[LAYER 4] Executing asset caching...")
    work_specs = upstream_data['work_specifications']
    capability_map = own_config.get("capability_implementation_map", {})
    cache_files_config = own_config.get("cache_files", {})
    # Manifest will track the logical location of each asset.
    cache_manifest: Dict[Tuple[str, str], Tuple[str, str]] = {}
    # This new dictionary will collect all tensors to be written to each file.
    tensors_to_write_by_file = defaultdict(dict)

    metadata_to_write_by_file = defaultdict(lambda: defaultdict(dict))
    # This will hold the metadata dictionaries to be written to each cache file.
    # The structure will be: { "filename.safetensors": {"aggregates": {...}} }
    print(f"  - Received {len(work_specs)} generic work specifications.")

    # Group work by the target consumer function
    work_by_consumer = defaultdict(list)
    for spec in work_specs:
        implementation = find_matching_capability(spec['required_capabilities'], capability_map)
        if implementation:
            consumer_name = implementation['consumer_function']
            work_by_consumer[consumer_name].append(spec)
    
    # Process each block of work. There is only one path now.
    for consumer_name, specs_for_consumer in work_by_consumer.items():
        consumer_func = ASSET_CONSUMER_FUNCTIONS.get(consumer_name)
        if not consumer_func:
            print(f"  - WARNING: Consumer function '{consumer_name}' not found."); continue

        print(f"  - Delegating block of {len(specs_for_consumer)} items to '{consumer_name}'...")
        implementation_config = find_matching_capability(specs_for_consumer[0]['required_capabilities'], capability_map)
        kwargs_for_consumer = {**implementation_config, "data_iterator": iter(specs_for_consumer)}
        
        results_by_work_id = consumer_func(**kwargs_for_consumer)

        #new bespoke functionality which supports metadata mapping tensors
        #  *written into our cachefiles in the safetensors configheader*!!!
        if "aggregate_result" in results_by_work_id:
            agg_data = results_by_work_id["aggregate_result"]
            tensor_dict = agg_data["asset_data"]
            asset_type = agg_data["asset_type"]
            contributing_ids = agg_data["contributing_work_ids"]

            target_cache_file = cache_files_config.get(asset_type, "default_cache.safetensors")
            agg_id = hashlib.sha1("".join(sorted(contributing_ids)).encode()).hexdigest()

            # --- THE KEY CHANGE IS HERE ---
            # Create the metadata payload we WANT to store.
            metadata_payload = {
                "asset_type": asset_type,
                "contributing_work_ids": contributing_ids
            }
            # Serialize the entire payload into a single JSON string.
            # The key is the aggregate asset ID, the value is the string.
            metadata_to_write_by_file[target_cache_file][agg_id] = json.dumps(metadata_payload)
            # --- End of Change ---

            location_recipe = {}
            for sub_key, tensor_data in tensor_dict.items():
                location_in_file = f"{agg_id}_{asset_type}_{sub_key}"
                tensors_to_write_by_file[target_cache_file][location_in_file] = tensor_data
                location_recipe[sub_key] = f"loc:{target_cache_file}|key:{location_in_file}"
            
            for c_id in contributing_ids:
                cache_manifest[(c_id, asset_type)] = location_recipe

        # Path 2: Handle all "normal" any_in -> tensor_out functions.
        else:
            for work_id, assets in results_by_work_id.items():
                for asset_type, asset_data in assets.items():
                    target_cache_file = cache_files_config.get(asset_type, "default_cache.safetensors")
                    # Check if the asset data is a dictionary (our nested case)
                    if isinstance(asset_data, dict):
                        location_recipe = {}
                        for sub_key, tensor_data in asset_data.items():
                            location_in_file = f"{work_id}_{asset_type}_{sub_key}"
                            tensors_to_write_by_file[target_cache_file][location_in_file] = tensor_data
                            location_recipe[sub_key] = f"loc:{target_cache_file}|key:{location_in_file}"
                        cache_manifest[(work_id, asset_type)] = location_recipe
                    else:
                        # This is the original path for simple, non-nested tensor assets.
                        tensor_data = asset_data
                        location_in_file = f"{work_id}_{asset_type}"
                        cache_manifest[(work_id, asset_type)] = f"loc:{target_cache_file}|key:{location_in_file}"
                        tensors_to_write_by_file[target_cache_file][location_in_file] = tensor_data

    # --- STEP 2: Perform batched WRITES to disk ---
    print("  - All asset functions executed. Writing caches to disk...")
    for target_file, tensors_to_write in tensors_to_write_by_file.items():
        if not tensors_to_write: continue
        
        print(f"    - Writing {len(tensors_to_write)} assets to {target_file}")
        output_path = Path(target_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # To prevent overwriting, we first load existing tensors from the file if it exists.
        existing_tensors = {}
        if output_path.exists():
            with safetensors.safe_open(output_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    existing_tensors[key] = f.get_tensor(key)
                existing_metadata = f.metadata() if f.metadata() is not None else {}
        
        final_tensors = {**existing_tensors, **tensors_to_write}

        # --- THE FINAL METADATA MERGE ---
        new_metadata_for_this_file = metadata_to_write_by_file.get(target_file, {})
        # Now it's a simple, flat dictionary merge. No deep nesting needed.
        final_metadata = {**existing_metadata, **new_metadata_for_this_file}

        temp_output_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
        safetensors.torch.save_file(final_tensors, temp_output_path, metadata=final_metadata)
        os.replace(temp_output_path, output_path)

    print(f"  - Caching complete. Generated manifest with {len(cache_manifest)} asset locations.")
    return {'asset_cache_manifest': cache_manifest}


# --- Layer 5: Training Dataset Assembly (v5.5) ---
def assemble_training_dataset(own_config: Dict, **upstream_data) -> Dict[str, Any]:
    """
    LAYER 5 (v5.5): Assembles final data consumer batches by following an explicit
    list of instructions. It is now fully decoupled from the planning logic of Layer 3.
    """
    print("[LAYER 5] Assembling final data batches...")
    # ARCH-FIX: Consume the new, explicit and decoupled data contract.
    assembly_instructions = upstream_data['assembly_instructions']
    cache_manifest = upstream_data['asset_cache_manifest']
    print(f"  - Received instructions for {len(assembly_instructions)} items and manifest for {len(cache_manifest)} assets.")

    def final_dataset_generator() -> Iterator[Dict]:
        for i, item_instruction in enumerate(assembly_instructions):
            # ARCH-FIX: The assembly logic is now a simple, direct loop over a list of instructions.
            # It no longer needs to know about tuple keys or other L3 implementation details.
            batch = {"sample_id": i}
            all_assets_found = True

            for asset_req in item_instruction.get('assets_to_assemble', []):
                final_key = asset_req['final_asset_key']
                work_id = asset_req['lookup_work_id']
                asset_type = asset_req['lookup_asset_type']
                
                manifest_key = (work_id, asset_type)
                if manifest_key in cache_manifest:
                    # --- THE FIX IS HERE ---
                    # The value from the manifest can now be a string OR a dictionary (our recipe).
                    asset_location_info = cache_manifest[manifest_key]
                    batch[final_key] = asset_location_info
                    # --- End of Fix ---
                else:
                    # This part remains the same.
                    print(f"  - WARNING: Missing cached asset for work_id '{work_id}' (asset: {asset_type}).")
                    all_assets_found = False
                    break
            
            if all_assets_found:
                yield batch

    print("  - Assembly complete. Returning final dataset iterator.")
    return {'final_training_dataset': final_dataset_generator()}


# --- Layer 6: DAG Orchestration Engine ---
# ARCH-FIX: This engine remains generic. Its correctness depends on the strict
# interface contracts of the layers it calls. It correctly injects the 'own_config'
# for each stage, ensuring configuration doesn't leak between layers.

# BUGFIX: The main map now ONLY contains the layer functions.
STRATEGIZER_FUNCTION_MAP = {
    "get_required_asset_spec": get_required_asset_spec,
    "discover_slider_pairs_from_filesystem": discover_slider_pairs_from_filesystem,
    "discover_reconstruction_pairs_from_manifests": discover_reconstruction_pairs_from_manifests,
    "plan_asset_workload": plan_asset_workload,
    "execute_asset_caching": execute_asset_caching,
    "assemble_training_dataset": assemble_training_dataset,
    "compute_validation_metrics": compute_validation_metrics,
    'discover_assets_from_manifest': discover_assets_from_manifest,
}

def execute_dag_from_manifest(manifest_path: Path) -> Dict[str, Any]:
    """
    LAYER 6 (v5.5 - Corrected): Executes the full pipeline based on a manifest's
    dependency graph, using an explicit input/output mapping to prevent name collisions.
    """
    with open(manifest_path, 'r') as f:
        manifest = yaml.safe_load(f)
    
    graph_stages = manifest['execution_graph']
    execution_results: Dict[str, Any] = {}
    processed_stages: Set[str] = set()
    
    # Map all possible global output keys to the stage that produces them
    key_producer_map = {
        global_name: stage['stage_name']
        for stage in graph_stages
        # ARCH-FIX: Read the values (global names) from the new output map
        for global_name in stage.get('outputs', {}).values()
    }

    for _ in range(len(graph_stages)):
        newly_processed = False
        for stage in graph_stages:
            stage_name = stage['stage_name']
            if stage_name in processed_stages:
                continue

            # Determine dependencies from the values (global names) of the input map
            input_map = stage.get('inputs', {})
            dep_stages = {key_producer_map[k] for k in input_map.values() if k in key_producer_map}
            
            if dep_stages.issubset(processed_stages):
                print(f"\n[DAG RUNNER] Executing stage: '{stage_name}'")
                fn = STRATEGIZER_FUNCTION_MAP[stage['function']]
                
                # ARCH-FIX: Assemble kwargs using the new explicit input map
                # The stage function receives keys like 'required_assets_spec'.
                kwargs = {local_name: execution_results[global_name] for local_name, global_name in input_map.items()}
                
                stage_outputs = fn(own_config=stage.get('config', {}), **kwargs)

                # ARCH-FIX: Store outputs using the new explicit output map.
                # This correctly remaps the function's local key (e.g., 'required_assets_spec')
                # to the desired global key (e.g., 'decoder_assets_spec').
                output_map = stage.get('outputs', {})
                for local_key, global_key in output_map.items():
                    if local_key in stage_outputs:
                        execution_results[global_key] = stage_outputs[local_key]

                processed_stages.add(stage_name)
                newly_processed = True
        
        if not newly_processed and len(processed_stages) < len(graph_stages):
            unprocessed = {s['stage_name'] for s in graph_stages} - processed_stages
            raise RuntimeError(f"DAG execution failed. Deadlock detected. Unprocessed stages: {unprocessed}")

    return execution_results

# --- Main for Self-Testing ---
def main():

    # --- BEGIN WARNING SUPPRESSION PATCH ---
    # This section silences specific, high-volume warnings that are not relevant
    # to this workflow and only add noise to the logs.

    # 1. Silence the torchmetrics `torch.load` pickle warning.
    # Rationale: We are providing our own model files and operate in an environment
    # where we already trust the code and artifacts being used.
    warnings.filterwarnings(
        "ignore",
        message=".*You are using `torch.load` with `weights_only=False`.*",
        category=FutureWarning,
        module="torchmetrics.*"
    )

    # 2. Silence the `torch.tensor(sourceTensor)` copy warning.
    # Rationale: This warning is for performance optimization in computation graphs.
    # We are using torch.tensor to create detached data-holding tensors, where
    # this specific copy pattern is intentional and correct.
    warnings.filterwarnings(
        "ignore",
        message=".*To copy construct from a tensor, it is recommended to use.*",
        category=UserWarning
    )
    # --- END WARNING SUPPRESSION PATCH ---
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--manifest', default="run_artifacts/yamlzoo/experiment_manifest_dagv5.8.yaml")
    args = parser.parse_args()
    project_root = Path(os.getcwd())
    manifest_path = project_root / args.manifest
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
    # Set the start method to 'spawn' for CUDA safety and cross-platform consistency.
    # This must be done once, at the very beginning of the main execution block.
    # A try/except block is used for robustness in environments where the context
    # might already be set (e.g., in some interactive shells or test runners).
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass # Context already set, which is fine.

    # To run this, you will need:
    # 1. The `experiment_manifest_refactored.yaml` file in the same directory.
    # 2. A `discovery_scanner.py` file with a `discover_data_pool` function.
    # 3. The dataset and prompt files referenced in the manifest.
    main() # Commented out to prevent execution without setup.