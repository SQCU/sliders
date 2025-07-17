# data_pipeline.py
# Replaces data_schedule.py, embedding_cache_util.py, and parts of data_processing_utils.py
# The robust, high-performance, and architecturally sound version.

import torch
import yaml
import random
import json
import os
import gc
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from tqdm import tqdm
from safetensors.torch import save_file, load_file
from PIL import Image
from .batch_train_util import create_batched_prompt_embeddings
from .batch_train_util import neo_create_batched_prompt_embeddings, get_add_time_ids
#from .data_processing_utils import encode_images_to_latents # Still needed
#not needed anymore!

# --- UTILITY: VAE BATCH SIZE OPTIMIZATION ---

def find_optimal_vae_batch_size(environment: Dict[str, Any], sample_image_path: str):
    """
    Finds the optimal VAE encoding batch size for the current hardware.
    """
    print("--- Finding optimal VAE batch size... ---")
    vae = environment['vae'].to(environment['device'])
    
    # Create a sample image list
    sample_image = Image.open(sample_image_path).convert("RGB").resize(
        tuple(environment['config']['train']['resolution']), Image.Resampling.LANCZOS
    )
    
    batch_size = 1
    max_batch_size = 256 # A reasonable upper limit
    best_batch_size = 1
    best_throughput = 0.0

    while batch_size <= max_batch_size:
        images = [sample_image] * batch_size
        gc.collect()
        torch.cuda.empty_cache()

        
        try:
            start_time = time.time()
            # Perform a test encoding run
            with torch.no_grad():
                dummy_latents = encode_images_to_latents(images, **environment)
                del dummy_latents
            end_time = time.time()
            
            throughput = batch_size / (end_time - start_time)
            print(f"  - VAE Batch Size: {batch_size}, Throughput: {throughput:.2f} images/sec")
            
            if throughput > best_throughput:
                best_throughput = throughput
                best_batch_size = batch_size
            
            # If performance starts dropping, we've likely passed the peak
            if throughput < best_throughput * 0.95 and batch_size > best_batch_size:
                print("  - Performance degrading, stopping search.")
                break
            
            batch_size *= 2 # Exponential search
            
        except torch.cuda.OutOfMemoryError:
            print(f"  - OOM at batch size {batch_size}. Halting search.")
            gc.collect()
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"  - Error at batch size {batch_size}: {e}. Halting search.")
            break
            
    print(f"--- Optimal VAE batch size found: {best_batch_size} (Throughput: {best_throughput:.2f} images/sec) ---")
    vae.to('cpu') # Offload VAE after testing
    return best_batch_size


# --- STAGE 1: CREATE THE REPRODUCIBLE SCHEDULE ---
# (The create_training_schedule function remains exactly the same as the previous version. 
#  It correctly generates the pure Python dictionary plan, which is our ground truth.)
def create_training_schedule(config: Dict[str, Any]) -> Dict[str, Any]:
    # ... (code from previous response, unchanged)
    print("--- Stage 1: Creating Reproducible Training Schedule ---")
    
    seed = config['train']['seed']
    rng = random.Random(seed)
    print(f"Using random seed: {seed}")

    data_pool = defaultdict(dict)
    subfolder_names = [f.strip() for f in config['dataset']['folders'].split(',')]
    scales_raw = config['dataset']['scales']
    scale_values_str_list = []

    if isinstance(scales_raw, str):
        # Handles the case where scales is a single string: '-2, -1, 1, 2'
        scale_values_str_list = scales_raw.split(',')
    elif isinstance(scales_raw, list):
        # Handles the case where scales is already a list: [-2, -1, 1, 2]
        scale_values_str_list = scales_raw
    else:
        # If it's something else, we should fail loudly.
        raise TypeError(f"Config 'dataset.scales' must be a comma-separated string or a list, but got {type(scales_raw)}")

    # Now, convert the sanitized list of strings to floats, stripping whitespace from each element.
    # The str(s) is a safeguard in case the list is already numbers.
    scale_values = [float(str(s).strip()) for s in scale_values_str_list]
    
    for folder_name, scale in zip(subfolder_names, scale_values):
        subfolder_path = Path(config['dataset']['folder_main']) / folder_name
        for image_path in subfolder_path.glob("*"):
            if image_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
                data_pool[image_path.name][scale] = str(image_path)
    
    all_scales = sorted(list(set(scale_values)))
    valid_filenames = [f for f, scales_map in data_pool.items() if all(s in scales_map for s in all_scales)]

    with open(config['dataset']['prompts_file'], 'r') as f:
        all_prompts = yaml.safe_load(f)
    prompt_map = {f"prompt_{i}": prompt for i, prompt in enumerate(all_prompts)}

    schedule_dict = {
        "config_snapshot": config,
        "prompt_map": prompt_map,
        "schedule": []
    }

    total_steps = config['train']['iterations']
    batch_size = config['train']['batch_size']
    if batch_size % 2 != 0:
        raise ValueError("Batch size must be an even number for paired training.")

    print(f"Generating schedule for {total_steps} iterations with batch size {batch_size}...")
    for _ in tqdm(range(total_steps), desc="Building schedule"):
        batch_items = []
        for pair_j in range(batch_size // 2):
            selected_filename = rng.choice(valid_filenames)
            selected_prompt_key = rng.choice(list(prompt_map.keys()))
            low_scale, high_scale = sorted(rng.sample(all_scales, 2))

            shared_noise_seed = rng.randint(0, 2**32 - 1)

            item_high = {
                "image_path": data_pool[selected_filename][high_scale],
                "scale": high_scale,
                "prompt_key": selected_prompt_key,
                "pair_index": pair_j,
                "is_low_case": False,
                "noise_seed": shared_noise_seed
            }
            item_low = {
                "image_path": data_pool[selected_filename][low_scale],
                "scale": low_scale,
                "prompt_key": selected_prompt_key,
                "pair_index": pair_j,
                "is_low_case": True,
                "noise_seed": shared_noise_seed
            }
            batch_items.extend([item_high, item_low])
        
        # NO SHUFFLE HERE. The order is determined by this loop.
        schedule_dict["schedule"].append(batch_items)

    print("--- Schedule creation complete. ---")
    return schedule_dict


# MANDATORY HELPER FUNCTION FROM OLD DATA_PROCESSING_UTILS.PY.
def encode_images_to_latents(images, **environment):
    vae = environment["vae"]
    #device=environment["device"]
    #weight_dtype=environment["weight_dtype"]
    weight_dtype = torch.bfloat16
    
    from diffusers.image_processor import VaeImageProcessor
    start_time = time.time()
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True)
    image_tensors = [image_processor.preprocess(image) for image in images]
    image_batch = torch.cat(image_tensors, dim=0).to(device=vae.device, dtype=vae.dtype)
    latents = vae.encode(image_batch).latent_dist.sample(None)
    del image_batch
    #...huh?
    latents = vae.config.scaling_factor * latents
    end_time = time.time()
    return latents.to(device=torch.device("cpu")), (end_time - start_time)


def _get_unique_items_from_schedule(schedule_dict: Dict[str, Any]):
    all_items = [item for batch in schedule_dict['schedule'] for item in batch]
    unique_image_paths = sorted(list(set(item['image_path'] for item in all_items)))
    unique_prompt_keys = sorted(list(set(item['prompt_key'] for item in all_items)))
    return unique_image_paths, unique_prompt_keys

def materialize_static_batches(schedule_dict: Dict[str, Any], environment: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
    print("--- Stage 2: Materializing Schedule into Tensor Batches ---")
    config = environment['config']
    
    unique_image_paths, unique_prompt_keys = _get_unique_items_from_schedule(schedule_dict)

    # --- LATENT CACHING & ENCODING ---
    dataset_name = Path(config['dataset']['folder_main']).name
    latent_cache_file = Path(config['dataset']['folder_main']) / f"latents_{dataset_name}.safetensors"
    latent_index_file = Path(config['dataset']['folder_main']) / f"latents_{dataset_name}.json"
    
    # 1. CHECKING: Load existing latent cache if it exists
    image_latents_cache = {}
    if latent_cache_file.exists() and latent_index_file.exists():
        print(f"Loading existing latent cache from {latent_cache_file}")
        latent_tensors = load_file(latent_cache_file, device='cpu')
        with open(latent_index_file, 'r') as f:
            latent_index = json.load(f)
        
        # Populate in-memory cache
        for path, data in latent_index.items():
            image_latents_cache[path] = {
                "latent": latent_tensors[data['tensor_key']],
                "original_size": tuple(data['original_size'])
            }

    # 2. ENCODING: Find which images are missing and encode them in batches
    missing_image_paths = [p for p in unique_image_paths if p not in image_latents_cache]
    if missing_image_paths:
        print(f"Found {len(missing_image_paths)} images missing from latent cache. Encoding now...")
        
        # Find optimal batch size only if we need to encode
        vae_batch_size = config['train'].get('vae_batch_size')
        if not vae_batch_size:
            vae_batch_size = find_optimal_vae_batch_size(environment, missing_image_paths[0])
            config['train']['vae_batch_size'] = vae_batch_size # Save it back to config for this run
        
        newly_encoded_latents = {}
        for i in tqdm(range(0, len(missing_image_paths), vae_batch_size), desc="Encoding image batches"):
            batch_paths = missing_image_paths[i:i+vae_batch_size]
            images = []
            original_sizes = []
            for path in batch_paths:
                img = Image.open(path).convert("RGB")
                original_sizes.append(img.size)
                images.append(img.resize(tuple(config['train']['resolution']), Image.Resampling.LANCZOS))
            
            # This function returns latents on CPU
            latents_batch, _ = encode_images_to_latents(images, **environment)
            
            # Map back to cache
            for j, path in enumerate(batch_paths):
                tensor_key = f"latent_{hashlib.sha1(path.encode()).hexdigest()}"
                cloned_latent = latents_batch[j].clone().contiguous()
                newly_encoded_latents[tensor_key] = cloned_latent
                image_latents_cache[path] = {
                    "latent": latents_batch[j],
                    "original_size": original_sizes[j]
                }
            del latents_batch, _
        
        # 3. PERSISTENCE: Save the newly encoded latents
        print(f"Saving {len(newly_encoded_latents)} new latents to {latent_cache_file}...")
        save_file(newly_encoded_latents, latent_cache_file)
        # Update the JSON index
        new_index = {
            path: {
                "tensor_key": f"latent_{hashlib.sha1(path.encode()).hexdigest()}",
                "original_size": data['original_size']
            } for path, data in image_latents_cache.items()
        }
        with open(latent_index_file, 'w') as f:
            json.dump(new_index, f, indent=2)

    # --- TEXT EMBEDDING CACHING ---
    # Batched encoding for text, assuming batch size is not a bottleneck here
    text_embeddings_cache = {}
    print(f"Encoding {len(unique_prompt_keys)} unique text prompts...")
    prompt_map = schedule_dict['prompt_map']
    for key in tqdm(unique_prompt_keys, desc="Encoding prompts"):
        prompt_dict = prompt_map[key]
        # create_batched_prompt_embeddings is already batched internally for its 3 prompts
        text_encoder_kwargs = {
        "tokenizers": environment['tokenizers'],
        "text_encoders": environment['text_encoders'],
        "prompts": prompt_dict,
        }
        embeds = create_batched_prompt_embeddings(**text_encoder_kwargs)
        #returns (text_emb, text_pool)
        text_embeddings_cache[key] = tuple(t.cpu() for t in embeds)

    # --- MATERIALIZE BATCHES (Identical to previous implementation) ---
    print("Assembling final tensor batches...")
    static_batches = []
    noise_generator = torch.Generator()

    for batch_items in tqdm(schedule_dict['schedule'], desc="Materializing batches"):
        # ... (This entire loop is the same as the one from the previous response)
        batch_tensors = defaultdict(list)
        prompt_key = batch_items[0]['prompt_key']
        guidance_scale = schedule_dict['prompt_map'][prompt_key].get('guidance_scale', 1.0)

        for item in batch_items:
            latent_data = image_latents_cache[item['image_path']]
            embed_data = text_embeddings_cache[item['prompt_key']]
            #returns (text_emb, text_pool)
            
            batch_tensors['latents'].append(latent_data['latent'])
            batch_tensors['scales'].append(item['scale'])
            batch_tensors['pair_indices'].append(item['pair_index'])
            batch_tensors['is_low_cases'].append(item['is_low_case'])

            noise_generator.manual_seed(item['noise_seed'])
            noise = torch.randn(latent_data['latent'].shape, generator=noise_generator)
            batch_tensors['noise'].append(noise)

            timesteps_to = torch.randint(
                1,
                config['train']['max_denoising_steps'],
                (1,), # Match the batch size
                device='cpu' # Generate on CPU, it will be moved to GPU later
            ).long(),
            batch_tensors['timesteps_to'].append(timesteps_to)
            
            #timestep = torch.randint(1, config['train']['max_denoising_steps'], (1,)).long()
            #batch_tensors['timesteps_to'].append(timestep)

            cond_text = embed_data[0][2] if item['is_low_case'] else embed_data[0][0]
            cond_pooled = embed_data[1][2] if item['is_low_case'] else embed_data[1][0]
            uncond_text_embeds, uncond_pooled = embed_data[0][1], embed_data[1][1]
            
            batch_tensors['cfg_text_embeddings'].append(torch.stack([uncond_text_embeds, cond_text]))
            batch_tensors['cfg_pooled_embeds'].append(torch.stack([uncond_pooled, cond_pooled]))

            #get_add_time_ids(
            #    original_image_size[0], original_image_size[1], False, dtype=torch.float32
            #)

            add_time_ids = get_add_time_ids(
                latent_data['original_size'][0], 
                latent_data['original_size'][1],
                False,
                dtype=torch.float32
            )
            batch_tensors['add_time_ids'].append(add_time_ids)

        # --- INSERT THIS DIAGNOSTIC BLOCK ---
        if not static_batches:
            print(f"verbose logging? {config['logging']['verbose']}")
        if not static_batches and config["logging"]["verbose"]: # Only print for the very first batch
            print("\n--- DIAGNOSTICS: SHAPES OF TENSORS IN LISTS BEFORE FINAL COMBINATION ---")
            for key, tensor_list in batch_tensors.items():
                if tensor_list and isinstance(tensor_list[0], torch.Tensor):
                    shapes = [str(t.shape) for t in tensor_list]
                    print(f"  - batch_tensors['{key}'] contains {len(shapes)} tensors with shapes: {shapes}")
            print("------------------------------------------------------------------------\n")
        # --- END OF DIAGNOSTIC BLOCK ---

        final_batch = {
            "latents": torch.stack(batch_tensors['latents']),
            "scales": torch.tensor(batch_tensors['scales']),
            "pair_indices": torch.tensor(batch_tensors['pair_indices'], dtype=torch.long),
            "is_low_cases": torch.tensor(batch_tensors['is_low_cases'], dtype=torch.bool),
            "noise": torch.stack(batch_tensors['noise']),
            "timesteps_to": torch.randint(
                1,
                config['train']['max_denoising_steps'],
                (len(batch_tensors['latents']),), # Match the batch size
                device='cpu' # Generate on CPU, it will be moved to GPU later
            ).long(),
            "cfg_text_embeddings": torch.cat(batch_tensors['cfg_text_embeddings']),
            "cfg_pooled_embeds": torch.cat(batch_tensors['cfg_pooled_embeds']),
            "add_time_ids": torch.cat(batch_tensors['add_time_ids']),
            "guidance_scale": guidance_scale
        }

        # --- THE ONE DIAGNOSTIC PRINT BLOCK ---
        # We only print this for the very first batch to avoid spamming the log.
        if not static_batches and config["logging"]["verbose"]:
            print("\n--- DIAGNOSTICS: Shape of Tensors AFTER FINAL COMBINATION  ---")
            for key, tensor in final_batch.items():
                if isinstance(tensor, torch.Tensor):
                    print(f"  - batch['{key}'].shape: {tensor.shape}")
                else:
                    print(f"  - batch['{key}']: {tensor}")
            print("------------------------------------------------------------------\n")
        # --- END OF DIAGNOSTIC PRINT BLOCK ---

        static_batches.append(final_batch)

    print("--- Materialization complete. ---")
    return static_batches



###
###
###
#inside batch_data_pipeline.py 
import torch
from .batch_dataset_utils import ShamImageDataset
from tqdm import tqdm

def materialize_sham_dataset(config: dict, environment: dict) -> list[dict]:
    """
    Implements the unified data pipeline interface for the ShamImageDataset.
    It produces a list of static batches, just like the real data pipeline.
    """
    print("--- [Data Pipeline] Materializing ShamImageDataset into static batches... ---")
    dataset = ShamImageDataset(num_samples=config['num_samples'], size=config['size'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'])
    
    static_batches = []
    # A dummy timestep tensor, since it's generated live during training anyway
    # but might be needed by the evaluation workload.
    dummy_timesteps = torch.tensor([500], dtype=torch.long) 

    for clean_images in tqdm(dataloader, desc="Materializing Sham Batches"):
        # The batch now mimics the structure of an SDXL workload dict.
        # This is the key to making the data flow model-agnostic.
        batch_dict = {
            'clean_images': clean_images, # For the training loop
            'initial_latents': clean_images, # For the evaluation orchestrator
            'timesteps': dummy_timesteps.expand(clean_images.shape[0]), # For eval
            # Add a placeholder for conditioning, which TesterUViT will ignore.
            'conditioning': {}, 
        }
        static_batches.append(batch_dict)
        
    print(f"--- Materialization complete. Created {len(static_batches)} static batches. ---")
    return static_batches

#in batch_data_pipeline.py
import torch
import random
import os
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def _get_composed_prompt(item_path: Path, prompt_config: dict, root_prompts: dict, metadata: dict, rng: random.Random) -> dict:
    """
    Selects a final prompt configuration for an item based on a hierarchy and sampling importance.
    """
    item_filename = item_path.name
    folder_name = item_path.parent.name

    # 1. Gather all potential prompt sources for this item
    sources = []
    weights = []

    # Root prompt (always available as a fallback)
    sources.append(root_prompts['default_prompt'])
    weights.append(prompt_config['root']['importance'])

    # Folder-level prompt
    if folder_name in metadata.get('folders', {}):
        folder_prompt_key = metadata['folders'][folder_name]['prompt_key']
        sources.append(root_prompts[folder_prompt_key])
        weights.append(prompt_config['metadata']['folder_importance'])
    
    # Item-level prompt
    if item_filename in metadata.get('items', {}):
        item_prompt_key = metadata['items'][item_filename]['prompt_key']
        sources.append(root_prompts[item_prompt_key])
        weights.append(prompt_config['metadata']['item_importance'])

    # 2. Perform weighted random sampling to choose one prompt source
    chosen_prompt_recipe = rng.choices(sources, weights=weights, k=1)[0]

    # Return the full recipe and the final guidance
    return chosen_prompt_recipe

def get_recipe_key(recipe):
        return hashlib.sha1(json.dumps(recipe, sort_keys=True).encode()).hexdigest()

def materialize_sdxl_latent_dataset(config: dict, environment: dict) -> list[dict]:
    """
    Creates a static list of tensor batches for SDXL slider training.
    This pipeline is configurable, reproducible, and cache-aware.
    """
    print("--- [SDXL Data Pipeline] Beginning materialization... ---")
    
    # --- STAGE 0: LOAD ALL CONFIGURATION AND METADATA ---
    rng = random.Random(config['seed'])
    with open(config['prompt_sources']['root']['file'], 'r') as f:
        root_prompts = yaml.safe_load(f)
    with open(config['prompt_sources']['metadata']['file'], 'r') as f:
        metadata = json.load(f)

    # Get the weight_dtype from the environment.
    weight_dtype = environment.get("weight_dtype", torch.bfloat16)

    # ==========================================================================
    # STAGE 1: DISCOVER & FILTER DATA POOL (UNSTUBBED)
    # ==========================================================================
    print("--- Stage 1: Discovering and filtering image pool... ---")
    rng = random.Random(config['seed'])
    data_pool = defaultdict(dict)
    #are some of our scales being read as integers? we can maybe fix this in our data pool filtering + discovery.
    config['scales'] = [float(scale) for scale in config['scales']]
    
    for folder_name, scale in zip(config['folders'], config['scales']):
        subfolder_path = Path(config['folder_main']) / folder_name
        if not subfolder_path.exists():
            print(f"Warning: Folder not found, skipping: {subfolder_path}")
            continue
        for image_path in subfolder_path.glob("*"):
            if image_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
                data_pool[image_path.name][scale] = str(image_path)
    
    all_scales = config['scales']
    if config['pairing_strategy'] == 'strict':
        valid_filenames = [f for f, s_map in data_pool.items() if len(s_map) == len(all_scales)]
    elif config['pairing_strategy'] == 'relaxed':
        valid_filenames = [f for f, s_map in data_pool.items() if len(s_map) >= config['min_scales_required']]
    else:
        raise ValueError(f"Unknown pairing_strategy: {config['pairing_strategy']}")

    with open(config['prompts_file'], 'r') as f:
        prompt_map = yaml.safe_load(f)

    print(f"Discovered {len(data_pool)} unique filenames. Filtered to {len(valid_filenames)} valid filenames for training.")
    
    # ==========================================================================
    # STAGE 2: CREATE THE DETERMINISTIC TRAINING SCHEDULE
    # ==========================================================================
    print("--- Stage 2: Building deterministic training schedule... ---")
    training_schedule = [] # A list of simple instruction dicts, not tensors
    
    for step_index in tqdm(range(config['iterations']), desc="Scheduling Training Units"):
        batch_of_units = []
        for i in range(config['batch_size'] // 2):
            selected_filename = rng.choice(valid_filenames)
            available_scales = sorted(data_pool[selected_filename].keys())
            low_scale, high_scale = sorted(rng.sample(available_scales, 2))

            high_path = Path(data_pool[selected_filename][high_scale])
            low_path = Path(data_pool[selected_filename][low_scale])
            
            # Use the new composition helper for each item
            high_prompt_recipe = _get_composed_prompt(high_path, config['prompt_sources'], root_prompts, metadata, rng)
            low_prompt_recipe = _get_composed_prompt(low_path, config['prompt_sources'], root_prompts, metadata, rng)
            
            item_instruction_high = {"image_path": str(high_path), "prompt_recipe": high_prompt_recipe, "recipe_key": get_recipe_key(high_prompt_recipe), "scale": high_scale, "role": "high_scale_target"}
            item_instruction_low = {"image_path": str(low_path), "prompt_recipe": low_prompt_recipe, "recipe_key": get_recipe_key(low_prompt_recipe), "scale": low_scale, "role": "low_scale_target"}
            
            training_unit = {"unit_id": f"unit_{step_index}_{i}_{selected_filename}", "type": "slider_pair", "items": [item_instruction_high, item_instruction_low]}
            batch_of_units.append(training_unit)
        training_schedule.append(batch_of_units)
        
    # ==========================================================================
    # STAGE 3: MATERIALIZE & CACHE ASSETS (UNSTUBBED)
    # ==========================================================================
    print("--- Stage 3: Materializing and caching heavy assets... ---")
    
    # Extract all unique assets required by the full schedule
    all_items = [item for batch in training_schedule for unit in batch for item in unit['items']]
    unique_image_paths = sorted(list(set(item['image_path'] for item in all_items)))
    # We no longer need to recalculate keys. We just grab the ones we already made.
    unique_recipes = {item['recipe_key']: item['prompt_recipe'] for item in all_items}

    # --- Latent Caching ---
    latent_cache_dir = Path(config['latent_cache_dir'])
    latent_cache_dir.mkdir(parents=True, exist_ok=True)
    latent_cache_file = latent_cache_dir / "latents.safetensors"
    latent_index_file = latent_cache_dir / "latents_index.json"
    
    # 1. Load existing cache
    latent_cache = {}
    if latent_cache_file.exists() and latent_index_file.exists():
        print("Loading existing latent cache...")
        index = json.loads(latent_index_file.read_text())
        tensors = load_file(latent_cache_file, device='cpu')
        # Ensure that loading from cache creates the same dictionary structure
        # as the new-image encoding path.
        for path, data in index.items():
            # Instead of just storing the tensor, wrap it in our standard dict format.
            latent_cache[path] = {
                'latent': tensors[data['tensor_key']], 
                'original_size': tuple(data['original_size'])
            }

    # 2. Identify and encode missing latents
    missing_paths = [p for p in unique_image_paths if p not in latent_cache]
    if missing_paths:
        print(f"Encoding {len(missing_paths)} new images into latents...")
        vae = environment['vae'].to(environment['device'], dtype=torch.bfloat16)
        
        # NOTE: A real implementation could use the ThroughputBatchFinder here.
        # For simplicity, we use a fixed VAE batch size from the manifest.
        vae_batch_size = config.get('vae_batch_size', 4)
        newly_encoded_tensors, newly_created_index = {}, {}
        
        for i in tqdm(range(0, len(missing_paths), vae_batch_size), desc="Encoding Latents"):
            batch_paths = missing_paths[i:i + vae_batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            original_sizes = [img.size for img in images]

            latents_tensor, _latents_batch = encode_images_to_latents(images, vae=vae)#, config['resolution'])
            
            for j, path in enumerate(batch_paths):
                tensor_key = f"latent_{hashlib.sha1(path.encode()).hexdigest()}"
                latent_cache[path] = {'latent': latents_tensor[j], 'original_size': original_sizes[j]}
                newly_encoded_tensors[tensor_key] = latents_tensor[j]
                newly_created_index[path] = {'tensor_key': tensor_key, 'original_size': original_sizes[j]}

        # 3. Save new additions to cache
        if newly_encoded_tensors:
            print(f"Saving {len(newly_encoded_tensors)} new latents to cache...")
            # 1. Save the tensor data (this part was already correct).
            save_file(newly_encoded_tensors, latent_cache_file, metadata={'format': 'pt'})
            
            # 2. Load the existing index, if it exists, or start with an empty dict.
            existing_index = json.loads(latent_index_file.read_text()) if latent_index_file.exists() else {}
            
            # 3. Merge the existing index with the newly created index entries.
            #    The `newly_created_index` already has the correct nested structure.
            full_index = {**existing_index, **newly_created_index}
            
            # 4. Write the complete, correctly formatted index back to disk.
            latent_index_file.write_text(json.dumps(full_index, indent=2))
        
        # Cleanup VRAM
        vae.to('cpu'); gc.collect(); torch.cuda.empty_cache()

    # --- Text Embedding Caching (In-Memory for this run) ---
    print(f"Encoding {len(unique_recipes)} unique text prompt recipes...")
    text_embed_cache = {} # -> {recipe_key: (embeds, pooled_embeds)}
    text_encoders = [te.to(environment['device']) for te in environment['text_encoders']]
    
    for recipe_key, recipe in tqdm(unique_recipes.items(), desc="Encoding Prompts"):
        # This function handles the positive, negative, and neutral prompts internally
        embeds, pooled_embeds = neo_create_batched_prompt_embeddings(
            recipe, environment['tokenizers'], text_encoders
        )
        # Store on CPU to conserve VRAM
        text_embed_cache[recipe_key] = (embeds.cpu(), pooled_embeds.cpu())

    # Cleanup VRAM
    for te in text_encoders: te.to('cpu')
    gc.collect(); torch.cuda.empty_cache()

    # ==========================================================================
    # STAGE 4: ASSEMBLE FINAL TENSOR BATCHES (UNSTUBBED)
    # ==========================================================================
    print("--- Stage 4: Assembling final tensor batches... ---")
    static_batches = []
    noise_generator = torch.Generator()

    # We need the scheduler from the environment to perform the projection
    noise_scheduler = environment['scheduler'] 
    max_steps = config['max_denoising_steps']

    for batch_of_units in tqdm(training_schedule, desc="Assembling Batches"):
        batch_tensor_lists = defaultdict(list)
        
        all_items_in_batch = [item for unit in batch_of_units for item in unit['items']]

        # Assuming guidance_scale is per-prompt, take the first one
        # The prompt_recipe is attached to each item, but guidance_scale is constant per prompt_recipe
        # So we get it once per *batch*, assuming all items in a batch might share a common prompt recipe,
        # or it is defined per recipe. Let's simplify and take from the first item's recipe for now.
        # It's safer to get the guidance scale per item from its recipe.
        
        for item in all_items_in_batch:
            latent_data = latent_cache[item['image_path']]
            recipe_key = item['recipe_key']
            text_data = text_embed_cache[recipe_key] # (embeds, pooled_embeds)
            
            # --- Collate Core Tensors for this item ---
            batch_tensor_lists['clean_latents'].append(latent_data['latent'])
            batch_tensor_lists['scales'].append(item['scale'])
            batch_tensor_lists['is_low_cases'].append(item['role'] == 'low_scale_target') # Store bool

            # Reproducible noise for training based on a unit ID
            unit_id = next(u['unit_id'] for u in batch_of_units if item in u['items'])
            noise_generator.manual_seed(hash(unit_id))
            latent_shape = latent_data['latent'].shape
            noise = torch.randn(latent_shape, generator=noise_generator, dtype=weight_dtype)
            batch_tensor_lists['noise'].append(noise)
            
            # Additive time embeddings
            add_time_ids = get_add_time_ids(
                latent_data['original_size'][0], 
                latent_data['original_size'][1], 
                False, dtype=torch.float32 # Ensure dtype consistency for this operation
            )
            batch_tensor_lists['add_time_ids'].append(add_time_ids)
            
            # Individual Text/Pooled Embeddings (NOT YET CFG STACKED)
            # text_data is (embeds, pooled_embeds)
            # embeds is (pos_embed, uncond_embed, neutral_embed)
            # pooled_embeds is (pos_pooled, uncond_pooled, neutral_pooled)
            
            batch_tensor_lists['text_embeddings_cond'].append(text_data[0][0])
            batch_tensor_lists['text_embeddings_uncond'].append(text_data[0][1])
            batch_tensor_lists['text_embeddings_neutral'].append(text_data[0][2])

            batch_tensor_lists['pooled_embeds_cond'].append(text_data[1][0])
            batch_tensor_lists['pooled_embeds_uncond'].append(text_data[1][1])
            batch_tensor_lists['pooled_embeds_neutral'].append(text_data[1][2])

            # Store the individual guidance scale for this prompt recipe
            batch_tensor_lists['guidance_scale'].append(item['prompt_recipe'].get('guidance_scale', config.get('guidance_scale', 1.0)))

            # Original image sizes for evaluation's VAE decoder
            batch_tensor_lists['original_sizes'].append(torch.tensor(latent_data['original_size']))


            # Timestep calculations for training. These are specific to *training*.
            ts_generator = torch.Generator().manual_seed(hash(unit_id))
            
            timesteps_to = torch.randint(1, max_steps, (1,), generator=ts_generator).long()
            
            noise_scheduler.set_timesteps(max_steps, device='cpu')
            noise_level_timestep = noise_scheduler.timesteps[timesteps_to]

            noise_scheduler.set_timesteps(1000, device='cpu')
            normalized_tsteps = torch.round(timesteps_to.float() * 1000 / max_steps).long()
            unet_input_timestep = noise_scheduler.timesteps[normalized_tsteps]

            batch_tensor_lists['noise_level_timesteps'].append(noise_level_timestep)
            batch_tensor_lists['unet_input_timesteps'].append(unet_input_timestep)


        # --- Assemble the final, universal batch dictionary ---
        if not batch_tensor_lists: continue 
        
        final_batch = {
            "clean_latents": torch.stack(batch_tensor_lists['clean_latents']),
            "noise": torch.stack(batch_tensor_lists['noise']),
            "noise_level_timesteps": torch.cat(batch_tensor_lists['noise_level_timesteps']),
            "unet_input_timesteps": torch.cat(batch_tensor_lists['unet_input_timesteps']),

            "text_embeddings_cond": torch.stack(batch_tensor_lists['text_embeddings_cond']),
            "text_embeddings_uncond": torch.stack(batch_tensor_lists['text_embeddings_uncond']),
            "text_embeddings_neutral": torch.stack(batch_tensor_lists['text_embeddings_neutral']),

            "pooled_embeds_cond": torch.stack(batch_tensor_lists['pooled_embeds_cond']),
            "pooled_embeds_uncond": torch.stack(batch_tensor_lists['pooled_embeds_uncond']),
            "pooled_embeds_neutral": torch.stack(batch_tensor_lists['pooled_embeds_neutral']),
            
            "add_time_ids": torch.stack(batch_tensor_lists['add_time_ids']), # Stack them now
            "guidance_scale": torch.tensor(batch_tensor_lists['guidance_scale']),
            "scales": torch.tensor(batch_tensor_lists['scales']),
            "is_low_cases": torch.tensor(batch_tensor_lists['is_low_cases'], dtype=torch.bool),
            "original_sizes": torch.stack(batch_tensor_lists['original_sizes']),
        }
        
        static_batches.append(final_batch)

    print("--- SDXL Data Pipeline Materialization Complete. ---")
    return static_batches

def prepare_training_batch(raw_batch: dict, scheduler: Any, device: torch.device, weight_dtype: torch.dtype) -> dict:
    """
    Transforms a universal raw batch into the kwargfood format for training.
    """
    # Move to scheduler's device for add_noise, then potentially back for kwargfood
    scheduler_device = scheduler.device

    # 1. Add noise to clean latents
    noisy_latents = scheduler.add_noise(
        raw_batch['clean_latents'].to(scheduler_device),
        raw_batch['noise'].to(scheduler_device),
        raw_batch['noise_level_timesteps'].to(scheduler_device)
    ).to(device) # Move result to the target training device (GPU)

    # 2. Select appropriate text/pooled embeddings based on is_low_cases
    num_items = raw_batch['scales'].shape[0]
    
    # Initialize lists to gather CFG-ready embeddings
    cfg_text_embeddings_list = []
    cfg_pooled_embeds_list = []

    for i in range(num_items):
        uncond_text = raw_batch['text_embeddings_uncond'][i]
        uncond_pooled = raw_batch['pooled_embeds_uncond'][i]
        
        if raw_batch['is_low_cases'][i]:
            cond_text = raw_batch['text_embeddings_neutral'][i]
            cond_pooled = raw_batch['pooled_embeds_neutral'][i]
        else:
            cond_text = raw_batch['text_embeddings_cond'][i]
            cond_pooled = raw_batch['pooled_embeds_cond'][i]
        
        cfg_text_embeddings_list.append(torch.stack([uncond_text, cond_text]))
        cfg_pooled_embeds_list.append(torch.stack([uncond_pooled, cond_pooled]))

    # 3. Stack for CFG and prepare kwargfood
    kwargfood = {
        "sample": torch.cat([noisy_latents, noisy_latents], dim=0),
        "timestep": torch.cat([raw_batch['unet_input_timesteps'].to(device), raw_batch['unet_input_timesteps'].to(device)], dim=0),
        "encoder_hidden_states": torch.cat(cfg_text_embeddings_list).to(device),
        "added_cond_kwargs": {
            "text_embeds": torch.cat(cfg_pooled_embeds_list).to(device),
            "time_ids": torch.cat([raw_batch['add_time_ids'].to(device), raw_batch['add_time_ids'].to(device)], dim=0)
        }
    }

    return {
        "kwargfood": kwargfood,
        "target_noise": raw_batch['noise'].to(device),
        "scales": raw_batch['scales'].to(device),
        "guidance_scale": raw_batch['guidance_scale'].to(device)
    }