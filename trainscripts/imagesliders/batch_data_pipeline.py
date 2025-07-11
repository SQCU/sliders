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

from .batch_train_util import create_batched_prompt_embeddings, get_add_time_ids
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


# --- STAGE 2: MATERIALIZE THE SCHEDULE INTO TENSORS ---

# MANDATORY HELPER FUNCTION FROM OLD DATA_PROCESSING_UTILS.PY.
def encode_images_to_latents(images, **environment):
    vae = environment["vae"]
    #device=environment["device"]
    weight_dtype=environment["weight_dtype"]
    
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
            #"timesteps_to": torch.cat(batch_tensors['timesteps_to']),
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