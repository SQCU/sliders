#embedding_cache_util.py
import torch
import os
from pathlib import Path
import time
from typing import Any, Dict, Tuple, Union, Callable

from PIL import Image

# Assuming these are available from data_processing_utils and batch_train_util
from .data_processing_utils import resize_image_if_needed, encode_images_to_latents
from . import batch_train_util
from tqdm import tqdm

def process_and_cache_item(
    item_identifier: Any,
    encoder_fn: Callable, # Function to encode the item
    cache_map: Dict[Any, Any], # The dictionary to store cached data (in-memory)
    cache_key_fn: Callable, # Function to get a unique key for the item
    invalidation_check_fn: Callable, # Function to check if existing cached data is valid
    save_to_persistent_storage_fn: Callable = None, # Optional: Function to save to disk
    load_from_persistent_storage_fn: Callable = None, # Optional: Function to load from disk
    environment: Dict[str, Any] = None, # Pass environment if needed by encoder/invalidation/save/load
    config: Any = None, # Pass config if needed by encoder/invalidation/save/load
    item_type: str = "item", # For logging/context
    force_reencode: bool = False # New parameter: if True, bypass invalidation check
) -> None:
    """
    Processes an item, encodes it, and caches it if not already validly cached.
    Updates the cache_map in place.
    """
    key = cache_key_fn(item_identifier)
    
    # Try to load from in-memory cache first
    if key in cache_map:
        return

    existing_data = None
    if load_from_persistent_storage_fn:
        existing_data = load_from_persistent_storage_fn(item_identifier, environment, config) # Pass item_identifier to load_fn

    if existing_data is not None and not force_reencode and invalidation_check_fn(item_identifier, existing_data, environment, config):
        cache_map[key] = existing_data
        return
    else:
        if existing_data is not None and not force_reencode:
            print(f"[{item_type}] Cached data for {key} is invalid or not found. Re-encoding.")
        elif force_reencode:
            print(f"[{item_type}] Forcing re-encoding for {key} due to invalidate_latent_cache flag.")

    encoded_data = encoder_fn(item_identifier, environment, config)
    cache_map[key] = encoded_data

    if save_to_persistent_storage_fn:
        save_to_persistent_storage_fn(item_identifier, encoded_data, environment, config) # Pass item_identifier to save_fn

# --- Helper functions for Images ---

def _image_encoder_fn(image_path: str, environment: Dict[str, Any], config: Any) -> Dict[str, Any]:
    vae = environment['vae']
    device = environment['device']
    weight_dtype = environment['weight_dtype']
    target_resolution = tuple(config.train.get("resolution", (512, 512))) # This is the resolution we are currently encoding for

    image = Image.open(image_path).convert("RGB")
    original_image_size = image.size # Capture original size

    # Resize image to the target resolution for encoding
    resized_image = resize_image_if_needed(image, target_resolution)
    
    # Encode the resized image to latent
    # encode_images_to_latents expects a list of images and returns a list of latents
    new_latent_gpu = encode_images_to_latents([resized_image], vae, device, weight_dtype)[0]
    
    # Store the latent keyed by its resolution
    latents_by_resolution = {target_resolution: new_latent_gpu.cpu()}
    
    return {"latents": latents_by_resolution, "original_image_size": original_image_size}

def _image_cache_key_fn(image_path: str) -> str:
    return image_path

def _image_invalidation_check_fn(image_path: str, existing_data: Dict[str, Any], environment: Dict[str, Any], config: Any) -> bool:
    target_resolution = tuple(config.train.get("resolution", (512, 512)))
    
    if not existing_data or "latents" not in existing_data or target_resolution not in existing_data["latents"]:
        print(f"Latent for {image_path} at resolution {target_resolution} not found in cache. Re-encoding.")
        return False

    # --- THE FIX IS HERE ---
    # Call the encoder function and treat its return as a dictionary
    newly_encoded_data = _image_encoder_fn(image_path, environment, config)
    new_latents_by_resolution = newly_encoded_data["latents"]
    # Now you can correctly access the latent by its resolution key
    new_latent = new_latents_by_resolution[target_resolution]
    # --- END OF FIX ---

    existing_latent = existing_data["latents"][target_resolution]

    are_close = torch.allclose(existing_latent, new_latent, atol=1e-4, rtol=1e-3)
    if not are_close:
        diff = torch.mean(torch.abs(existing_latent - new_latent))
        print(f"Latent mismatch for {image_path} at resolution {target_resolution}. Mean absolute difference: {diff.item()}. Re-encoding.")
    return are_close

def _image_save_fn(image_path: str, encoded_data: Dict[str, Any], environment: Dict[str, Any], config: Any) -> None:
    main_folder = Path(config.dataset.folder_main)
    relative_image_path = Path(image_path).relative_to(main_folder)
    latent_path = main_folder / "latents" / relative_image_path.with_suffix(".pt")

    os.makedirs(latent_path.parent, exist_ok=True)

    latents_by_resolution = encoded_data["latents"]
    original_image_size = encoded_data["original_image_size"]

    # Load existing data if any, to merge new resolution latents
    existing_data = _image_load_fn(image_path, environment, config)
    if existing_data:
        existing_latents = existing_data["latents"]
        existing_latents.update(latents_by_resolution) # Merge new latents
        latents_to_save = existing_latents
    else:
        latents_to_save = latents_by_resolution

    data_to_save = {
        "original_image_size": original_image_size,
        "latents": latents_to_save
    }
    torch.save(data_to_save, latent_path)


def _image_load_fn(image_path: str, environment: Dict[str, Any], config: Any) -> Union[Dict[str, Any], None]:
    main_folder = Path(config.dataset.folder_main)
    relative_image_path = Path(image_path).relative_to(main_folder)
    latent_path = main_folder / "latents" / relative_image_path.with_suffix(".pt")

    if os.path.exists(latent_path):
        try:
            loaded_data = torch.load(latent_path, map_location='cpu')
            if isinstance(loaded_data, torch.Tensor):
                # Old format detected, invalidate by returning None to force re-encoding
                print(f"Warning: Old latent format detected for {image_path}. Invalidating cache to force re-encoding.")
                return None
            return loaded_data
        except Exception as e:
            print(f"Warning: Could not load existing latent for {image_path}: {e}")
            return None
    return None

# --- Helper functions for Text ---

def _text_encoder_fn(prompt_dict: Dict[str, Any], environment: Dict[str, Any], config: Any) -> Tuple[torch.Tensor, ...]:
    tokenizers = environment['tokenizers']
    text_encoders = environment['text_encoders']
    
    text_embeddings, pooled_embeds = batch_train_util.create_batched_prompt_embeddings(
        tokenizers,
        text_encoders,
        prompt_dict,
    )
    # Split and return on CPU
    positive_text_embeds, uncond_text_embeds, neutral_text_embeds = text_embeddings.chunk(3)
    positive_pooled_embeds, uncond_pooled_embeds, neutral_pooled_embeds = pooled_embeds.chunk(3)
    return (
        positive_text_embeds.cpu(),
        positive_pooled_embeds.cpu(),
        uncond_text_embeds.cpu(),
        uncond_pooled_embeds.cpu(),
        neutral_text_embeds.cpu(),
        neutral_pooled_embeds.cpu(),
    )

def _text_cache_key_fn(prompt_dict: Dict[str, Any]) -> frozenset:
    return frozenset(prompt_dict.items())

def _text_invalidation_check_fn(prompt_dict: Dict[str, Any], existing_embeddings: Tuple[torch.Tensor, ...], environment: Dict[str, Any], config: Any) -> bool:
    # For text, if it exists in the cache, it's considered valid.
    # This function is primarily for demonstrating the unified interface.
    # In a real scenario, you might add a hash check or versioning for text embeddings.
    return existing_embeddings is not None

# --- Main Unified Cache Builder ---

def build_unified_embedding_cache(config: Any, environment: Dict[str, Any], training_schedule: Any) -> Dict[str, Any]:
    """
    Builds a unified cache for image latents and text embeddings.
    Returns a dictionary containing 'image_latents' and 'text_embeddings' maps.
    """
    unified_cache = {
        "image_latents": {}, # Map from image_path to latent tensor
        "text_embeddings": {} # Map from frozenset(prompt.items()) to tuple of embeddings
    }

    # 1. Get unique items to encode
    unique_image_paths = sorted(list(set([item.image_path for batch in training_schedule for item in batch])))
    unique_prompts = training_schedule.get_unique_prompts()

    print(f"Found {len(unique_image_paths)} unique images and {len(unique_prompts)} unique prompts to process for unified cache.")

    # 2. Process unique images
    for image_path in tqdm(unique_image_paths, desc="Caching image latents"):
        process_and_cache_item(
            item_identifier=image_path,
            encoder_fn=_image_encoder_fn,
            cache_map=unified_cache["image_latents"],
            cache_key_fn=_image_cache_key_fn,
            invalidation_check_fn=_image_invalidation_check_fn,
            save_to_persistent_storage_fn=_image_save_fn,
            load_from_persistent_storage_fn=_image_load_fn,
            environment=environment,
            config=config,
            item_type="image",
            force_reencode=config.train.get("invalidate_latent_cache", False)
        )
    
    # 3. Process unique prompts
    for prompt_dict in tqdm(unique_prompts, desc="Caching text embeddings"):
        process_and_cache_item(
            item_identifier=prompt_dict,
            encoder_fn=_text_encoder_fn,
            cache_map=unified_cache["text_embeddings"],
            cache_key_fn=_text_cache_key_fn,
            invalidation_check_fn=_text_invalidation_check_fn,
            save_to_persistent_storage_fn=None, # No persistent storage for text embeddings by default
            load_from_persistent_storage_fn=None, # No persistent storage for text embeddings by default
            environment=environment,
            config=config,
            item_type="text"
        )
    
    print("Unified embedding cache built.")
    return unified_cache
