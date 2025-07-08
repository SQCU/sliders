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
    item_type: str = "item" # For logging/context
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

    if existing_data is not None and invalidation_check_fn(item_identifier, existing_data, environment, config):
        cache_map[key] = existing_data
        return
    else:
        if existing_data is not None:
            print(f"[{item_type}] Cached data for {key} is invalid or not found. Re-encoding.")

    encoded_data = encoder_fn(item_identifier, environment, config)
    cache_map[key] = encoded_data

    if save_to_persistent_storage_fn:
        save_to_persistent_storage_fn(item_identifier, encoded_data, environment, config) # Pass item_identifier to save_fn

# --- Helper functions for Images ---

def _image_encoder_fn(image_path: str, environment: Dict[str, Any], config: Any) -> torch.Tensor:
    vae = environment['vae']
    device = environment['device']
    weight_dtype = environment['weight_dtype']
    max_resolution = tuple(config.train.get("resolution", (512, 512)))

    image = Image.open(image_path).convert("RGB")
    image = resize_image_if_needed(image, max_resolution)
    
    # encode_images_to_latents expects a list of images
    new_latents_gpu, _ = encode_images_to_latents([image], vae, device, weight_dtype)
    return new_latents_gpu.cpu().squeeze(0) # Return single latent on CPU

def _image_cache_key_fn(image_path: str) -> str:
    return image_path

def _image_invalidation_check_fn(image_path: str, existing_latent: torch.Tensor, environment: Dict[str, Any], config: Any) -> bool:
    # Re-encode the image to compare with existing_latent
    new_latent = _image_encoder_fn(image_path, environment, config)
    are_close = torch.allclose(existing_latent, new_latent, atol=1e-4, rtol=1e-3)
    if not are_close:
        diff = torch.mean(torch.abs(existing_latent - new_latent))
        print(f"Latent mismatch for {image_path}. Mean absolute difference: {diff.item()}.")
    return are_close

def _image_save_fn(image_path: str, latent: torch.Tensor, environment: Dict[str, Any], config: Any) -> None:
    output_dir = Path(config.dataset_config.dataset.folder_main) / "latents" # Assuming this path structure
    os.makedirs(output_dir, exist_ok=True)
    latent_filename = os.path.splitext(os.path.basename(image_path))[0] + ".pt"
    latent_path = os.path.join(output_dir, latent_filename)
    torch.save(latent, latent_path)

def _image_load_fn(image_path: str, environment: Dict[str, Any], config: Any) -> Union[torch.Tensor, None]:
    output_dir = Path(config.dataset_config.dataset.folder_main) / "latents"
    latent_filename = os.path.splitext(os.path.basename(image_path))[0] + ".pt"
    latent_path = os.path.join(output_dir, latent_filename)
    if os.path.exists(latent_path):
        try:
            return torch.load(latent_path, map_location='cpu')
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
            item_type="image"
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
