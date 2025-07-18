from typing import Callable, Dict, Any, Tuple, List, Optional
import torch
from diffusers import UNet2DConditionModel, AutoencoderKL
from pathlib import Path
import json

# --- Model Signature Definitions ---
# These functions define the expected input signatures for different model architectures.
# They act as a "warning-not-exception type checker" by providing a specification.

def _get_sdxl_unet_signature() -> Dict[str, Any]:
    """
    Returns the expected forward() signature for an SDXL UNet.
    This includes positional arguments, keyword arguments, and their expected tensor shapes/types.
    """
    # Based on diffusers.UNet2DConditionModel.forward signature for SDXL
    return {
        "positional_args": [
            {"name": "sample", "shape": "B, 4, H/8, W/8", "dtype": torch.float32},
            {"name": "timestep", "shape": "B", "dtype": torch.int64},
            {"name": "encoder_hidden_states", "shape": "B, seq_len, 1280", "dtype": torch.float32}, # SDXL text embedding dim
        ],
        "keyword_args": {
            "added_cond_kwargs": {
                "text_embeds": {"shape": "B, 1280", "dtype": torch.float32}, # Pooled text embeddings
                "time_ids": {"shape": "B, 6", "dtype": torch.float32}, # Original image size, crop, and aesthetic score
            },
            "cross_attention_kwargs": None, # Placeholder for LoRA/PEFT
            "class_labels": None, # Optional
            "attention_mask": None, # Optional
            "down_block_additional_residuals": None, # Optional
            "mid_block_additional_residual": None, # Optional
            "return_dict": True, # Default
        },
        "output_shape": "B, 4, H/8, W/8",
        "output_dtype": torch.float32,
    }

def _get_sd1_unet_signature() -> Dict[str, Any]:
    """
    Returns the expected forward() signature for an SD1.x UNet.
    Note the differences in text embedding shapes and added conditioning.
    """
    # Based on diffusers.UNet2DConditionModel.forward signature for SD1.x
    return {
        "positional_args": [
            {"name": "sample", "shape": "B, 4, H/8, W/8", "dtype": torch.float32},
            {"name": "timestep", "shape": "B", "dtype": torch.int64},
            {"name": "encoder_hidden_states", "shape": "B, seq_len, 768", "dtype": torch.float32}, # SD1.x text embedding dim
        ],
        "keyword_args": {
            "cross_attention_kwargs": None, # Placeholder for LoRA/PEFT
            "class_labels": None, # Optional
            "attention_mask": None, # Optional
            "return_dict": True, # Default
        },
        "output_shape": "B, 4, H/8, W/8",
        "output_dtype": torch.float32,
    }

def _get_sd1_vae_decoder_signature() -> Dict[str, Any]:
    """
    Returns the expected forward() signature for an SD1.x VAE decoder.
    This is relevant for training the VAE decoder stage.
    """
    # Based on diffusers.AutoencoderKL.decode signature
    return {
        "positional_args": [
            {"name": "latents", "shape": "B, 4, H/8, W/8", "dtype": torch.float32},
        ],
        "keyword_args": {
            "return_dict": True, # Default
        },
        "output_shape": "B, 3, H, W", # RGB image output
        "output_dtype": torch.float32,
    }

# --- Net-Type-Mapper (Stubbed) ---
# This function maps our TrainingUnit data to the model's expected input format.

def _map_training_unit_to_model_inputs(
    training_unit: Dict[str, Any],
    model_signature: Dict[str, Any],
    model_architecture: str,
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Given a TrainingUnit and a model's expected signature, maps the relevant data
    from the TrainingUnit into a format directly consumable by the model's forward() call.
    Returns (args, kwargs) tuple.
    """
    print(f"Stub: _map_training_unit_to_model_inputs called for {model_architecture}.")
    print(f"  Training Unit Sample: {list(training_unit.keys())}")
    print(f"  Model Signature Sample: {list(model_signature.keys())}")

    # This is where the complex mapping logic would go.
    # For now, we'll return placeholders based on the architecture.
    args = []
    kwargs = {}

    if model_architecture == "SDXL_UNET":
        # Example: Mapping from training_unit to SDXL UNet inputs
        # This would involve looking up latents, timesteps, and text embeddings
        # based on the training_unit's high_path/low_path, prompt_recipe, etc.
        # and then shaping them according to the model_signature.
        pass
    elif model_architecture == "SD1_UNET":
        pass
    elif model_architecture == "SD1_VAE_DECODER":
        pass
    else:
        raise ValueError(f"Unknown model architecture: {model_architecture}")

    # The output should be directly unzippable into model.forward(*args, **kwargs)
    return args, kwargs

# --- Main Strategizer Function ---

def d_model_strategizer(model_architecture: str) -> Callable[[Dict[str, Any]], Tuple[List[Any], Dict[str, Any]]]:
    """
    Returns a callable function that, given a TrainingUnit, will produce the
    (args, kwargs) tuple required for the specified model_architecture's forward() call.

    Args:
        model_architecture (str): The target model architecture (e.g., "SDXL_UNET", "SD1_UNET", "SD1_VAE_DECODER").

    Returns:
        Callable[[Dict[str, Any]], Tuple[List[Any], Dict[str, Any]]]: A function that takes
        a TrainingUnit dictionary and returns a tuple of (positional_args, keyword_args)
        ready for a model's forward() call.
    """
    print(f"d_model_strategizer called for architecture: {model_architecture}")

    # 1. Get the model's expected signature
    model_signature = {}
    if model_architecture == "SDXL_UNET":
        model_signature = _get_sdxl_unet_signature()
    elif model_architecture == "SD1_UNET":
        model_signature = _get_sd1_unet_signature()
    elif model_architecture == "SD1_VAE_DECODER":
        model_signature = _get_sd1_vae_decoder_signature()
    else:
        raise ValueError(f"Unsupported model architecture: {model_architecture}")

    # 2. Return a closure that performs the mapping for each training unit
    def strategize_inputs(training_unit: Dict[str, Any]) -> Tuple[List[Any], Dict[str, Any]]:
        return _map_training_unit_to_model_inputs(training_unit, model_signature, model_architecture)

    return strategize_inputs

# --- Example Usage (for self-testing) ---
if __name__ == "__main__":
    print("--- Testing d_model_strategizer ---")

    batch_size = 1 # Moved to global scope for all tests

    # Example Training Unit (simplified for stub testing)
    sample_training_unit = {
        "unit_id": "test_unit_0",
        "high_path": "/path/to/high_image.png",
        "low_path": "/path/to/low_image.png",
        "high_scale": 1.0,
        "low_scale": 0.5,
        "high_prompt_recipe": {"positive": "a dog", "unconditional": "", "neutral": ""},
        "low_prompt_recipe": {"positive": "a cat", "unconditional": "", "neutral": ""},
        "high_recipe_key": "hash_dog",
        "low_recipe_key": "hash_cat",
        "seed": 12345,
    }

    # --- Stubbed Model Creation and Testing ---
    # For SDXL UNet
    print("\nTesting SDXL UNet:")
    try:
        # Load SDXL UNet config from canonical file
        sdxl_unet_config_path = Path("F:/dox/ai/gemmy/sliders/canon_configs/stabilityai__stable_diffusion_xl_base_1.0/unet/config.json")
        with open(sdxl_unet_config_path, 'r') as f:
            sdxl_unet_config = json.load(f)

        # Create a dummy SDXL UNet using the loaded config
        sdxl_unet = UNet2DConditionModel.from_config(sdxl_unet_config)
        print("  SDXL UNet created successfully from canonical config.")

        # Generate random tensors based on signature and loaded config
        sample_size = sdxl_unet_config["sample_size"]
        cross_attention_dim = sdxl_unet_config["cross_attention_dim"]
        projection_class_embeddings_input_dim = sdxl_unet_config["projection_class_embeddings_input_dim"]
        addition_time_embed_dim = sdxl_unet_config["addition_time_embed_dim"]

        sample_tensor = torch.randn(batch_size, 4, sample_size, sample_size, dtype=torch.float32)
        timestep_tensor = torch.randint(0, 1000, (batch_size,), dtype=torch.int64)
        encoder_hidden_states_tensor = torch.randn(batch_size, 77, cross_attention_dim, dtype=torch.float32)
        text_embeds_tensor = torch.randn(batch_size, projection_class_embeddings_input_dim - addition_time_embed_dim * 6, dtype=torch.float32) # Adjusted based on config
        time_ids_tensor = torch.randn(batch_size, 6, dtype=torch.float32)

        # Test forward pass
        _ = sdxl_unet(sample_tensor, timestep_tensor, encoder_hidden_states_tensor,
                      added_cond_kwargs={"text_embeds": text_embeds_tensor, "time_ids": time_ids_tensor})
        print("  SDXL UNet forward pass executed successfully with correct tensor shapes.")

    except Exception as e:
        import traceback
        print(f"  Error creating/testing SDXL UNet: {e}")
        print(traceback.format_exc())

    # For SD1.x UNet
    print("\nTesting SD1.x UNet:")
    try:
        # Load SD1.x UNet config from canonical file
        sd1_unet_config_path = Path("F:/dox/ai/gemmy/sliders/canon_configs/runwayml__stable_diffusion_v1_5/unet/config.json")
        with open(sd1_unet_config_path, 'r') as f:
            sd1_unet_config = json.load(f)

        # Create a dummy SD1.x UNet using the loaded config
        sd1_unet = UNet2DConditionModel.from_config(sd1_unet_config)
        print("  SD1.x UNet created successfully from canonical config.")

        # Generate random tensors based on signature and loaded config
        sample_size = sd1_unet_config["sample_size"]
        cross_attention_dim = sd1_unet_config["cross_attention_dim"]

        sample_tensor = torch.randn(batch_size, 4, sample_size, sample_size, dtype=torch.float32)
        timestep_tensor = torch.randint(0, 1000, (batch_size,), dtype=torch.int64)
        encoder_hidden_states_tensor = torch.randn(batch_size, 77, cross_attention_dim, dtype=torch.float32)

        # Test forward pass
        _ = sd1_unet(sample_tensor, timestep_tensor, encoder_hidden_states_tensor)
        print("  SD1.x UNet forward pass executed successfully with correct tensor shapes.")

    except Exception as e:
        import traceback
        print(f"  Error creating/testing SD1.x UNet: {e}")
        print(traceback.format_exc())

    # For SD1.x VAE Decoder
    print("\nTesting SD1.x VAE Decoder:")
    try:
        # Load SD1.x VAE config from canonical file
        sd1_vae_config_path = Path("F:/dox/ai/gemmy/sliders/canon_configs/madebyollin__sdxl_vae_fp16_fix/config.json")
        with open(sd1_vae_config_path, 'r') as f:
            sd1_vae_config = json.load(f)

        # Create a dummy SD1.x VAE Decoder using the loaded config
        sd1_vae_decoder = AutoencoderKL.from_config(sd1_vae_config)
        print("  SD1.x VAE Decoder created successfully from canonical config.")

        # Generate random tensors based on signature and loaded config
        sample_size = sd1_vae_config["sample_size"]

        latent_sample_tensor = torch.randn(batch_size, 4, sample_size, sample_size, dtype=torch.float32)

        # Test decode pass
        _ = sd1_vae_decoder.decode(latent_sample_tensor)
        print("  SD1.x VAE Decoder decode pass executed successfully with correct tensor shapes.")

    except Exception as e:
        import traceback
        print(f"  Error creating/testing SD1.x VAE Decoder: {e}")
        print(traceback.format_exc())

