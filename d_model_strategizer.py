from typing import Callable, Dict, Any, Tuple, List, Optional
import torch
from diffusers import UNet2DConditionModel, AutoencoderKL
from pathlib import Path
import json
import os

# --- Model Signature Definitions ---
# These functions define the expected input signatures for different model architectures.
# They act as a "warning-not-exception type checker" by providing a specification.

def _get_sdxl_unet_signature() -> Dict[str, Any]:
    """
    Returns the expected forward() signature for an SDXL UNet.
    This includes positional arguments, keyword arguments, and their expected tensor shapes/types.
    Specifies minimum precision (bfloat16) for relevant tensors.
    """
    # Based on diffusers.UNet2DConditionModel.forward signature for SDXL
    return {
        "positional_args": [
            {"name": "sample", "shape": "B, 4, H/8, W/8", "dtype": torch.bfloat16},
            {"name": "timestep", "shape": "B", "dtype": torch.int64},
            {"name": "encoder_hidden_states", "shape": "B, seq_len, 1280", "dtype": torch.bfloat16}, # SDXL text embedding dim
        ],
        "keyword_args": {
            "added_cond_kwargs": {
                "text_embeds": {"shape": "B, 1280", "dtype": torch.bfloat16}, # Pooled text embeddings
                "time_ids": {"shape": "B, 6", "dtype": torch.float32}, # Original image size, crop, and aesthetic score (often float32)
            },
            "cross_attention_kwargs": None, # Placeholder for LoRA/PEFT
            "class_labels": None, # Optional
            "attention_mask": None, # Optional
            "down_block_additional_residuals": None, # Optional
            "mid_block_additional_residual": None, # Optional
            "return_dict": True, # Default
        },
        "output_shape": "B, 4, H/8, W/8",
        "output_dtype": torch.bfloat16,
    }

def _get_sd1_unet_signature() -> Dict[str, Any]:
    """
    Returns the expected forward() signature for an SD1.x UNet.
    Note the differences in text embedding shapes and added conditioning.
    Specifies minimum precision (bfloat16) for relevant tensors.
    """
    # Based on diffusers.UNet2DConditionModel.forward signature for SD1.x
    return {
        "positional_args": [
            {"name": "sample", "shape": "B, 4, H/8, W/8", "dtype": torch.bfloat16},
            {"name": "timestep", "shape": "B", "dtype": torch.int64},
            {"name": "encoder_hidden_states", "shape": "B, seq_len, 768", "dtype": torch.bfloat16}, # SD1.x text embedding dim
        ],
        "keyword_args": {
            "cross_attention_kwargs": None, # Placeholder for LoRA/PEFT
            "class_labels": None, # Optional
            "attention_mask": None, # Optional
            "return_dict": True, # Default
        },
        "output_shape": "B, 4, H/8, W/8",
        "output_dtype": torch.bfloat16,
    }

def _get_sd1_vae_decoder_signature() -> Dict[str, Any]:
    """
    Returns the expected forward() signature for an SD1.x VAE decoder.
    This is relevant for training the VAE decoder stage.
    Specifies minimum precision (bfloat16) for relevant tensors.
    """
    # Based on diffusers.AutoencoderKL.decode signature
    return {
        "positional_args": [
            {"name": "latents", "shape": "B, 4, H/8, W/8", "dtype": torch.bfloat16},
        ],
        "keyword_args": {
            "return_dict": True, # Default
        },
        "output_shape": "B, 3, H, W", # RGB image output
        "output_dtype": torch.bfloat16,
    } # VAEs can sometimes be sensitive to bfloat16. If numerical issues arise, consider using float32 for VAE inputs/outputs.

def _get_sdxl_vae_encoder_signature() -> Dict[str, Any]:
    """
    Returns the expected forward() signature for an SDXL VAE encoder (AutoencoderKL.encode).
    """
    # Based on diffusers.AutoencoderKL.encode signature
    return {
        "positional_args": [
            {"name": "x", "shape": "B, 3, H, W", "dtype": torch.float32}, # VAE typically expects float32 image input
        ],
        "keyword_args": {
            "return_dict": True, # Default
        },
        "output_shape": "B, 4, H/8, W/8", # Latent distribution
        "output_dtype": torch.float32, # Latent distribution is typically float32
    }

# --- Net-Type-Mapper (Stubbed) ---
# This function maps our TrainingUnit data to the model's expected input format.

def _map_data_unit_to_model_inputs(
    data_unit: Dict[str, Any],
    model_signature: Dict[str, Any],
    model_architecture: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Given a data_unit and a model's expected signature, maps the relevant data
    from the data_unit into a format directly consumable by the model's forward() call.
    This function is responsible for:
    1. Materializing data (e.g., loading latents, embeddings from cache/disk).
    2. Ensuring correct tensor shapes.
    3. Ensuring correct tensor dtypes (precision).
    4. Placing tensors on the correct device.
    Returns (args, kwargs) tuple.
    """
    # Initialize positional and keyword arguments
    args = []
    kwargs = {}

    # Iterate through expected positional arguments from the model signature
    for arg_spec in model_signature.get("positional_args", []):
        arg_name = arg_spec["name"]
        expected_dtype = arg_spec["dtype"]
        # expected_shape = arg_spec["shape"] # Not directly used for validation here, but for reference

        # Retrieve tensor from data_unit, cast to expected dtype, and move to device
        if arg_name in data_unit:
            tensor = data_unit[arg_name].to(device=device, dtype=expected_dtype)
            args.append(tensor)
        else:
            raise ValueError(f"Positional argument '{arg_name}' not found in data_unit.")

    # Iterate through expected keyword arguments from the model signature
    for kwarg_name, kwarg_spec in model_signature.get("keyword_args", {}).items():
        if kwarg_spec is None:
            kwargs[kwarg_name] = None
            continue

        if kwarg_name == "added_cond_kwargs":
            added_cond_kwargs = {}
            for sub_kwarg_name, sub_kwarg_spec in kwarg_spec.items():
                if sub_kwarg_spec is None:
                    added_cond_kwargs[sub_kwarg_name] = None
                    continue
                expected_dtype = sub_kwarg_spec["dtype"]
                # expected_shape = sub_kwarg_spec["shape"]

                if sub_kwarg_name in data_unit.get("added_cond_kwargs", {}):
                    tensor = data_unit["added_cond_kwargs"][sub_kwarg_name].to(device=device, dtype=expected_dtype)
                    added_cond_kwargs[sub_kwarg_name] = tensor
                else:
                    raise ValueError(f"Added conditional argument '{sub_kwarg_name}' not found in data_unit['added_cond_kwargs'].")
            kwargs[kwarg_name] = added_cond_kwargs
        elif kwarg_name == "cross_attention_kwargs":
            # This would be for LoRA/PEFT, likely passed through directly if available in data_unit
            kwargs[kwarg_name] = data_unit.get("cross_attention_kwargs", None) # Retrieve from data_unit or default to None
        # Add other top-level keyword arguments as needed
        else:
            kwargs[kwarg_name] = data_unit.get(kwarg_name, None) # Retrieve from data_unit or default to None

    # The output should be directly unzippable into model.forward(*args, **kwargs)
    return args, kwargs

# --- Main Strategizer Function ---

def d_model_strategizer(model_architecture: str, device: torch.device, dtype: torch.dtype) -> Callable[[Dict[str, Any]], Tuple[List[Any], Dict[str, Any]]]:
    """
    Returns a callable function that, given a TrainingUnit, will produce the
    (args, kwargs) tuple required for the specified model_architecture's forward() call.

    Args:
        model_architecture (str): The target model architecture (e.g., "SDXL_UNET", "SD1_UNET", "SD1_VAE_DECODER", "SDXL_VAE_ENCODER").

    Returns:
        Callable[[Dict[str, Any]], Tuple[List[Any], Dict[str, Any]]]: A function that takes
        a TrainingUnit dictionary and returns a tuple of (positional_args, keyword_args)
        ready for a model's forward() call.
    """
    # 1. Get the model's expected signature
    model_signature = {}
    if model_architecture == "SDXL_UNET":
        model_signature = _get_sdxl_unet_signature()
    elif model_architecture == "SD1_UNET":
        model_signature = _get_sd1_unet_signature()
    elif model_architecture == "SD1_VAE_DECODER":
        model_signature = _get_sd1_vae_decoder_signature()
    elif model_architecture == "SDXL_VAE_ENCODER":
        model_signature = _get_sdxl_vae_encoder_signature()
    else:
        raise ValueError(f"Unsupported model architecture: {model_architecture}")

    # 2. Return a closure that performs the mapping for each training unit
    def strategize_inputs(data_unit: Dict[str, Any]) -> Tuple[List[Any], Dict[str, Any]]:
        return _map_data_unit_to_model_inputs(data_unit, model_signature, model_architecture, device, dtype)

    return strategize_inputs

# --- Example Usage (for self-testing) ---
def main():    
    print("--- Testing d_model_strategizer ---")    
    batch_size = 1 # Moved to global scope for all tests    
    #Determine device and dtype    
    device = "cuda" if torch.cuda.is_available() else "cpu"    
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32    
    print(f"Using device: {device}, dtype: {dtype})")    

    # Load model configurations to get dimension parameters
    # Using relative paths now
    project_root = Path(__file__).parent # Corrected path to be the 'sliders' directory
    
    sdxl_unet_config_path = project_root / "canon_configs" / "stabilityai__stable_diffusion_xl_base_1.0" / "unet" / "config.json"
    with open(sdxl_unet_config_path, 'r') as f:
        sdxl_unet_config = json.load(f)

    sd1_unet_config_path = project_root / "canon_configs" / "runwayml__stable_diffusion_v1_5" / "unet" / "config.json"
    sd1_unet_config = {}
    with open(sd1_unet_config_path, 'r') as f:            
        sd1_unet_config = json.load(f)        

    sd1_vae_config_path = project_root / "canon_configs" / "madebyollin__sdxl_vae_fp16_fix" / "config.json"
    sd1_vae_config = {}
    with open(sd1_vae_config_path, 'r') as f:
        sd1_vae_config = json.load(f)

    # Define dimension parameters from loaded configs
    sample_size = sdxl_unet_config["sample_size"] # Assuming sample_size is consistent across UNets for dummy data
    sdxl_cross_attention_dim = sdxl_unet_config["cross_attention_dim"] # 1280
    sd1_cross_attention_dim = sd1_unet_config["cross_attention_dim"] # 768
    
    # Dummy materialized data units, simulating output of an AssetMaterializer
    # These directly provide the tensors needed by _map_data_unit_to_model_inputs
    
    # For SDXL
    materialized_sdxl_unit = {
        "high_latents": torch.randn(batch_size, 4, sample_size, sample_size, dtype=dtype),
        "high_positive_embeddings": torch.randn(batch_size, 77, sdxl_cross_attention_dim, dtype=dtype),
        "high_pooled_embeddings": torch.randn(batch_size, 1280, dtype=dtype),
        "original_size": (sample_size * 8, sample_size * 8) # Example original image size
    }

    # For SD1.x
    materialized_sd1_unit = {
        "high_latents": torch.randn(batch_size, 4, sample_size, sample_size, dtype=dtype),
        "high_positive_embeddings": torch.randn(batch_size, 77, sd1_cross_attention_dim, dtype=dtype),
        "original_size": (sample_size * 8, sample_size * 8) # Example original image size
    }

    # Example Data Units (simulating output of an Asset Materializer)
    # These data units contain pre-materialized tensors with correct shapes and dtypes.

    # For SDXL UNet
    sdxl_sample_data_unit = {
        "sample": materialized_sdxl_unit["high_latents"],
        "timestep": torch.randint(0, 1000, (batch_size,), dtype=torch.int64),
        "encoder_hidden_states": materialized_sdxl_unit["high_positive_embeddings"],
        "added_cond_kwargs": {
            "text_embeds": materialized_sdxl_unit["high_pooled_embeddings"],
            "time_ids": torch.randn(batch_size, 6, dtype=torch.float32), # time_ids remain float32
        },
        # "cross_attention_kwargs": None, # Optional, can be added if needed
    }

    # For SD1.x UNet
    sd1_unet_sample_data_unit = {
        "sample": materialized_sd1_unit["high_latents"],
        "timestep": torch.randint(0, 1000, (batch_size,), dtype=torch.int64),
        "encoder_hidden_states": materialized_sd1_unit["high_positive_embeddings"],
        # "cross_attention_kwargs": None, # Optional, can be added if needed
    }

    # For SD1.x VAE Decoder
    sd1_vae_decoder_sample_data_unit = {
        "latents": materialized_sd1_unit["high_latents"],
    }

    # --- Stubbed Model Creation and Testing ---
    # For SDXL UNet
    print("Testing SDXL UNet:")
    try:
        # Create a dummy SDXL UNet using the loaded config
        sdxl_unet = UNet2DConditionModel.from_config(sdxl_unet_config)
        sdxl_unet.to(device=device, dtype=dtype)
        sdxl_unet.enable_gradient_checkpointing() # Enable gradient checkpointing
        print("  SDXL UNet created successfully from canonical config and moved to device with correct dtype.")

        # Get the strategizer for SDXL UNet
        sdxl_unet_strategizer = d_model_strategizer("SDXL_UNET", device, dtype)

        # Use the strategizer to get args and kwargs from the sample data unit
        args, kwargs = sdxl_unet_strategizer(sdxl_sample_data_unit)

        # Test forward pass with no_grad
        with torch.no_grad():
            _ = sdxl_unet(*args, **kwargs)
            print("  SDXL UNet forward pass executed successfully with correct tensor shapes and no_grad.")
    except Exception as e:
        import traceback
        print(f"  Error creating/testing SDXL UNet: {e}")
        print(traceback.format_exc())

    # For SD1.x UNet
    print("Testing SD1.x UNet:")
    try:
        # Create a dummy SD1.x UNet using the loaded config
        sd1_unet = UNet2DConditionModel.from_config(sd1_unet_config)
        sd1_unet.to(device=device, dtype=dtype)
        sd1_unet.enable_gradient_checkpointing()
        # Enable gradient checkpointing
        print("  SD1.x UNet created successfully from canonical config and moved to device with correct dtype.")

        # Get the strategizer for SD1.x UNet
        sd1_unet_strategizer = d_model_strategizer("SD1_UNET", device, dtype)

        # Use the strategizer to get args and kwargs from the sample data unit
        args, kwargs = sd1_unet_strategizer(sd1_unet_sample_data_unit)

        # Test forward pass with no_grad
        with torch.no_grad():
            _ = sd1_unet(*args, **kwargs)
            print("  SD1.x UNet forward pass executed successfully with correct tensor shapes and no_grad.")
    except Exception as e:
        import traceback
        print(f"  Error creating/testing SD1.x UNet: {e}")
        print(traceback.format_exc())

    # For SD1.x VAE Decoder
    print("Testing SD1.x VAE Decoder:")
    try:
        # Create a dummy SD1.x VAE Decoder using the loaded config
        sd1_vae_decoder = AutoencoderKL.from_config(sd1_vae_config)
        sd1_vae_decoder.to(device=device, dtype=dtype)
        print("  SD1.x VAE Decoder created successfully from canonical config and moved to device with correct dtype.")

        # Get the strategizer for SD1.x VAE Decoder
        sd1_vae_decoder_strategizer = d_model_strategizer("SD1_VAE_DECODER", device, dtype)

        # Use the strategizer to get args and kwargs from the sample data unit
        args, kwargs = sd1_vae_decoder_strategizer(sd1_vae_decoder_sample_data_unit)

        # Test decode pass with no_grad
        with torch.no_grad():
            _ = sd1_vae_decoder.decode(*args, **kwargs)
            print("  SD1.x VAE Decoder decode pass executed successfully with correct tensor shapes and no_grad.")
    except Exception as e:
        import traceback
        print(f"  Error creating/testing SD1.x VAE Decoder: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()