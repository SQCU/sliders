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


# --- Net-Type-Mapper (Stubbed) ---
# This function maps our TrainingUnit data to the model's expected input format.

def _map_data_unit_to_model_inputs(
    data_unit: Dict[str, Any],
    model_signature: Dict[str, Any],
    model_architecture: str,
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
    # Placeholder for device and dtype, these would typically be passed in or determined globally
    # For now, we'll assume they are available from the calling context (e.g., main function)
    # In a real scenario, this function might receive a 'device' and 'dtype' argument.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    # Initialize positional and keyword arguments
    args = []
    kwargs = {}

    # Iterate through expected positional arguments from the model signature
    for arg_spec in model_signature.get("positional_args", []):
        arg_name = arg_spec["name"]
        expected_dtype = arg_spec["dtype"]
        # expected_shape = arg_spec["shape"] # Not directly used for materialization here, but for validation

        # --- Materialization and Type/Device Conversion Logic ---
        # This is where the data_unit would be processed to get the actual tensor.
        # For now, we'll use dummy tensors for illustration.
        # In a real implementation, you'd fetch data_unit[arg_name] or derive it.
        if arg_name == "sample":
            # Example: Latent sample for UNet
            # This would come from a materialized latent cache based on data_unit
            tensor = torch.randn(1, 4, 64, 64, device=device, dtype=expected_dtype) # Dummy shape
        elif arg_name == "timestep":
            # Example: Timestep for UNet
            tensor = torch.randint(0, 1000, (1,), device=device, dtype=expected_dtype)
        elif arg_name == "encoder_hidden_states":
            # Example: Text embeddings for UNet
            tensor = torch.randn(1, 77, 768, device=device, dtype=expected_dtype) # Dummy shape
        elif arg_name == "latents":
            # Example: Latents for VAE decoder
            tensor = torch.randn(1, 4, 64, 64, device=device, dtype=expected_dtype) # Dummy shape
        else:
            raise NotImplementedError(f"Materialization for positional argument '{arg_name}' not implemented.")

        args.append(tensor)

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

                if sub_kwarg_name == "text_embeds":
                    tensor = torch.randn(1, 1280, device=device, dtype=expected_dtype) # Dummy shape
                elif sub_kwarg_name == "time_ids":
                    tensor = torch.randn(1, 6, device=device, dtype=expected_dtype) # Dummy shape
                else:
                    raise NotImplementedError(f"Materialization for added_cond_kwargs sub-argument '{sub_kwarg_name}' not implemented.")
                added_cond_kwargs[sub_kwarg_name] = tensor
            kwargs[kwarg_name] = added_cond_kwargs
        elif kwarg_name == "cross_attention_kwargs":
            # This would be for LoRA/PEFT, likely passed through directly if available in data_unit
            kwargs[kwarg_name] = None # Placeholder
        # Add other top-level keyword arguments as needed
        else:
            kwargs[kwarg_name] = None # Default to None for other optional kwargs

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
    def strategize_inputs(data_unit: Dict[str, Any]) -> Tuple[List[Any], Dict[str, Any]]:
        return _map_data_unit_to_model_inputs(data_unit, model_signature, model_architecture)

    return strategize_inputs

# --- Example Usage (for self-testing) ---
def main():    
    print("--- Testing d_model_strategizer ---")    
    batch_size = 1 # Moved to global scope for all tests    
    #Determine device and dtype    
    device = "cuda" if torch.cuda.is_available() else "cpu"    
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32    
    print(f"Using device: {device}, dtype: {dtype}")    # Example Training Unit (simplified for stub testing)
    sample_training_unit = {     
        "unit_id": "test_unit_0",        
        "high_path": "/path/to/high_image.png",        
        "low_path": "/path/to/low_image.png",        
        "high_scale": 1.0,        
        "low_scale": 0.5,        
        "high_prompt_recipe": 
        {"positive": "a dog", "unconditional": "", "neutral": ""},        
        "low_prompt_recipe": {"positive": "a cat", "unconditional": "", "neutral": ""},        
        "high_recipe_key": "hash_dog",        "low_recipe_key": "hash_cat",        
        "seed": 12345,    
    }    
    # --- Stubbed Model Creation and Testing ---    
    # For SDXL UNet    
    print("\nTesting SDXL UNet:")    
    try:                
        sdxl_unet_config_path = Path(
            "F:/dox/ai/gemmy/sliders/canon_configs/stabilityai__stable_diffusion_xl_base_1.0/unet/config.json"
            )        
        with open(sdxl_unet_config_path, 'r') as f:
            sdxl_unet_config = json.load(f)
            # Create a dummy SDXL UNet using the loaded config
        sdxl_unet = UNet2DConditionModel.from_config(sdxl_unet_config)
        sdxl_unet.to(device=device, dtype=dtype)
        sdxl_unet.enable_gradient_checkpointing() # Enable gradient checkpointing
        print("  SDXL UNet created successfully from canonical config and moved to device with correct dtype.")
        # Generate random tensors based on signature and loaded config
        sample_size = sdxl_unet_config["sample_size"]
        cross_attention_dim = sdxl_unet_config["cross_attention_dim"]
        projection_class_embeddings_input_dim = sdxl_unet_config["projection_class_embeddings_input_dim"]
        addition_time_embed_dim = sdxl_unet_config["addition_time_embed_dim"]
        sample_tensor = torch.randn(batch_size, 4, sample_size, sample_size, dtype=dtype, device=device)
        timestep_tensor = torch.randint(0, 1000, (batch_size,), dtype=torch.int64, device=device)
        encoder_hidden_states_tensor = torch.randn(batch_size, 77, cross_attention_dim, dtype=dtype, device=device)
        text_embeds_tensor = torch.randn(batch_size, projection_class_embeddings_input_dim - addition_time_embed_dim * 6, dtype=dtype, device=device)
        time_ids_tensor = torch.randn(batch_size, 6, dtype=torch.float32, device=device) # time_ids remain float32        # Test forward pass with no_grad        
        with torch.no_grad():            
            _ = sdxl_unet(sample_tensor, timestep_tensor, encoder_hidden_states_tensor,
            added_cond_kwargs={"text_embeds": text_embeds_tensor, "time_ids": time_ids_tensor})
            print("  SDXL UNet forward pass executed successfully with correct tensor shapes and no_grad.")
    except Exception as e:
        import traceback
        print(f"  Error creating/testing SDXL UNet: {e}")
        print(traceback.format_exc())
        # For SD1.x UNet    
    print("\nTesting SD1.x UNet:")
    try:        
    # Load SD1.x UNet config from canonical file        
        sd1_unet_config_path = Path("F:/dox/ai/gemmy/sliders/canon_configs/runwayml__stable_diffusion_v1_5/unet/config.json")
        sd1_unet_config = {}
        with open(sd1_unet_config_path, 'r') as f:            
            sd1_unet_config = json.load(f)        
        # Create a dummy SD1.x UNet using the loaded config        
        sd1_unet = UNet2DConditionModel.from_config(sd1_unet_config)
        sd1_unet.to(device=device, dtype=dtype)
        sd1_unet.enable_gradient_checkpointing() 
        # Enable gradient checkpointing        
        print("  SD1.x UNet created successfully from canonical config and moved to device with correct dtype.")
        # Generate random tensors based on signature and loaded config        
        sample_size = sd1_unet_config["sample_size"]        
        cross_attention_dim = sd1_unet_config["cross_attention_dim"]        
        sample_tensor = torch.randn(batch_size, 4, sample_size, sample_size, dtype=dtype, device=device)        
        timestep_tensor = torch.randint(0, 1000, (batch_size,), dtype=torch.int64, device=device)        
        encoder_hidden_states_tensor = torch.randn(batch_size, 77, cross_attention_dim, dtype=dtype, device=device)        
        # Test forward pass with no_grad
        with torch.no_grad():
            _ = sd1_unet(sample_tensor, timestep_tensor, encoder_hidden_states_tensor)
            print("  SD1.x UNet forward pass executed successfully with correct tensor shapes and no_grad.")
    except Exception as e:
        import traceback
        print(f"  Error creating/testing SD1.x UNet: {e}")
        print(traceback.format_exc())
    # For SD1.x VAE Decoder
    print("\nTesting SD1.x VAE Decoder:")
    try:        
        # Load SD1.x VAE config from canonical file
        sd1_vae_config_path = Path("F:/dox/ai/gemmy/sliders/canon_configs/madebyollin__sdxl_vae_fp16_fix/config.json")
        sd1_vae_config = {}
        with open(sd1_vae_config_path, 'r') as f:
            sd1_vae_config = json.load(f)
    # Create a dummy SD1.x VAE Decoder using the loaded config 
        sd1_vae_decoder = AutoencoderKL.from_config(sd1_vae_config)
        sd1_vae_decoder.to(device=device, dtype=dtype)
        print("  SD1.x VAE Decoder created successfully from canonical config and moved to device with correct dtype.")
        # Generate random tensors based on signature and loaded config
        sample_size = sd1_vae_config["sample_size"]
        latent_sample_tensor = torch.randn(batch_size, 4, sample_size // 8, sample_size // 8, dtype=dtype, device=device)
        # Test decode pass with no_grad
        with torch.no_grad():
            _ = sd1_vae_decoder.decode(latent_sample_tensor)
            print("  SD1.x VAE Decoder decode pass executed successfully with correct tensor shapes and no_grad.")
    except Exception as e:
        import traceback
        print(f"  Error creating/testing SD1.x VAE Decoder: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()