# fuse_vae_lora.py
import argparse
import torch
from safetensors.torch import load_file, save_file
from collections import defaultdict
from tqdm import tqdm
import torch.nn.functional as F

def extract_vae_state_dict(model_path: str, device: str = "cpu") -> dict:
    """
    Intelligently loads a .safetensors file and extracts the VAE state dictionary.
    
    Handles two cases:
    1. The file is a standalone VAE.
    2. The file is a full Stable Diffusion checkpoint with a prefixed VAE.
    
    Returns a clean state dictionary with VAE keys (e.g., 'encoder.conv_in...').
    """
    full_state_dict = load_file(model_path, device=device)
    
    # --- Determine if this is a full checkpoint by checking for known prefixes ---
    vae_prefix = None
    if any(key.startswith("first_stage_model.") for key in full_state_dict.keys()):
        vae_prefix = "first_stage_model."
    elif any(key.startswith("vae.") for key in full_state_dict.keys()):
        vae_prefix = "vae."

    if vae_prefix:
        print(f"Detected full checkpoint with VAE prefix: '{vae_prefix}'")
        vae_state_dict = {}
        for key, tensor in full_state_dict.items():
            if key.startswith(vae_prefix):
                # Strip the prefix to get the clean VAE key
                clean_key = key[len(vae_prefix):]
                vae_state_dict[clean_key] = tensor
        
        if not vae_state_dict:
            raise ValueError("VAE prefix found, but no VAE keys were extracted. Check the file content.")
            
        print(f"✅ Successfully extracted {len(vae_state_dict)} VAE tensors.")
        return vae_state_dict
    else:
        print("✅ Detected standalone VAE file. Loading directly.")
        # Check if it looks like a VAE
        if not any(key.startswith(("encoder.", "decoder.", "quant_conv.")) for key in full_state_dict.keys()):
             raise ValueError("File does not appear to be a standalone VAE or a full checkpoint. No VAE keys found.")
        return full_state_dict


def fuse_vae_lora(
    base_model_path: str,
    vae_adapter_path: str,
    output_path: str,
    device: str = "cpu"
):
    """
    Loads a base VAE (from standalone or full checkpoint), fuses a trained 
    LoRA/GLoRA adapter, and saves a new, standalone VAE model file.
    """
    print(f"Loading base VAE weights from: {base_model_path}")
    base_vae_state_dict = extract_vae_state_dict(base_model_path, device=device)
    
    print(f"Loading VAE adapter from: {vae_adapter_path}")
    adapter_state_dict = load_file(vae_adapter_path, device=device)

    # --- 1. Group Adapter Tensors by Target Module ---
    grouped_tensors = defaultdict(dict)
    known_param_suffixes = [
        'lora_down.weight', 'lora_up.weight', 'glora_a1.weight', 'glora_a2.weight', 
        'glora_b1.weight', 'glora_b2.weight', 'alpha'
    ]
    
    for key, tensor in adapter_state_dict.items():
        found_suffix = None
        for suffix in known_param_suffixes:
            if key.endswith(suffix):
                found_suffix = suffix
                break
        
        if found_suffix:
            # Reconstruct the original module path for the VAE state_dict
            # e.g., 'lora_vae_decoder_down_blocks_0_resnets_0_conv1' -> 'decoder.down_blocks.0.resnets.0.conv1'
            adapter_prefix = "lora_vae_"
            module_key_base = key[:-(len(found_suffix) + 1)] # strip suffix and dot
            if module_key_base.startswith(adapter_prefix):
                clean_key_base = module_key_base[len(adapter_prefix):].replace('_', '.')
            else:
                print(f"  > WARNING: Unrecognized adapter key format, skipping: {key}")
                continue

            param_name = found_suffix.replace('glora_', '')
            grouped_tensors[clean_key_base][param_name] = tensor

    print(f"Found {len(grouped_tensors)} LoRA/GLoRA modules to fuse.")

    # --- 2. Create a copy of the base VAE state to modify ---
    fused_vae_state_dict = base_vae_state_dict.copy()

    # --- 3. Iterate Through Modules and Fuse Weights ---
    for module_key, params in tqdm(grouped_tensors.items(), desc="Fusing VAE Layers"):
        base_weight_key = f"{module_key}.weight"
        
        if base_weight_key not in base_vae_state_dict:
            print(f"  > WARNING: Base weight for module '{module_key}' not found in VAE. Skipping.")
            continue
            
        W_base = base_vae_state_dict[base_weight_key]
        original_dtype = W_base.dtype
        W_base = W_base.to(torch.float32)
        
        delta_W = None
        is_glora = 'a1.weight' in params
        
        with torch.no_grad():
            if is_glora:
                rank = params['b2.weight'].shape[0]
                alpha = params.get('alpha', torch.tensor(rank)).item()
                a1, a2 = params['a1.weight'].to(torch.float32), params['a2.weight'].to(torch.float32)
                b1, b2 = params['b1.weight'].to(torch.float32), params['b2.weight'].to(torch.float32)
                
                delta_W_A = (W_base @ a1) @ a2
                delta_W_B = b1 @ b2
                delta_W = (delta_W_A + delta_W_B) * (alpha / rank)

            else: # Standard LoRA
                rank = params['lora_down.weight'].shape[0]
                alpha = params.get('alpha', torch.tensor(rank)).item()
                lora_down, lora_up = params['lora_down.weight'].to(torch.float32), params['lora_up.weight'].to(torch.float32)
                
                if W_base.dim() == 4: # Conv2D
                    delta_W = F.conv2d(lora_down.permute(1, 0, 2, 3), lora_up).permute(1, 0, 2, 3)
                else: # Linear
                    delta_W = lora_up @ lora_down
                
                delta_W *= (alpha / rank)

        if delta_W is not None:
            fused_vae_state_dict[base_weight_key] = (W_base + delta_W).to(original_dtype)

    # --- 4. Save the new, fused state dictionary ---
    print(f"Saving fused VAE to: {output_path}")
    save_file(fused_vae_state_dict, output_path)
    print("✅ Fusion complete. The new file is a standalone, drop-in replacement VAE.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fuse a trained VAE adapter into a base VAE, from either a standalone VAE or a full SD checkpoint."
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="Path to the base model file in .safetensors format. Can be a full SDXL checkpoint or a standalone VAE.",
    )
    parser.add_argument(
        "--vae_adapter_path",
        type=str,
        required=True,
        help="Path to the trained VAE adapter in .safetensors format.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the new, fused, standalone VAE .safetensors file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for the fusion process.",
    )
    
    args = parser.parse_args()

    fuse_vae_lora(
        base_model_path=args.base_model_path,
        vae_adapter_path=args.vae_adapter_path,
        output_path=args.output_path,
        device=args.device
    )

    # Example Usage 1: Fusing into a standalone VAE
    # python fuse_vae_lora.py \
    #   --base_model_path "path/to/sd_xl_vae.safetensors" \
    #   --vae_adapter_path "models/rehab_run/rehab_run_last.safetensors" \
    #   --output_path "models/fused_vaes/sd_xl_vae_rehab_fused.safetensors"

    # Example Usage 2: Fusing into a VAE from a full SDXL checkpoint
    # python fuse_vae_lora.py \
    #   --base_model_path "path/to/full_sdxl_model.safetensors" \
    #   --vae_adapter_path "models/rehab_run/rehab_run_last.safetensors" \
    #   --output_path "models/fused_vaes/sd_xl_vae_rehab_fused.safetensors"```