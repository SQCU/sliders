# fuse_vae_lora.py
import argparse
import torch
from safetensors.torch import load_file, save_file
from collections import defaultdict
from tqdm import tqdm
import torch.nn.functional as F
import re # <-- Import the regular expression module
from typing import List, Tuple, Dict, Any, Union
import copy

class ModuleNode:
    """
    An abstract representation of a single module.
    Can be initialized from a live module OR directly from shape information.
    """
    def __init__(self, full_key: str, name_map: Dict[str, str], known_tokens: List[str], shape: Tuple[int, int] = None):
        self.raw_key = full_key
        self.shape = shape
        # 1. Tokenize the raw key using the intelligent parser
        raw_components = intelligent_parse(self.raw_key, known_tokens)
        
        # 2. Normalize the list of components and identify their types
        self.signature = self._create_normalized_signature(raw_components, name_map)

    def _parse_key(self, key: str) -> List[Tuple[str, Union[str, int]]]:
        """
        Parses a key like 'decoder.up.0.block.1' into:
        [('name', 'decoder'), ('name', 'up'), ('ordinal', 0), ('name', 'block'), ('ordinal', 1)]
        """
        parts = key.replace('_', '.').split('.')
        components = []
        for part in parts:
            if part.isdigit():
                components.append(('ordinal', int(part)))
            elif part.startswith('-') and part[1:].isdigit(): # Handle negative indices
                components.append(('ordinal', int(part)))
            else:
                components.append(('name', part))
        return components

    def _get_shape(self, module: torch.nn.Module) -> Tuple[int, int]:
        """Extracts (in_features, out_features) from a module."""
        if hasattr(module, 'in_channels') and hasattr(module, 'out_channels'):
            return (module.in_channels, module.out_channels)
        if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
            return (module.in_features, module.out_features)
        return (0, 0)

    def _create_normalized_signature(self, components: List[str], name_map: Dict[str, str]) -> Tuple:
        normalized_signature = []
        for part in components:
            if part.isdigit() or (part.startswith('-') and part[1:].isdigit()):
                normalized_signature.append(('ordinal', int(part)))
            else:
                # Apply semantic mapping to the component
                normalized_name = name_map.get(part, part)
                normalized_signature.append(('name', normalized_name))
        return tuple(normalized_signature)

    def __repr__(self):
        return f"ModuleNode(key='{self.raw_key}', signature={self.signature}, shape={self.shape})"

def intelligent_parse(key: str, known_tokens: List[str]) -> List[str]:
    """
    A greedy, separator-aware tokenizer. It splits a key into components
    without destroying names that contain underscores (like 'norm_out').
    
    1. It prioritizes matching longer, known multi-word tokens (e.g., 'up_blocks').
    2. It treats '.' and '_' as potential separators.
    """
    components = []
    remaining_key = key

    # Known tokens should be sorted by length, descending, for greedy matching
    sorted_tokens = sorted(known_tokens, key=len, reverse=True)

    while remaining_key:
        found_match = False

        # Rule 1: Try to match a known multi-word token first
        for token in sorted_tokens:
            if remaining_key.startswith(token):
                components.append(token)
                remaining_key = remaining_key[len(token):]
                if remaining_key and (remaining_key.startswith('_') or remaining_key.startswith('.')):
                    remaining_key = remaining_key[1:]
                found_match = True
                break
        
        if found_match:
            continue

        # Rule 2: If no known token, find the next segment up to a separator
        # --- THIS IS THE CORRECTED REGEX ---
        match = re.match(r"([^._]+)", remaining_key)
        if match:
            segment = match.group(1)
            components.append(segment)
            remaining_key = remaining_key[len(segment):]
            if remaining_key and (remaining_key.startswith('_') or remaining_key.startswith('.')):
                remaining_key = remaining_key[1:]
        else:
            # This can happen if a key starts with a separator, just consume it
            if remaining_key and (remaining_key.startswith('_') or remaining_key.startswith('.')):
                 remaining_key = remaining_key[1:]
            else:
                 # End of string or malformed key
                 break
            
    return components

# =================================================================================
# == Graph Building Functions
# =================================================================================

def ordinal_transform_graph(input_graph: Dict[str, ModuleNode], transform_map: Dict[str, Dict]) -> Dict[str, ModuleNode]:
    """
    Applies ordinal transformations to a graph, returning a new, transformed graph.
    This handles architectural inversions (e.g., CompVis vs. Diffusers decoders).
    """
    print("Applying ordinal graph transformation...")
    transformed_graph = {}
    for original_key, node in input_graph.items():
        transformed_signature_list = list(node.signature)
        
        for i, (comp_type, comp_value) in enumerate(transformed_signature_list):
            if comp_type == 'name' and comp_value in transform_map:
                transform_info = transform_map[comp_value]
                # Find the next component, which should be the ordinal to transform
                if (i + 1) < len(transformed_signature_list) and transformed_signature_list[i + 1][0] == 'ordinal':
                    ordinal_to_transform = transformed_signature_list[i + 1][1]
                    new_ordinal = transform_info["transform_fn"](ordinal_to_transform)
                    transformed_signature_list[i + 1] = ('ordinal', new_ordinal)

        # Create a new node with the transformed signature
        new_node = copy.copy(node) # Shallow copy is sufficient
        new_node.signature = tuple(transformed_signature_list)
        transformed_graph[original_key] = new_node

    return transformed_graph

def build_base_graph(base_vae_state_dict: Dict[str, torch.Tensor]) -> Dict[str, ModuleNode]:
    """
    Builds an abstract graph for a standard model state_dict (like a VAE).
    Base model keys are considered canonical, so no name mapping is needed.
    """
    print("Building abstract graph for base VAE...")
    base_graph = {}
    for key, tensor in base_vae_state_dict.items():
        if not key.endswith('.weight'):
            continue

        module_key = key[:-len('.weight')]
        shape = (0, 0)

        if tensor.dim() == 4:  # Conv2d: (out, in, kH, kW)
            shape = (tensor.shape[1], tensor.shape[0])
        elif tensor.dim() == 2:  # Linear: (out, in)
            shape = (tensor.shape[1], tensor.shape[0])

        if shape != (0, 0):
            # Base model keys are already clean, so pass an empty map and token list.
            base_graph[module_key] = ModuleNode(module_key, {}, [], shape)
    
    print(f"✅ Successfully built base graph with {len(base_graph)} nodes.")
    return base_graph

def build_adapter_graph(adapter_state_dict: Dict[str, torch.Tensor], name_map: Dict[str, str], known_tokens: List[str]) -> Tuple[Dict[str, ModuleNode], Dict[str, Tuple[int, int]]]:
    """
    Builds an abstract graph for a LoRA adapter's state_dict.
    """
    print("Building abstract graph from adapter state_dict...")
    adapter_graph = {}
    adapter_shapes = {}
    processed_modules = set()
    prefix_to_strip = "lora_vae_"

    for key in adapter_state_dict.keys():
        is_glora = '.a1.weight' in key
        is_lora = 'lora_down.weight' in key

        if is_glora or is_lora:
            module_key_base = key.rsplit('.', 2)[0]
            if module_key_base in processed_modules:
                continue
            
            clean_adapter_key = module_key_base[len(prefix_to_strip):] if module_key_base.startswith(prefix_to_strip) else module_key_base
            shape = (0, 0)
            if is_glora:
                in_ch = adapter_state_dict[key].shape[0]
                out_ch = adapter_state_dict.get(f"{module_key_base}.b1.weight", torch.empty(0)).shape[0]
                shape = (in_ch, out_ch) if out_ch > 0 else (0,0)
            else: # is_lora
                in_ch = adapter_state_dict[key].shape[1]
                up_key = key.replace('lora_down.weight', 'lora_up.weight')
                out_ch = adapter_state_dict.get(up_key, torch.empty(0)).shape[0]
                shape = (in_ch, out_ch) if out_ch > 0 else (0,0)

            adapter_graph[module_key_base] = ModuleNode(clean_adapter_key, name_map, known_tokens, shape)
            adapter_shapes[module_key_base] = shape
            processed_modules.add(module_key_base)
            
    print(f"✅ Successfully built adapter graph with {len(adapter_graph)} nodes.")
    return adapter_graph, adapter_shapes

def remap_graphs(adapter_graph: Dict[str, ModuleNode], base_graph: Dict[str, ModuleNode]) -> Dict[str, str]:
    """
    Maps nodes from the adapter graph to the base graph by finding an
    isomorphism between their normalized signatures.
    """
    print("🧠 Performing semantic graph-based remapping...")
    
    base_groups = defaultdict(list)
    for key, node in base_graph.items():
        base_groups[node.signature].append(key)

    final_mapping = {}
    unmapped_nodes = []
    for adapter_key, adapter_node in adapter_graph.items():
        if adapter_node.signature in base_groups:
            possible_matches = base_groups[adapter_node.signature]
            if len(possible_matches) == 1:
                final_mapping[adapter_key] = possible_matches[0]
            else:
                unmapped_nodes.append(f"Ambiguous match for {adapter_key}: maps to {len(possible_matches)} base modules.")
        else:
            unmapped_nodes.append(f"No match for {adapter_key} (Signature: {adapter_node.signature})")

    if unmapped_nodes:
        print(f"\n⚠️  Could not map {len(unmapped_nodes)} adapter modules:")
        for info in unmapped_nodes[:10]:
            print(f"  - {info}")
    
    print(f"✅ Successfully generated a definitive map for {len(final_mapping)} modules.")
    return final_mapping

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

def translate_adapter_key_to_vae_key(pytorch_path: str) -> str:
    """
    Translates adapter key format to VAE state_dict format.
    IMPROVED HEURISTIC: Fixes common dot-vs-underscore errors.
    """
    key = pytorch_path
    
    # Step 1: Handle middle block (no changes needed here)
    key = key.replace('mid_block_attentions_0_to_q', 'mid.attn_1.q')
    key = key.replace('mid_block_attentions_0_to_k', 'mid.attn_1.k')
    key = key.replace('mid_block_attentions_0_to_v', 'mid.attn_1.v')
    key = key.replace('mid_block_attentions_0_to_out_0', 'mid.attn_1.proj_out')
    key = key.replace('mid_block_resnets_0', 'mid.block_1')
    key = key.replace('mid_block_resnets_1', 'mid.block_2')
    
    # Step 2: Handle up_blocks (no changes needed here)
    for i in range(4):
        for j in range(3):
            key = key.replace(f'up_blocks_{i}_resnets_{j}', f'up.{i}.block.{j}')
        key = key.replace(f'up_blocks_{i}_upsamplers_0', f'up.{i}.upsampler')

    # --- NEW: Step 3: Fix remaining underscore-to-dot mismatches ---
    # These are the errors revealed by the --debug_keys output.
    # We replace them AFTER the block replacements to avoid unintended side effects.
    key = key.replace('_conv1', '.conv1')
    key = key.replace('_conv2', '.conv2')
    key = key.replace('upsampler_conv', 'upsampler.conv')
    key = key.replace('conv_shortcut', 'nin_shortcut') # This was already here, but fits logically
    
    # The old `nin_shortcut` replacement might need to be more specific
    # to avoid conflicts, but let's correct the main `conv` issue first.
    # A more robust version would be:
    key = key.replace('_nin_shortcut', '.nin_shortcut')


    # Step 4: Add decoder prefix
    final_key = f"decoder.{key}"
    
    return final_key

def fuse_vae_lora(
    base_model_path: str,
    vae_adapter_path: str,
    output_path: str,
    device: str = "cpu",
    pickletensor: bool = False,
    debug_keys: bool = False,
    ordinal_transform: str = None
):
    # ========================================================================
    # == SEMANTIC CONFIGURATION: The only part you may need to edit.        ==
    # ========================================================================
    # Maps naming conventions from the training framework (keys) to the
    # standard diffusers VAE naming (values).
    SEMANTIC_NAME_MAP = {
        # General
        "mid_block": "mid",
        # Blocks
        "up_blocks": "up",
        "down_blocks": "down",
        "resnets": "block",
        # Attention
        "attentions": "attn_1", # Assuming a single attention block per level
        "to_q": "q",
        "to_k": "k",
        "to_v": "v",
        "to_out_0": "proj_out",
        # Other
        "upsamplers": "upsampler",
        "conv_shortcut": "nin_shortcut"
    }
    # This is crucial for the parser to prioritize longer matches
    KNOWN_ADAPTER_TOKENS = list(SEMANTIC_NAME_MAP.keys())
 
    # This map defines the "unnatural, bestial" architectural transformations.
    PREDEFINED_ORDINAL_TRANSFORMS = {
        "compvis2hface": {
            # In CompVis, `up` block 0 is near the bottleneck (high channels).
            # In Diffusers/HF, `up` block 0 is near the output (low channels).
            # This lambda reverses the 4-block index (0->3, 1->2, etc.)
            "up": {"transform_fn": lambda x: 3 - x}
        }
        # New transformations can be added here, e.g., "hface2diffusers_v2"
    }
    # --- 1. Load Models & Build Graphs ---
    base_vae_state_dict = extract_vae_state_dict(base_model_path, device=device)
    base_graph = build_base_graph(base_vae_state_dict)

    if pickletensor:
        adapter_state_dict = torch.load(vae_adapter_path, map_location=device)
    else:
        adapter_state_dict = load_file(vae_adapter_path, device=device)

    adapter_graph, adapter_shapes = build_adapter_graph(adapter_state_dict, SEMANTIC_NAME_MAP, KNOWN_ADAPTER_TOKENS)

    # --- 2. Apply Architectural Transformation (if requested) ---
    if ordinal_transform:
        if ordinal_transform in PREDEFINED_ORDINAL_TRANSFORMS:
            print(f"Applying architectural transformation: '{ordinal_transform}' to adapter graph.")
            transform_map = PREDEFINED_ORDINAL_TRANSFORMS[ordinal_transform]
            adapter_graph = ordinal_transform_graph(adapter_graph, transform_map)
        else:
            print(f"⚠️ WARNING: Unknown ordinal transform '{ordinal_transform}'. Proceeding without transformation.")

    # --- 2. Generate the Definitive Mapping ---
    final_adapter_to_base_map = remap_graphs(adapter_graph, base_graph)

    if debug_keys:
        print("\n--- DEBUG: Semantic Graph Mapping ---")
        for adapter_key, base_key in sorted(final_adapter_to_base_map.items()):
            adapter_node = adapter_graph[adapter_key]
            base_node = base_graph[base_key]
            shape_match_str = "✅" if adapter_node.shape == base_node.shape else "🚨 MISMATCH"
            print(f"'{adapter_key}' (shape {adapter_node.shape}) -> '{base_key}' (shape {base_node.shape}) {shape_match_str}")
        return

    # --- 4. Group Tensors and Validate Shapes ---
    grouped_tensors = defaultdict(dict)
    known_param_suffixes = ['a1.weight', 'a2.weight', 'b1.weight', 'b2.weight', 'lora_down.weight', 'lora_up.weight', 'alpha']
    
    for key, tensor in adapter_state_dict.items():
        for suffix in known_param_suffixes:
            if key.endswith(suffix):
                module_key_base = key[:-(len(suffix) + 1)]
                
                if module_key_base in final_adapter_to_base_map:
                    base_module_key = final_adapter_to_base_map[module_key_base]
                    
                    # SHAPE VALIDATION SAFEGUARD
                    adapter_shape = adapter_shapes.get(module_key_base)
                    base_shape = base_graph[base_module_key].shape
                    if adapter_shape and base_shape != (0,0) and adapter_shape != base_shape:
                        print(f"  > 🚨 SHAPE MISMATCH WARNING for {module_key_base} -> {base_module_key}:")
                        print(f"    Adapter expects shape {adapter_shape}, but base module has {base_shape}.")
                        print("    Skipping this module due to high risk of error.")
                        continue
                    
                    param_name = key[len(module_key_base)+1:]
                    grouped_tensors[base_module_key][param_name] = tensor
                break
    
    # --- 5. Fuse Weights ---
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
                # --- THIS IS THE FINAL, CORRECTED GLORA BLOCK ---
                rank = params['b2.weight'].shape[0]
                alpha = params.get('alpha', torch.tensor(rank)).item()
                
                a1 = params['a1.weight'].to(torch.float32)
                a2 = params['a2.weight'].to(torch.float32)
                b1 = params['b1.weight'].to(torch.float32)
                b2 = params['b2.weight'].to(torch.float32)
                print(f" fuck it lets print all of the relevant shapes:\nW_base{W_base.shape},a1{a1.shape},a2{a2.shape}")
                if W_base.dim() == 4: # Conv2d Layer
                    # A-Term: (W * A1) @ A2, where * and @ are tensor contractions
                    # W: (out_c, in_c, kH, kW), a1: (in_c, r), a2: (r, in_c)
                    # This requires einsum for a clean implementation.
                    delta_W_A = torch.einsum(
                        "o j k h, j i -> o i k h",
                        torch.einsum("o i k h, i j -> o j k h", W_base, a1),
                        a2
                    )
                    
                    # B-Term: B1 @ B2, reshaped to a 1x1 kernel
                    # b1: (out_c, r), b2: (r, in_c) -> (out_c, in_c)
                    delta_W_B = (b1 @ b2).unsqueeze(-1).unsqueeze(-1)

                    # For conv layers, the B-term needs to match the kernel size of the A-term if it's not 1x1
                    # However, GLoRA's B-term is typically treated as a 1x1 conv. 
                    # Let's assume the A-term keeps the original kernel size.
                    
                    # A check for kernel size mismatch is needed if B is not 1x1
                    # But for this implementation, we assume B is a 1x1 conv adjustment
                    # and A adjusts the full kernel. This is a common GLoRA pattern.
                    # We might need to pad delta_W_B if stride > 1
                    
                    delta_W = delta_W_A + delta_W_B

                else: # Linear Layer
                    delta_W_A = (W_base @ a1) @ a2 #was called w_base_32 here but w_base was already upcast either way
                    delta_W_B = b1 @ b2
                    delta_W = delta_W_A + delta_W_B
                
                delta_W *= (alpha / rank)

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
    parser.add_argument(
        "--ordinal_transform",
        type=str,
        choices=["compvis2hface"],
        default="compvis2hface",
        help="Apply a predefined architectural transformation to the adapter keys to bridge known incompatibilities (e.g., CompVis VAE indexing vs. modern Diffusers VAE)."
    )
    # --- NEW ARGUMENT ---
    parser.add_argument(
        "--pickletensor",
        action="store_true",
        help="Load the VAE adapter as a PyTorch pickle file instead of safetensors. Use this for files saved with the '.safenetensors' typo."
    )
    # --- END NEW ---
    parser.add_argument(
        "--debug_keys",
        action="store_true",
        help="dump matched key targets before attempting fusion because hey it's string type inference code"
    )
    args = parser.parse_args()

    fuse_vae_lora(**vars(args))

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