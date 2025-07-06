import os
import torch
from safetensors import safe_open
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from typing import Dict, Any, Union, List
import yaml

# Define a type for the text encoders used in SDXL
SDXL_TEXT_ENCODER_TYPE = Union[CLIPTextModel, CLIPTextModelWithProjection]

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = AttrDict(value)

def load_config_from_yaml(filepath):
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

class ModelMap:
    """
    A class to hold metadata about the models within a single checkpoint file,
    without loading the actual model weights into memory.
    """
    def __init__(self, model_path: str, metadata: Dict[str, Any], tensor_info: Dict[str, Dict[str, Any]]):
        self.model_path = model_path
        self.metadata = metadata
        self.tensor_info = tensor_info
        self._submodel_keys = self._infer_submodel_keys()

    def _infer_submodel_keys(self) -> Dict[str, List[str]]:
        """
        Infers which tensors belong to which submodel (UNet, VAE, Text Encoders).
        This is a heuristic-based approach and might need refinement based on
        actual checkpoint structures.
        """
        submodel_keys = {
            "unet": [],
            "vae": [],
            "text_encoder": [],
            "text_encoder_2": [],
            "tokenizer": [], # Tokenizers don't have weights, but we can note their presence
            "tokenizer_2": [],
        }

        for key in self.tensor_info.keys():
            if key.startswith("model.diffusion_model."):
                submodel_keys["unet"].append(key)
            elif key.startswith("first_stage_model."):
                submodel_keys["vae"].append(key)
            elif key.startswith("cond_stage_model.transformer.text_model."):
                submodel_keys["text_encoder"].append(key)
            elif key.startswith("cond_stage_model.model.text_model."):
                submodel_keys["text_encoder_2"].append(key)
            # Tokenizers are usually loaded separately and don't have weights in the safetensors file
            # We'll assume their presence based on the text encoders.

        return submodel_keys

    def get_submodel_keys(self, submodel_name: str) -> List[str]:
        """
        Returns the list of tensor keys associated with a given submodel.
        """
        return self._submodel_keys.get(submodel_name, [])

def sliders_from_single_file(model_path: str) -> ModelMap:
    """
    Reads a single checkpoint file (.safetensors) and extracts metadata and
    tensor information without loading the full model into memory.
    Returns a ModelMap object.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    metadata = {}
    tensor_info = {}

    with safe_open(model_path, framework="pt") as f:
        metadata = f.metadata()
        for key in f.keys():
            tensor_info[key] = {
                "shape": f.get_slice(key).get_shape(),
                "dtype": str(f.get_slice(key).get_dtype()) # Convert torch.dtype to string
            }
    
    return ModelMap(model_path, metadata, tensor_info)

def sliders_loadmodels(
    model_map: ModelMap,
    components_to_load: List[str],
    device: torch.device,
    weight_dtype: torch.dtype,
    sdxl_config_path: str, # New parameter for the SDXL config
) -> Dict[str, Any]:
    """
    Loads specified components from a checkpoint file into memory.
    """
    loaded_components = {}
    sdxl_config = AttrDict(load_config_from_yaml(sdxl_config_path))

    with safe_open(model_map.model_path, framework="pt") as f:
        for component_name in components_to_load:
            if component_name == "unet":
                unet_config = sdxl_config.model.network_config.params
                unet = UNet2DConditionModel(
                    sample_size=unet_config.get("sample_size", 128), # Default if not in config
                    in_channels=unet_config.in_channels,
                    out_channels=unet_config.out_channels,
                    down_block_types=["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"],
                    up_block_types=["UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"],
                    block_out_channels=unet_config.channel_mult, # This needs to be mapped correctly
                    layers_per_block=unet_config.num_res_blocks,
                    cross_attention_dim=unet_config.context_dim,
                    attention_head_dim=unet_config.num_head_channels,
                )
                
                unet_state_dict = {}
                for key in model_map.get_submodel_keys("unet"):
                    # Remove the 'model.diffusion_model.' prefix
                    unet_state_dict[key.replace("model.diffusion_model.", "")] = f.get_tensor(key)
                unet.load_state_dict(unet_state_dict)
                loaded_components["unet"] = unet.to(device, dtype=weight_dtype)

            elif component_name == "vae":
                vae_config = sdxl_config.model.first_stage_config.params.ddconfig
                vae = AutoencoderKL(
                    in_channels=vae_config.in_channels,
                    out_channels=vae_config.out_ch,
                    down_block_types=["DownEncoderBlock2D"] * len(vae_config.ch_mult),
                    up_block_types=["UpDecoderBlock2D"] * len(vae_config.ch_mult),
                    block_out_channels=vae_config.ch_mult,
                    latent_channels=vae_config.z_channels,
                    layers_per_block=vae_config.num_res_blocks,
                )

                vae_state_dict = {}
                for key in model_map.get_submodel_keys("vae"):
                    # Remove the 'first_stage_model.' prefix
                    vae_state_dict[key.replace("first_stage_model.", "")] = f.get_tensor(key)
                vae.load_state_dict(vae_state_dict)
                loaded_components["vae"] = vae.to(device, dtype=weight_dtype)

            elif component_name == "text_encoder":
                # This corresponds to FrozenCLIPEmbedder
                # The config doesn't directly provide the model name, so we infer from the architecture
                text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14") # This still uses from_pretrained for config
                text_encoder_state_dict = {}
                for key in model_map.get_submodel_keys("text_encoder"):
                    # Remove the 'cond_stage_model.transformer.text_model.' prefix
                    text_encoder_state_dict[key.replace("cond_stage_model.transformer.text_model.", "")] = f.get_tensor(key)
                text_encoder.load_state_dict(text_encoder_state_dict)
                loaded_components["text_encoder"] = text_encoder.to(device, dtype=weight_dtype)

            elif component_name == "text_encoder_2":
                # This corresponds to FrozenOpenCLIPEmbedder2
                text_encoder_2 = CLIPTextModelWithProjection.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k") # This still uses from_pretrained for config
                text_encoder_2_state_dict = {}
                for key in model_map.get_submodel_keys("text_encoder_2"):
                    # Remove the 'cond_stage_model.model.text_model.' prefix
                    text_encoder_2_state_dict[key.replace("cond_stage_model.model.text_model.", "")] = f.get_tensor(key)
                text_encoder_2.load_state_dict(text_encoder_2_state_dict)
                loaded_components["text_encoder_2"] = text_encoder_2.to(device, dtype=weight_dtype)

            elif component_name == "tokenizer":
                loaded_components["tokenizer"] = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

            elif component_name == "tokenizer_2":
                loaded_components["tokenizer_2"] = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
            
    return loaded_components