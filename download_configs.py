from huggingface_hub import hf_hub_download
import os

CONFIG_MODELS = {
    "stabilityai/stable-diffusion-xl-base-1.0": "unet/config.json",
    "runwayml/stable-diffusion-v1-5": "unet/config.json",
    "madebyollin/sdxl-vae-fp16-fix": "config.json", # Corrected repo_id
}

CANON_CONFIGS_DIR = "F:/dox/ai/gemmy/sliders/canon_configs"

os.makedirs(CANON_CONFIGS_DIR, exist_ok=True)

for repo_id, config_path in CONFIG_MODELS.items():
    # Create a clean directory name from the repo_id
    repo_dir_name = repo_id.replace("/", "__").replace("-", "_") # Replace / and - for valid dir names
    target_dir = os.path.join(CANON_CONFIGS_DIR, repo_dir_name)
    os.makedirs(target_dir, exist_ok=True)

    # Determine the filename within the repo (e.g., config.json or unet/config.json)
    # and save it directly into the target_dir
    filename_in_repo = os.path.basename(config_path)

    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=config_path,
            local_dir=target_dir, # Download directly into the new subdirectory
            local_dir_use_symlinks=False # Avoid symlinks for direct file copy
        )
        print(f"Downloaded {config_path} from {repo_id} to {downloaded_path}")
    except Exception as e:
        print(f"Error downloading {config_path} from {repo_id}: {e}")