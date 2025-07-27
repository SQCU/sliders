from huggingface_hub import hf_hub_download, snapshot_download
import os

# Cleaned up list of assets. Removed non-existent files like scheduler/config.json.
# Added wildcards for tokenizer directories, which snapshot_download can handle.
CONFIG_MODELS = {
    "stabilityai/stable-diffusion-xl-base-1.0": [
        "unet/config.json",
        "text_encoder/config.json",
        "text_encoder_2/config.json",
        "vae/config.json",
        "tokenizer/*",    # Glob pattern for the first tokenizer directory
        "tokenizer_2/*",  # Glob pattern for the second tokenizer directory
        "scheduler/*",    # sneakily reintroduce and make explicit the noise scheduler to formalize our model specs.
    ],
    "runwayml/stable-diffusion-v1-5": ["unet/config.json", "tokenizer/*"], # Also grab its tokenizer
    "madebyollin/sdxl-vae-fp16-fix": ["config.json"],
}

CANON_CONFIGS_DIR = "F:/dox/ai/gemmy/sliders/canon_configs"
os.makedirs(CANON_CONFIGS_DIR, exist_ok=True)
print("--- Yoinking Canon Configs and Tokenizers ---")

for repo_id, paths in CONFIG_MODELS.items():
    repo_dir_name = repo_id.replace("/", "__")
    target_dir = os.path.join(CANON_CONFIGS_DIR, repo_dir_name)
    os.makedirs(target_dir, exist_ok=True)
    print(f"\nProcessing repo: {repo_id} -> {target_dir}")

    for path_pattern in paths:
        try:
            # ** THE FIX **: Differentiate between downloading a single file and a directory pattern.
            if path_pattern.endswith('*'):
                # This is a glob pattern for a directory. Use snapshot_download.
                print(f"  - Downloading directory pattern: {path_pattern}")
                # We tell it to only fetch files matching this pattern.
                snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=[path_pattern],
                    local_dir=target_dir,
                    local_dir_use_symlinks=False,
                )
                print(f"    ... Success.")
            else:
                # This is a single file. Use hf_hub_download.
                print(f"  - Downloading single file: {path_pattern}")
                # hf_hub_download correctly places it in the nested subdirectory inside target_dir.
                hf_hub_download(
                    repo_id=repo_id,
                    filename=path_pattern,
                    local_dir=target_dir,
                    local_dir_use_symlinks=False,
                )
                print(f"    ... Success.")
        except Exception as e:
            print(f"    ... ERROR downloading '{path_pattern}' from '{repo_id}': {e}")