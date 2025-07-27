from huggingface_hub import snapshot_download
import os

# The list of assets is perfect. We will now use it correctly.
CONFIG_MODELS = {
    "stabilityai/stable-diffusion-xl-base-1.0": [
        "unet/config.json",
        "text_encoder/config.json",
        "text_encoder_2/config.json",
        "vae/config.json",
        "tokenizer/*",
        "tokenizer_2/*",
        "scheduler/scheduler_config.json", # Be explicit with the filename
    ],
    "runwayml/stable-diffusion-v1-5": ["unet/config.json", "tokenizer/*"],
    "madebyollin/sdxl-vae-fp16-fix": ["config.json"],
}

CANON_CONFIGS_DIR = "F:/dox/ai/gemmy/sliders/canon_configs"
os.makedirs(CANON_CONFIGS_DIR, exist_ok=True)
print("--- Yoinking Canon Configs and Tokenizers (Corrected Method) ---")

for repo_id, patterns in CONFIG_MODELS.items():
    repo_dir_name = repo_id.replace("/", "__")
    target_dir = os.path.join(CANON_CONFIGS_DIR, repo_dir_name)
    
    # No need to create the target_dir beforehand, snapshot_download will do it.
    print(f"\nProcessing repo: {repo_id} -> {target_dir}")

    try:
        # --- ** THE FIX IS HERE ** ---
        # Call snapshot_download ONCE per repo, giving it the full list of patterns.
        # This is the idiomatic and correct way to use the function.
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=patterns,       # Pass the entire list of files and globs
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            # It's also good practice to ignore other files that might match
            ignore_patterns=["*.md", ".gitattributes"],
        )
        print(f"  ... Success. All assets for {repo_id} downloaded.")
    except Exception as e:
        print(f"  ... ERROR downloading from '{repo_id}': {e}")