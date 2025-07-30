# dset_visualize_single_latent.py
# A tool to render the 4 channels of a single latent tensor from a safetensors file.
# USAGE: uv run python dset_visualize_single_latent.py -c <cache_file> -k <tensor_key>

import safetensors
import torch
import numpy as np
import argparse
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Matplotlib is required. Please run: pip install matplotlib")
    exit()

def main():
    parser = argparse.ArgumentParser(description="Visualize the channels of a single latent tensor.")
    parser.add_argument('-c', '--cache_file', type=str, required=True, help="Path to the safetensors cache file containing the latent.")
    parser.add_argument('-k', '--key', type=str, required=True, help="The key of the latent tensor within the cache file.")
    args = parser.parse_args()

    cache_path = Path(args.cache_file)
    if not cache_path.exists():
        print(f"Error: Cache file not found at '{cache_path}'")
        return

    try:
        with safetensors.safe_open(cache_path, framework="pt", device="cpu") as f:
            # Check if the key exists before trying to get it
            if args.key not in f.keys():
                print(f"Error: Key '{args.key}' not found in '{cache_path}'.")
                print(f"Available keys: {list(f.keys())[:5]}...") # Show some available keys
                return
            latent_tensor = f.get_tensor(args.key)
    except Exception as e:
        print(f"An error occurred while loading the tensor: {e}")
        return

    print(f"Loaded latent tensor '{args.key}' with shape: {latent_tensor.shape}")
    if latent_tensor.dim() != 3 or latent_tensor.shape[0] != 4:
        print("Error: Expected a 3D tensor with 4 channels, e.g., (4, H, W).")
        return

    # --- Visualization ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f"Latent Tensor Visualization\nFile: {cache_path.name}\nKey: {args.key}", fontsize=14)

    display_tensor = latent_tensor.to(torch.float32)
    # Normalize the color mapping based on the min/max of the entire tensor for consistent viz
    vmin = display_tensor.min()
    vmax = display_tensor.max()

    for i, ax in enumerate(axes.flat):
        channel_data = display_tensor[i]
        im = ax.imshow(channel_data, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f"Channel {i}\nMean: {channel_data.mean():.3f}, Std: {channel_data.std():.3f}")
        ax.axis('off')

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    output_filename = f"latent_viz_{args.key}.png"
    plt.savefig(output_filename)
    print(f"\n--- Visualization of latent '{args.key}' saved to: {output_filename} ---")
    plt.show()

if __name__ == "__main__":
    main()