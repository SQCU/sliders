import torch
import os
import matplotlib.pyplot as plt
from PIL import Image

def analyze_tensor(tensor, name):
    print(f"\n--- Analysis for {name} ---")
    print(f"Shape: {tensor.shape}")
    print(f"Data Type: {tensor.dtype}")
    print(f"Min: {tensor.min().item()}")
    print(f"Max: {tensor.max().item()}")
    print(f"Mean: {tensor.mean().item()}")
    print(f"Std Dev: {tensor.std().item()}")

    # Heuristic for image-like tensors (e.g., latents are 4D)
    if tensor.dim() == 4 and tensor.shape[1] in [1, 3, 4]: # Channels first
        print("This tensor appears to be image-like (e.g., latents). Consider visualizing it.")
        print("To visualize, you might need to denormalize and convert to a displayable format (e.g., PIL Image).")
    else:
        print("This tensor is not directly image-like. Its values represent numerical differences.")

def plot_difference_tensor(tensor, name, output_dir):
    if tensor.dim() == 4 and tensor.shape[1] in [1, 3, 4]:
        # Assuming the tensor is [batch_size, channels, height, width]
        # Take the first image in the batch and convert to numpy
        # For visualization, we might want to average across channels or select one
        if tensor.shape[1] == 4: # Latent space, often 4 channels
            # Simple approach: take the mean across channels for grayscale visualization
            image_data = tensor[0].mean(dim=0).cpu().numpy()
        else:
            image_data = tensor[0].permute(1, 2, 0).cpu().numpy() # HWC for matplotlib

        plt.figure(figsize=(6, 6))
        plt.imshow(image_data, cmap='viridis') # Use a colormap suitable for differences
        plt.title(f"Difference: {name}")
        plt.colorbar()
        plt.axis('off')
        plot_filename = os.path.join(output_dir, f"{name}.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved plot: {plot_filename}")
    else:
        print(f"Skipping plot for {name}: Not a 4D image-like tensor.")

def main():
    print("Loading and analyzing difference tensors...")

    output_dir = "F:/dox/ai/gemmy/sliders/tensor_logs"
    state_dict_filename = os.path.join(output_dir, "diff_debugger_tensors.pt")

    if os.path.exists(state_dict_filename):
        try:
            loaded_tensors = torch.load(state_dict_filename)
            print(f"\n--- Analyzing contents of {state_dict_filename} ---")

            for name, tensor in loaded_tensors.items():
                analyze_tensor(tensor, name)

            # Calculate and print Frobenius norms of differences between per-element squared errors
            print("\n--- Frobenius Norms of Per-Element Squared Error Differences ---")
            try:
                diff_high_orig_refactored_per_element = loaded_tensors["loss_high_orig_per_element"] - loaded_tensors["loss_high_refactored_per_element"]
                diff_low_orig_refactored_per_element = loaded_tensors["loss_low_orig_per_element"] - loaded_tensors["loss_low_refactored_per_element"]
                diff_high_orig_super_per_element = loaded_tensors["loss_high_orig_per_element"] - loaded_tensors["loss_high_super_per_element"]
                diff_low_orig_super_per_element = loaded_tensors["loss_low_orig_per_element"] - loaded_tensors["loss_low_super_per_element"]

                print(f"Frobenius Norm (high_orig vs high_refactored): {torch.linalg.norm(diff_high_orig_refactored_per_element).item()}")
                print(f"Frobenius Norm (low_orig vs low_refactored): {torch.linalg.norm(diff_low_orig_refactored_per_element).item()}")
                print(f"Frobenius Norm (high_orig vs high_super): {torch.linalg.norm(diff_high_orig_super_per_element).item()}")
                print(f"Frobenius Norm (low_orig vs low_super): {torch.linalg.norm(diff_low_orig_super_per_element).item()}")

                # Plotting subtractive differences
                print("\n--- Plotting Subtractive Differences ---")
                plot_difference_tensor(diff_high_orig_refactored_per_element, "diff_high_orig_refactored_per_element", output_dir)
                plot_difference_tensor(diff_low_orig_refactored_per_element, "diff_low_orig_refactored_per_element", output_dir)
                plot_difference_tensor(diff_high_orig_super_per_element, "diff_high_orig_super_per_element", output_dir)
                plot_difference_tensor(diff_low_orig_super_per_element, "diff_low_orig_super_per_element", output_dir)

            except KeyError as ke:
                print(f"Missing tensor in state dict for Frobenius norm calculation or plotting: {ke}")
            except Exception as e:
                print(f"Error calculating Frobenius norms or plotting: {e}")

        except Exception as e:
            print(f"Error loading or analyzing {state_dict_filename}: {e}")
    else:
        print(f"State dictionary file not found: {state_dict_filename}")

    print("\n--- Future Enhancements ---")
    print("Tensor visualization: Implement functions to convert image-like tensors to displayable formats.")
    print("EMA for sampled tensors: Integrate into training loop to track moving averages of tensor statistics.")
    print("Metadata labeling: Enhance analysis with explicit metadata about tensor origin and purpose.")

if __name__ == "__main__":
    main()

