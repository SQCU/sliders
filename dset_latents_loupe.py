# dset_latents_loupe.py (v2)
# An enhanced tool to visualize detailed, per-channel latent statistics.
# USAGE: python latents_loupe.py -f ./stats/final_stats_report.safetensors

import safetensors
import torch
import numpy as np
import argparse
from pathlib import Path
import json

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError:
    print("Matplotlib is required. Please run: pip install matplotlib")
    exit()

def plot_histogram(ax, counts, bins, title, x_label, ideal_line):
    """Helper function to plot a single histogram."""
    ax.bar(bins[:-1], counts, width=np.diff(bins), edgecolor='black', align='edge')
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("Image Count")
    ax.set_xlabel(x_label)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.axvline(x=ideal_line, color='r', linestyle='--', label=f'Ideal ({ideal_line})')
    ax.legend(fontsize=8)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.tick_params(axis='x', labelsize=8)

def main():
    parser = argparse.ArgumentParser(description="Visualize aggregate latent statistics from a report file.")
    parser.add_argument('-s', '--stats_report', type=str, required=True, help="Path to the final_stats_report.safetensors file.")
    parser.add_argument('-l', '--latent_cache', type=str, required=True, help="Path to the safetensors cache file containing the original latents.")
    args = parser.parse_args()

    report_path = Path(args.stats_report)
    if not report_path.exists():
        print(f"Error: Stats report file not found at '{report_path}'"); return

    print(f"--- Loading Latent Statistics Report from: {report_path} ---")

    # --- Step 1: Load all data from the file in a single, simple block ---
    report_key = None
    contributing_work_ids = []
    outlier_report = []
    histogram_data = {}
    
    try:
        with safetensors.safe_open(report_path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            if not metadata:
                print("Error: No metadata found in report file."); return
            
            report_key = next(iter(metadata.keys()))
            report_metadata = json.loads(metadata[report_key])
            contributing_work_ids = report_metadata.get("contributing_work_ids", [])

            # Load ESSENTIAL tensors for the outlier report
            outlier_indices = f.get_tensor(f"{report_key}_aggregate_latent_stats_report_outlier_indices")
            outlier_scores = f.get_tensor(f"{report_key}_aggregate_latent_stats_report_outlier_scores")
            outlier_stats_vectors = f.get_tensor(f"{report_key}_aggregate_latent_stats_report_outlier_stats_vectors")
            sample_count = int(f.get_tensor(f"{report_key}_aggregate_latent_stats_report_sample_count").item())
            
            # Reliably determine num_channels from the essential stats vector
            num_channels = outlier_stats_vectors.shape[1] // 2

            # Try to load OPTIONAL histogram tensors
            try:
                for i in range(num_channels):
                    histogram_data[f'ch{i}_means_counts'] = f.get_tensor(f"{report_key}_aggregate_latent_stats_report_ch{i}_means_hist_counts")
                    histogram_data[f'ch{i}_means_bins'] = f.get_tensor(f"{report_key}_aggregate_latent_stats_report_ch{i}_means_hist_bins")
                    histogram_data[f'ch{i}_stds_counts'] = f.get_tensor(f"{report_key}_aggregate_latent_stats_report_ch{i}_stds_hist_counts")
                    histogram_data[f'ch{i}_stds_bins'] = f.get_tensor(f"{report_key}_aggregate_latent_stats_report_ch{i}_stds_hist_bins")
            except KeyError:
                print("\n[INFO] Report file does not contain histogram data. Skipping plot generation.")
                histogram_data = {} # Clear partial data if any key was missing
    
    except Exception as e:
        print(f"\n[ERROR] An error occurred while loading the report: {e}")
        return

    # --- Step 2: Process loaded data to create the readable outlier report ---
    for i, original_index in enumerate(outlier_indices):
        stats_vec = outlier_stats_vectors[i]
        outlier_report.append({
            "work_id": contributing_work_ids[original_index.item()],
            "outlier_score": outlier_scores[i].item(),
            "means_per_channel": stats_vec[:num_channels].tolist(),
            "stds_per_channel": stats_vec[num_channels:].tolist(),
        })

    # --- Step 3: Present the results ---
    print("\n--- Top Outlier Images (Most distant from ideal Gaussian stats) ---")
    if not outlier_report:
        print("  - No outlier information found in the report.")
    else:
        for item in outlier_report:
            means_str = ", ".join([f"{m:.3f}" for m in item['means_per_channel']])
            stds_str = ", ".join([f"{s:.3f}" for s in item['stds_per_channel']])
            print(f"  - Work ID: {item['work_id']}")
            print(f"    - Score: {item['outlier_score']:.4f}, Means: [{means_str}], Stds: [{stds_str}]")
    print("---------------------------------------------------------------------")

    if outlier_report:
        top_outlier_work_id = outlier_report[0]['work_id']
        latent_key = f"{top_outlier_work_id}_latent"
        print("\n[ACTION] To visualize the channels of the top outlier latent, run this command:")
        print(f"uv run python dset_visualize_single_latent.py -c \"{args.latent_cache}\" -k \"{latent_key}\"")

    # --- Step 4: Visualize if possible ---
    if histogram_data:
        fig, axes = plt.subplots(num_channels, 2, figsize=(12, 3 * num_channels), sharex='col')
        fig.suptitle(f"Per-Channel Latent Statistics Distribution for {sample_count} Images", fontsize=16)

        for i in range(num_channels):
            plot_histogram(axes[i, 0], histogram_data[f'ch{i}_means_counts'].numpy(), histogram_data[f'ch{i}_means_bins'].numpy(), f"Channel {i} Mean Distribution", "Mean Value", 0.0)
            plot_histogram(axes[i, 1], histogram_data[f'ch{i}_stds_counts'].numpy(), histogram_data[f'ch{i}_stds_bins'].numpy(), f"Channel {i} Std. Dev. Distribution", "Std. Dev. Value", 1.0)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        output_filename = report_path.parent / f"{report_path.stem}_per_channel_viz.png"
        plt.savefig(output_filename)
        print(f"\n--- Detailed per-channel visualization saved to: {output_filename} ---")

if __name__ == "__main__":
    main()