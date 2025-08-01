# dset_latents_loupe.py (v2)
# An enhanced tool to visualize detailed, per-channel latent statistics.
# USAGE: python dset_latents_loupe.py -f ./stats/final_stats_report.safetensors

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
    channelwise_outlier_report = []
    global_outlier_report = []
    histogram_data = {}
    global_histogram_data = {}
    
    try:
        with safetensors.safe_open(report_path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            if not metadata:
                print("Error: No metadata found in report file."); return
            
            report_key = next(iter(metadata.keys()))
            report_metadata = json.loads(metadata[report_key])
            contributing_work_ids = report_metadata.get("contributing_work_ids", [])

            # Load Channel-wise outlier data
            chwise_indices = f.get_tensor(f"{report_key}_aggregate_latent_stats_report_channelwise_outlier_indices")
            chwise_scores = f.get_tensor(f"{report_key}_aggregate_latent_stats_report_channelwise_outlier_scores")
            chwise_stats_vectors = f.get_tensor(f"{report_key}_aggregate_latent_stats_report_channelwise_outlier_stats_vectors")
            num_channels = chwise_stats_vectors.shape[1] // 2
            
            # Load Global outlier data
            global_indices = f.get_tensor(f"{report_key}_aggregate_latent_stats_report_global_outlier_indices")
            global_scores = f.get_tensor(f"{report_key}_aggregate_latent_stats_report_global_outlier_scores")
            global_stats_vectors = f.get_tensor(f"{report_key}_aggregate_latent_stats_report_global_outlier_stats_vectors")
            sample_count = int(f.get_tensor(f"{report_key}_aggregate_latent_stats_report_sample_count").item())

            # take a peek at the channel-squashed histogram tensors
            try:
                global_histogram_data['means_counts'] = f.get_tensor(f"{report_key}_aggregate_latent_stats_report_global_means_hist_counts")
                global_histogram_data['means_bins'] = f.get_tensor(f"{report_key}_aggregate_latent_stats_report_global_means_hist_bins")
                global_histogram_data['stds_counts'] = f.get_tensor(f"{report_key}_aggregate_latent_stats_report_global_stds_hist_counts")
                global_histogram_data['stds_bins'] = f.get_tensor(f"{report_key}_aggregate_latent_stats_report_global_stds_hist_bins")
            except KeyError:
                print("[INFO] Report file does not contain global histogram data. Skipping global plot.")
                global_histogram_data = {}

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
     # --- Step 2: Process data into human-readable reports ---
    for i, idx in enumerate(chwise_indices):
        vec = chwise_stats_vectors[i]
        channelwise_outlier_report.append({ "work_id": contributing_work_ids[idx.item()], 
        "score": chwise_scores[i].item(), "means": vec[:num_channels].tolist(), "stds": vec[num_channels:].tolist() })
    
    for i, idx in enumerate(global_indices):
        vec = global_stats_vectors[i]
        global_outlier_report.append({ "work_id": contributing_work_ids[idx.item()], 
        "score": global_scores[i].item(), "mean": vec[0].item(), "std": vec[1].item() })

    # --- Step 3: Present the results ---
    print("\n--- Top Outlier Images (Most distant from ideal Gaussian stats) ---")
    print("\n--- Top Outliers (by Per-Channel Stats) ---")
    for item in channelwise_outlier_report:
        means_str = ", ".join([f"{m:.3f}" for m in item['means']])
        stds_str = ", ".join([f"{s:.3f}" for s in item['stds']])
        print(f"  - Work ID: {item['work_id']} | Score: {item['score']:.4f}\n    - Means: [{means_str}]\n    - Stds:  [{stds_str}]")

    print("\n--- Top Outliers (by Global Stats) ---")
    for item in global_outlier_report:
        print(f"  - Work ID: {item['work_id']} | Score: {item['score']:.4f} (Mean: {item['mean']:.3f}, Std: {item['std']:.3f})")

    # --- Actionable command for the top of EACH list ---
    if channelwise_outlier_report:
        top_chwise_id = channelwise_outlier_report[0]['work_id']
        print("\n[ACTION] To visualize the top CHANNEL-WISE outlier, run:")
        print(f"python dset_visualize_single_latent.py -c \"{args.latent_cache}\" -k \"{top_chwise_id}_latent\"")
    
    if global_outlier_report:
        top_global_id = global_outlier_report[0]['work_id']
        print("\n[ACTION] To visualize the top GLOBAL outlier, run:")
        print(f"python dset_visualize_single_latent.py -c \"{args.latent_cache}\" -k \"{top_global_id}_latent\"")

    # --- Step 4: Visualize if possible ---
    if histogram_data:
        num_plot_rows = num_channels + 1 if global_histogram_data else num_channels
        fig, axes = plt.subplots(num_plot_rows, 2, figsize=(12, 3 * num_plot_rows), sharex='col')
        fig.suptitle(f"Latent Statistics Distribution for {sample_count} Images", fontsize=16)

        plot_row_offset = 0
        if global_histogram_data:
            # --- NEW: Plot Global Stats in the first row ---
            plot_row_offset = 1
            ax_mean_global = axes[0, 0]
            ax_std_global = axes[0, 1]
            plot_histogram(ax_mean_global, global_histogram_data['means_counts'].numpy(), global_histogram_data['means_bins'].numpy(), "Global Mean Distribution (All Channels)", "Mean Value", 0.0)
            plot_histogram(ax_std_global, global_histogram_data['stds_counts'].numpy(), global_histogram_data['stds_bins'].numpy(), "Global Std. Dev. Distribution (All Channels)", "Std. Dev. Value", 1.0)

        for i in range(num_channels):
            ax_mean = axes[i + plot_row_offset, 0]
            ax_std = axes[i + plot_row_offset, 1]
            plot_histogram(ax_mean, histogram_data[f'ch{i}_means_counts'].numpy(), histogram_data[f'ch{i}_means_bins'].numpy(), f"Channel {i} Mean Distribution", "Mean Value", 0.0)
            plot_histogram(ax_std, histogram_data[f'ch{i}_stds_counts'].numpy(), histogram_data[f'ch{i}_stds_bins'].numpy(), f"Channel {i} Std. Dev. Distribution", "Std. Dev. Value", 1.0)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        output_filename = report_path.parent / f"{report_path.stem}_per_channel_viz.png"
        plt.savefig(output_filename)
        print(f"\n--- Detailed per-channel visualization saved to: {output_filename} ---")

if __name__ == "__main__":
    main()