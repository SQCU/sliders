import ast
from pathlib import Path
import gc, os
import random
import torch
from tqdm import tqdm
from collections import defaultdict
import torch.nn.functional as F
import numpy as np

# IMPORTANT: Add these imports
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# logging_utils.py

class TimestepStratifiedLossTracker:
    """
    Track loss progression stratified by timestep difficulty.
    Implements the analysis method mentioned in Karras et al. but missing from libraries.
    """
    def __init__(self, max_timesteps=1000, n_buckets=10):
        self.max_timesteps = max_timesteps
        self.n_buckets = n_buckets
        
        # Define timestep buckets (e.g., [0-100], [100-200], ..., [900-1000])
        self.bucket_edges = torch.linspace(0, max_timesteps, n_buckets + 1)
        
        # Storage: losses[bucket_idx][iteration] = [loss1, loss2, ...]
        self.losses_by_bucket = {i: defaultdict(list) for i in range(n_buckets)}
        
        # Predicted difficulty curve (fit from initial iterations)
        self.difficulty_curve = None
    
    def log(self, losses, timesteps, iteration):
        """
        Log losses stratified by timestep bucket.
        
        Args:
            losses: (B,) tensor of per-sample losses
            timesteps: (B,) tensor of timestep indices
            iteration: current training iteration
        """
        for loss, t in zip(losses, timesteps):
            bucket_idx = self._timestep_to_bucket(t)
            self.losses_by_bucket[bucket_idx][iteration].append(loss.item())
    
    def _timestep_to_bucket(self, t):
        """Map timestep to bucket index."""
        bucket_idx = torch.searchsorted(self.bucket_edges, t.float()) - 1
        return bucket_idx.clamp(0, self.n_buckets - 1).item()
    
    def fit_difficulty_curve(self, iterations_range=(0, 100)):
        """
        Fit piecewise function: difficulty(bucket) from early training.
        
        This captures the 'baseline difficulty' of each timestep range
        before significant learning has occurred.
        """
        start_iter, end_iter = iterations_range
        
        baseline_losses = []
        for bucket_idx in range(self.n_buckets):
            # Aggregate losses from early training
            bucket_losses = []
            for iter_idx in range(start_iter, end_iter):
                if iter_idx in self.losses_by_bucket[bucket_idx]:
                    bucket_losses.extend(self.losses_by_bucket[bucket_idx][iter_idx])
            
            if bucket_losses:
                # Mean and variance as difficulty measures
                baseline_mean = np.mean(bucket_losses)
                baseline_std = np.std(bucket_losses)
                baseline_losses.append({
                    'bucket': bucket_idx,
                    'mean': baseline_mean,
                    'std': baseline_std,
                    'difficulty': baseline_mean  # Simple: use mean as difficulty
                })
        
        self.difficulty_curve = baseline_losses
        return self.difficulty_curve
    
    def compute_difficulty_adjusted_progress(self, current_iteration):
        """
        For each bucket, compute: (baseline_loss - current_loss) / baseline_loss
        
        This tells you: "Given the intrinsic difficulty of this timestep range,
        how much has the model improved?"
        """
        if self.difficulty_curve is None:
            raise ValueError("Must fit difficulty curve first")
        
        progress_by_bucket = []
        
        for bucket_data in self.difficulty_curve:
            bucket_idx = bucket_data['bucket']
            baseline_loss = bucket_data['mean']
            
            # Get recent losses for this bucket
            recent_losses = []
            for iter_idx in range(max(0, current_iteration - 50), current_iteration):
                if iter_idx in self.losses_by_bucket[bucket_idx]:
                    recent_losses.extend(self.losses_by_bucket[bucket_idx][iter_idx])
            
            if recent_losses:
                current_loss = np.mean(recent_losses)
                
                # Relative improvement
                relative_improvement = (baseline_loss - current_loss) / baseline_loss
                
                progress_by_bucket.append({
                    'bucket': bucket_idx,
                    'baseline_loss': baseline_loss,
                    'current_loss': current_loss,
                    'relative_improvement': relative_improvement,
                    'absolute_improvement': baseline_loss - current_loss,
                })
        
        return progress_by_bucket
    
    def plot_stratified_learning_curves(self, save_path=None):
        """
        Create the 'famous plot' showing:
        1. Loss vs iteration for each timestep bucket
        2. Variance vs timestep bucket
        3. Learning rate (slope) vs timestep bucket
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Learning curves by bucket
        ax1 = axes[0, 0]
        for bucket_idx in range(self.n_buckets):
            iterations = []
            mean_losses = []
            
            # Aggregate by iteration
            all_iters = sorted(self.losses_by_bucket[bucket_idx].keys())
            for iter_idx in all_iters:
                losses = self.losses_by_bucket[bucket_idx][iter_idx]
                if losses:
                    iterations.append(iter_idx)
                    mean_losses.append(np.mean(losses))
            
            if iterations:
                bucket_range = f"{int(self.bucket_edges[bucket_idx])}-{int(self.bucket_edges[bucket_idx+1])}"
                ax1.plot(iterations, mean_losses, label=f"t={bucket_range}", alpha=0.7)
        
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Loss")
        ax1.set_title("Learning Curves Stratified by Timestep")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Variance vs timestep bucket
        ax2 = axes[0, 1]
        if self.difficulty_curve:
            buckets = [d['bucket'] for d in self.difficulty_curve]
            variances = [d['std']**2 for d in self.difficulty_curve]
            ax2.bar(buckets, variances, alpha=0.7)
            ax2.set_xlabel("Timestep Bucket")
            ax2.set_ylabel("Loss Variance")
            ax2.set_title("Task Variance by Timestep")
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Difficulty curve (baseline)
        ax3 = axes[1, 0]
        if self.difficulty_curve:
            buckets = [d['bucket'] for d in self.difficulty_curve]
            difficulties = [d['difficulty'] for d in self.difficulty_curve]
            ax3.plot(buckets, difficulties, 'o-', linewidth=2)
            ax3.set_xlabel("Timestep Bucket")
            ax3.set_ylabel("Baseline Loss (Difficulty)")
            ax3.set_title("Task Difficulty by Timestep")
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Relative improvement by bucket
        ax4 = axes[1, 1]
        progress = self.compute_difficulty_adjusted_progress(
            max([iter_idx for bucket in self.losses_by_bucket.values() 
                 for iter_idx in bucket.keys()])
        )
        if progress:
            buckets = [p['bucket'] for p in progress]
            improvements = [p['relative_improvement'] * 100 for p in progress]
            colors = ['green' if imp > 0 else 'red' for imp in improvements]
            ax4.bar(buckets, improvements, color=colors, alpha=0.7)
            ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
            ax4.set_xlabel("Timestep Bucket")
            ax4.set_ylabel("Relative Improvement (%)")
            ax4.set_title("Learning Progress by Timestep Bucket")
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        return fig


def fit_piecewise_difficulty(timesteps, losses, n_pieces=5):
    """
    Fit piecewise linear function: loss = f(timestep)
    
    This captures how task difficulty scales with noise level.
    """
    # Sort by timestep
    sorted_indices = torch.argsort(timesteps)
    timesteps_sorted = timesteps[sorted_indices]
    losses_sorted = losses[sorted_indices]
    
    # Split into pieces
    n_samples = len(timesteps)
    piece_size = n_samples // n_pieces
    
    piece_functions = []
    for i in range(n_pieces):
        start_idx = i * piece_size
        end_idx = (i + 1) * piece_size if i < n_pieces - 1 else n_samples
        
        t_piece = timesteps_sorted[start_idx:end_idx]
        l_piece = losses_sorted[start_idx:end_idx]
        
        # Fit linear: loss = a * t + b
        A = torch.stack([t_piece, torch.ones_like(t_piece)], dim=1)
        coeffs = torch.linalg.lstsq(A, l_piece).solution
        
        piece_functions.append({
            'range': (t_piece[0].item(), t_piece[-1].item()),
            'coeffs': coeffs,
        })
    
    return piece_functions

def predict_expected_loss(timestep, piece_functions):
    """Given a timestep, predict expected loss from difficulty curve."""
    for piece in piece_functions:
        t_min, t_max = piece['range']
        if t_min <= timestep <= t_max:
            a, b = piece['coeffs']
            return a * timestep + b
    return None

def plot_loss_variance_by_timestep(losses_dict):
    """
    Plot that shows: higher timesteps → higher loss variance.
    
    This is the 'famous (100 people read it)' plot demonstrating
    that task difficulty isn't just about mean loss, but uncertainty.
    """
    timesteps = []
    means = []
    stds = []
    
    for t, loss_list in sorted(losses_dict.items()):
        timesteps.append(t)
        means.append(np.mean(loss_list))
        stds.append(np.std(loss_list))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Top: Mean ± Std
    ax1.plot(timesteps, means, 'o-', label='Mean Loss')
    ax1.fill_between(timesteps, 
                      np.array(means) - np.array(stds),
                      np.array(means) + np.array(stds),
                      alpha=0.3, label='±1 Std Dev')
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Distribution by Timestep")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom: Coefficient of Variation (std/mean)
    cv = np.array(stds) / np.array(means)
    ax2.plot(timesteps, cv, 'o-', color='red')
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Coefficient of Variation")
    ax2.set_title("Relative Uncertainty by Timestep")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig