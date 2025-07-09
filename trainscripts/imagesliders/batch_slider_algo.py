import torch
from typing import List, Dict, Tuple, Any
from collections import deque

class GradientNoiseScaleEstimator:
    def __init__(self, beta: float = 0.999):
        self.beta = beta
        self.exp_avg_sq_grad_norm_global = 0.0  # Exponentially weighted average of squared global gradient norm (|G_B_big|^2)
        self.exp_avg_sq_grad_norm_local = 0.0   # Exponentially weighted average of squared local gradient norm (|G_B_small|^2)
        self.t = 0  # Timestep counter

    def update(self, model: torch.nn.Module):
        self.t += 1

        # Calculate global gradient norm (L2 norm of all gradients)
        global_grad_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                global_grad_norm_sq += p.grad.data.norm(2).item()**2
        
        # Calculate local gradient norm (average of squared L2 norms of individual parameter gradients)
        # This is a heuristic approximation for B_small=1 in a single-device context.
        local_grad_norm_sq_sum = 0.0
        num_params_with_grad = 0
        for p in model.parameters():
            if p.grad is not None:
                local_grad_norm_sq_sum += p.grad.data.norm(2).item()**2
                num_params_with_grad += 1
        
        local_grad_norm_sq = local_grad_norm_sq_sum / num_params_with_grad if num_params_with_grad > 0 else 0.0

        # Update exponentially weighted moving averages
        self.exp_avg_sq_grad_norm_global = self.beta * self.exp_avg_sq_grad_norm_global + (1 - self.beta) * global_grad_norm_sq
        self.exp_avg_sq_grad_norm_local = self.beta * self.exp_avg_sq_grad_norm_local + (1 - self.beta) * local_grad_norm_sq

    def get_noise_scale(self, B_big: int, B_small: int = 1) -> float:
        # Bias correction for moving averages
        bias_correction = 1 - self.beta**self.t
        
        # Estimated |G|^2 (true gradient norm squared)
        # From paper: |G|^2 = (B_big * |G_B_big|^2 - B_small * |G_B_small|^2) / (B_big - B_small)
        # Using corrected moving averages as estimates for |G_B_big|^2 and |G_B_small|^2
        
        # Note: The paper's formula for |G|^2 and tr(Sigma) assumes |G_B_small|^2 is from a batch of size B_small.
        # Our local_grad_norm_sq is an average per parameter, not per sample.
        # This is a pragmatic adaptation for a single-device setup.
        
        # For the purpose of this estimation, we'll use the global_grad_norm_sq as |G_B_big|^2
        # and the local_grad_norm_sq as |G_B_small|^2 (assuming B_small=1 for per-parameter average)
        
        # We need to be careful with the interpretation of |G_B_small|^2 here.
        # The paper's formula for S and |G|^2 is based on the expectation E[|G_est|^2] = |G|^2 + tr(Sigma)/B.
        # If we assume our `local_grad_norm_sq` is a proxy for E[|G_est|^2] for B=1, and `global_grad_norm_sq` for B=B_big,
        # then we can use the formulas from Appendix A.1.

        # Use the raw (uncorrected) values for the calculation as per the paper's formula for S and |G|^2
        # The bias correction is applied to the individual estimates before they are used in the ratio.
        
        # Estimated |G|^2 (true gradient norm squared)
        # This is the |G|^2 from the paper's formula, not the raw squared norm.
        # It's derived from the two batch sizes.
        
        # We need to use the *current* estimates of E[|G_est|^2] for B_big and B_small.
        # Let's use the bias-corrected exponential averages.
        
        # Corrected estimates for E[|G_est|^2]
        corrected_global_grad_norm_sq = self.exp_avg_sq_grad_norm_global / bias_correction
        corrected_local_grad_norm_sq = self.exp_avg_sq_grad_norm_local / bias_correction

        if B_big == B_small: # Avoid division by zero
            return 0.0

        # Estimated |G|^2 (true gradient norm squared)
        # |G|^2 = (B_big * E[|G_B_big|^2] - B_small * E[|G_B_small|^2]) / (B_big - B_small)
        estimated_G_sq = (B_big * corrected_global_grad_norm_sq - B_small * corrected_local_grad_norm_sq) / (B_big - B_small)
        
        # Estimated tr(Sigma)
        # tr(Sigma) = (1/B_small - 1/B_big)^-1 * (E[|G_B_small|^2] - E[|G_B_big|^2])
        estimated_tr_Sigma = (1 / (1/B_small - 1/B_big)) * (corrected_local_grad_norm_sq - corrected_global_grad_norm_sq)

        if estimated_G_sq <= 0: # Avoid division by zero or negative noise scale
            return 0.0

        noise_scale = estimated_tr_Sigma / estimated_G_sq
        return noise_scale if noise_scale > 0 else 0.0 # Ensure non-negative noise scale

def _cfg_duplicate(tensor: torch.Tensor) -> torch.Tensor:
    """Duplicates a tensor along the batch dimension for CFG."""
    return torch.cat([tensor, tensor], dim=0)



def form_cfg_microbatch(
    unswizzled_data: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Forms the final microbatch for the UNet based on the paired indices and low/high case flags.

    Args:
        unswizzled_data (dict): Output from unswizzle_conditioning_data.
        batch (dict): The original batch, needed for latents, scales, pair_indices, and is_low_cases.

    Returns:
        A tuple containing:
        - Dict[str, torch.Tensor]: A dict of CFG-ready tensors for the UNet.
          Keys: 'latents_cfg', 'text_embeds_cfg', 'pooled_embeds_cfg', 'add_time_ids_cfg'.
        - torch.Tensor: A tensor of the scales, reordered to match the microbatch.
    """
    pair_indices = batch["pair_indices"]
    is_low_cases = batch["is_low_cases"]

    ordered_indices = pair_indices
    selected_cond_text = []
    selected_cond_pool = []

    for i, idx in enumerate(ordered_indices):
        if is_low_cases[i]:
            selected_cond_text.append(unswizzled_data["neutral_text"][idx])
            selected_cond_pool.append(unswizzled_data["neutral_pool"][idx])
        else:
            selected_cond_text.append(unswizzled_data["positive_text"][idx])
            selected_cond_pool.append(unswizzled_data["positive_pool"][idx])

    if not ordered_indices.numel():
        return {}, torch.empty(0)

    # Gather data in the new microbatch order
    ordered_latents = batch["latents"][ordered_indices]
    ordered_scales = batch["scales"][ordered_indices]
    ordered_add_time_ids = batch["add_time_ids"][ordered_indices]
    ordered_uncond_text = unswizzled_data["unconditional_text"][ordered_indices]
    ordered_uncond_pool = unswizzled_data["unconditional_pool"][ordered_indices]

    # Assemble the final conditional embeddings
    final_cond_text = torch.cat(selected_cond_text, dim=0)
    final_cond_pool = torch.cat(selected_cond_pool, dim=0)

    # Build CFG tensors: cat([unconditional, conditional])
    microbatch = {
        "latents_cfg": _cfg_duplicate(ordered_latents),
        "add_time_ids_cfg": _cfg_duplicate(ordered_add_time_ids),
        "text_embeds_cfg": torch.cat([ordered_uncond_text, final_cond_text], dim=0),
        "pooled_embeds_cfg": torch.cat([ordered_uncond_pool, final_cond_pool], dim=0),
    }
    
    # Return the microbatch and the corresponding ordered latents/scales needed by the main loop
    return microbatch, ordered_latents, ordered_scales, ordered_indices

def calculate_paired_loss(
    predicted_noise: torch.Tensor,
    target_noise: torch.Tensor,
    pair_indices: torch.Tensor,
    is_low_cases: torch.Tensor,
    model: torch.nn.Module = None, # Added for gradient noise estimation
    noise_estimator: GradientNoiseScaleEstimator = None, # Added for gradient noise estimation
) -> torch.Tensor:
    """
    Calculates the training loss, summing losses for paired items before averaging.

    Args:
        predicted_noise (torch.Tensor): Noise from the UNet (shape 2*N_eff, C, H, W).
        target_noise (torch.Tensor): Ground truth noise, duplicated for CFG.
        pair_indices (torch.Tensor): Indices of the paired items.
        is_low_cases (torch.Tensor): Boolean tensor indicating if an item is a low case.
        model (torch.nn.Module): The model whose gradients will be used for noise estimation.
        noise_estimator (GradientNoiseScaleEstimator): An instance of the noise estimator.

    Returns:
        torch.Tensor: The final scalar loss for the batch.
    """
    # Calculate MSE loss per-element, then reduce to a per-item scalar.
    # This is the generative error for each individual training data sample.
    loss_per_sample = (predicted_noise - target_noise).pow(2).mean(dim=[1, 2, 3])

    # The 'scale-tuple' (or pair, for now) is the smallest semantically valid unit of training.
    # We sum the generative errors for all samples within the same 'scale-tuple'.
    # This corresponds to the joint minimization of the generative error across related samples.
    unique_pair_indices = torch.unique(pair_indices)
    summed_pair_losses = torch.zeros(len(unique_pair_indices), device=predicted_noise.device, dtype=predicted_noise.dtype)

    for i, p_idx in enumerate(unique_pair_indices):
        # Select losses corresponding to the current pair index
        current_pair_losses = loss_per_sample[pair_indices == p_idx]
        # Sum them up to get the loss for this 'scale-tuple'
        summed_pair_losses[i] = current_pair_losses.sum()

    # Finally, mean reduce the summed 'scale-tuple' losses to get the batch loss.
    # This is analogous to averaging policy optimization losses over multiple rollouts
    # or averaging GPT losses over many context-continuation pairs in a batch.
    final_loss = summed_pair_losses.mean()
    return final_loss


# from trainscripts/imagesliders/batch_loss_functions.py
# Defines stateless loss functions.
# semantically meaningful only for 'textsliders'

def calculate_erase_loss(
    target_latents: torch.FloatTensor,
    positive_latents: torch.FloatTensor,
    unconditional_latents: torch.FloatTensor,
    neutral_latents: torch.FloatTensor,
    guidance_scale: float,
    loss_fn: torch.nn.Module,
) -> torch.FloatTensor:
    """Target latents are going not to have the positive concept."""
    return loss_fn(
        target_latents,
        neutral_latents
        - guidance_scale * (positive_latents - unconditional_latents)
    )

def calculate_enhance_loss(
    target_latents: torch.FloatTensor,
    positive_latents: torch.FloatTensor,
    unconditional_latents: torch.FloatTensor,
    neutral_latents: torch.FloatTensor,
    guidance_scale: float,
    loss_fn: torch.nn.Module,
) -> torch.FloatTensor:
    """Target latents are going to have the positive concept."""
    return loss_fn(
        target_latents,
        neutral_latents
        + guidance_scale * (positive_latents - unconditional_latents)
    )
