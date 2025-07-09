import torch
from typing import List, Dict, Tuple, Any
from collections import deque

class GradientNoiseScaleEstimator:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.gradient_norms_squared = deque(maxlen=window_size)
        self.gradient_norms = deque(maxlen=window_size)

    def update(self, model: torch.nn.Module):
        # Calculate the squared L2 norm of the gradients for all parameters
        total_grad_norm_squared = 0.0
        total_grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm_squared = p.grad.data.norm(2).item()**2
                total_grad_norm_squared += grad_norm_squared
                total_grad_norm += p.grad.data.norm(2).item()

        self.gradient_norms_squared.append(total_grad_norm_squared)
        self.gradient_norms.append(total_grad_norm)

    def get_noise_scale(self) -> float:
        if len(self.gradient_norms_squared) < 2:
            return 0.0  # Not enough data to estimate variance

        # Estimate tr(Sigma) using variance of gradient norms squared
        # This is a simplification and not the exact method from the paper's Appendix A.1
        # which requires gradients from different batch sizes.
        # Here, we're using the variance of the full batch gradient norms as a proxy for noise.
        tr_sigma_estimate = torch.tensor(list(self.gradient_norms_squared)).var().item()

        # Estimate |G|^2 using mean of gradient norms squared
        g_squared_estimate = torch.tensor(list(self.gradient_norms)).mean().item()**2

        if g_squared_estimate == 0:
            return 0.0

        # B_simple = tr(Sigma) / |G|^2
        noise_scale = tr_sigma_estimate / g_squared_estimate
        return noise_scale

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

    # Update gradient noise estimator if provided
    if noise_estimator is not None and model is not None:
        # Perform backward pass to get gradients before updating estimator
        # This is done here because the estimator needs access to computed gradients
        final_loss.backward(retain_graph=True) # Retain graph for subsequent backward if needed
        noise_estimator.update(model)
        # Zero out gradients after update to prevent accumulation if not handled elsewhere
        model.zero_grad()

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
