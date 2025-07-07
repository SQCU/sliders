import torch
from typing import List, Dict, Tuple, Any

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
) -> torch.Tensor:
    """
    Calculates the training loss, summing losses for paired items before averaging.

    Args:
        predicted_noise (torch.Tensor): Noise from the UNet (shape 2*N_eff, C, H, W).
        target_noise (torch.Tensor): Ground truth noise, duplicated for CFG.
        pair_indices (torch.Tensor): Indices of the paired items.
        is_low_cases (torch.Tensor): Boolean tensor indicating if an item is a low case.

    Returns:
        torch.Tensor: The final scalar loss for the batch.
    """
    # Calculate MSE loss per-element, then reduce to a per-item scalar
    loss_per_item_cfg = (predicted_noise - target_noise).pow(2).mean(dim=[1, 2, 3])
    
    # Average the loss from the unconditional and conditional forward passes
    uncond_loss, cond_loss = loss_per_item_cfg.chunk(2)
    loss_per_item = (uncond_loss + cond_loss) / 2.0  # Shape: (N_eff,)
    
    # Separate losses for low and high cases
    low_case_losses = loss_per_item[is_low_cases]
    high_case_losses = loss_per_item[~is_low_cases]

    # Assuming low and high cases are perfectly interleaved for pairing
    num_pairs = min(len(low_case_losses), len(high_case_losses))
    
    if num_pairs == 0:
        return loss_per_item.mean() if loss_per_item.numel() > 0 else torch.tensor(0.0)

    # Sum the losses for each pair
    summed_pair_losses = low_case_losses[:num_pairs] + high_case_losses[:num_pairs]
    
    # Get losses for any leftover items (if any)
    unpaired_losses = torch.cat([
        low_case_losses[num_pairs:],
        high_case_losses[num_pairs:]
    ])
    
    # Combine the pair-wise summed losses and the individual unpaired losses
    all_losses_to_average = torch.cat([summed_pair_losses, unpaired_losses])
    
    return all_losses_to_average.mean()


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
