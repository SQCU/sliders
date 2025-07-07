import torch
from typing import List, Dict, Tuple, Any

def _unswizzle_tensor(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Reshapes a (B*3, ...) tensor to (B, 3, ...)."""
    expected_dim0 = batch_size * 3
    if tensor.shape[0] != expected_dim0:
        raise ValueError(
            f"Expected tensor dim 0 to be {expected_dim0}, but got {tensor.shape[0]}"
        )
    return tensor.view(batch_size, 3, *tensor.shape[1:])

def _cfg_duplicate(tensor: torch.Tensor) -> torch.Tensor:
    """Duplicates a tensor along the batch dimension for CFG."""
    return torch.cat([tensor, tensor], dim=0)

def create_pairing_map(scales: torch.Tensor) -> Dict[str, List[Any]]:
    """
    Creates a map of pairs and unpaired items from a batch based on their scales.

    Pairs are formed greedily between items of two different, adjacent scales.
    An item with a lower scale is paired with an item with a higher scale.

    Args:
        scales (torch.Tensor): A 1D tensor of scale values for each item in the batch.

    Returns:
        Dict[str, List[Any]]: A dictionary with two keys:
            - 'pairs': A list of tuples, where each tuple is (low_scale_idx, high_scale_idx).
            - 'unpaired': A list of indices for items that could not be paired.
    """
    if scales.ndim != 1:
        raise ValueError(f"scales tensor must be 1D, but got shape {scales.shape}")

    scales_list = scales.tolist()
    unique_scales_list = sorted(list(set(scales_list)))
    unique_scales = torch.tensor(unique_scales_list, dtype=scales.dtype, device=scales.device)
    if len(unique_scales) < 2:
        return {"pairs": [], "unpaired": list(range(len(scales)))}

    pairs = []
    is_paired = torch.zeros_like(scales, dtype=torch.bool)

    # Iterate through all possible pairings of unique scales (s_low < s_high)
    for i in range(len(unique_scales)):
        for j in range(i + 1, len(unique_scales)):
            low_scale_val = unique_scales[i]
            high_scale_val = unique_scales[j]

            # Find available items for this pair of scales
            low_indices = torch.where((scales == low_scale_val) & ~is_paired)[0]
            high_indices = torch.where((scales == high_scale_val) & ~is_paired)[0]

            num_new_pairs = min(len(low_indices), len(high_indices))

            if num_new_pairs > 0:
                new_low_indices = low_indices[:num_new_pairs]
                new_high_indices = high_indices[:num_new_pairs]

                for low_idx, high_idx in zip(new_low_indices, new_high_indices):
                    pairs.append((low_idx.item(), high_idx.item()))
                    is_paired[low_idx] = True
                    is_paired[high_idx] = True

    unpaired_indices = torch.where(~is_paired)[0].tolist()
    
    return {"pairs": pairs, "unpaired": unpaired_indices}

def unswizzle_conditioning_data(batch: dict) -> Dict[str, torch.Tensor]:
    """
    Unswizzles conditioning tensors from a batch into a structured dictionary.

    Assumes that text_embeddings and pooled_embeds are concatenated in the order:
    [positive, unconditional, neutral].

    Args:
        batch (dict): The original batch containing 'latents', 'text_embeddings',
                      and 'pooled_embeds'.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing the separated embedding tensors:
            - 'positive_text', 'unconditional_text', 'neutral_text'
            - 'positive_pool', 'unconditional_pool', 'neutral_pool'
    """
    batch_size = batch["latents"].shape[0]

    # Unswizzle from (B*3, ...) to (B, 3, ...)
    text_embeds_split = _unswizzle_tensor(batch["text_embeddings"], batch_size)
    pooled_embeds_split = _unswizzle_tensor(batch["pooled_embeds"], batch_size)

    # Separate into the three conditioning types
    unswizzled = {
        "positive_text": text_embeds_split[:, 0],
        "unconditional_text": text_embeds_split[:, 1],
        "neutral_text": text_embeds_split[:, 2],
        "positive_pool": pooled_embeds_split[:, 0],
        "unconditional_pool": pooled_embeds_split[:, 1],
        "neutral_pool": pooled_embeds_split[:, 2],
    }
    return unswizzled

def form_cfg_microbatch(
    unswizzled_data: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    pairing_map: Dict[str, List[Any]],
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Forms the final microbatch for the UNet based on the pairing map.

    Args:
        unswizzled_data (dict): Output from unswizzle_conditioning_data.
        batch (dict): The original batch, needed for latents, scales, etc.
        pairing_map (dict): Output from create_pairing_map.

    Returns:
        A tuple containing:
        - Dict[str, torch.Tensor]: A dict of CFG-ready tensors for the UNet.
          Keys: 'latents_cfg', 'text_embeds_cfg', 'pooled_embeds_cfg', 'add_time_ids_cfg'.
        - torch.Tensor: A tensor of the scales, reordered to match the microbatch.
    """
    ordered_indices = []
    selected_cond_text = []
    selected_cond_pool = []

    # Process pairs: high-scale item (positive) followed by low-scale item (neutral)
    for low_idx, high_idx in pairing_map["pairs"]:
        ordered_indices.extend([high_idx, low_idx])
        selected_cond_text.append(unswizzled_data["positive_text"][high_idx])
        selected_cond_pool.append(unswizzled_data["positive_pool"][high_idx])
        selected_cond_text.append(unswizzled_data["neutral_text"][low_idx])
        selected_cond_pool.append(unswizzled_data["neutral_pool"][low_idx])

    # Process unpaired items (use positive conditioning)
    for unpaired_idx in pairing_map["unpaired"]:
        ordered_indices.append(unpaired_idx)
        selected_cond_text.append(unswizzled_data["positive_text"][unpaired_idx])
        selected_cond_pool.append(unswizzled_data["positive_pool"][unpaired_idx])

    if not ordered_indices:
        return {}, torch.empty(0)

    # Gather data in the new microbatch order
    ordered_indices = torch.tensor(ordered_indices, device=batch["latents"].device)
    ordered_latents = batch["latents"][ordered_indices]
    ordered_scales = batch["scales"][ordered_indices]
    ordered_add_time_ids = batch["add_time_ids"][ordered_indices]
    ordered_uncond_text = unswizzled_data["unconditional_text"][ordered_indices]
    ordered_uncond_pool = unswizzled_data["unconditional_pool"][ordered_indices]

    # Assemble the final conditional embeddings
    final_cond_text = torch.stack(selected_cond_text, dim=0)
    final_cond_pool = torch.stack(selected_cond_pool, dim=0)

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
    pairing_map: dict
) -> torch.Tensor:
    """
    Calculates the training loss, summing losses for paired items before averaging.

    Args:
        predicted_noise (torch.Tensor): Noise from the UNet (shape 2*N_eff, C, H, W).
        target_noise (torch.Tensor): Ground truth noise, duplicated for CFG.
        pairing_map (dict): The map of paired and unpaired indices.

    Returns:
        torch.Tensor: The final scalar loss for the batch.
    """
    # Calculate MSE loss per-element, then reduce to a per-item scalar
    loss_per_item_cfg = (predicted_noise - target_noise).pow(2).mean(dim=[1, 2, 3])
    
    # Average the loss from the unconditional and conditional forward passes
    uncond_loss, cond_loss = loss_per_item_cfg.chunk(2)
    loss_per_item = (uncond_loss + cond_loss) / 2.0  # Shape: (N_eff,)
    
    num_pairs = len(pairing_map["pairs"])
    if num_pairs == 0:
        return loss_per_item.mean() if loss_per_item.numel() > 0 else torch.tensor(0.0)

    # The first (num_pairs * 2) items in loss_per_item correspond to the pairs
    # Reshape to (num_pairs, 2) where each row is [loss_high, loss_low]
    paired_losses = loss_per_item[:num_pairs * 2].view(num_pairs, 2)
    
    # Sum the losses for each pair to get a single value per pair
    summed_pair_losses = paired_losses.sum(dim=1)
    
    # Get losses for any leftover items
    unpaired_losses = loss_per_item[num_pairs * 2:]
    
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
