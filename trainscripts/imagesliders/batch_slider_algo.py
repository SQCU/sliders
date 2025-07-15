import torch
from typing import List, Dict, Tuple, Any
from collections import deque
import math

class GradientNoiseEstimator:
    """
    Implements gradient noise scale estimation for serial gradient accumulation.
    
    This class performs a periodic, memory-intensive profiling step to estimate
    the optimal batch size, as described in 'An Empirical Model of Large-Batch Training'.
    """
    def __init__(self, 
                 model, 
                 micro_batch_size, 
                 profile_freq=10, 
                 zoomy_alpha=0.1,
                 ponderous_alpha=0.01,
                 # update_cooldown is now removed from args
                 change_threshold=0.1
                ):
        self.model = model
        self.micro_batch_size = micro_batch_size
        self.profile_freq = profile_freq
       
        self.is_profiling = False
        self._step_count = 0
        
        # --- NEW: Dual EMA state ---
        self.zoomy_alpha = zoomy_alpha
        self.ponderous_alpha = ponderous_alpha
        self.ema_zoomy_b_crit = None
        self.ema_ponderous_b_crit = None
        self.instant_b_crit = None

        # Buffers for the profiling step
        self._grad_sum_buffer = None
        self._micro_norm_sq_values = []

        # --- NEW: Control logic state ---

        self.update_cooldown = math.ceil(1.0 / max(self.zoomy_alpha, self.ponderous_alpha))
        self.change_threshold = change_threshold
        self._profile_steps_since_last_update = 0

    def _get_full_grad_norm_sq(self, use_buffer=False):
        """Calculates the squared L2 norm of the full gradient."""
        norm_sq = torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')
        params_source = self._grad_sum_buffer if use_buffer else [p for p in self.model.parameters() if p.grad is not None]
        
        for p in params_source:
            grad = p if use_buffer else p.grad
            if grad is not None:
                norm_sq += torch.sum(grad.pow(2))
        return norm_sq

    def pre_accumulate_step(self):
        """Call this before starting a gradient accumulation loop."""
        self._step_count +=1
        self.is_profiling = (self._step_count % self.profile_freq == 0)
        
        if self.is_profiling:
            print(f"--- [Step {self._step_count}] Starting gradient noise profiling ---")
            # Allocate buffer only when needed
            self._grad_sum_buffer = [torch.zeros_like(p.data) for p in self.model.parameters()]
            self._micro_norm_sq_values = []

    def post_micro_backward_step(self):
        """Call this after each micro-batch's backward() call."""
        if self.is_profiling:
            # 1. Calculate and store the squared norm of the micro-batch gradient
            micro_norm_sq = self._get_full_grad_norm_sq(use_buffer=False)
            self._micro_norm_sq_values.append(micro_norm_sq.item())
            
            # 2. Manually accumulate the gradient into our buffer
            with torch.no_grad():
                for p, p_sum in zip(self.model.parameters(), self._grad_sum_buffer):
                    if p.grad is not None:
                        p_sum.add_(p.grad)
            
            # 3. CRITICAL: Zero out the model's grad so the next backward() call is clean
            self.model.zero_grad()


    def post_accumulate_step(self, accumulation_steps):
        """Call this after the full accumulation loop, before optimizer.step()."""
        if self.is_profiling:
            # --- Finalize Profiling ---
            
            # 1. Load the accumulated gradient from our buffer back into the model
            with torch.no_grad():
                for p, p_sum in zip(self.model.parameters(), self._grad_sum_buffer):
                    p.grad = p_sum
            
            # 2. Calculate the statistics
            # We have the sum of gradients, so G_macro = (1/N) * sum(g_micro)
            # ||G_macro||^2 = (1/N^2) * ||sum(g_micro)||^2
            macro_norm_sq = self._get_full_grad_norm_sq(use_buffer=False) / (accumulation_steps ** 2)
            
            # E[||g_micro||^2]
            if self._micro_norm_sq_values:
                mean_micro_norm_sq = sum(self._micro_norm_sq_values) / len(self._micro_norm_sq_values)
            else:
                mean_micro_norm_sq = 0.0

            if macro_norm_sq > 1e-8: # Avoid division by zero
                b_crit = self.micro_batch_size * (mean_micro_norm_sq / macro_norm_sq.item() - 1)
                self.instant_b_crit = max(0, b_crit) # Clamp at 0

                    # --- NEW: Update dual EMAs ---
                if self.ema_zoomy_b_crit is None:
                    self.ema_zoomy_b_crit = self.instant_b_crit
                    self.ema_ponderous_b_crit = self.instant_b_crit
                else:
                    self.ema_zoomy_b_crit = self.zoomy_alpha * self.instant_b_crit + (1 - self.zoomy_alpha) * self.ema_zoomy_b_crit
                    self.ema_ponderous_b_crit = self.ponderous_alpha * self.instant_b_crit + (1 - self.ponderous_alpha) * self.ema_ponderous_b_crit

                print(f"--- [Step {self._step_count}] Profiling Results ---")
                print(f"    Mean micro-grad norm^2: {mean_micro_norm_sq:.4f}")
                print(f"    Macro-grad norm^2:      {macro_norm_sq.item():.4f}")
                print(f"    Noise/Signal Ratio:     {(mean_micro_norm_sq / macro_norm_sq.item()):.4f}")
                print(f"    Instant B_crit: {self.instant_b_crit:.2f}")
                print(f"    Zoomy EMA B_crit (α={self.zoomy_alpha}): {self.ema_zoomy_b_crit:.2f}")
                print(f"    Ponderous EMA B_crit (α={self.ponderous_alpha}): {self.ema_ponderous_b_crit:.2f}")

            # Clean up memory
            self._grad_sum_buffer = None
            self._micro_norm_sq_values = []
            self.is_profiling = False

    def propose_new_accumulation_steps(self, current_steps: int, min_steps: int, max_steps: int) -> int:
        """
        Proposes a new number of accumulation steps based on EMA trends.
        Call this *after* post_accumulate_step during a profiling step.
        """
        self._profile_steps_since_last_update += 1
        if self.ema_ponderous_b_crit is None or self._profile_steps_since_last_update < self.update_cooldown:
            return current_steps

        self.current_steps = current_steps
        current_effective_batch_size = self.micro_batch_size * current_steps
        
        # Trend confirmation from both EMAs
        suggests_increase = self.ema_zoomy_b_crit * self.current_steps > self.current_steps and \
                            self.ema_ponderous_b_crit * self.current_steps > self.current_steps
        
        suggests_decrease = self.ema_zoomy_b_crit * self.current_steps < self.current_steps and \
                            self.ema_ponderous_b_crit * self.current_steps < self.current_steps

        # Use the stable (ponderous) EMA to determine the target
        # Add micro_batch_size to avoid division by zero or tiny values
        target_steps = math.ceil(self.ema_ponderous_b_crit * self.current_steps)
        
        # accelerando: only 125% increase per update:
        ratio = 1.25
        if abs(target_steps) / (current_steps + 1e-6) > ratio:
            target_steps = math.ceil(ratio * self.current_steps)
            print(f"RATIO BLOCKED! target steps above scaling ratio {ratio} folded to {ratio * self.current_steps}")

        # Hysteresis: Only act if the proposed change is significant
        if abs(target_steps - current_steps) / (current_steps + 1e-6) < self.change_threshold:
             return current_steps

        new_steps = current_steps
        if suggests_increase:
            print(f"📈 [Dynamic Accumulation] Trend suggests INCREASE. Current Eff. Batch: {current_effective_batch_size:.1f}, Ponderous B_crit: {self.ema_ponderous_b_crit*current_effective_batch_size:.1f}")
            new_steps = target_steps
        elif suggests_decrease:
            print(f"📉 [Dynamic Accumulation] Trend suggests DECREASE. Current Eff. Batch: {current_effective_batch_size:.1f}, Ponderous B_crit: {self.ema_ponderous_b_crit*current_effective_batch_size:.1f}")
            new_steps = target_steps
        else:
            # Trends disagree, hold steady
            return current_steps

        # Clamp and finalize
        new_steps = max(min_steps, min(max_steps, new_steps))

        if new_steps != current_steps:
            print(f"✅ [Dynamic Accumulation] Proposing change from {current_steps} to {new_steps} steps.")
            self._profile_steps_since_last_update = 0 # Reset cooldown
            return new_steps
        
        return current_steps


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
    #fine, fine, we can put the absurd stuff in the siglip auxiliary losss
    loss = torch.nn.functional.mse_loss(predicted_noise.float(), target_noise.float(), reduction="mean")
    return loss


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
