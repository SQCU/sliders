# Data Flow in `batched_training_loop.py`

This document outlines the current data flow within `batched_training_loop.py`, focusing on dataset construction, noise generation, CFG guidance, and identifying potential areas for simplification and resolving tensor shape mismatches.

## Overall Training Loop (`main` function)

1.  **Configuration and Environment Setup:**
    *   `config = config_io()`: Loads training configuration from a YAML file.
    *   `environment = config_envsetup(config)`: Initializes the training environment, including devices (CPU/GPU), models (UNet, VAE, Text Encoders), noise scheduler, etc.

2.  **VRAM Management (Manual Model Offloading/Loading):**
    *   UNet is moved to CPU (`tdcpu`) before batch preparation to free VRAM.
    *   `prepare_cached_batches()` is called.
    *   VAE and Text Encoders are moved to CPU after batch preparation.
    *   UNet is moved back to GPU for the main training loop.
    *   (Optional) UNet compilation with `torch.compile`.

3.  **Network and Optimizer Setup:**
    *   `network = lora.BatchedLoRANetwork(...)`: Initializes the LoRA network, attaching it to the UNet.
    *   Optimizer and learning rate scheduler (`lr_scheduler`) are configured based on the loaded `config`.

4.  **Training Execution:**
    *   `training_loop(environment, static_batches)`: Initiates the main training process.

5.  **Model Saving:**
    *   `graceful_shutdown(environment)`: Saves the trained LoRA network weights.

## Dataset Construction (`prepare_cached_batches`)

This function is responsible for pre-generating and caching all training batches based on a defined schedule and pre-encoded embeddings.

1.  **Schedule and Cache Initialization:**
    *   `training_schedule = TrainingSchedule(config)`: An object that dictates the order and composition of training items within batches. This is a critical component for understanding batch structure.
    *   `unified_cache = build_unified_embedding_cache(config, environment, training_schedule)`: This function populates two main caches:
        *   `image_latents_cache`: Stores pre-encoded latent representations of images.
        *   `text_embeddings_cache`: Stores pre-encoded text embeddings for various prompts (positive, unconditional, neutral). The keys are `frozenset(item.prompt.items())`.

2.  **Batch Assembly Loop:**
    *   The function iterates through `training_schedule` to create individual batches.
    *   For each `item` in a `batch_items` (from `training_schedule`):
        *   `latent = image_latents_cache[item.image_path]`: Retrieves the image latent.
        *   `scales.append(item.scale)`: Appends the associated scale.
        *   `pair_indices.append(item.pair_index)` and `is_low_cases.append(item.is_low_case)`: These are used for the `calculate_paired_loss` function and determine the "pairing" or "scale-tuple-grouping" logic.
        *   **Prompt Swizzling (Area for Simplification):**
            *   Text embeddings (positive, unconditional, neutral) are retrieved from `text_embeddings_cache`.
            *   `if item.is_low_case:`: This conditional logic *selects* either `neutral_text_embeds` or `positive_text_embeds` to be the `cond_text_embeddings` for the current item. This is the "swizzling" mentioned in the problem description, where one of the two conditional cases is chosen upfront.
            *   `all_uncond_text_embeddings` and `all_uncond_pooled_embeds` are always appended.
    *   **Batch Tensor Creation:**
        *   Individual latents, scales, and embeddings are concatenated using `torch.cat` to form batch tensors (`latents_batch`, `scales_batch`, `cond_text_embeddings_batch`, etc.).
        *   `add_time_ids`: Hardcoded for 1024x1024 resolution, repeated for the batch size.
        *   A `batch` dictionary is created containing all these tensors, along with `pair_indices`, `is_low_cases`, and `guidance_scale`.

## Training Step (`train_step`)

This function performs a single optimization step for a given batch.

1.  **Data Transfer to Device:**
    *   All relevant tensors from the `batch` dictionary are moved to the specified `device` (GPU) and `weight_dtype`.

2.  **Noise Generation:**
    *   `noise_scheduler.set_timesteps(config.train.max_denoising_steps, device=device)`: Sets the timesteps for the noise scheduler.
    *   `timesteps_to = torch.randint(...)`: A random timestep is generated for *each item* in the batch. The shape is `(batch_size,)`.
    *   `noise = torch.randn(latents.shape, ..., generator=generator)`: Noise is generated with the same shape as the input `latents` (i.e., `(batch_size, channels, height, width)`).
    *   `noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps_to)`: Noise is added to the original latents.

3.  **CFG Microbatch Preparation (`prepare_cfg_batch`):**
    *   This function takes the `noisy_latents`, `timesteps_to`, and the conditional/unconditional embeddings from the `batch`.
    *   It concatenates the unconditional and conditional versions of `noisy_latents`, `text_embeddings`, `pooled_embeds`, `add_time_ids`, and `unet_timesteps` along the batch dimension (`dim=0`).
    *   The output tensors (e.g., `latents_cfg`, `text_embeddings_cfg`) will have a batch size of `2 * original_batch_size`.

4.  **LoRA Scale Setting:**
    *   `batched_scales_cfg = torch.cat([scales, scales], dim=0).unsqueeze(-1)`: The `scales` tensor is doubled along the batch dimension to match the CFG microbatch size, and then unsqueezed for broadcasting.
    *   `network.set_lora_scales(batched_scales_cfg)`: Sets the LoRA scales for the network.

5.  **UNet Forward Pass:**
    *   `predicted_noise = batch_train_util.batched_predict_noise_xl(...)`: The UNet performs a forward pass with the CFG-prepared inputs. The output `predicted_noise` will have a batch size of `2 * original_batch_size`.

6.  **Post-CFG Guidance:**
    *   `predicted_noise_uncond, predicted_noise_text = predicted_noise.chunk(2)`: The UNet output is split back into unconditional and conditional parts. Each chunk has the `original_batch_size`.
    *   `predicted_noise_cfg_reduced = (predicted_noise_uncond + guidance_scale * (predicted_noise_text - predicted_noise_uncond)).to(device)`: The CFG formula is applied to combine the unconditional and conditional predictions. The resulting `predicted_noise_cfg_reduced` tensor has the `original_batch_size`.

7.  **Loss Calculation:**
    *   `loss = calculate_paired_loss(predicted_noise_cfg_reduced, noise, pair_indices, is_low_cases)`: The loss is calculated between the `predicted_noise_cfg_reduced` and the original `noise`.
        *   **Tensor Shape Check:** At this point, `predicted_noise_cfg_reduced.shape` should be `(batch_size, channels, height, width)` and `noise.shape` is also `(batch_size, channels, height, width)`. If there's a mismatch, it's likely due to an error in an intermediate step or how `calculate_paired_loss` expects its inputs. The current code structure suggests they should match.

8.  **Backpropagation and Optimization:**
    *   `loss.backward()`: Computes gradients.
    *   `optimizer.step()`: Updates model weights.
    *   `lr_scheduler.step()`: Updates the learning rate.

## CFG Batch Preparation (`prepare_cfg_batch`)

This helper function prepares the inputs for the UNet's CFG pass.

1.  **Input Extraction:** Extracts conditional and unconditional embeddings, `noisy_latents`, `timesteps_to`, and `add_time_ids` from the provided arguments.
2.  **Concatenation:** Concatenates all these inputs along the batch dimension (`dim=0`) to create tensors that are twice the original batch size. This allows for a single forward pass through the UNet for both unconditional and conditional predictions.

## Areas for Simplification and Improvement

The primary area for simplification lies in the "swizzling" of text embeddings during batch preparation and the overall complexity introduced by the "dataset scale-tuple-grouping" (referred to as "pairs").

### Current "Swizzling" Problem:

In `prepare_cached_batches`, the `if item.is_low_case:` block pre-selects either `neutral_text_embeds` or `positive_text_embeds` as the `cond_text_embeddings` for a given batch item. This means:
*   The `text_embeddings_cache` already contains all three types of embeddings (positive, unconditional, neutral).
*   We are effectively discarding one of the two conditional embeddings (positive or neutral) *before* the training loop even begins for each batch item.
*   This pre-selection adds complexity to `prepare_cached_batches` and makes the data flow less transparent.

### Proposed Simplification:

Instead of pre-swizzling, we should aim to carry *all necessary* embeddings (positive, neutral, unconditional) for each batch item through the `batch` dictionary. Then, the selection of the *actual* conditional embedding for the UNet input can happen dynamically *just before* the `prepare_cfg_batch` call in `train_step`.

**Revised Data Flow for Prompt Handling:**

1.  **`prepare_cached_batches`:**
    *   When populating the `batch` dictionary, instead of:
        ```python
        if item.is_low_case:
            all_cond_text_embeddings.append(neutral_text_embeds)
            all_cond_pooled_embeds.append(neutral_pooled_embeds)
        else:
            all_cond_text_embeddings.append(positive_text_embeds)
            all_cond_pooled_embeds.append(positive_pooled_embeds)
        ```
    *   Store all three relevant embeddings for each item:
        ```python
        batch = {
            # ... other items ...
            "positive_text_embeddings": positive_text_embeds,
            "positive_pooled_embeds": positive_pooled_embeds,
            "neutral_text_embeddings": neutral_text_embeds,
            "neutral_pooled_embeds": neutral_pooled_embeds,
            "uncond_text_embeddings": uncond_text_embeds,
            "uncond_pooled_embeds": uncond_pooled_embeds,
            "is_low_cases": torch.tensor(is_low_cases, dtype=torch.bool, device=device), # Keep this
            # ...
        }
        ```
    *   This means `cond_text_embeddings_batch` and `cond_pooled_embeds_batch` would no longer be directly created in `prepare_cached_batches`.

2.  **`train_step`:**
    *   Before calling `prepare_cfg_batch`, dynamically select the conditional embeddings based on `is_low_cases`:
        ```python
        # Select the appropriate conditional embeddings based on is_low_cases
        # This will create tensors of shape (batch_size, ...)
        selected_cond_text_embeddings = torch.where(
            batch['is_low_cases'].unsqueeze(-1).unsqueeze(-1), # Expand dims to match embedding shape
            batch['neutral_text_embeddings'],
            batch['positive_text_embeddings']
        )
        selected_cond_pooled_embeds = torch.where(
            batch['is_low_cases'].unsqueeze(-1), # Expand dims to match pooled embedding shape
            batch['neutral_pooled_embeds'],
            batch['positive_pooled_embeds']
        )

        # Now pass these selected embeddings to prepare_cfg_batch
        latents_cfg, text_embeddings_cfg, pooled_embeds_cfg, add_time_ids_cfg, unet_timesteps_cfg = prepare_cfg_batch(
            batch, # Pass the original batch for other items
            noisy_latents,
            timesteps_to,
            noise_scheduler,
            weight_dtype,
            selected_cond_text_embeddings, # New argument
            selected_cond_pooled_embeds, # New argument
        )
        ```
    *   This would require modifying `prepare_cfg_batch` to accept these new arguments instead of extracting them from the `batch` dictionary.

### Benefits of Simplification:

*   **Reduced Code Complexity:** Eliminates the conditional logic for prompt selection within `prepare_cached_batches`, making it cleaner.
*   **Increased Transparency:** The decision of which conditional prompt to use is made explicitly at the point of use (`train_step`), rather than being hidden within batch construction.
*   **Reduced LoC:** Potentially reduces lines of code by removing redundant logic.
*   **Clearer Data Flow:** Makes it easier to trace how conditional prompts are handled.

### Addressing Tensor Shape Mismatch:

The current code structure for `noise` and `predicted_noise_cfg_reduced` suggests they should have matching shapes (`(batch_size, channels, height, width)`). The `print` statements added in `train_step` are crucial for debugging this. If a mismatch occurs, it's likely due to:
1.  An unexpected change in batch size or tensor dimensions during an intermediate operation.
2.  An issue within `calculate_paired_loss` if it expects a different shape or order.

The proposed simplification of prompt handling should not directly impact the noise or predicted noise shapes, but by making the code cleaner, it might expose other underlying issues more clearly.

## Function Call Chain (Simplified View)

`main()`
  -> `prepare_cached_batches()` (Dataset Construction, including simplified prompt handling)
  -> `training_loop()`
    -> `train_step()`
      -> Noise Generation (`torch.randn`, `noise_scheduler.add_noise`)
      -> Dynamic Conditional Prompt Selection (new step)
      -> `prepare_cfg_batch()` (CFG Microbatch Formation)
      -> `batch_train_util.batched_predict_noise_xl()` (UNet Forward Pass)
      -> Post-CFG Guidance (Chunking and CFG formula application)
      -> `calculate_paired_loss()` (Loss Calculation)
      -> Backpropagation & Optimization
