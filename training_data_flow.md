# Data Flow in `batched_training_loop.py` (Refactored)

This document outlines the refactored data flow within `batched_training_loop.py`, focusing on dataset construction, noise generation, and CFG guidance, aiming for simplification and resolving tensor shape mismatches.

## Overall Training Loop (`main` function)

1.  **Configuration and Environment Setup (`main` and `config_envsetup`):**
    *   `config = config_io()`: Loads training configuration from a YAML file.
    *   `environment = config_envsetup(config)`: Initializes the training environment, including devices (CPU/GPU), models (UNet, VAE, Text Encoders), noise scheduler, etc. **Crucially, the `torch.Generator` for noise generation is now created and seeded here, and added to the `environment` dictionary.**

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

This function is responsible for pre-generating and caching all training batches based on a defined schedule and pre-encoded embeddings. It now performs the "swizzling" to create CFG-ready embedding tensors.

1.  **Schedule and Cache Initialization:**
    *   `training_schedule = TrainingSchedule(config)`: An object that dictates the order and composition of training items within batches.
    *   `unified_cache = build_unified_embedding_cache(config, environment, training_schedule)`: This function populates two main caches:
        *   `image_latents_cache`: Stores pre-encoded latent representations of images.
        *   `text_embeddings_cache`: Stores pre-encoded text embeddings for various prompts (positive, unconditional, neutral). The keys are `frozenset(item.prompt.items())`.

2.  **Batch Assembly Loop (with CFG-ready embeddings):**
    *   The function iterates through `training_schedule` to create individual batches.
    *   For each `item` in a `batch_items` (from `training_schedule`):
        *   `latent = image_latents_cache[item.image_path]`: Retrieves the image latent.
        *   `scales.append(item.scale)`: Appends the associated scale.
        *   `pair_indices.append(item.pair_index)` and `is_low_cases.append(item.is_low_case)`: These are used for the `calculate_paired_loss` function.
        *   **Conditional Embedding Selection (Swizzling):**
            *   Text embeddings (positive, unconditional, neutral) are retrieved from `text_embeddings_cache`.
            *   `if item.is_low_case:`: This conditional logic *selects* either `neutral_text_embeds` or `positive_text_embeds` to be the `selected_cond_text_embeds` for the current item.
        *   **CFG-Ready Embedding Concatenation:**
            *   `cfg_text_embeds_for_item = torch.cat([uncond_text_embeds, selected_cond_text_embeds], dim=0)`: Concatenates the unconditional and selected conditional text embeddings for the current item. The order is `[unconditional, selected_conditional]`.
            *   `cfg_pooled_embeds_for_item = torch.cat([uncond_pooled_embeds, selected_cond_pooled_embeds], dim=0)`: Similar concatenation for pooled embeddings.
            *   These `cfg_text_embeds_for_item` and `cfg_pooled_embeds_for_item` are appended to `all_cfg_text_embeddings` and `all_cfg_pooled_embeds` lists, respectively.
    *   **Batch Tensor Creation:**
        *   Individual latents, scales, and the CFG-ready embeddings are concatenated using `torch.cat` to form batch tensors (`latents_batch`, `scales_batch`, `cfg_text_embeddings_batch`, `cfg_pooled_embeds_batch`).
        *   `cfg_text_embeddings_batch` and `cfg_pooled_embeds_batch` will have a shape of `(batch_size * 2, ...)`.
        *   `add_time_ids`: Hardcoded for 1024x1024 resolution, repeated for the batch size.
        *   A `batch` dictionary is created containing all these tensors, along with `pair_indices`, `is_low_cases`, and `guidance_scale`.

## Training Step (`train_step`)

This function performs a single optimization step for a given batch.

1.  **Batch Rectification (`rectify_batch_fn`):**
    *   `batch = rectify_batch_fn(batch, device, weight_dtype)`: A new helper function that moves all relevant tensors from the `batch` dictionary to the specified `device` (GPU) and `weight_dtype`.

2.  **Data Unpacking:**
    *   All necessary tensors (`latents`, `scales`, `pair_indices`, `is_low_cases`, `guidance_scale`, `cfg_text_embeddings`, `cfg_pooled_embeds`, `add_time_ids`) are unpacked from the `batch` dictionary.

3.  **CFG Embedding Splitting:**
    *   `uncond_text_embeddings, cond_text_embeddings = cfg_text_embeddings.chunk(2)`: The CFG-ready text embeddings are split back into their unconditional and conditional parts.
    *   `uncond_pooled_embeds, cond_pooled_embeds = cfg_pooled_embeds.chunk(2)`: Similar splitting for pooled embeddings.

4.  **Noise Generation:**
    *   `noise_scheduler.set_timesteps(config.train.max_denoising_steps, device=device)`: Sets the timesteps for the noise scheduler.
    *   `generator = environment["generator"]`: Retrieves the pre-initialized `torch.Generator` from the environment.
    *   `timesteps_to = torch.randint(...)`: A random timestep is generated for *each item* in the batch using the shared `generator`. The shape is `(batch_size,)`.
    *   `noise = torch.randn(latents.shape, ..., generator=generator)`: Noise is generated with the same shape as the input `latents` (i.e., `(batch_size, channels, height, width)`) using the shared `generator`.
    *   `noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps_to)`: Noise is added to the original latents.

5.  **CFG Microbatch Preparation (`prepare_cfg_batch`):**
    *   This function now directly accepts the `noisy_latents`, `timesteps_to`, `noise_scheduler`, `weight_dtype`, and the *split* unconditional and conditional embeddings (`uncond_text_embeddings`, `cond_text_embeddings`, `uncond_pooled_embeds`, `cond_pooled_embeds`, `add_time_ids`).
    *   It concatenates these inputs along the batch dimension (`dim=0`) to create tensors that are twice the original batch size, ready for the UNet's CFG pass.

6.  **LoRA Scale Setting:**
    *   `batched_scales_cfg = torch.cat([scales, scales], dim=0).unsqueeze(-1)`: The `scales` tensor is doubled along the batch dimension to match the CFG microbatch size, and then unsqueezed for broadcasting.
    *   `network.set_lora_scales(batched_scales_cfg)`: Sets the LoRA scales for the network.

7.  **UNet Forward Pass:**
    *   `predicted_noise = batch_train_util.batched_predict_noise_xl(...)`: The UNet performs a forward pass with the CFG-prepared inputs. The output `predicted_noise` will have a batch size of `2 * original_batch_size`.

8.  **Post-CFG Guidance:**
    *   `predicted_noise_uncond, predicted_noise_text = predicted_noise.chunk(2)`: The UNet output is split back into unconditional and conditional parts. Each chunk has the `original_batch_size`.
    *   `predicted_noise_cfg_reduced = (predicted_noise_uncond + guidance_scale * (predicted_noise_text - predicted_noise_uncond)).to(device)`: The CFG formula is applied to combine the unconditional and conditional predictions. The resulting `predicted_noise_cfg_reduced` tensor has the `original_batch_size`.

9.  **Loss Calculation:**
    *   `loss = calculate_paired_loss(predicted_noise_cfg_reduced, noise, pair_indices, is_low_cases)`: The loss is calculated between the `predicted_noise_cfg_reduced` and the original `noise`.

10. **Backpropagation and Optimization:**
    *   `loss.backward()`: Computes gradients.
    *   `optimizer.step()`: Updates model weights.
    *   `lr_scheduler.step()`: Updates the learning rate.

## CFG Batch Preparation (`prepare_cfg_batch`)

This helper function now directly accepts the necessary components for CFG, simplifying its role:

1.  **Input Acceptance:** Directly takes `noisy_latents`, `timesteps_to`, `noise_scheduler`, `weight_dtype`, `uncond_text_embeddings`, `cond_text_embeddings`, `uncond_pooled_embeds`, `cond_pooled_embeds`, and `add_time_ids` as arguments.
2.  **Concatenation:** Concatenates all these inputs along the batch dimension (`dim=0`) to create tensors that are twice the original batch size. This allows for a single forward pass through the UNet for both unconditional and conditional predictions.

## Areas for Simplification and Improvement (Addressed)

The primary area for simplification, the "swizzling" of text embeddings and the verbose handling of named variables, has been addressed by:

*   **Consolidating Conditional Embedding Selection:** The selection of positive or neutral conditional embeddings is now handled within `prepare_cached_batches`, creating a single CFG-ready embedding tensor for each batch item.
*   **Tensor-based Operations:** The use of `torch.cat` and `chunk` operations for handling unconditional and conditional embeddings streamlines the data flow and reduces the need for numerous named variables.
*   **`rectify_batch_fn`:** Isolates the device and dtype transfer logic, keeping `train_step` cleaner and more focused on the core training logic.
*   **Centralized `torch.Generator`:** Moving the `torch.Generator` creation and seeding to `config_envsetup` ensures consistent and reproducible noise generation across the entire training process.