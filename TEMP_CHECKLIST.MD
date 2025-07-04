# Checklist for Future Code Improvements

This checklist outlines the next steps for enhancing the training core loop to support batched training and GLORA features.

## 1. Batched Training of Multiple Training Pairs

**Goal:** Enable `train_lora-scale-xl.py` to process multiple image pairs within a single forward pass, leveraging GPU parallelism.

*   [x] **Modify `train_lora-scale-xl.py` (Main Training Loop):**
    *   [ ] Ensure the `dataloader` provides batches of `(img1_batch, scale1_batch), (img2_batch, scale2_batch)` where `img1_batch` and `img2_batch` have a batch dimension `B > 1`.
    *   [ ] Implement logic to select or construct `B` prompt embeddings, one for each pair in the batch.
    *   [ ] Pass these batched prompt embeddings (e.g., `batched_positive_embeds`, `batched_neutral_embeds`) to `superfunctional_train_step`.

*   [x] **Modify `superfunctional_train_step` Function:** (Batching of `get_noisy_image` completed, `predict_noise_xl` modularized but still called sequentially for different LoRA scales)
    *   [x] Adapt the function signature to accept batched inputs for images, scales, and prompt embeddings. (Implicitly handled by existing signature)
    *   [x] Concatenate `img1_batch` and `img2_batch` along the batch dimension for a single UNet input. (Handled by `batch_train_util.get_batched_noisy_images`)
    *   [ ] Perform a single call to `predict_noise_xl` with the concatenated batched inputs. (Still sequential calls for now due to LoRA application; `predict_noise_xl` is now modularized in `batch_train_util.py`)
    *   [ ] Split the output (e.g., `noise_pred`) back into `B` components. (Not applicable as `predict_noise_xl` is sequential)
    *   [ ] Calculate the loss for each pair in the batch and then average or sum them. (Loss calculation is still separate for high/low.)

*Note: The LoRA application within `superfunctional_train_step` is currently sequential (setting slider for high, then low) and `predict_noise_xl` is called twice. True batched LoRA application, which is necessary to fully leverage the modularized `predict_noise_xl` in a single batched call for multiple LoRA scales, requires modifications to `LoRAModule` and `LoRANetwork` to accept a batch of scales. This is a separate task under "Batched LoRA Application" below.*

## 2. GLORA Features (Sigmoid Contrastive Residual Loss & Batched LoRA Application)

**Goal:** Integrate the Sigmoid Contrastive Residual Loss for improved conceptual slider learning and implement batched LoRA application for performance.

*   [ ] **Sigmoid Contrastive Residual Loss:**
    *   [ ] **Residual Extraction:** Modify `LoRANetwork` and `LoRAModule` (or UNet forward pass) to expose intermediate activations (residuals) before and after LoRA application.
    *   [ ] **New Loss Function:** Implement `SigmoidContrastiveLoss` module/function that takes pairs of residual tensors and their "matching" labels.
    *   [ ] **Integration into Training Loop:**
        *   [ ] Extract necessary residuals in `superfunctional_train_step`.
        *   [ ] Form "matched" and "unmatched" pairs of residuals.
        *   [ ] Calculate `SigmoidContrastiveLoss` and add it to the total loss.
    *   [ ] **Learnable `t'` and `b` Terms:** Implement as `nn.Parameter` layers within `SigmoidContrastiveLoss`.
    *   [ ] **Linear Probes for Monitoring:** Add `nn.Linear` layers to predict slider magnitude from residual features and track their accuracy.

*   [x] **Batched LoRA Application:** (Completed)
    *   [x] **`BatchedLoRAModule` Creation:** Created `trainscripts/imagesliders/batch_lora.py` with `BatchedLoRAModule` to handle batched multipliers using tensor operations.
    *   [x] **`BatchedLoRANetwork` Creation:** Created `trainscripts/imagesliders/batch_lora.py` with `BatchedLoRANetwork` to manage `BatchedLoRAModule` instances and accept a batch of scales.
    *   [x] **`superfunctional_train_step` Integration:** Integrated `BatchedLoRANetwork` into `batched_training_loop.py` to enable a single, truly batched `predict_noise_xl` call with multiple LoRA scales.

## 3. Triton Compatibility Improvements

*   [x] **Dynamic Shapes:** Removed `get_random_resolution_in_bucket()` and set `dynamic_resolution: false` in configuration to ensure static shapes for UNet inputs.

## 4. VRAM Optimization

*   [x] **Pre-encode and Cache VAE Latents:** Implemented in `map_data_to_latents.py` to pre-encode and cache VAE latents, reducing VRAM usage during training.