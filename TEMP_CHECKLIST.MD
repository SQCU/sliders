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

## 2. GLORA Integration

**Goal:** Integrate the GLORA framework, as detailed in `GLORA_NOTES.MD` and tracked in `GLORA_CHECKLIST.MD`, for enhanced conceptual slider learning.

*   [ ] **GLORA Framework Integration:** Refer to `GLORA_CHECKLIST.MD` for detailed tasks related to Sigmoid Contrastive Residual Loss and other GLORA components.
    *   **Status:** This integration can proceed now that the core batching fixes, including the Batched LoRA Application, are in place. The batched infrastructure provides a suitable foundation for implementing GLORA's requirements, such as exposing intermediate activations and handling batched data for contrastive loss calculations.

*   [x] **Batched LoRA Application:** (Completed - now considered part of core batching fixes, not solely a GLORA feature)
    *   [x] **`BatchedLoRAModule` Creation:** Created `trainscripts/imagesliders/batch_lora.py` with `BatchedLoRAModule` to handle batched multipliers using tensor operations.
    *   [x] **`BatchedLoRANetwork` Creation:** Created `trainscripts/imagesliders/batch_lora.py` with `BatchedLoRANetwork` to manage `BatchedLoRAModule` instances and accept a batch of scales.
    *   [x] **`superfunctional_train_step` Integration:** Integrated `BatchedLoRANetwork` into `batched_training_loop.py` to enable a single, truly batched `predict_noise_xl` call with multiple LoRA scales.

## 5. Torch Compile and SDPA Integration

**Goal:** Optimize training performance using `torch.compile` and Scaled Dot Product Attention (SDPA) with Flash Attention.

*   [ ] **Enable SDPA Backends:**
    *   [ ] Add necessary imports: `from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp`
    *   [ ] Call `enable_flash_sdp(True)` (and potentially others like `enable_cudnn_sdp(True)`) at the start of the script, before any model is loaded or compiled.
    *   **Rationale:** These calls enable PyTorch to use optimized attention kernels where available, with Flash Attention being particularly fast.

*   [ ] **`torch.compile` for UNet:**
    *   [ ] Identify the point where the `unet` model is instantiated.
    *   [ ] Apply `torch.compile` to the `unet` instance: `compiled_unet = torch.compile(unet_instance)`.
    *   [ ] Ensure the compiled `unet` is passed to `BatchedLoRANetwork` and `superfunctional_train_step`.
    *   **Rationale:** The UNet is the most computationally intensive part of the diffusion model, and compiling it can yield significant speedups.

*   [ ] **`torch.compile` for VAE (Latent Precomputation Stage):**
    *   [ ] Identify where the `vae` model is instantiated.
    *   [ ] Apply `torch.compile` to the `vae` instance: `compiled_vae = torch.compile(vae_instance)`.
    *   [ ] Ensure the compiled `vae` is used in `batch_train_util.get_batched_noisy_images`.
    *   **Rationale:** If VAE encoding is a bottleneck during data loading or precomputation, compiling it can improve overall throughput. This is especially relevant if VAE latents are not fully cached.

*   [ ] **`torch.compile` for Text Conditioning Models (CLIPTextModel, CLIPTokenizer):**
    *   [ ] Identify where the `text_encoders` (CLIPTextModel) are instantiated.
    *   [ ] Apply `torch.compile` to each `text_encoder` instance: `compiled_text_encoder = torch.compile(text_encoder_instance)`.
    *   [ ] Ensure the compiled `text_encoders` are used in `train_util.encode_prompts_xl`.
    *   **Rationale:** Similar to VAE, if text encoding is a bottleneck, compiling these models can offer speedups.

*   [ ] **Text Encoder Call Analysis and Optimization:**
    *   [ ] **Identify Call Locations:** Pinpoint exactly where and when the text encoders are invoked within the training loop (e.g., during data loading, prompt processing, or within `superfunctional_train_step`).
    *   [ ] **Usage Pattern Analysis:** Determine if text encoders are being loaded and intermittently used, or if their outputs are pre-computed and cached.
    *   [ ] **Ensure Detached/NoGrad:** If text encoders are called during the training forward pass (i.e., not just for pre-computation), ensure they are run within `torch.no_grad()` context and/or their outputs are `.detach()`ed to prevent unnecessary gradient computation.
    *   [ ] **Caching Strategy:** In the optimal case, explore caching text encoder outputs (embeddings) alongside VAE encodings during the latent precomputation stage to avoid re-computation during training.
    *   **Rationale:** Text encoding can be computationally expensive. Avoiding redundant computations and unnecessary gradient tracking is crucial for performance and memory efficiency.

*   [ ] **Testing and Benchmarking:**
    *   [ ] Run benchmarks with and without `torch.compile` for each component to quantify performance gains.
    *   [ ] Monitor VRAM usage, as compilation can sometimes increase memory footprint.
    *   [ ] Verify correctness after compilation (e.g., ensure training loss behaves as expected).
    *   **Rationale:** Compilation introduces overhead and can sometimes lead to unexpected behavior or increased memory usage. Thorough testing is crucial.

## 6. Triton Compatibility Improvements

**Goal:** Integrate the SIGLIP framework, as detailed in `SIGLIP_NOTES.MD` and tracked in `SIGLIP_CHECKLIST.MD`, for improved image-text alignment.

*   [ ] **SIGLIP Framework Integration:** Refer to `SIGLIP_CHECKLIST.MD` for detailed tasks related to SIGLIP components.
    *   **Status:** This integration can proceed independently of the core batching fixes, as it primarily involves changes to the loss function and potentially the data pipeline, rather than the core UNet forward pass or LoRA application. It can be pursued in parallel or after the batching work.

## 4. Triton Compatibility Improvements

*   [x] **Dynamic Shapes:** Removed `get_random_resolution_in_bucket()` and set `dynamic_resolution: false` in configuration to ensure static shapes for UNet inputs.

## 4. VRAM Optimization

*   [x] **Pre-encode and Cache VAE Latents:** Implemented in `map_data_to_latents.py` to pre-encode and cache VAE latents, reducing VRAM usage during training.