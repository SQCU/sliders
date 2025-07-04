# Checklist for Future Code Improvements

This checklist outlines the next steps for enhancing the training core loop to support batched training and GLORA features.

## 1. Batched Training of Multiple Training Pairs

**Goal:** Enable `train_lora-scale-xl.py` to process multiple image pairs within a single forward pass, leveraging GPU parallelism.

*   [x] **Batched LoRA Application:** (Completed - now considered part of core batching fixes)
    *   [x] **`BatchedLoRAModule` Creation:** Created `trainscripts/imagesliders/batch_lora.py` with `BatchedLoRAModule` to handle batched multipliers using tensor operations.
    *   [x] **`BatchedLoRANetwork` Creation:** Created `trainscripts/imagesliders/batch_lora.py` with `BatchedLoRANetwork` to manage `BatchedLoRAModule` instances and accept a batch of scales.
    *   [x] **`superfunctional_train_step` Integration:** Integrated `BatchedLoRANetwork` into `batched_training_loop.py` to enable a single, truly batched `predict_noise_xl` call with multiple LoRA scales.
*   [] **Review batched_lora_testing_checklist.md** Refer to batched_lora_testing_checklist.md for detailed tasks related to batchifying our training loop, in terms of using concurrent calls where the original trainer used serial blocking calls, and also in terms of allowing training over multiple high/low scale pairs per training iteration, a curious absence from the original work.

## 2. Torch Compile and SDPA Integration

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

## 3. GLORA Integration

**Goal:** Integrate the GLORA framework, as detailed in `GLORA_NOTES.MD` and tracked in `GLORA_CHECKLIST.MD`, for enhanced conceptual slider learning.

*   [ ] **GLORA Framework Integration:** Refer to `GLORA_CHECKLIST.MD` for detailed tasks related to Sigmoid Contrastive Residual Loss and other GLORA components.
    *   **Status:** This integration can proceed now that the core batching fixes, including the Batched LoRA Application, are in place. The batched infrastructure provides a suitable foundation for implementing GLORA's requirements, such as exposing intermediate activations and handling batched data for contrastive loss calculations.

## 4. SIGLIP??

**Goal:** Integrate the SIGLIP framework, as detailed in `SIGLIP_NOTES.MD` and tracked in `SIGLIP_CHECKLIST.MD`, for improved image-text alignment.

*   [ ] **SIGLIP Framework Integration:** Refer to `SIGLIP_CHECKLIST.MD` for detailed tasks related to SIGLIP components.
    *   **Status:** This integration can proceed independently of the core batching fixes, as it primarily involves changes to the loss function and potentially the data pipeline, rather than the core UNet forward pass or LoRA application. It can be pursued in parallel or after the batching work.

## Done. Triton Compatibility Improvements

*   [x] **Dynamic Shapes:** Removed `get_random_resolution_in_bucket()` and set `dynamic_resolution: false` in configuration to ensure static shapes for UNet inputs.

## Done. VRAM Optimization

*   [x] **Pre-encode and Cache VAE Latents:** Implemented in `map_data_to_latents.py` to pre-encode and cache VAE latents, reducing VRAM usage during training.