# Batched LoRA Testing Expectations

This document outlines the expected behavior when running a test data sample of 3 image pairs through the `batched_training_loop.py` and compares its expected output to an accumulated run of 3 image pairs using `refactored_training_loop.py`.

## Test Setup

*   **Test Data Sample:** 3 distinct image pairs (e.g., `(img_high_1, img_low_1)`, `(img_high_2, img_low_2)`, `(img_high_3, img_low_3)`).
*   **LoRA Scales:** Each image pair will have its own associated LoRA scales. For instance, `(scale_high_1, scale_low_1)`, `(scale_high_2, scale_low_2)`, `(scale_high_3, scale_low_3)`. These scales can be arbitrary, demonstrating the multi-scale capability.
*   **Batched Trainer:** `batched_training_loop.py` will process these 3 image pairs in a single batch. This means the `img_batches` input to `superfunctional_train_step` will contain 6 images (3 high, 3 low), and the `scales` input will be a tensor of 6 corresponding LoRA scales.
*   **Reference Trainer:** `refactored_training_loop.py` will process each of the 3 image pairs sequentially.

## Expected Behavior of `batched_training_loop.py`

When `batched_training_loop.py` processes the 3 image pairs in a single batch:

1.  **`get_batched_noisy_images`:** This function will receive the 6 images (3 high, 3 low) and generate noise for all of them in a single batched operation. It will return 6 denoised latents and 6 noise tensors.
2.  **Prompt Embeddings:** The prompt embeddings for all 6 images (3 positive, 3 neutral) will be concatenated and passed to the `superfunctional_train_step`.
3.  **LoRA Scale Application:** The `BatchedLoRANetwork` will receive a tensor of 6 LoRA scales. Its `set_lora_scales` method will correctly distribute these scales to the `current_multiplier` of each `BatchedLoRAModule` instance.
4.  **`batched_predict_noise_xl_modular`:** This function will perform a single UNet forward pass for all 6 latents. Crucially, because `BatchedLoRAModule` now correctly applies the `current_multiplier` (which is a tensor of scales), each of the 6 images in the batch will have its corresponding LoRA scale applied during the UNet inference.
5.  **Loss Calculation:** The output of `batched_predict_noise_xl_modular` will be 6 predicted noise tensors. The loss will then be calculated for each of the 3 image pairs (high vs. noise, low vs. noise) and then potentially averaged or summed across the batch.

## Comparison with `refactored_training_loop.py` (Float-for-Float Identical Results)

**Expectation:** We **do not expect** float-for-float identical results between the batched trainer and the accumulated sequential runs of the reference trainer.

**Reasons for Expected Differences:**

1.  **Floating Point Precision:** Even with identical operations, the order of operations in batched versus sequential processing can lead to minor differences in floating-point precision. Accumulating errors over multiple operations can result in noticeable deviations.
2.  **Batch Normalization/Layer Normalization (if present):** If the UNet or LoRA modules contain batch normalization or layer normalization layers, their behavior can differ between single-item processing and batched processing, leading to different activations and gradients.
3.  **LoRA Application Order:** While the `BatchedLoRANetwork` applies scales concurrently, the exact numerical outcome might slightly vary compared to applying them sequentially in separate forward passes, even if the mathematical intent is the same.
4.  **Randomness:** While we control the `torch.manual_seed` for noise generation, other sources of randomness (e.g., within PyTorch operations, CUDA non-determinism) can contribute to minor differences, especially when batching changes the execution path.

**Verification Strategy:**

Instead of float-for-float identity, the primary goal will be to verify **functional correctness** and **convergence**.

*   **Functional Correctness:** Ensure that the batched trainer produces sensible outputs and that the LoRA scales are indeed being applied as intended to their respective batch items. This can be verified by inspecting intermediate tensor shapes and values, and by ensuring the loss behaves as expected.
*   **Convergence:** The batched trainer should exhibit similar training dynamics (e.g., loss reduction over epochs) and ultimately produce models with comparable performance to the sequential trainer. This is the more important metric for a batched training loop.

