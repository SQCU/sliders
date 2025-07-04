# Batched LoRA Testing Checklist

## Reference Source Files for Debugging and Comparison

*   `trainscripts/imagesliders/batch_lora.py` (New Batched LoRA implementation)
*   `trainscripts/imagesliders/batch_train_util.py` (New Batched Training Utilities)
*   `batched_training_loop.py` (The main training loop with batched LoRA and utilities)
*   `refactored_training_loop.py` (The reference sequential training loop)
*   `trainscripts/imagesliders/lora.py` (Original LoRA implementation for architectural comparison)
*   `trainscripts/imagesliders/train_util.py` (Original Training Utilities for functional comparison)

## Test Setup

*   [ ] Prepare a test data sample of 3 distinct image pairs: `(img_high_1, img_low_1)`, `(img_high_2, img_low_2)`, `(img_high_3, img_low_3)`.
*   [ ] Define corresponding LoRA scales for each pair: `(scale_high_1, scale_low_1)`, `(scale_high_2, scale_low_2)`, `(scale_high_3, scale_low_3)`.
*   [ ] Ensure the `batched_training_loop.py` is configured to process these 3 image pairs in a single batch.
    *   [ ] Verify `img_batches` input to `superfunctional_train_step` contains 6 images (3 high, 3 low).
    *   [ ] Verify `scales` input to `superfunctional_train_step` is a tensor of 6 corresponding LoRA scales.

## Expected Behavior of `batched_training_loop.py`

*   [ ] **`get_batched_noisy_images`:**
    *   [ ] Verify it receives all 6 images.
    *   [ ] Verify it generates noise for all 6 images in a single batched operation.
    *   [ ] Verify it returns 6 denoised latents and 6 noise tensors.
*   [ ] **Prompt Embeddings:**
    *   [ ] Verify prompt embeddings for all 6 images (3 positive, 3 neutral) are concatenated.
    *   [ ] Verify these concatenated embeddings are passed to `superfunctional_train_step`.
*   [ ] **LoRA Scale Application (`BatchedLoRANetwork`):**
    *   [ ] Verify `BatchedLoRANetwork` receives the tensor of 6 LoRA scales.
    *   [ ] Verify `set_lora_scales` correctly distributes these scales to the `current_multiplier` of each `BatchedLoRAModule` instance.
*   [ ] **`batched_predict_noise_xl_modular`:**
    *   [ ] Verify it performs a single UNet forward pass for all 6 latents.
    *   [ ] Verify that each of the 6 images in the batch has its corresponding LoRA scale applied during the UNet inference (via `BatchedLoRAModule`'s `current_multiplier`).
*   [ ] **Loss Calculation:**
    *   [ ] Verify the output of `batched_predict_noise_xl_modular` consists of 6 predicted noise tensors.
    *   [ ] Verify loss is calculated for each of the 3 image pairs (high vs. noise, low vs. noise).
    *   [ ] Verify losses are then appropriately averaged or summed across the batch.

## Comparison with `refactored_training_loop.py`

*   [ ] **Expected Outcome:** Confirm that float-for-float identical results are **NOT** expected.
*   [ ] **Reasons for Differences (Acknowledge):**
    *   [ ] Floating Point Precision differences due to order of operations.
    *   [ ] Batch Normalization/Layer Normalization behavior differences (if applicable).
    *   [ ] LoRA Application Order nuances.
    *   [ ] Minor sources of randomness.
*   [ ] **Verification Strategy:** Focus on:
    *   [ ] **Functional Correctness:** Inspect intermediate tensor shapes and values to ensure LoRA scales are applied as intended.
    *   [ ] **Convergence:** Observe similar training dynamics (e.g., loss reduction) and comparable model performance.

## Testing Implementation

*   [ ] Read `batched_training_loop.py`.
*   [ ] Identify missing or miswritten features, and write a new timestamped batched_training_hypothesis statement as file.
*   [ ] Run `batched_training_loop.py` to test your hypothesis before thinking of any code edits.
    *   [ ] Write a checklist of revisions, additions, or further tests
    *   [ ] Check off that checklist before attempting revisions
    *   [ ] Make revisions by adding new functions to batch_* libraries or adding new batch_* versions of existing libraries, never changing common library function interfaces
*   [ ] Peace out early if you find yourself skipping steps or writing code line edits that just don't work, appending a little 'peace out notice' to your timestamped hypothesis statement
*   [ ] Most importantly, have fun!
