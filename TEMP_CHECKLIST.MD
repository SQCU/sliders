# Checklist for Code Improvements

This checklist summarizes the issues that need to be addressed to improve the performance, scalability, and Triton compatibility of the codebase.

## Triton Compatibility

*   [ ] **Remove Dynamic Shapes:**
    *   [ ] Replace `get_random_resolution_in_bucket()` with a fixed resolution or a pre-defined set of resolutions that can be batched together.
    *   [ ] Ensure that all tensors passed to the UNet have static shapes.
*   [ ] **Minimize CPU/GPU Synchronization:**
    *   [ ] **Fix `refactored_training_loop.py` execution:**
        *   [ ] **Immediate:** Correct `superfunctional_train_step` to use `predict_noise_xl` with `add_time_ids`.
            *   [ ] Run `uv run python F:/dox/ai/gemmy/sliders/refactored_training_loop.py > refactored_training_loop_test.log 2>&1`.
            *   [ ] Review `refactored_training_loop_test.log` for errors.
            *   [ ] Manually apply necessary code changes based on the error.
        *   [ ] **Intermediate:** Ensure `refactored_training_loop.py` runs without errors.
        *   [ ] **Final:** Verify refactored loops are float-for-float identical to original.
    *   [ ] Keep all tensors on the GPU as much as possible.
    *   [ ] Move loss calculation to the GPU.
    *   [ ] Remove unnecessary `.to("cpu")` calls.

## General Performance and Scalability

*   [x] **Implement a `torch.utils.data.Dataset` and `DataLoader`:**
    *   [x] Create a custom `Dataset` to load and preprocess images.
    *   [x] Use a `DataLoader` to load data in a separate process and pre-fetch batches.
*   [x] **Implement True Batching:**
    *   [x] Modify the data loading and training loop to handle true batches of data (i.e., tensors with a batch dimension greater than 1).
    *   [x] Remove the manual gradient accumulation.
*   [x] **Refactor the Training Loop:**
    *   [x] Move all data loading and preprocessing out of the main loop.
    *   [x] Ensure that all operations within the loop are performed on the GPU.
*   [x] **Make Parameters Configurable:**
    *   [x] Replace hardcoded values (e.g., image resolution) with configurable parameters.

## VRAM Optimization

*   [ ] **Pre-encode and Cache VAE Latents:**
    *   [ ] Implement `map_data_to_latents` function to encode and cache VAE latents to disk.
    *   [ ] Ensure VAE encoder is loaded with highest possible batch size during pre-encoding.
    *   [ ] Implement checksum verification for cached latents (VAE state_dict, image checksum, encoded latent checksum).
*   [ ] **Optimize VAE Decoder Usage:**
    *   [ ] Ensure VAE decoder is only loaded when necessary (e.g., for post-training analysis), not during the core training loop.
*   [ ] **Review UNet and LoRANetwork Instantiation:**
    *   [ ] Verify that the UNet model is not being inadvertently copied or duplicated.
    *   [ ] Investigate `LoRANetwork` constructor to ensure proper memory management and garbage collection of UNet instances.

## Diff Debugger Suite

*   [x] **Basic Statistics:** Implement loading and analysis of difference tensors (min, max, mean, std dev).
*   [ ] **Tensor Visualization:** Implement functions to convert image-like tensors to displayable formats.
*   [ ] **Tensor Statistics and EMAs:** Integrate into training loop to track moving averages of tensor statistics.
*   [ ] **Metadata Labeling:** Enhance analysis with explicit metadata about tensor origin and purpose.
