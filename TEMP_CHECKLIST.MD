# Checklist for Code Improvements

This checklist summarizes the issues that need to be addressed to improve the performance, scalability, and Triton compatibility of the codebase.

## Triton Compatibility

*   [x] **Remove Dynamic Shapes:**
    *   [ ] Replace `get_random_resolution_in_bucket()` with a fixed resolution or a pre-defined set of resolutions that can be batched together.
    *   [ ] Ensure that all tensors passed to the UNet have static shapes.
*   [x] **Minimize CPU/GPU Synchronization:**
    *   [x] **Fix `refactored_training_loop.py` execution:**
        *   [x] **Immediate:** Correct `superfunctional_train_step` to use `predict_noise_xl` with `add_time_ids`.
            *   [x] Run `uv run python F:/dox/ai/gemmy/sliders/refactored_training_loop.py > refactored_training_loop_test.log 2>&1`.
            *   [x] Review `refactored_training_loop_test.log` for errors.
            *   [x] Manually apply necessary code changes based on the error.
        *   [x] **Intermediate:** Ensure `refactored_training_loop.py` runs without errors.
        *   [x] **Final:** Verify `functional_train_step` is float-for-float identical to original.
        *   [x] **Final:** Verify `superfunctional_train_step` is float-for-float identical to original.
    *   [x] Keep all tensors on the GPU as much as possible.
    *   [x] Move loss calculation to the GPU.
    *   [x] Remove unnecessary `.to("cpu")` calls.

## General Performance and Scalability

*   [x] **Implement a `torch.utils.data.Dataset` and `DataLoader`:**
    *   [x] Create a custom `Dataset` to load and preprocess images.
    *   [x] Use a `DataLoader` to load data in a separate process and pre-fetch batches.
*   [x] **Implement True Batching:**
    *   [x] Modify the data loading and training loop to handle true batches of data (i.e., tensors with a batch dimension greater than 1).
    *   [x] Remove the manual gradient accumulation.
*   [x] **Optimize Sequential Tensor Operations:**
    *   [ ] Identify sequential calls to functions that operate on compatible tensors (e.g., `func(tensor1); func(tensor2)`).
    *   [ ] Refactor these calls to batch the tensors (e.g., `func(torch.cat([tensor1, tensor2]))`) and process them concurrently as a single execution step, then split results if necessary.
    *   [ ] Prioritize pairs of identical tensors for this optimization to maximize cache hits and accelerator utilization.
*   [x] **Refactor the Training Loop:**
    *   [x] Move all data loading and preprocessing out of the main loop.
    *   [x] Ensure that all operations within the loop are performed on the GPU.
*   [x] **Make Parameters Configurable:**
    *   [x] Replace hardcoded values (e.g., image resolution) with configurable parameters.

## VRAM Optimization

*   [x] **Pre-encode and Cache VAE Latents:**
    *   [x] Implement `map_data_to_latents` function to encode and cache VAE latents to disk.
    *   [x] Ensure VAE encoder is loaded with highest possible batch size during pre-encoding.
    *   [x] Implement checksum verification for cached latents (VAE state_dict, image checksum, encoded latent checksum).
*   [x] **Optimize VAE Decoder Usage:**
    *   [x] Ensure VAE decoder is only loaded when necessary (e.g., for post-training analysis), not during the core training loop.
*   [x] **Review UNet and LoRANetwork Instantiation:**
    *   [x] Verify that the UNet model is not being inadvertently copied or duplicated.
    *   [x] Investigate `LoRANetwork` constructor to ensure proper memory management and garbage collection of UNet instances.
    * Integrate Pre-encoded Latents into Training Loop:
       * [x] Add latent_cache_dir to RootConfig in config_util.py (both imagesliders and textsliders).
       * [x] Import map_data_to_latents in train_lora.py and train_lora_xl.py.
       * [ ] Add image_path: Optional[str] to PromptSettings in prompt_util.py.
       * [ ] Add image_path: Optional[str] to PromptEmbedsPair class definition in prompt_util.py.
       * [ ] Update PromptEmbedsPair.__init__ to assign self.image_path = settings.image_path.
       * [ ] Modify load_prompts_from_yaml to read image_path from the YAML and pass it to PromptSettings.
       * [ ] Implement logic in train_lora.py and train_lora_xl.py to load latents from config.latent_cache_dir using image_path if available, otherwise generate them.

## Diff Debugger Suite

*   [x] **Basic Statistics:** Implement loading and analysis of difference tensors (min, max, mean, std dev).
*   [x] **Tensor Visualization:** Implement functions to convert image-like tensors to displayable formats.
*   [x] **Tensor Statistics and EMAs:** Integrate into training loop to track moving averages of tensor statistics.
*   [x] **Metadata Labeling:** Enhance analysis with explicit metadata about tensor origin and purpose.
