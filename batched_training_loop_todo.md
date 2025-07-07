# Refactoring batched_training_loop.py - To Do List

*   **Modify `form_cfg_microbatch` in `batched_training_loop.py`**:
    *   Remove `pairing_map` from its arguments.
    *   Update the logic to use `batch["pair_indices"]` and `batch["is_low_cases"]` to determine the order of `ordered_indices`, `selected_cond_text`, and `selected_cond_pool`.
    *   Adjust the return signature to match the new `form_cfg_microbatch` definition.
*   **Modify `calculate_paired_loss` in `batched_training_loop.py`**:
    *   Remove `pairing_map` from its arguments.
    *   Add `pair_indices` and `is_low_cases` to its arguments.
    *   Update the loss calculation logic to use `pair_indices` and `is_low_cases` to correctly sum losses for paired items.
*   **Update calls to `form_cfg_microbatch` and `calculate_paired_loss` in `train_step`**:
    *   Remove `pairing_map` from the arguments passed to these functions.
    *   Pass `pair_indices` and `is_low_cases` to `calculate_paired_loss`.
*   **Remove `pairing_map` from the `batch` dictionary creation in `prepare_cached_batches`**.
*   **Remove the `create_pairing_map` function from `batched_training_loop.py`**.
*   **Remove the `create_pairing_map`, `form_cfg_microbatch`, and `calculate_paired_loss` functions from `batch_slider_algo.py`**.