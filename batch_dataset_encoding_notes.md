# Batch Dataset Encoding Refactor Plan

## Problem Statement

The current dataset creation and batching mechanism is inefficient and overly complex. Key issues include:

1.  **Inefficient Batch Creation:** The script iteratively calls a constructor to generate one batch at a time, which is a slow and unscalable approach.
2.  **Coupled Logic:** The process of creating the training schedule (the "map" of which data to use for each step) is tightly coupled with the actual data loading and materialization. This prevents parallelization of these distinct tasks.
3.  **Redundant Computation:** The input embeddings for the training images are being recalculated arbitrarily and redundantly, leading to wasted compute resources and memory bandwidth.
4.  **Excessive Complexity:** The current approach is convoluted, making the code difficult to read, debug, and maintain. This complexity was a major issue in `batched_training_loop.py`.

## Proposed Solution

The goal is to refactor and simplify the training data pipeline by decoupling the training schedule from data materialization.

1.  **Create a Data Map/Schedule (`data_schedule.py`):**
    *   A new module, `data_schedule.py`, now houses the `TrainingSchedule` class.
    *   This class is responsible for generating a deterministic, pseudo-randomized training schedule based on a configurable seed.
    *   It creates a master pool of all possible training data combinations (image paths, scales, and prompts).
    *   For each training step, it samples from this master pool to construct a batch of `TrainingItem` objects.
    *   The `TrainingItem` objects encapsulate all necessary metadata for a single training example within a batch (image path, scale, prompt, pairing information, and whether it's a 'low case' for data augmentation).

2.  **Bundle Metadata:**
    *   The schedule includes all necessary metadata for each training step, such as:
        *   Microbatch pairings (e.g., which indices form a pair).
        *   Instructions for data augmentation (e.g., which cases to apply `scrongle_blimbify()` to).

3.  **Decouple Data Loading:**
    *   The `prepare_cached_batches` function in `batch_dataset_encoding.py` now directly consumes the `TrainingSchedule`.
    *   It iterates through the pre-defined schedule, loads the necessary latents and prompts for each `TrainingItem`, and then constructs the final batch dictionary.
    *   This ensures that the "what to train on" (the schedule) is entirely separate from the "how to get the data" (the loading and encoding logic), enabling cleaner code and future parallelization.

## Immediate Task: Fix VAE Batch Size Bounds Tester

The immediate priority is to fix the dynamic VAE batch size bounds tester in `batch_dataset_encoding.py`. The current implementation allocates excessive VRAM without incremental checks, leading to memory overruns and preventing effective profiling. This needs to be rewritten to test memory usage incrementally.

### VRAM Spillover Detection

A critical consideration is that PyTorch's CUDA memory utilities (`torch.cuda.memory_allocated`, etc.) do not report memory that has been offloaded to shared/paged GPU memory. This "spillover" can cause a severe performance degradation (e.g., 10-20x slowdown) without triggering an explicit Out-of-Memory (OOM) error. Relying solely on OOM errors or VRAM allocation figures is therefore insufficient for finding the true optimal batch size.

To address this, the rewritten bounds tester will incorporate a performance-based heuristic:

1.  **Incremental Testing:** Test batch sizes one by one, starting from 1.
2.  **Throughput Monitoring:** Track the peak throughput (images per second).
3.  **Slowdown Detection:** If the throughput for a new, larger batch size drops significantly (e.g., below 50% of the observed peak), the test will halt. This sudden drop is a strong indicator that memory spillover has occurred.
4.  **Recommendation:** The optimal batch size will be recommended based on the highest value achieved *before* the slowdown or an OOM error occurs.

A complex, multiprocessing-based watchdog to monitor for this slowdown was considered but deemed overly complex and brittle for this use case.
