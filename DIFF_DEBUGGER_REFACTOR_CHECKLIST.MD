# Refactoring `diff_debugger.py` for Config-Driven Metadata

## 1. Define the Configuration Structure
- [ ] Design a JSON schema for the `diff_debugger_config.json` file. This schema will define:
    - The path to the `diff_debugger_tensors.pt` file.
    - The output directory for visualizations.
    - A list of tensor groups to analyze/visualize, each with:
        - A `prefix` (e.g., "orig", "refactored", "super").
        - A list of `tensor_types` (e.g., "denoised_latents", "noise", "target_latents", "loss_per_element").
        - A `visualization_type` (e.g., "image", "plot", "none").
        - Any specific `metadata_filters` if needed (e.g., `{"scale": "high"}`).

## 2. Create `configs/diff_debugger_config.json`
- [x] Create the JSON file based on the defined schema, populating it with the current hardcoded values and tensor names.

## 3. Modify `diff_debugger.py` to Load Config
- [ ] Import the `json` module.
- [ ] Load the `diff_debugger_config.json` file at the start of `main()`.
- [ ] Replace hardcoded paths and tensor lists with values from the loaded config.

## 4. Implement a Metadata Decoder/Helper Function
- [ ] Create a function (e.g., `get_tensor_info(loaded_tensors, prefix, tensor_type, scale=None)`) that takes the `loaded_tensors` dictionary and config parameters, and returns the `tensor` and its `metadata` dictionary. This function will handle the nested dictionary access.
- [ ] This function should also handle filtering by `scale` or other metadata if specified in the config.

## 5. Refactor Tensor Analysis and Visualization Loop
- [ ] Iterate through the `tensor_groups` defined in the config.
- [ ] For each group, use the metadata decoder function to retrieve the relevant tensors.
- [ ] Dynamically call `analyze_tensor`, `plot_difference_tensor`, or `visualize_tensor_as_image` based on the `visualization_type` in the config.
- [ ] Ensure that `analyze_tensor` and visualization functions correctly use the extracted metadata for logging and titles.

## 6. Refactor Frobenius Norm Calculations
- [ ] Update the Frobenius norm calculations to use the `get_tensor_info` helper function for retrieving the `loss_per_element` tensors.

## 7. Test and Verify
- [ ] Run `refactored_training_loop.py` to generate a new `diff_debugger_tensors.pt` with the metadata.
- [ ] Run `diff_debugger.py` with the new config to ensure all analyses and visualizations are generated correctly and reflect the metadata.

## STOPPING POINT: Context Limit Reached
- [ ] Resume refactoring `diff_debugger.py` from a fresh context, starting with ensuring `diff_debugger.py` is in a clean state (e.g., `git restore`).