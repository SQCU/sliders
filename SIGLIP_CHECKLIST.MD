## Checklist for Generic PyTorch SigLIPper Module Implementation

### 1. Module Structure and Initialization

*   [ ] Define a `SigLIPLoss` PyTorch module (e.g., `torch.nn.Module`).
*   [ ] `__init__` method should accept:
    *   `model_to_regularize`: The model whose hidden states will be regularized (e.g., UNet with LoRA).
    *   `reference_model`: Optional. The frozen reference model for residual calculation. If `None`, direct hidden state comparison is used.
    *   `layers_to_consider`: A list of layer names or indices within the `model_to_regularize` from which to extract hidden states. This allows for arbitrary layer selection.
    *   `comparison_mode`: Enum/string ('residual', 'direct').
        *   'residual': `H_tuned - H_ref` (requires `reference_model`).
        *   'direct': `H_tuned` directly.
    *   `loss_aggregation_mode`: Enum/string ('all_layers', 'attention_only', 'selected_layers', 'final_state_only').
        *   'all_layers': Apply loss to all specified `layers_to_consider`.
        *   'attention_only': Filter `layers_to_consider` to only include attention blocks (requires a mechanism to identify them).
        *   'selected_layers': Use `layers_to_consider` as is.
        *   'final_state_only': Only apply loss to the last specified layer in `layers_to_consider`.
    *   `temperature_init`: Initial value for `t_l` (e.g., `log(10)`).
    *   `bias_init`: Initial value for `b_l` (e.g., `0`).
    *   `weight`: Scalar weighting for this auxiliary loss when combined with other losses.
    *   `max_pairs_per_batch`: Integer. Limits the number of pairs considered for loss calculation to manage computational cost.
*   [ ] Initialize learnable `t_l` and `b_l` parameters for each selected layer. These should be `torch.nn.Parameter`s.

### 2. Hidden State Extraction Mechanism

*   [ ] Implement a mechanism to register forward hooks on the `model_to_regularize` (and `reference_model` if `comparison_mode` is 'residual') to capture intermediate hidden states.
*   [ ] The hooks should store the hidden states in a temporary buffer accessible by the `SigLIPLoss` module.
*   [ ] Ensure hooks are properly removed after each forward pass to prevent memory leaks.

### 3. Forward Pass (`forward` method)

*   [ ] The `forward` method should accept:
    *   `input_data`: The input to the `model_to_regularize` (and `reference_model`).
    *   `slider_directions`: A tensor/list indicating the slider direction (`+1` for `+scale`, `-1` for `-scale`) for each item in the batch.
*   [ ] Perform a forward pass through `model_to_regularize` (and `reference_model` in `torch.no_grad()` if `comparison_mode` is 'residual').
*   [ ] Retrieve the captured hidden states for the specified `layers_to_consider`.
*   [ ] Based on `comparison_mode`:
    *   If 'residual': Compute `H_tuned - H_ref` for each layer.
    *   If 'direct': Use `H_tuned` directly.
*   [ ] Based on `loss_aggregation_mode`, select the relevant hidden states for loss calculation.
*   [ ] For each selected layer:
    *   Construct positive and negative pairs based on `slider_directions`.
    *   Apply `max_pairs_per_batch` sampling if the number of pairs is too large.
    *   Calculate the dot product similarity for each pair.
    *   Apply `t_l` and `b_l` for the current layer.
    *   Compute the SigLIP loss for the layer: `-log(sigmoid(y_AB * s_AB))`.
*   [ ] Aggregate the loss across selected layers (e.g., sum or mean).
*   [ ] Return the total auxiliary SigLIP loss.

### 4. Utility Functions

*   [ ] `_get_layer_by_name_or_index(model, layer_identifier)`: Helper to retrieve a specific layer from a model.
*   [ ] `_identify_attention_layers(model)`: Helper to programmatically identify attention layers within a UNet-like architecture.
*   [ ] `_compute_dot_product(tensor1, tensor2)`: Simple dot product calculation.

### 5. Integration with Training Loop

*   [ ] Instantiate `SigLIPLoss` module.
*   [ ] Add `SigLIPLoss` parameters to the optimizer.
*   [ ] In the training loop, after the main diffusion loss calculation, call the `SigLIPLoss` module's `forward` method.
*   [ ] Add the returned auxiliary loss (multiplied by its `weight`) to the total loss.

### 6. Testing and Validation

*   [ ] Write comprehensive unit tests for each component (hidden state extraction, pair construction, loss calculation).
*   [ ] Integrate into a small-scale training script to verify end-to-end functionality.
*   [ ] Monitor `t_l` and `b_l` values during training.
*   [ ] Implement linear probes as described in `SIGLIP_NOTES.MD` to monitor the effectiveness of the regularization.
