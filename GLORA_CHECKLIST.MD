## Checklist for GLORA Integration (Revised)

### 1. Core GLORA Module Design

*   [ ] **Define a `GLORAAdapter` Base Class:**
    *   This class should define the common interface for all GLORA adapter types (e.g., `forward`, `get_trainable_parameters`).
    *   It should handle the application of the adapter to a given input tensor.
*   [ ] **Implement Specific GLORA Support Tensor Types:**
    *   Implement the different types of support tensors (`A`, `B`, `C`, `D`, `E`) as subclasses or distinct implementations.
    *   For each, support the various forms they can take (e.g., LoRA-type, vector, scalar, `none`).
    *   Each implementation should handle its specific low-rank decomposition (if applicable) and forward pass logic.
*   [ ] **`GLORALayer` Wrapper:**
    *   Create a wrapper module (e.g., `GLORALayer`) that can replace existing linear or convolutional layers in the base model.
    *   This module should internally manage the specific combination of `A`, `B`, `C`, `D`, and `E` support tensors for that layer, as defined by the configuration.
    *   It should handle the precise mathematical combination of the base layer's output with the outputs of the active support tensors, as per GLORA's equations (e.g., `W_uni = W_0 + W_A + B`, `b_uni = CW_0 + D_B_0 + E_b_0`).

### 2. Model Patching and Configuration

*   [ ] **Comprehensive GLORA Configuration System:**
    *   Develop a robust configuration system (e.g., using a structured YAML or Python dataclasses) that allows for precise definition of the GLORA architecture.
    *   This system must specify, for *each target layer* in the base model:
        *   The *type* of support tensor (`A`, `B`, `C`, `D`, `E`) to apply (e.g., 'LoRA', 'vector', 'scalar', 'none').
        *   The *hyperparameters* for each active support tensor (e.g., `rank` for LoRA-type, `dimension` for vector).
        *   The *initialization strategy* for each support tensor.
    *   This system should be designed to easily load configurations derived from evolutionary searches.
*   [ ] **Automated Model Patching Utility:**
    *   Develop a utility function (e.g., `apply_glora_config_to_model`) that takes a pre-trained model and the detailed GLORA configuration.
    *   This utility should iterate through the model's modules and replace specified layers with `GLORALayer` instances, configuring each `GLORALayer` with its specific set of support tensors.
    *   Ensure that the original weights (`W_0`, `b_0`) of the replaced layers are preserved and correctly integrated into the `GLORALayer` as frozen components.
*   [ ] **Handling Different Architectures:**
    *   The patching utility should be generic enough to work with various PyTorch models (e.g., UNets, Transformers), identifying target layers by name or type.

### 3. Training Loop Integration

*   [ ] **Precise Parameter Management:**
    *   Ensure that *only* the parameters of the active `GLORAAdapter` instances (i.e., the `A`, `B`, `C`, `D`, `E` support tensors that are not `none`) are marked as trainable. The base model parameters (`W_0`, `b_0`) must remain frozen.
    *   Modify the optimizer to exclusively optimize these trainable GLORA parameters.
*   [ ] **Saving and Loading GLORA Weights:**
    *   Implement functionality to save *only* the trainable GLORA support tensor weights (and not the entire base model).
    *   Implement functionality to load these GLORA weights and apply them to a pre-trained base model, reconstructing the `GLORALayer` instances based on the saved configuration.
*   [ ] **Compatibility with Existing Training Scripts:**
    *   The integration should aim to minimize changes to the existing training loop. The `GLORALayer` should behave like the original module from the perspective of the training script.

### 4. Testing and Validation

*   [ ] **Preliminary Testing with Lycoris Idiom:**
    *   Test the implementation of `kohakublueleaf/lycoris` parameter efficient fine-tuning adapters. A GLORA implementation that uses the same idiom as Lycoris is pre-ported to many inference backends, and reduces the complexity of getting basic tests running.
*   [ ] **Unit Tests for Each Support Tensor Type:**
    *   Write comprehensive unit tests for each `A`, `B`, `C`, `D`, `E` support tensor type (LoRA, vector, scalar, none) to verify their forward pass, parameter count, and correct interaction with base weights/biases.
*   [ ] **Unit Tests for `GLORALayer`:**
    *   Test the `GLORALayer`'s ability to correctly combine different support tensor types based on configuration.
    *   Verify that trainable parameters are correctly identified and that the mathematical operations (Equations 10, 12, 13) are accurately implemented.
*   [ ] **Integration Test (Small Scale):**
    *   Apply GLORA with a specific configuration to a small, simple model.
    *   Run a minimal training loop to ensure that the model trains correctly with GLORA.
    *   Verify that the loss decreases and the model learns.
*   [ ] **End-to-End Training and Evaluation:**
    *   Integrate GLORA into the main diffusion model training pipeline using a configuration derived from the paper or a hypothetical evolutionary search.
    *   Conduct full training runs and evaluate the performance against baseline LoRA and full fine-tuning.
    *   Monitor key metrics (e.g., image quality, FID, CLIP score, concept slider effectiveness).
*   [ ] **Ablation Studies (Simulated Evolutionary Search):**
    *   Experiment with different GLORA configurations (varying support tensor types and hyperparameters per layer) to understand their impact on performance. This can simulate the insights gained from an evolutionary search.

### 5. Documentation

*   [ ] Update `README.md` with detailed instructions on how to configure and use GLORA, including examples of different support tensor configurations.
*   [ ] Add comprehensive docstrings and type hints to all GLORA-related code.
*   [ ] Document the GLORA configuration schema in detail.