## Algorithm Notes for Auxiliary SigLIP Loss

**Objective:** To regularize the internal representations (layerwise hidden states) of a fine-tuned model (e.g., LoRA-adapted UNet) such that the "residual" hidden states (difference from a reference model) align according to the slider direction. This encourages same-direction residuals to be similar and opposite-direction residuals to be dissimilar.

**Key Components:**

1.  **Reference Model:** A frozen, pre-trained diffusion model (e.g., the base UNet) to generate reference hidden states. This model will operate in `torch.no_grad()` mode.
2.  **Trained Model:** The fine-tuned diffusion model (e.g., UNet with LoRA adapters) whose hidden states will be regularized.
3.  **Hidden State Extraction:** During the forward pass of both the reference and trained models, extract the hidden states from selected intermediate layers (e.g., after attention blocks, after ResNet blocks). These extracted states will serve as the "embeddings" for our SigLIP loss.
4.  **Residual Calculation:** For each extracted hidden state `H_tuned` from the trained model and `H_ref` from the reference model at the same layer and input, compute the residual `R = H_tuned - H_ref`. This `R` will be the "embedding" used in the SigLIP formulation.
5.  **Pair Construction (within a batch):**
    *   Assume a training batch contains images generated with both `+scale` and `-scale` slider applications.
    *   **Positive Pairs:**
        *   `(R_i+, R_j+)`: Pairs of residuals where both `i` and `j` correspond to `+scale` slider applications.
        *   `(R_i-, R_j-)`: Pairs of residuals where both `i` and `j` correspond to `-scale` slider applications.
    *   **Negative Pairs:**
        *   `(R_i+, R_j-)`: Pairs of residuals where `i` corresponds to a `+scale` and `j` to a `-scale` slider application.
6.  **SigLIP Loss Formulation (per layer):**
    *   For each selected layer `l`:
        *   Let `R_l` be the set of residual hidden states extracted from layer `l` for the current batch.
        *   For every pair `(R_A, R_B)` from `R_l`:
            *   Calculate the raw similarity score: `s_AB = dot_product(R_A, R_B) * exp(t_l) + b_l`
                *   The `dot_product` is the core similarity measure, analogous to `torch.matmul(image_embeds, text_embeds.T)` in the original SigLIP.
                *   `t_l` and `b_l` are learnable temperature and bias parameters *specific to layer `l`*.
            *   Determine the ground truth label `y_AB`:
                *   `y_AB = 1` if `R_A` and `R_B` are from the same slider direction (both `+scale` or both `-scale`).
                *   `y_AB = -1` if `R_A` and `R_B` are from opposite slider directions.
            *   The loss for this pair is: `L_pair = -log(sigmoid(y_AB * s_AB))`
                *   This is equivalent to `F.binary_cross_entropy_with_logits(s_AB, (y_AB + 1) / 2)`, where `(y_AB + 1) / 2` converts -1 to 0 and 1 to 1 for BCE.
    *   The total SigLIP loss for layer `l` is the average `L_pair` over all selected pairs in the batch.
    *   The overall auxiliary SigLIP loss is the sum or average of losses across all selected layers.

**Implementation Considerations:**

*   **Learnable Parameters:** `t_l` and `b_l` should be `torch.nn.Parameter`s, initialized as `t_l = log(10)` and `b_l = 0` (or similar small values).
*   **Gradient Flow:** Ensure gradients only flow through the trained model and the `t_l`, `b_l` parameters. The reference model should be detached.
*   **Computational Efficiency:** The number of possible pairs grows quadratically with batch size. Implement strategies to limit pair comparisons (e.g., random sampling of negative pairs) using a "sigmoid-loss-comparison-richness" hyperparameter as suggested in `TEMP.MD`.
*   **Integration:** This auxiliary loss will be added to the primary diffusion loss, likely with a tunable weighting hyperparameter.

## Testing Plan

1.  **Unit Tests for Residual Calculation:**
    *   Create mock hidden states for a reference and trained model.
    *   Assert that the residual `H_tuned - H_ref` is computed correctly.
2.  **Unit Tests for Pair Construction:**
    *   Given a mock batch with `+scale` and `-scale` indicators, verify that positive and negative pairs are correctly identified and indexed.
3.  **Unit Tests for SigLIP Loss Component:**
    *   Create mock residual embeddings and ground truth labels.
    *   Verify the `dot_product`, scaling, bias, and `binary_cross_entropy_with_logits` calculation for various scenarios (e.g., perfect alignment, perfect anti-alignment, random values).
    *   Test with different initializations and values of `t_l` and `b_l`.
4.  **Integration Test (Small Scale):**
    *   Modify a simplified training loop (e.g., a single layer, small batch size) to incorporate the auxiliary SigLIP loss.
    *   Monitor the loss values and gradients to confirm correct flow.
    *   Observe if the `t_l` and `b_l` parameters are learning and converging.
5.  **Monitoring with Linear Probes:**
    *   Implement linear probes that attempt to predict the slider direction (`+scale` or `-scale`) solely from the residual hidden states.
    *   Monitor the accuracy of these probes during training. An increasing accuracy suggests the SigLIP loss is effectively shaping the internal representations.
    *   Compare probe accuracy with and without the SigLIP loss to quantify its impact.
6.  **Ablation Studies:**
    *   Train models with and without the auxiliary SigLIP loss to evaluate its impact on:
        *   Final image quality and consistency of slider effects.
        *   Generalization to unseen prompts/images.
        *   Training stability and convergence.
    *   Experiment with different weighting hyperparameters for the auxiliary loss.
    *   Experiment with different "sigmoid-loss-comparison-richness" settings to find an optimal balance between computational cost and regularization strength.