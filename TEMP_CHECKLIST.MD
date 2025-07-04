# Checklist for Future Code Improvements

This checklist outlines the next steps for enhancing the training core loop to support batched training and GLORA features.

## 1. Batched Training of Multiple Training Pairs

**Goal:** Enable `train_lora-scale-xl.py` to process multiple image pairs within a single forward pass, leveraging GPU parallelism.

*   [ ] **Modify `train_lora-scale-xl.py` (Main Training Loop):**
    *   [ ] Ensure the `dataloader` provides batches of `(img1_batch, scale1_batch), (img2_batch, scale2_batch)` where `img1_batch` and `img2_batch` have a batch dimension `B > 1`.
    *   [ ] Implement logic to select or construct `B` prompt embeddings, one for each pair in the batch.
    *   [ ] Pass these batched prompt embeddings (e.g., `batched_positive_embeds`, `batched_neutral_embeds`) to `superfunctional_train_step`.

*   [ ] **Modify `superfunctional_train_step` Function:**
    *   [ ] Adapt the function signature to accept batched inputs for images, scales, and prompt embeddings.
    *   [ ] Concatenate `img1_batch` and `img2_batch` along the batch dimension for a single UNet input.
    *   [ ] Concatenate corresponding batched prompt embeddings and `add_time_ids`.
    *   [ ] Perform a single call to `predict_noise_xl` with the concatenated batched inputs.
    *   [ ] Split the output (e.g., `noise_pred`) back into `B` components.
    *   [ ] Calculate the loss for each pair in the batch and then average or sum them.

## 2. GLORA Features (Sigmoid Contrastive Residual Loss & Batched LoRA Application)

**Goal:** Integrate the Sigmoid Contrastive Residual Loss for improved conceptual slider learning and implement batched LoRA application for performance.

*   [ ] **Sigmoid Contrastive Residual Loss:**
    *   [ ] **Residual Extraction:** Modify `LoRANetwork` and `LoRAModule` (or UNet forward pass) to expose intermediate activations (residuals) before and after LoRA application.
    *   [ ] **New Loss Function:** Implement `SigmoidContrastiveLoss` module/function that takes pairs of residual tensors and their "matching" labels.
    *   [ ] **Integration into Training Loop:**
        *   [ ] Extract necessary residuals in `superfunctional_train_step`.
        *   [ ] Form "matched" and "unmatched" pairs of residuals.
        *   [ ] Calculate `SigmoidContrastiveLoss` and add it to the total loss.
    *   [ ] **Learnable `t'` and `b` Terms:** Implement as `nn.Parameter` layers within `SigmoidContrastiveLoss`.
    *   [ ] **Linear Probes for Monitoring:** Add `nn.Linear` layers to predict slider magnitude from residual features and track their accuracy.

*   [ ] **Batched LoRA Application:**
    *   [ ] **`LoRAModule` Modification:** Update `LoRAModule`'s `forward` method to accept a `torch.Tensor` of `multiplier` values (one per batch item).
    *   [ ] **`LoRANetwork` Modification:** Change `set_lora_slider` to accept a batch of scales.
    *   [ ] **`superfunctional_train_step` Integration:** Ensure `network.set_lora_slider` takes a batch of scales, and the UNet forward pass implicitly applies different LoRA scales to different items in the batch.
