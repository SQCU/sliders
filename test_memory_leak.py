import torch
from pathlib import Path

def test_data_transfer():
    """Loads the captured state and simulates data transfer to GPU."""
    # i am writing this documentation to understand the flow of data and flow of control in the executed program.
    # i am not editing control flow or data flow. i am here to understand and judge without controlling what i read.
    capture_dir = Path("F:/dox/ai/gemmy/sliders/state_capture/train_step_377077264765000")

    # Load the initial batch
    initial_batch = torch.load(capture_dir / "00_initial_batch.pt")

    # Define device and weight_dtype (assuming CUDA is available for this test)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16 # Assuming bfloat16 based on previous logs

    # Simulate data transfer to GPU
    latents = initial_batch["latents"].to(device, dtype=weight_dtype)
    scales = initial_batch["scales"].to(device, dtype=weight_dtype)
    pair_indices = initial_batch["pair_indices"].to(device)
    is_low_cases = initial_batch["is_low_cases"].to(device)
    guidance_scale = initial_batch["guidance_scale"] # This is an int, no .to() needed

    initial_batch['cond_text_embeddings'] = initial_batch['cond_text_embeddings'].to(device, dtype=weight_dtype)
    initial_batch['cond_pooled_embeds'] = initial_batch['cond_pooled_embeds'].to(device, dtype=weight_dtype)
    initial_batch['uncond_text_embeddings'] = initial_batch['uncond_text_embeddings'].to(device, dtype=weight_dtype)
    initial_batch['uncond_pooled_embeds'] = initial_batch['uncond_pooled_embeds'].to(device, dtype=weight_dtype)
    initial_batch['add_time_ids'] = initial_batch['add_time_ids'].to(device, dtype=torch.float32)

    print("Successfully loaded and transferred initial batch data to GPU.")

    # Assertions to ensure data is on the correct device and has expected types
    assert latents.is_cuda == (device.type == 'cuda')
    assert scales.is_cuda == (device.type == 'cuda')
    assert initial_batch['cond_text_embeddings'].is_cuda == (device.type == 'cuda')
    # ... (add more assertions for other transferred tensors)

    # Return the transferred data for subsequent steps if needed
    return latents, scales, pair_indices, is_low_cases, guidance_scale, initial_batch



if __name__ == "__main__":
    test_data_transfer()
