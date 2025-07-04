# VAE Caching Checklist

*   [x] **Create a `map_data_to_latents.py` script:**
    *   [x] Implement a function to load images from a directory.
    *   [x] Implement a function to encode images to latents using the VAE.
    *   [x] Implement a function to save the latents to disk.
    *   [x] Implement a function to load the latents from disk.
*   [x] **Integrate into the training loop:**
    *   [x] Modify the `Dataset` to load cached latents instead of images.
    *   [x] Add a command-line argument to enable or disable VAE caching.
*   [x] **Implement checksum verification:**
    *   [x] Calculate and store a checksum for the VAE state_dict.
    *   [x] Calculate and store a checksum for each image.
    *   [x] Calculate and store a checksum for each encoded latent.
    *   [x] Verify the checksums when loading cached latents.
