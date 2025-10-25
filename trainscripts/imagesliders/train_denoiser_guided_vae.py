# train_denoiser_guided_vae.py
import argparse
from pathlib import Path
import torch
from tqdm import tqdm

# --- Reuse all the battle-tested utilities ---
import train_util
import model_util
import config_util
from config_util import RootConfig
from train_lora_scale_xl import (
    batch_generator, 
    create_prefetch_generator,
    generate_ar_buckets,
    log_attention_backend_status,
)

# --- Import our specialized helpers ---
from lora_alt import AltLoRANetwork
import lora_alt_vae
import optimizer_util

def train_denoiser_guided_vae(
    config: RootConfig,
    device,
    folder_main: str,
    folders,
    scales,
    vae_lora_type: str
):
    """
    Main training function for Denoiser-Guided VAE Decoder Rehabilitation.
    """
    scales_unique = list(set(scales))
    save_path = Path(config.save.path)
    weight_dtype = config_util.parse_precision(config.train.precision)
    
    # --- 1. Load ALL Models ---
    print("Loading all models (U-Net, VAE, Scheduler, Text Encoders)...")
    tokenizers, text_encoders, unet, noise_scheduler, vae = model_util.load_models_xl(
        config.pretrained_model.name_or_path,
        scheduler_name=config.train.noise_scheduler,
    )

    # --- 2. Freeze Models & Set Modes ---
    # The U-Net is our frozen, expert teacher
    unet.to(device, dtype=weight_dtype).eval().requires_grad_(False)
    # The VAE encoder is only for getting initial latents
    vae.encoder.to(device, dtype=weight_dtype).eval().requires_grad_(False)
    # Text encoders are needed for the U-Net's conditioning
    for te in text_encoders:
        te.to(device, dtype=weight_dtype).eval().requires_grad_(False)

    # The VAE decoder is our student
    vae.decoder.to(device, dtype=weight_dtype)
    print("Models loaded and frozen.")

    # --- 3. Prepare VAE Decoder for LoRA fine-tuning ---
    print(f"Applying '{vae_lora_type.upper()}' adapters to VAE Decoder...")
    vae_decoder_config = lora_alt_vae.create_vae_lora_config_map(
        vae.decoder,
        rank=config.network.rank,
        alpha=config.network.alpha,
        lora_type=vae_lora_type
    )
    network = AltLoRANetwork(vae.decoder, resolved_config={"lora_map": vae_decoder_config})
    network.to(device, dtype=weight_dtype).train()

    # --- 4. Setup Optimizer with Corrected Logic ---
    print("Setting up hybrid optimizer...")
    optimizer = optimizer_util.setup_vae_optimizer(network, lr=config.train.lr)
    lr_scheduler = train_util.get_lr_scheduler(config.train.lr_scheduler, optimizer, config.train.iterations, config.train.lr / 100)
    
    # --- 5. Setup Data and Prompts ---
    # We still need prompts for the U-Net teacher
    prompts = config_util.load_config_from_yaml(config.prompts_file).prompts
    prompt_pair = train_util.prepare_prompt_embeds(prompts[0], tokenizers, text_encoders, device)
    
    print("Initializing data generator...")
    TARGET_AREA = config.train.train_resolution * config.train.train_resolution
    ASPECT_RATIO_BUCKETS = generate_ar_buckets(TARGET_AREA)
    data_gen = batch_generator(folder_main, folders, scales, scales_unique, ASPECT_RATIO_BUCKETS, k=4, sharpness=5.0, batch_size=config.train.batch_size, n_tuple=config.train.n_tuple, max_denoising_steps=config.train.max_denoising_steps)
    prefetcher = create_prefetch_generator(data_gen, num_prefetch=2)

    # --- 6. The Corrected Training Loop ---
    criteria = torch.nn.MSELoss(reduction='none')
    pbar = tqdm(range(config.train.iterations))

    for i in pbar:
        optimizer.zero_grad()
        
        batch_packet = next(prefetcher)
        flat_images = [img for tpl in batch_packet["images"] for img in tpl]
        image_processor = train_util.VaeImageProcessor(vae_scale_factor=8)
        image_tensor = image_processor.preprocess(flat_images).to(device=device, dtype=weight_dtype)

        with torch.no_grad():
            # A. Get clean latents and create noisy versions
            x0_latents = vae.encode(image_tensor).latent_dist.sample() * vae.config.scaling_factor
            t_indices = torch.randint(0, noise_scheduler.config.num_train_timesteps, (x0_latents.shape[0],), device=device)
            noise = torch.randn_like(x0_latents)
            noisy_latents = noise_scheduler.add_noise(x0_latents, noise, t_indices)

            # B. Get the "teacher's" prediction from the U-Net
            # Prepare conditioning inputs for the U-Net
            time_ids = train_util.batch_add_time_ids(batch_packet["original_sizes"], batch_packet["crop_coords"], batch_packet["target_sizes"], dtype=weight_dtype, device=device)
            text_embeds, pooled_embeds = train_util.broadcast_prompts_to_n_tuple(prompt_pair.positive.text_embeds, prompt_pair.positive.pooled_embeds, prompt_pair.neutral.text_embeds, prompt_pair.neutral.pooled_embeds, batch_packet["scales"])
            
            # Use a helper for clarity
            unet_args, unet_kwargs = train_util.sdxl_condnet_batchjoin(noisy_latents, text_embeds, pooled_embeds, time_ids, noise_scheduler, t_indices)
            
            # This is the U-Net's prediction of the noise
            predicted_noise = unet(*unet_args, **unet_kwargs).sample

            # C. Calculate the denoiser's intended clean latent (the decoder's target)
            target_latents_for_decoder = train_util.get_x0_from_xt_eps(noisy_latents, predicted_noise, t_indices, noise_scheduler)

            # D. Calculate SNR-based loss weight
            sigmas = noise_scheduler.sigmas.to(device)[t_indices]
            loss_weights = (1.0 / (sigmas**2 + 1.0)).detach().view(-1, 1, 1, 1)

        # E. Forward pass: Decode the U-Net's intended latent
        # The 'network' IS the LoRA-adapted vae.decoder
        reconstructed_pixels = network(target_latents_for_decoder)
        
        # F. Calculate the weighted loss against the ORIGINAL clean image
        pixel_loss = criteria(reconstructed_pixels, image_tensor)
        weighted_pixel_loss = pixel_loss * loss_weights
        loss = weighted_pixel_loss.mean()

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        pbar.set_description(f"VAE Rehab Loss: {loss.item():.6f}")

        if (i % config.save.per_steps == 0 and i != 0):
            save_path.mkdir(parents=True, exist_ok=True)
            network.save_weights(
                save_path / f"{config.save.name}_denoiser_guided_vae_{i}steps.safenetensors",
                dtype=weight_dtype
            )

    print("Final VAE LoRA save...")
    save_path.mkdir(parents=True, exist_ok=True)
    network.save_weights(
        save_path / f"{config.save.name}_denoiser_guided_vae_last.safenetensors",
        dtype=weight_dtype
    )
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add a new argument for VAE LoRA type
    parser.add_argument("--vae_lora_type", type=str, default='glora', choices=['lora', 'glora'], help="Type of LoRA adapter for the VAE.")
    # (Keep all other arguments from the original script)
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--folder_main", type=str, required=True)
    parser.add_argument("--folders", type=str, default='verylow, low, high, veryhigh')
    parser.add_argument("--scales", type=float, nargs='*', default=[-2, -1, 1, 2])
    
    args = parser.parse_args()
    
    config = config_util.load_config_from_yaml(args.config_file)
    if args.name:
        config.save.name = args.name

    config.save.name += f'_alpha{args.alpha}_rank{args.rank}_vae_guided_rehab'
    config.save.path += f'/{config.save.name}'
    
    device = torch.device(f"cuda:{args.device}")
    
    train_denoiser_guided_vae(
        config=config, 
        device=device, 
        folder_main=args.folder_main, 
        folders=[f.strip() for f in args.folders.split(',')], 
        scales=args.scales,
        vae_lora_type=args.vae_lora_type
    )