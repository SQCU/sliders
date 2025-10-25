# train_scaledataset_vae_lora-xl.py
import argparse
import ast
from pathlib import Path
import gc, os
import random
import torch
from tqdm import tqdm

# --- Reuse all the battle-tested utilities ---
import train_util
import model_util
import config_util
import prompt_util  # <--- ADD THIS IMPORT
NUM_IMAGES_PER_PROMPT = 1
from prompt_util import (
    PromptEmbedsCache,
    PromptEmbedsPair,
    PromptSettings,
    PromptEmbedsXL,
)
from config_util import RootConfig
from train_lora_scale_xl import (
    batch_generator, 
    create_prefetch_generator,
    parse_adapters,
    generate_ar_buckets,
    log_attention_backend_status,
    parse_adapters,  # Ensure this is imported
)

from logging_utils import TimestepStratifiedLossTracker

# --- Import our new, specialized helpers ---
import lora_alt
from lora_alt import AltLoRANetwork
import lora_alt_vae
import optimizer_util


def flush():
    torch.cuda.empty_cache()
    gc.collect()

from torch.utils.checkpoint import checkpoint

def apply_gradient_checkpointing_to_vae_decoder(decoder):
    """
    Manually applies gradient checkpointing to the memory-intensive blocks
    of a VAE decoder. This is necessary because the VAE class in diffusers
    lacks a built-in .enable_gradient_checkpointing() method.
    """
    # The main memory hogs are the ResnetBlocks and AttentionBlocks
    from diffusers.models.resnet import ResnetBlock2D
    from diffusers.models import Transformer2DModel

    def checkpoint_forward(self, hidden_states, temb=None):
        return checkpoint(self._original_forward, hidden_states, temb)
    
    def checkpoint_forward_transformer(self, hidden_states, encoder_hidden_states=None, timestep=None, class_labels=None, cross_attention_kwargs=None, return_dict=True):
        # Transformer2DModel has a more complex forward signature
        return checkpoint(self._original_forward, hidden_states, encoder_hidden_states, timestep, class_labels, cross_attention_kwargs, return_dict)

    blocks_checkpointed = 0
    for module in decoder.modules():
        if isinstance(module, ResnetBlock2D):
            # Monkey-patch the forward method
            module._original_forward = module.forward
            module.forward = checkpoint_forward.__get__(module, type(module))
            blocks_checkpointed += 1
        # The VAE decoder uses Transformer2DModel for its attention blocks
        if isinstance(module, Transformer2DModel):
            module._original_forward = module.forward
            module.forward = checkpoint_forward_transformer.__get__(module, type(module))
            blocks_checkpointed += 1
            
    print(f"✅ Applied gradient checkpointing to {blocks_checkpointed} ResNet/Transformer blocks in the VAE decoder.")


def train_denoiser_guided_vae(
    config: RootConfig,
    device,
    folder_main: str,
    folders,
    scales,
    vae_lora_type: str,
    unet_base_adapters_str: str = None  # <-- NEW ARGUMENT
):
    """
    Main training function for Denoiser-Guided VAE Decoder Rehabilitation.
    **FINAL version with stable scheduler and correct step logic.**
    """
    scales_unique = list(set(scales))
    save_path = Path(config.save.path)
    weight_dtype = config_util.parse_precision(config.train.precision)

    device = torch.device(f"cuda:{args.device}")
    worker_pid = os.getpid()
    log_attention_backend_status(worker_pid)
    
    # --- 1. Load ALL Models ---
    print("Loading all models...")
    tokenizers, text_encoders, unet, noise_scheduler, vae = model_util.load_models_xl(
        config.pretrained_model.name_or_path,
        scheduler_name=config.train.noise_scheduler,
    )

    # --- NEW: Apply pre-trained UNet adapters if provided ---
    if unet_base_adapters_str:
        print(f"Fusing pre-trained adapters into the UNet from: {unet_base_adapters_str}")
        base_adapter_list = parse_adapters(unet_base_adapters_str)
        lora_alt.apply_base_adapters_reparam(unet, base_adapter_list)
        print("✅ UNet adapters fused successfully. The UNet teacher is now specialized.")
    # --- END NEW ---

    # --- 2. Freeze Models & Set Modes ---
    unet.to(device, dtype=weight_dtype).eval().requires_grad_(False)
    # --- CRITICAL FIX: Cast the ENTIRE VAE to the correct dtype ---
    vae.to(device, dtype=weight_dtype)
    # --- END FIX ---
    # The encoder will be used in no_grad mode, so we can set its state here
    vae.encoder.eval().requires_grad_(False)
    vae.decoder.eval().requires_grad_(False)
    for te in text_encoders:
        te.to(device, dtype=weight_dtype).eval().requires_grad_(False)
    print("Models loaded and frozen.")
    
    # --- CRITICAL: Set scheduler to the full training timestep range ONCE ---
    noise_scheduler.set_timesteps(config.train.max_denoising_steps, device=device)
    print(f"✅ Noise scheduler configured for {config.train.max_denoising_steps} training steps.")

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

    apply_gradient_checkpointing_to_vae_decoder(vae.decoder)

    # --- 4. Setup Optimizer ---
    print("Setting up hybrid optimizer...")
    optimizer = optimizer_util.setup_vae_optimizer(network, lr=config.train.lr)
    lr_scheduler = train_util.get_lr_scheduler(config.train.lr_scheduler, optimizer, config.train.iterations, config.train.lr / 100)
    criteria = torch.nn.MSELoss(reduction='none')
    
    # --- 5. Setup Data and Prompts ---
    prompts = prompt_util.load_prompts_from_yaml(config.prompts_file)
    # Assuming you want to use the first valid prompt set, which is standard.
    cache = PromptEmbedsCache()
    prompt_pairs: list[PromptEmbedsPair] = []
    
    with torch.no_grad():
        for settings in prompts:
            print(settings)
            for prompt in [
                settings.target,
                settings.positive,
                settings.neutral,
                settings.unconditional,
            ]:
                if cache[prompt] == None:
                    tex_embs, pool_embs = train_util.encode_prompts_xl(
                            tokenizers,
                            text_encoders,
                            [prompt],
                            num_images_per_prompt=NUM_IMAGES_PER_PROMPT,
                        )
                    cache[prompt] = PromptEmbedsXL(
                        tex_embs,
                        pool_embs
                    )

            prompt_pairs.append(
                PromptEmbedsPair(
                    criteria,
                    cache[settings.target],
                    cache[settings.positive],
                    cache[settings.unconditional],
                    cache[settings.neutral],
                    settings,
                )
            )

    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        del tokenizer, text_encoder

    flush()

    if hasattr(config.train, "grad_accum"):
        ACCUMULATION_STEPS = config.train.grad_accum
    else:
        ACCUMULATION_STEPS = 8 # Or 8, 32, etc.
        print(f"defaulted ur gradaccum to n:{ACCUMULATION_STEPS}")

    # ======================== GLUON RESEARCH PATCH ========================
    # The original script uses schedulers (like DDIM) that operate in the alpha/beta
    # space and do not have a .sigmas attribute.
    #
    # Our experiment with Karras-style loss weighting requires sigmas.
    # We will therefore compute them from the scheduler's alphas_cumprod
    # and monkey-patch the attribute onto the object. This allows us to
    # implement the advanced loss weighting without changing the core scheduler behavior.
    # The relationship is: sigma^2 = (1 - alpha_cumprod) / alpha_cumprod

    if not hasattr(noise_scheduler, 'sigmas'):
        print("⚠️  Warning: Scheduler is missing .sigmas. Calculating from alphas_cumprod for loss weighting.")
        
        # Move alphas_cumprod to CPU to avoid device mismatches, then compute sigmas
        alphas_cumprod = noise_scheduler.alphas_cumprod.to(device="cpu")
        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        
        # Monkey-patch the sigmas attribute onto the scheduler instance
        noise_scheduler.sigmas = sigmas.to(device=device) # Move back to the training device

        print("✅  Successfully monkey-patched .sigmas onto the scheduler.")
    # ======================================================================


    print("Initializing data generator...")
    TARGET_AREA = config.train.train_resolution * config.train.train_resolution
    ASPECT_RATIO_BUCKETS = generate_ar_buckets(TARGET_AREA)
    data_gen = batch_generator(folder_main, folders, scales, scales_unique, ASPECT_RATIO_BUCKETS, k=4, sharpness=5.0, batch_size=config.train.batch_size, n_tuple=config.train.n_tuple, max_denoising_steps=config.train.max_denoising_steps)
    prefetcher = create_prefetch_generator(data_gen, num_prefetch=2)
    
    # Initialize tracker
    loss_tracker = TimestepStratifiedLossTracker(
    max_timesteps=config.train.max_denoising_steps,
    n_buckets=10
    )

    # --- 6. The Final, Stabilized Training Loop ---
    pbar = tqdm(range(config.train.iterations))

    for i in pbar:
        optimizer.zero_grad()
        
        batch_packet = next(prefetcher)
        # We need the clean image tensor for the final loss calculation
        flat_images = [img for tpl in batch_packet["images"] for img in tpl]
        image_processor = train_util.VaeImageProcessor(vae_scale_factor=8)
        image_tensor_target = image_processor.preprocess(flat_images).to(device=device, dtype=weight_dtype)
        #reference implementation...
        prompt_pair: PromptEmbedsPair = random.choice(prompt_pairs)
        with torch.no_grad():
            # A. Get noisy latents (zt) and clean latents (x0) using the robust utility
            t_indices = batch_packet["timesteps_to"][:, 0] # Shape: (B,)
            latent_packet = train_util.prepare_correlated_noisy_latents(
                batch_packet=batch_packet,
                vae=vae,
                scheduler=noise_scheduler,
                timesteps_to=t_indices,
                dtype=weight_dtype
            )
            noisy_latents = latent_packet["noisy_latents"] # This is our z_t

            # B. Get the U-Net's prediction of noise from zt
            # --- CRITICAL: Use the INDEX to look up the VALUE ---
            # The scheduler's `timesteps` tensor is now our lookup table.
            t_values = noise_scheduler.timesteps[t_indices] # Shape: (B,)
            b_dim, n_dim = batch_packet["scales"].shape
            flat_t_values = t_values.unsqueeze(1).expand(-1, n_dim).reshape(-1)
            # --- END ---

            time_ids = train_util.batch_add_time_ids(batch_packet["original_sizes"], batch_packet["crop_coords"], batch_packet["target_sizes"], dtype=weight_dtype, device=device)
            text_embeds, pooled_embeds = train_util.broadcast_prompts_to_n_tuple(prompt_pair.positive.text_embeds, prompt_pair.positive.pooled_embeds, prompt_pair.neutral.text_embeds, prompt_pair.neutral.pooled_embeds, batch_packet["scales"])
            
            unet_args, unet_kwargs = train_util.sdxl_condnet_batchjoin(noisy_latents, text_embeds, pooled_embeds, time_ids, noise_scheduler, flat_t_values)
            predicted_noise = unet(*unet_args, **unet_kwargs).sample

            # C. Calculate the STABLE target for the decoder: z_{t-1}
            # We pass the VALUE of the timestep to the step function.
            # --- CRITICAL REVISION: Loop over unique timesteps ---
            latents_prev_step = torch.zeros_like(noisy_latents)
            unique_timesteps = torch.unique(flat_t_values)

            for t in unique_timesteps:
                mask = (flat_t_values == t)
                
                noise_slice = predicted_noise[mask]
                latents_slice = noisy_latents[mask]
                
                # Call step with a scalar timestep for the slice
                prev_sample_slice = noise_scheduler.step(noise_slice, t.item(), latents_slice).prev_sample
                
                # Place the results back into the full tensor
                latents_prev_step[mask] = prev_sample_slice
            # --- END REVISION ---
            
            # D. Calculate SNR-based loss weight based on the ORIGINAL timestep t's index
            flat_t_indices = t_indices.unsqueeze(1).expand(-1, n_dim).reshape(-1)
            sigmas = noise_scheduler.sigmas.to(device)[flat_t_indices]
            loss_weights = (1.0 / (sigmas**2 + 1.0)).detach().view(-1, 1, 1, 1)

        # E. Forward pass: Decode the stable, one-step-denoised latent z_{t-1}
        reconstructed_pixels = reconstructed_pixels = vae.decoder(latents_prev_step)
        

        # F. Calculate the weighted loss against the ORIGINAL clean image
        pixel_loss = criteria(reconstructed_pixels, image_tensor_target)
        weighted_pixel_loss = pixel_loss * loss_weights
        # 4 logging 4 the claudes
        per_sample_loss = weighted_pixel_loss.detach().mean(dim=(1,2,3))
        loss = weighted_pixel_loss.mean()/ ACCUMULATION_STEPS

        # Log to stratified tracker
        loss_tracker.log(
            losses=per_sample_loss,
            timesteps=t_indices,
            iteration=i
        )

        loss.backward()
        
        if (i + 1) % ACCUMULATION_STEPS == 0:
            # The profiler's step method is called here, seeing the accumulated grad
            GRAD_CLIP_MAX_NORM = getattr(config.train, 'grad_clip', 0.1)

            if GRAD_CLIP_MAX_NORM is not None and GRAD_CLIP_MAX_NORM > 0:
                # This is the list of parameters whose gradients will be clipped
                params_to_clip = [p for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
                # The single, canonical function call for gradient clipping
                torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=GRAD_CLIP_MAX_NORM)

            optimizer.step() 
            lr_scheduler.step()
            optimizer.zero_grad()

        pbar.set_description(f"VAE Rehab Loss: {loss.item():.6f}")

        if (i % config.save.per_steps == 0 and i != 0):
            print(f"\nSaving VAE LoRA at step {i}...")
            save_path.mkdir(parents=True, exist_ok=True)
            network.save_weights(
                save_path / f"{config.save.name}_denoiser_guided_vae_{i}steps.safenetensors",
                dtype=weight_dtype
            )
            # Periodic analysis
        if i == 100:
            # Fit difficulty curve from early training
            loss_tracker.fit_difficulty_curve(iterations_range=(0, 100))
            print("\n=== Difficulty Curve Fitted ===")
            for bucket_data in loss_tracker.difficulty_curve:
                print(f"Bucket {bucket_data['bucket']}: "
                    f"difficulty={bucket_data['difficulty']:.4f}, "
                    f"std={bucket_data['std']:.4f}")
        
        if i % 500 == 0 and i > 100:
            # Analyze progress
            progress = loss_tracker.compute_difficulty_adjusted_progress(i)
            print(f"\n=== Progress Report (Iteration {i}) ===")
            for p in progress:
                print(f"Bucket {p['bucket']}: "
                    f"{p['relative_improvement']*100:.1f}% improvement "
                    f"({p['baseline_loss']:.4f} → {p['current_loss']:.4f})")
            
            # Generate plots
            loss_tracker.plot_stratified_learning_curves(
                save_path=save_path / f"{config.save.name}_loss_analysis_iter{i}.png"
            )
        del reconstructed_pixels, latent_packet, latents_prev_step, noisy_latents, loss, per_sample_loss
    print("Final VAE LoRA save...")
    save_path.mkdir(parents=True, exist_ok=True)
    network.save_weights(
        save_path / f"{config.save.name}_denoiser_guided_vae_last.safenetensors",
        dtype=weight_dtype
    )
    print("Done.")


if __name__ == "__main__":
    # Same arg parser as before, no changes needed here.
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_lora_type", type=str, default='glora', choices=['lora', 'glora'])
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--folder_main", type=str, required=True)
    parser.add_argument("--folders", type=str, default='verylow, low, high, veryhigh')
    parser.add_argument("--scales", type=float, nargs='*', default=[-2, -1, 1, 2])
    # --- NEW ARGUMENT ---
    parser.add_argument(
        "--unet_base_adapters",
        type=str,
        required=False,
        default=None,
        help="Pre-trained UNet adapters to fuse before training. Renamed to clarify target. Format: 'path1:scale1,path2:scale2,...'",
    )
    # --- END NEW ---

    args = parser.parse_args()
    
    config = config_util.load_config_from_yaml(args.config_file)
    if args.name:
        config.save.name = args.name

    config.save.name += f'_alpha{args.alpha}_rank{args.rank}_vae_guided_rehab_stable'
    config.save.path += f'/{config.save.name}'
    
    device = torch.device(f"cuda:{args.device}")
    
    train_denoiser_guided_vae(
        config=config, 
        device=device, 
        folder_main=args.folder_main, 
        folders=[f.strip() for f in args.folders.split(',')], 
        scales=args.scales,
        vae_lora_type=args.vae_lora_type,
        unet_base_adapters_str=args.unet_base_adapters  # Pass the new 
    )