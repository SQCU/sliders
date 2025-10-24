# train_lora-scale-xl.py
# ref:
# - https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L566
# - https://huggingface.co/spaces/baulab/Erasing-Concepts-In-Diffusion/blob/main/train.py

from typing import List, Optional
import argparse
import ast
from pathlib import Path
import gc, os
import numpy as np
import math

import torch
from tqdm import tqdm
from PIL import Image

#prefetch data
import queue
import threading


import train_util
import random
import model_util
import prompt_util
from prompt_util import (
    PromptEmbedsCache,
    PromptEmbedsPair,
    PromptSettings,
    PromptEmbedsXL,
)
import debug_util
import config_util
from config_util import RootConfig

import wandb

NUM_IMAGES_PER_PROMPT = 1
from lora import LoRANetwork, DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV
import lora_alt


import torch._dynamo
torch._dynamo.config.recompile_limit = 16 # Optional: Increase the limit


def select_target_bucket(
    original_size: tuple[int, int],
    buckets: list[tuple[int, int]],
    k_nearest: int = 3,
    sharpness: float = 1.0,
) -> tuple[int, int]:
    """
    Selects a target aspect ratio bucket for an entire batch based on a
    representative image's dimensions.
    """
    original_w, original_h = original_size
    original_ar = original_w / original_h

    # --- Calculate distance to each bucket's aspect ratio ---
    bucket_distances = []
    for bucket_w, bucket_h in buckets:
        bucket_ar = bucket_w / bucket_h
        distance = abs(math.log(original_ar) - math.log(bucket_ar))
        bucket_distances.append((distance, (bucket_w, bucket_h)))

    # --- Probabilistically sample a target bucket ---
    bucket_distances.sort(key=lambda x: x[0])
    nearest_buckets = bucket_distances[:k_nearest]
    distances = np.array([-d for d, _ in nearest_buckets])
    probabilities = np.exp((distances - np.max(distances)) * sharpness)
    probabilities /= probabilities.sum()
    chosen_idx = np.random.choice(len(nearest_buckets), p=probabilities)
    _, target_size = nearest_buckets[chosen_idx]
    
    return target_size

def determine_transform_params(
    original_size: tuple[int, int],
    target_size: tuple[int, int], # This is now a fixed input
    prescale_perc: float = 0.2,
) -> tuple[tuple, tuple, tuple]:
    """
    Determines the full transformation parameters for a single tuple, respecting
    the locked batch target size. It correctly prescales to circumscribe the target.
    
    Returns:
        prescale_size (tuple or None)
        crop_box (tuple)
        reported_original_size (tuple)
    """
    original_w, original_h = original_size
    target_w, target_h = target_size

    
    # --- NEW: Safeguard for small images ---
    is_too_small = original_w < target_w or original_h < target_h
    
    if is_too_small or (random.random() < prescale_perc):
        # --- MODE A: Prescale-then-Crop (Global View) ---
        # 1. Correctly determine the scale factor to make the image *larger* than the target crop.
        #    The goal is to make the smaller dimension of the prescaled image match the target.
        scale_factor = max(target_w / original_w, target_h / original_h)
        prescale_w = int(math.ceil(original_w * scale_factor))
        prescale_h = int(math.ceil(original_h * scale_factor))
        prescale_size = (prescale_w, prescale_h)
        
        # 2. Jitter the crop window within this new, larger prescaled image.
        max_left = prescale_w - target_w
        max_top = prescale_h - target_h
        crop_left = random.randint(0, max_left)
        crop_top = random.randint(0, max_top)
        
        crop_box = (crop_left, crop_top, crop_left + target_w, crop_top + target_h)
        
        # 3. Report the prescaled size to the model.
        reported_original_size = prescale_size
        
    else:
        # --- MODE B: Crop-from-Original (Detail View) ---
        # 1. No prescaling is done.
        prescale_size = None
        
        # 2. Determine the crop window that has the target aspect ratio.
        target_ar = target_w / target_h
        original_ar = original_w / original_h
        
        if original_ar > target_ar: # Original is wider than target AR
            crop_h = original_h
            crop_w = int(target_ar * crop_h)
        else: # Original is taller or same
            crop_w = original_w
            crop_h = int(crop_w / target_ar)

        # 3. Jitter the crop window within the original image.
        max_left = original_w - crop_w
        max_top = original_h - crop_h
        crop_left = random.randint(0, max_left)
        crop_top = random.randint(0, max_top)
        
        crop_box = (crop_left, crop_top, crop_left + crop_w, crop_top + crop_h)
        
        # 4. Report the true original size to the model.
        reported_original_size = original_size
        
    return prescale_size, crop_box, reported_original_size

def generate_ar_buckets(target_area: int, step: int = 64, max_ratio: float = 4.0):
    """
    Generates a list of (width, height) buckets with a target area, where dimensions
    are multiples of a given step.
    """
    buckets = set()
    min_dim = int(math.sqrt(target_area / max_ratio) // step) * step
    max_dim = int(math.sqrt(target_area * max_ratio) // step) * step

    for w in range(min_dim, max_dim + step, step):
        if w == 0: continue
        # Calculate height that gets closest to the target area
        h = int(round(target_area / w / step)) * step
        if h == 0: continue
        buckets.add((w, h))
        
    # Also consider swapped aspect ratios
    for h in range(min_dim, max_dim + step, step):
        if h == 0: continue
        w = int(round(target_area / h / step)) * step
        if w == 0: continue
        buckets.add((w, h))
        
    return sorted(list(buckets))

#hopefully deprecated
def get_transform_params(
    original_size: tuple[int, int],
    buckets: list[tuple[int, int]],
    k_nearest: int = 3,
    sharpness: float = 1.0,
    prescale_perc: float = 0.2, # The new hyperparameter for global (downscaled) views vs superresolution cropped views of data
):
    """
    Determines a full set of transformation parameters for an image, choosing between
    a "global view" (prescale-then-crop) and a "detail view" (crop-from-original).

    Returns:
        prescale_size (tuple or None): The size for the initial resize.
        crop_box (tuple): The box to crop from the (potentially prescaled) image.
        target_size (tuple): The final resolution to resize to.
        reported_original_size (tuple): The "original size" to report to add_time_ids.
    """
    original_w, original_h = original_size
    original_ar = original_w / original_h

    # Calculate distance to each bucket's aspect ratio
    bucket_distances = []
    for bucket_w, bucket_h in buckets:
        bucket_ar = bucket_w / bucket_h
        # Use log-space difference for better perceptual distance
        distance = abs(math.log(original_ar) - math.log(bucket_ar))
        bucket_distances.append((distance, (bucket_w, bucket_h)))

    # Select the k-nearest buckets
    bucket_distances.sort(key=lambda x: x[0])
    nearest_buckets = bucket_distances[:k_nearest]

    # Create a probability distribution using softmax
    distances = np.array([-d for d, _ in nearest_buckets])
    probabilities = np.exp((distances - np.max(distances)) * sharpness)
    probabilities /= probabilities.sum()

    # Sample a target bucket based on the distribution
    chosen_idx = np.random.choice(len(nearest_buckets), p=probabilities)
    _, target_size = nearest_buckets[chosen_idx]
    target_w, target_h = target_size
    target_ar = target_w / target_h

    # Determine the crop box to match the target aspect ratio
    if original_ar > target_ar:
        # Original is wider than target -> crop width
        new_w = int(target_ar * original_h)
        offset = (original_w - new_w) // 2
        crop_box = (offset, 0, offset + new_w, original_h)
    else:
        # Original is taller than target -> crop height
        new_h = int(original_w / target_ar)
        offset = (original_h - new_h) // 2
        crop_box = (0, offset, original_w, offset + new_h)
        
    return crop_box, target_size


def process_image(
    image: Image.Image, 
    prescale_size: Optional[tuple], 
    crop_box: tuple, 
    target_size: tuple
):
    """
    Applies an optional prescale, a crop, and a final resize, handling both
    "global view" (prescale-then-crop) and "detail view" (crop-then-resize) modes.
    """
    # --- MODE A: Prescale-then-Crop (Global View) ---
    if prescale_size is not None:
        # 1. First, resize the entire image to the calculated prescale size.
        image = image.resize(prescale_size, Image.LANCZOS)
        
        # 2. Then, crop the target window from the prescaled image.
        # The result of this crop is already at the target_size, so no final resize is needed.
        return image.crop(crop_box)
        
    # --- MODE B: Crop-from-Original (Detail View) ---
    else:
        # 1. First, crop the detail window from the full-resolution original image.
        cropped_image = image.crop(crop_box)
        
        # 2. Then, resize that (potentially large) crop down to the target bucket size.
        return cropped_image.resize(target_size, Image.LANCZOS)

def create_prefetch_generator(iterable, num_prefetch=2):
    """
    Creates a generator that fetches items in a background thread.
    'iterable' should be a generator or iterator that yields your data.
    """
    q = queue.Queue(maxsize=num_prefetch)
    sentinel = object()  # Marker for the end of the iterator

    def producer():
        for item in iterable:
            q.put(item)
        q.put(sentinel)

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    def consumer():
        while True:
            item = q.get()
            if item is sentinel:
                break
            yield item
            q.task_done()

    return consumer()

def batch_generator(
    folder_main: str,
    folders: list,
    scales: list,
    scales_unique: list,
    buckets: list,
    k: int,
    sharpness: float,
    batch_size: int,
    n_tuple: int,
    max_denoising_steps: int,
    lock_target_size:bool = True,
):
    """
    Yields packets of structured data for N-tuple contrastive training.
    Respects the arbitrary mapping between scales and folder names.
    """
    # Create numpy arrays for efficient lookup
    scales_array = np.array(scales)
    folders_array = np.array(folders)
    
    while True:
        try:
            # --- Initialize lists for the batch ---
            batch_images = []         # Shape: (B, N) of PILs
            batch_scales = []         # Shape: (B, N) of floats
            batch_orig_sizes = []     # Shape: (B, 2) of ints
            batch_crop_boxes = []     # Shape: (B, 4) of ints
            batch_target_sizes = []   # Shape: (B, 2) of ints
            batch_seeds = []          # Shape: (B,) of ints
            batch_timesteps = []

            if lock_target_size:
                # === BATCH-LEVEL SETUP ===
                # 1. Select a representative image to decide the AR for the whole batch.
                # (This logic to get a random image can be optimized, but is clear for now)
                sample_folder = folders_array[0]
                ims_path = os.path.join(folder_main, sample_folder)
                ims = [f for f in os.listdir(ims_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
                if not ims: continue
                rep_image_name = random.choice(ims)
                rep_img_path = os.path.join(ims_path, rep_image_name)
                with Image.open(rep_img_path) as rep_img:
                    rep_original_size = rep_img.size

                # 2. Lock the target size for this entire microbatch.
                target_size = select_target_bucket(
                    rep_original_size, buckets, k_nearest=k, sharpness=sharpness
                )
            else:
                print(f"bro u need to lock target sizes... or implement a batching semantics like paged attention supporting mixed tensor shapes as inputs...") 
                raise NotImplementedError 

            # === TUPLE-LEVEL PROCESSING ===
            for _ in range(batch_size):
                # 1. Select the image name for this specific tuple.
                image_name = random.choice(ims)
                
                # 2. Get its true original size.
                img_path_for_size = os.path.join(ims_path, image_name)
                with Image.open(img_path_for_size) as img_for_size:
                    original_size = img_for_size.size

                # 3. Determine the transform for this tuple using the locked target size.
                # The transform applied to all N images in this tuple will be identical.
                (
                    prescale_size, 
                    crop_box, 
                    reported_original_size
                ) = determine_transform_params(
                    original_size, target_size, prescale_perc=0.2
                )
                # 4. Select N concepts (scales) for this tuple
                sampled_scales = sorted(random.sample(scales_unique, n_tuple))
                
                # 5. Load the N corresponding images using the scale-to-folder map
                images_for_tuple = []
                # 4. Load the N images for the tuple and apply the *exact same transform* to all.
                for scale in sampled_scales:
                    folder_name = folders_array[scales_array == scale][0]
                    img_path = os.path.join(folder_main, folder_name, image_name)
                    with Image.open(img_path) as img:
                        processed_img = process_image(
                            img.convert("RGB"), prescale_size, crop_box, target_size
                        )
                        images_for_tuple.append(processed_img)
                
                # For each tuple, sample a timestep index
                tuple_timestep_index = random.randint(1, max_denoising_steps - 1)
                # 6. Append all data for this tuple to the batch lists
                batch_images.append(images_for_tuple)
                batch_scales.append(sampled_scales)
                batch_orig_sizes.append(reported_original_size)
                batch_crop_boxes.append(crop_box)
                batch_target_sizes.append(target_size)
                batch_seeds.append(random.randint(0, 2**32 - 1))
                batch_timesteps.append(tuple_timestep_index)

            if not batch_images: continue

            # --- Final Tensor Conversion and Broadcasting ---
            b_dim, n_dim = len(batch_scales), len(batch_scales[0])

            timesteps_tensor = torch.tensor(batch_timesteps, dtype=torch.int64)
            broadcast_timesteps = timesteps_tensor.unsqueeze(1).expand(-1, n_tuple)
            
            yield {
                "images":           batch_images,
                "scales":           torch.tensor(batch_scales, dtype=torch.float32),
                "target_sizes":     torch.tensor(batch_target_sizes, dtype=torch.int32).unsqueeze(1).expand(b_dim, n_dim, 2),
                "original_sizes":   torch.tensor(batch_orig_sizes, dtype=torch.int32).unsqueeze(1).expand(b_dim, n_dim, 2),
                "crop_coords":      torch.tensor(batch_crop_boxes, dtype=torch.int32).unsqueeze(1).expand(b_dim, n_dim, 4),
                "tuple_prng_seeds": torch.tensor(batch_seeds, dtype=torch.int64).unsqueeze(1).expand(b_dim, n_dim),
                "timesteps_to":     broadcast_timesteps
            }
            
        except (IOError, FileNotFoundError, IndexError) as e:
            print(f"Warning: Skipping batch due to error in generator: {e}")
            continue

def unflatten_and_split(
        flat_tensor: torch.Tensor, 
        batch_size: int, 
        n_tuple: int
    ) -> tuple[torch.Tensor, ...]:
        """
        Takes a flat tensor of shape (B*N, ...) and splits it into a tuple
        of N tensors, each of shape (B, ...).
        """
        # 1. Get the shape of the trailing dimensions (...)
        trailing_dims = flat_tensor.shape[1:]
        
        # 2. Reshape the flat tensor back to its logical (B, N, ...) structure
        logical_shape = (batch_size, n_tuple) + trailing_dims
        logical_tensor = flat_tensor.view(logical_shape)
        
        # 3. Split along the N_TUPLE dimension (dim=1)
        # This returns a tuple of N tensors, each of shape (B, 1, ...)
        split_tensors = torch.split(logical_tensor, 1, dim=1)
        
        # 4. Squeeze to remove the leftover 'N' dimension from each tensor
        # The final result is a tuple of N tensors, each of shape (B, ...)
        return tuple(tensor.squeeze(1) for tensor in split_tensors)


def flush():
    torch.cuda.empty_cache()
    gc.collect()


def train(
    config: RootConfig,
    prompts: list[PromptSettings],
    device,
    folder_main: str,
    folders,
    scales,
    
):
    scales = np.array(scales)   #why was nparray?
    folders = np.array(folders)
    scales_unique = list(scales)

    metadata = {
        "prompts": ",".join([prompt.json() for prompt in prompts]),
        "config": config.json(),
    }
    save_path = Path(config.save.path)

    #obsolete module selection code!
    #modules = DEFAULT_TARGET_REPLACE
    #if config.network.type == "c3lier":
    #    modules += UNET_TARGET_REPLACE_MODULE_CONV

    if config.logging.verbose:
        print(metadata)

    #if config.logging.use_wandb:
    #    wandb.init(project=f"LECO_{config.save.name}", config=metadata)

    weight_dtype = config_util.parse_precision(config.train.precision)
    save_weight_dtype = config_util.parse_precision(config.train.precision)

    (
        tokenizers,
        text_encoders,
        unet,
        noise_scheduler,
        vae
    ) = model_util.load_models_xl(
        config.pretrained_model.name_or_path,
        scheduler_name=config.train.noise_scheduler,
    )

        
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

    # --- NEW: Define bucketing parameters ---
    TARGET_AREA = 512 * 512
    BUCKET_STEP = 64
    K_NEAREST_BUCKETS = 4 # How many nearby aspect ratios to consider for sampling
    SAMPLING_SHARPNESS = 5.0 # Higher value -> more likely to pick the closest AR

    print(f"Generating aspect ratio buckets for target area {TARGET_AREA}...")
    ASPECT_RATIO_BUCKETS = generate_ar_buckets(TARGET_AREA, BUCKET_STEP)
    print(f"✅ Generated {len(ASPECT_RATIO_BUCKETS)} buckets.")

    LOGGING_INTERVAL = 100  # Log stats every 100 steps
    sampled_ar_counts = {}

    # ===============================================================================
    # VVVVVVVVVVVVVVVVVVVVVVVV MODIFICATION START VVVVVVVVVVVVVVVVVVVVVVVVVV
    # ===============================================================================
    if hasattr(config.train, "n_tuple"):
        N_TUPLE = config.train.n_tuple
    else:
        N_TUPLE = 2 # Or 8, 32, etc.
        print(f"defaulted ur n_tuple to n:{N_TUPLE}")
    if hasattr(config.train, "batch_size"):
        BATCH_SIZE = config.train.batch_size
    else:
        BATCH_SIZE = 2 # Or 8, 32, etc.
        print(f"defaulted ur batchsize to n:{BATCH_SIZE}")

    # --- Initialize the new batch-aware prefetcher ---
    data_gen = batch_generator(
        folder_main, folders, scales, scales_unique,
        ASPECT_RATIO_BUCKETS, K_NEAREST_BUCKETS, SAMPLING_SHARPNESS,
        batch_size=BATCH_SIZE,
        n_tuple=N_TUPLE,
        max_denoising_steps = config.train.max_denoising_steps
    )
    prefetcher = create_prefetch_generator(data_gen, num_prefetch=2)
    # ----------------------------------------

    for text_encoder in text_encoders:
        text_encoder.to(device, dtype=weight_dtype)
        text_encoder.requires_grad_(False)
        text_encoder.eval()

    unet.to(device, dtype=weight_dtype)
    if config.other.use_pytorch_SDPA:
        from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
        enable_cudnn_sdp(True)
        enable_flash_sdp(True)
    elif config.other.use_xformers:
        unet.enable_xformers_memory_efficient_attention()

    unet.requires_grad_(False)
    unet.eval()
    
    vae.to(device, dtype=weight_dtype)
    vae.requires_grad_(False)
    vae.eval()

    if config.other.torch_compile:
        unet = torch.compile(unet)
    
    #lycorisized = False
    #no more lycorisize
    
    # --- Option 1: Use a predefined training method ---
    # This replicates the old behavior but correctly includes FFNs within the scope.
    # For example, 'xattn' will now correctly target Linear layers inside 'attn2' blocks,
    # including the FFNs if they were part of that block's naming scheme.
    #    resolved_config = {
    #    "lora_map": lora_config_util.get_lora_map_for_training_method(
    #        unet, rank=64, alpha=32.0, method='full' # or 'xattn', 'selfattn', etc.
    #    )
    #}

    # --- Option 2: Use custom, flexible filters (Recommended) ---
    # This is more powerful and explicit. Let's train only the FFNs and cross-attention 'to_q' and 'to_k'.
    resolved_config_custom = {
        "lora_map": lora_alt.create_lora_config_map(
            unet,
            rank=config.network.rank,
            alpha=config.network.alpha,
            lora_type=config.network.type,  # <-- SPECIFY THE ADAPTER TYPE HERE: 'lora', 'glora'
            include_substrings=['ff.net', 'attn2.to_q', 'attn2.to_k','attn1.to_v','attn1.to_out'],
            exclude_substrings=[] # Example: exclude the first downblock: ['down_blocks.0']
        )
    }

    network = lora_alt.AltLoRANetwork(
        unet, resolved_config=resolved_config_custom
    ).to(device, dtype=weight_dtype)

    optimizer_module = train_util.get_optimizer(config.train.optimizer)
    #optimizer_args
    optimizer_kwargs = {}
    if config.train.optimizer_args is not None and len(config.train.optimizer_args) > 0:
        for arg in config.train.optimizer_args.split(" "):
            key, value = arg.split("=")
            value = ast.literal_eval(value)
            optimizer_kwargs[key] = value
    
    # A1. Create a list to hold our new, granular parameter groups.
    optimizer_param_groups = []

    optimizer_param_groups = network.prepare_optimizer_params(
        #alpha_lr=config.train.alpha_lr # (assuming you add this to your config)
    )


    optimizer_kwargs['weight_decay'] = 0.0

    print(f"✅ Created {len(optimizer_param_groups)} granular parameter groups for the optimizer.")

    #trivial optimization over naive adamw
    if config.train.optimizer.lower() == "adamw" and device.type == 'cuda':
        optimizer_kwargs["fused"] = True
        print("✅ Using fused AdamW optimizer.")

    # A4. Pass the LIST of groups to the optimizer.
    optimizer = optimizer_module(
        optimizer_param_groups, 
        lr=config.train.lr, 
        **optimizer_kwargs
    )

    lr_scheduler = train_util.get_lr_scheduler(
        config.train.lr_scheduler,
        optimizer,
        max_iterations=config.train.iterations,
        lr_min=config.train.lr / 100,
    )
    # Change the reduction method to 'none'.
    # This will make criteria(pred, target) return a tensor with the same shape as the inputs.
    criteria = torch.nn.MSELoss(reduction='none')

    print("Prompts")
    for settings in prompts:
        print(settings)

    # debug
    debug_util.check_requires_grad(network)
    debug_util.check_training_mode(network)

    cache = PromptEmbedsCache()
    prompt_pairs: list[PromptEmbedsPair] = []

    # no more cudastreams

    #"for settings in prompts" seems to suggest u can run more than one prompt pair in a prompt config?
    #i think unconditional might be about specifying the negative prompt at inference time.
    #which leaves 'target' to be the positive prompt invariant and 'positive' to be the positive prompt variant. 
    #...
    #literally:loss = 
    #target_latents - (neutral_latents + self.guidance_scale * (positive_latents - unconditional_latents))
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
    pbar = tqdm(range(config.train.iterations))

    loss = None

    for i in pbar:
        with torch.no_grad():
            noise_scheduler.set_timesteps(
                config.train.max_denoising_steps, device=device
            )
            prompt_pair: PromptEmbedsPair = random.choice(prompt_pairs)
            # 1 ~ 49 からランダム
            #timesteps_to = torch.randint(
            #    1, config.train.max_denoising_steps, (1,)
            #).item()

            # --- Step A: Get the structured data packet from the generator ---
            batch_packet = next(prefetcher)

            #    Shape: (B, N) -> (B,)
            timesteps_to_indices = batch_packet["timesteps_to"][:, 0]
            
            # Update aspect ratio logging
            # We take the size from the first tuple in the batch
            ar_log_size = tuple(batch_packet["target_sizes"][0, 0].tolist())
            sampled_ar_counts[ar_log_size] = sampled_ar_counts.get(ar_log_size, 0) + 1
            
            # --- Step B: Prepare Correlated Noisy Latents ---
            # This single call replaces all the old manual VAE/noise logic.
            latent_packet = train_util.prepare_correlated_noisy_latents(
                batch_packet=batch_packet,
                vae=vae,
                scheduler=noise_scheduler,
                timesteps_to=timesteps_to_indices,
                dtype=weight_dtype
            )

            # Step C: Prepare Prompt Conditioning (The "shameful" explicit call)
            combined_text_embeds, combined_pooled_embeds = train_util.broadcast_prompts_to_n_tuple(
                positive_text_embeds=prompt_pair.positive.text_embeds,
                positive_pooled_embeds=prompt_pair.positive.pooled_embeds,
                neutral_text_embeds=prompt_pair.neutral.text_embeds,
                neutral_pooled_embeds=prompt_pair.neutral.pooled_embeds,
                scales=batch_packet["scales"]
            )

            # Step D: Prepare Time Conditioning (Fixing the SDXL API contract)
            combined_add_time_ids = train_util.batch_add_time_ids(
                original_sizes=batch_packet["original_sizes"],
                crop_coords=batch_packet["crop_coords"],
                target_sizes=batch_packet["target_sizes"],
                dtype=weight_dtype,
                device=device
            )

            #    Shape: (B,) -> (B,)
            kilostep_indices = (timesteps_to_indices * (1000 / config.train.max_denoising_steps)).long()
            #    Set the scheduler to the 1000-step 'pseudocontinuous' values.
            noise_scheduler.set_timesteps(1000)
            #    Shape: (B,)
            current_timesteps_per_tuple = noise_scheduler.timesteps[kilostep_indices]
            # Get the sigma for the current timestep
            current_sigmas_per_tuple = noise_scheduler.sigmas[kilostep_indices]
            unet_timesteps = current_timesteps_per_tuple.unsqueeze(1).expand(-1, N_TUPLE).reshape(-1)
            kilostep_indices = kilostep_indices.unsqueeze(1).expand(-1, N_TUPLE).reshape(-1)
            # Calculate a loss weight. A common formulation from Karras et al.
            # Shape: (B,) -> (B, 1) -> (B, N) -> (B*N, 1, 1, 1) for broadcasting with latents
            loss_weights = (1.0 / (current_sigmas_per_tuple**2 + 1.0))
            loss_weights = loss_weights.unsqueeze(1).expand(-1, N_TUPLE).reshape(-1, 1, 1, 1)

                # Assemble all UNet inputs in one final, clean call
            unet_args, unet_kwargs = train_util.sdxl_condnet_batchjoin(
                noisy_latents=latent_packet["noisy_latents"],
                text_embeddings=combined_text_embeds,
                pooled_embeddings=combined_pooled_embeds,
                time_ids=combined_add_time_ids,
                scheduler=noise_scheduler,
                timestep=unet_timesteps # Pass the 1000-step-schedule timestep
            )

        network.set_lora_scales(torch.flatten(batch_packet["scales"]).to(device)) # Set per-sample scales
        # Call the UNet directly and explicitly
        target_latents = unet(*unet_args, **unet_kwargs).sample.to(device, dtype=weight_dtype)
        
        # C, H, W = latent_packet["noisy_latents"].shape[1:]
        # --- 4. LOSS CALCULATION & BACKWARD PASS ---
        target_latents_high, target_latents_low = unflatten_and_split(target_latents, BATCH_SIZE, N_TUPLE)
        
        # We need to split all ground truth tensors from the latent_packet as well.
        high_noise, low_noise = unflatten_and_split(latent_packet["ground_truth_noise"], BATCH_SIZE, N_TUPLE)
        noisy_latents_high, noisy_latents_low = unflatten_and_split(latent_packet["noisy_latents"], BATCH_SIZE, N_TUPLE)
        x0_latents_high, x0_latents_low = unflatten_and_split(latent_packet["x0_latents"], BATCH_SIZE, N_TUPLE)
        # You will need to split the `loss_weights` tensor just like the others:
        loss_weights_high, loss_weights_low = unflatten_and_split(loss_weights, BATCH_SIZE, N_TUPLE)
        kilostep_indices_high, kilostep_indices_low = unflatten_and_split(kilostep_indices, BATCH_SIZE, N_TUPLE)

        # Loss calculation logic is the same, just applied to the split tensors
        aux_losses = {}

        # The forward pass and loss calculation are dispatched to s_high
        loss_eps_high = criteria(target_latents_high, high_noise.to(weight_dtype))
        # 2. Mean reduce over spatial and channel dims. Shape: (B,)
        # loss_weights_high has shape (B, 1, 1, 1), so we squeeze it to (B,)
        loss_eps_high = loss_weights_high.squeeze() * loss_eps_high.mean(dim=(1, 2, 3))
        # B. data prediction from eps extrapolation loss
        pred_x0_high = train_util.get_x0_from_xt_eps(
                noisy_latents_high, target_latents_high, kilostep_indices_high, noise_scheduler
            )
        loss_x0_high = criteria(pred_x0_high.to(weight_dtype), x0_latents_high.to(weight_dtype))
        loss_x0_high_karrasnormed = loss_weights_high.squeeze()*loss_x0_high.mean(dim=(1, 2, 3))
        aux_losses["high"]=(loss_x0_high_karrasnormed.detach().mean())
        # C. Combine the losses
        lambda_x0 = getattr(config.train, 'lambda_x0_loss', 0)
        lambda_std = getattr(config.train, 'lambda_std_loss', 0)
        mean_loss_high, std_loss_high = train_util.statistical_matching_loss(target_latents_high)
        aux_losses["high_std"]=(std_loss_high.detach().mean()) 
        #(Shape: B,)
        loss_high = loss_eps_high + lambda_x0 * loss_x0_high_karrasnormed + lambda_std * std_loss_high
        #(shape: 1)
        loss_high = loss_high.mean() / ACCUMULATION_STEPS
        
        loss_eps_low = criteria(target_latents_low, low_noise.to(weight_dtype))
        # 2. Mean reduce over spatial and channel dims. Shape: (B,)
        # loss_weights_high has shape (B, 1, 1, 1), so we squeeze it to (B,)
        loss_eps_low = loss_weights_low.squeeze() * loss_eps_low.mean(dim=(1, 2, 3))
        # B. data prediction from eps extrapolation loss
        lambda_x0 = getattr(config.train, 'lambda_x0_loss', 0)
        lambda_std = getattr(config.train, 'lambda_std_loss', 0)
        pred_x0_low = train_util.get_x0_from_xt_eps(
                noisy_latents_low, target_latents_low, kilostep_indices_low, noise_scheduler
            )
        loss_x0_low = criteria(pred_x0_low.to(weight_dtype), x0_latents_low.to(weight_dtype))
        loss_x0_low_karrasnormed = loss_weights_low.squeeze()*loss_x0_low.mean(dim=(1, 2, 3))
        aux_losses["low"]=(loss_x0_low_karrasnormed.detach().mean())
        mean_loss_low, std_loss_low = train_util.statistical_matching_loss(target_latents_low)
        aux_losses["low_std"]=(std_loss_low.detach().mean()) 
        # C. Combine the losses
        loss_low = loss_eps_low + lambda_x0 * loss_x0_low_karrasnormed + lambda_std * std_loss_low
        loss_low = loss_low.mean() / ACCUMULATION_STEPS

        total_loss = loss_high+loss_low
        total_loss.backward()

        if len(aux_losses.keys()) >= 4:
            pbar.set_description(f"Loss_high/low*1k: {loss_high.item()*1000:.1f}/{loss_low.item()*1000:.1f};{aux_losses['high'].item()*1000:.1f}/{aux_losses['low'].item()*1000:.1f};{aux_losses['high_std'].item()*1000:.1f}/{aux_losses['low_std'].item()*1000:.1f}")
        else:
            pbar.set_description(f"Loss_high/low*1k: {loss_high.item()*1000:.1f}/{loss_low.item()*1000:.1f}")
        aux_losses = {}
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
        
        del (
            target_latents_low,
            target_latents_high,
        )
        #flush()

        ### INTERVALED LOGGING AND PERSISTENCE:
        # --- NEW: Periodically log the sampled aspect ratio distribution ---
        if (i + 1) % LOGGING_INTERVAL == 0 and i > 0:
            total_samples = sum(sampled_ar_counts.values())
            print(f"\n--- Aspect Ratio Sampling Distribution (Step {i + 1}/{config.train.iterations}) ---")
            print(f"{'Resolution':<15} | {'AR':<8} | {'Count':<7} | {'Percent':<7}")
            print("-" * 50)
            
            # Sort by count in descending order for readability
            sorted_counts = sorted(sampled_ar_counts.items(), key=lambda item: item[1], reverse=True)
            
            for (w, h), count in sorted_counts:
                aspect_ratio = w / h
                percentage = (count / total_samples) * 100
                print(f"({w}, {h}){'':<6} | {aspect_ratio:.2f}:1{'':<3} | {count:<7} | {percentage:.2f}%")
            print("-" * 50)
        # -----------------------------------------------------------------
        if (
            i % config.save.per_steps == 0
            and i != 0
            and i != config.train.iterations - 1
        ):
            print("Saving...")
            save_path.mkdir(parents=True, exist_ok=True)
            network.save_weights(
                save_path / f"{config.save.name}_{i}steps.safetensors",
                dtype=save_weight_dtype,
                metadata=None,
            )

    print("Saving...")
    save_path.mkdir(parents=True, exist_ok=True)
    network.save_weights(
        save_path / f"{config.save.name}_last.safetensors",
        dtype=save_weight_dtype,
        metadata=None,
    )

    del (
        unet,
        noise_scheduler,
        loss,
        optimizer,
        network,
    )

    flush()

    print("Done.")


# uhhh check what attention backend is available
from torch.backends.cuda import sdp_kernel, SDPBackend
def log_attention_backend_status(worker_pid):
    """Checks and logs the status of the selected PyTorch attention backend."""
    if not torch.cuda.is_available():
        print(f"[{worker_pid}] CUDA not available. Using default CPU attention backend.")
        return

    # Check which backend was successfully enabled by the sdp_kernel context
    if torch.backends.cuda.flash_sdp_enabled():
        print(f"[{worker_pid}] ✅ PyTorch attention backend set to FLASH ATTENTION (Fastest)")
    elif torch.backends.cuda.mem_efficient_sdp_enabled():
        print(f"[{worker_pid}] ✅ PyTorch attention backend set to MEMORY-EFFICIENT (xFormers/Native)")
    else:
        print(f"[{worker_pid}] ⚠️ PyTorch attention backend fell back to MATH (Eager/Slowest)")

def main(args):
    config_file = args.config_file
    config = config_util.load_config_from_yaml(config_file)

    #meta
    if args.name is not None:
        config.save.name = args.name
    attributes = []
    if args.attributes is not None:
        attributes = args.attributes.split(',')
        attributes = [a.strip() for a in attributes]
    
    config.network.alpha = args.alpha
    config.network.rank = args.rank
    config.save.name += f'_alpha{args.alpha}'
    config.save.name += f'_rank{config.network.rank }'
    config.save.name += f'_{config.network.training_method}'
    config.save.path += f'/{config.save.name}'
    
    #cuda
    device = torch.device(f"cuda:{args.device}")
    worker_pid = os.getpid()
    log_attention_backend_status(worker_pid)

    #data
    prompts = prompt_util.load_prompts_from_yaml(config.prompts_file, attributes)

    
    folders = args.folders.split(',')
    folders = [f.strip() for f in folders]
    #why were these being passed as strings :(((
    #scales = args.scales.split(',')
    #scales = [f.strip() for f in scales]
    #scales = [int(s) for s in scales]
    #scales = [s for s in scales]
    scales = args.scales
    
    print(folders, scales)
    if len(scales) != len(folders):
        raise Exception('the number of folders need to match the number of scales')
    
    if args.stylecheck is not None:
        check = args.stylecheck.split('-')
        
        for i in range(int(check[0]), int(check[1])):
            folder_main = args.folder_main+ f'{i}'
            config.save.name = f'{os.path.basename(folder_main)}'
            config.save.name += f'_alpha{args.alpha}'
            config.save.name += f'_rank{config.network.rank }'
            config.save.path = f'models/{config.save.name}'
            train(config=config, prompts=prompts, device=device, folder_main = folder_main, folders = folders, scales = scales)
    else:
        train(config=config, prompts=prompts, device=device, folder_main = args.folder_main, folders = folders, scales = scales)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        required=True,
        help="Config file for training.",
    )
    # config_file 'data/config.yaml'
    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="LoRA weight.",
    )
    # --alpha 1.0
    parser.add_argument(
        "--rank",
        type=int,
        required=False,
        help="Rank of LoRA.",
        default=4,
    )
    # --rank 4
    parser.add_argument(
        "--device",
        type=int,
        required=False,
        default=0,
        help="Device to train on.",
    )
    # --device 0
    parser.add_argument(
        "--name",
        type=str,
        required=False,
        default=None,
        help="Device to train on.",
    )
    # --name 'eyesize_slider'
    parser.add_argument(
        "--attributes",
        type=str,
        required=False,
        default=None,
        help="attritbutes to disentangle (comma seperated string)",
    )
    parser.add_argument(
        "--folder_main",
        type=str,
        required=True,
        help="The folder to check",
    )
    
    parser.add_argument(
        "--stylecheck",
        type=str,
        required=False,
        default = None,
        help="The folder to check",
    )
    
    parser.add_argument(
        "--folders",
        type=str,
        required=False,
        default = 'verylow, low, high, veryhigh',
        help="folders with different attribute-scaled images",
    )
    parser.add_argument(
        "--scales",
        type=float, #this was string. why was string????
        required=False,
        nargs='*',
        default = '-2, -1, 1, 2',
        help="scales for different attribute-scaled images",
    )
    
    
    args = parser.parse_args()

    main(args)
