# d_dset_functions.py
# A library of pure, stateless asset-producing functions.
# Each function is a "tool" that can be called by the Asset Execution Engine (Layer 4).
# It knows nothing about the DAG, planning, or caching—only how to perform
# its specific computation on the primitives it receives.
import re
import os
import torch
import safetensors
from diffusers import UNet2DConditionModel, AutoencoderKL
from PIL import Image
from torchvision import transforms
import json
import time


"""
A note from gemini 2.5:
The True Power of the New Architecture
The most important part is what this enables for the future. Think back to our TheMask_Eval_Consumer test case.
    In v5.4: To add this text-to-text evaluation, a developer would have needed to modify the Python code in 
    Layer 3 and Layer 4 to make them aware of the new the_mask_language_objective interface.
    In v5.5: A developer needs to make zero changes to the Python source code. They simply:
        Add the TheMask_Eval_Consumer entry to the DATA_CONSUMER_REGISTRY in Layer 1.
        Add a new capability definition to the capability_implementation_map in the YAML manifest.
        Provide the actual Python function that performs the "masking" in d_dset_functions.py.
The core DAG runner and all its layer logic remain untouched. This is the "spiritual center" we were aiming for: 
a toolchain so generic and well-defined that it can orchestrate entirely
novel data processing tasks purely through declarative configuration.
"""

# ==============================================================================
# ==                      KEY REMAPPING TOOLKIT (v1)                          ==
# ==============================================================================
# This section implements the explicit key remapping strategy.
# It is designed to be self-contained and easily portable.
# ==============================================================================
# ==                      KEY REMAPPING TOOLKIT (v3 - Final)                  ==
# ==============================================================================
# This is the definitive, structure-aware translator that handles all known
# differences between the A1111/Kohya and Diffusers VAE naming conventions.

def translate_a1111_to_diffusers_vae(key: str) -> str:
    """Translates a single VAE key."""
    if not key.startswith("first_stage_model."):
        return key
    
    new_key = key[len("first_stage_model."):]

    # --- Pass 1: Handle the reversed decoder block indices ---
    if "decoder.up." in new_key:
        # Matches decoder.up.0.block.0... or decoder.up.0.upsample...
        match = re.search(r"decoder\.up\.(\d+)\.(block|upsample)", new_key)
        if match:
            block_index = int(match.group(1))
            block_type = match.group(2)
            
            # Reverse the index (3 -> 0, 2 -> 1, 1 -> 2, 0 -> 3)
            new_block_index = 3 - block_index
            
            if block_type == "block":
                # decoder.up.3.block.0 -> decoder.up_blocks.0.resnets.0
                new_key = new_key.replace(f"decoder.up.{block_index}.block", f"decoder.up_blocks.{new_block_index}.resnets")
            elif block_type == "upsample":
                # decoder.up.3.upsample -> decoder.up_blocks.0.upsamplers.0
                new_key = new_key.replace(f"decoder.up.{block_index}.upsample", f"decoder.up_blocks.{new_block_index}.upsamplers.0")
    
    # --- Pass 2: Handle general structural and name changes ---
    replacements = {
        # Structure
        "encoder.down.": "encoder.down_blocks.",
        "decoder.down.": "decoder.down_blocks.",
        "encoder.mid.block_1": "encoder.mid_block.resnets.0",
        "encoder.mid.block_2": "encoder.mid_block.resnets.1",
        "decoder.mid.block_1": "decoder.mid_block.resnets.0",
        "decoder.mid.block_2": "decoder.mid_block.resnets.1",
        "encoder.mid.attn_1": "encoder.mid_block.attentions.0",
        "decoder.mid.attn_1": "decoder.mid_block.attentions.0",
        ".downsample.": ".downsamplers.0.",

        # Naming
        ".block.": ".resnets.", # Must be after other block replacements
        "nin_shortcut": "conv_shortcut",
        "norm_out": "conv_norm_out",
    }
    for old, new in replacements.items():
        new_key = new_key.replace(old, new)
        
    # --- Pass 3: Handle the fine-grained attention block keys ---
    # This must be done last, after the main structure is in place.
    if "attentions" in new_key:
        attn_replacements = {
            ".norm.": ".group_norm.",
            ".q.": ".to_q.",
            ".k.": ".to_k.",
            ".v.": ".to_v.",
            ".proj_out.": ".to_out.0.",
        }
        for old, new in attn_replacements.items():
            new_key = new_key.replace(old, new)
            
    return new_key

# --- 3. Format Inference Logic ---
def infer_vae_sdxl_state_dict_format(keys: set) -> str:
    """Inspects a set of keys and returns the name of the inferred format."""
    # Check for A1111/Kohya signature key for VAE
    if "first_stage_model.encoder.down.0.block.0.norm1.weight" in keys:
        return "A1111_KOHYA"
    # Check for Diffusers signature key for VAE
    if "encoder.down_blocks.0.resnets.0.norm1.weight" in keys:
        return "Diffusers"
    # Add more checks for other formats here (e.g., "CompVis_v1")
    return "Unknown"

# --- 4. Key Application Logic ---
def apply_key_mapping(state_dict: dict, key_map: dict) -> dict:
    """Applies a key mapping to a state dictionary."""
    new_state_dict = {}
    for key, tensor in state_dict.items():
        new_key = key
        for old, new in key_map.items():
            if old in new_key:
                new_key = new_key.replace(old, new)
        new_state_dict[new_key] = tensor
    return new_state_dict


      
      
import torch
import multiprocessing as mp
from tensordict import TensorDict
from typing import Iterator, Callable, Dict, Tuple, List

# ==============================================================================
# ==        HIGH-PERFORMANCE PIPELINE UTILITIES (INTERNAL TO THIS MODULE)     ==
# ==============================================================================
# These are the reusable tools that our asset-producing functions will use.

def tensorize_and_collate_iterator(
    data_iterator: Iterator[Dict],
    preprocessing_fn: Callable[[Dict], Dict[str, torch.Tensor]]
) -> Iterator[TensorDict]:
    """Wraps a raw data iterator, yielding each item as a CPU TensorDict."""
    for item in data_iterator:
        try:
            tensor_data = preprocessing_fn(item)
            yield TensorDict(tensor_data, batch_size=[])
        except Exception as e:
            print(f"  - WARNING: Skipping item during tensorization: {e}")
            continue

def _loader_worker_process(
    cpu_queue: mp.Queue,
    raw_data_iterator: Iterator[Dict],
    preprocessing_fn: Callable,
    batch_size_queue: mp.Queue
):
    """
    A worker process that groups items into batches and puts CPU-side
    TensorDicts into a queue for the main process to transfer.
    It can dynamically change the batch size by reading from a queue.
    """
    current_bs = 1
    buffer = []
    while True:
        # Check for batch size updates without blocking
        try:
            current_bs = batch_size_queue.get_nowait()
        except mp.queues.Empty:
            pass

        try:
            # This is where the raw data is pulled and preprocessed into tensors
            item_td = next(tensorize_and_collate_iterator(raw_data_iterator, preprocessing_fn))
            buffer.append(item_td)

            if len(buffer) >= current_bs:
                # When batch is full, stack the TensorDicts and put on the queue
                batch_td = torch.stack(buffer, dim=0)
                cpu_queue.put(batch_td)
                buffer = []

        except StopIteration:
            # Handle the last partial batch
            if buffer:
                cpu_queue.put(torch.stack(buffer, dim=0))
            cpu_queue.put(None) # End-of-stream sentinel
            break

def _gpu_transfer_process(cpu_queue: mp.Queue, gpu_queue: mp.Queue, device: torch.device):
    """
    A dedicated process to transfer TensorDicts from a CPU queue to the GPU
    and put them on a GPU queue.
    """
    while True:
        cpu_td = cpu_queue.get()
        if cpu_td is None:
            gpu_queue.put(None) # Propagate the end-of-stream signal
            break
        
        # This process has its own CUDA context. It performs the transfer.
        gpu_td = cpu_td.pin_memory().to(device, non_blocking=True)
        gpu_queue.put(gpu_td)

class PipelinedTensorDictLoader:
    """
    Manages a truly concurrent data loading pipeline using dedicated processes
    for CPU preprocessing and GPU data transfer.
    """
    def __init__(self, raw_data_iterator, preprocessing_fn, device, initial_bs=1, buffer_count=2):
        self.device = device
        
        ctx = mp.get_context("spawn")
        self.cpu_queue = ctx.Queue(maxsize=buffer_count)
        self.gpu_queue = ctx.Queue(maxsize=buffer_count) # Queue for GPU-ready tensordicts
        self.batch_size_queue = ctx.Queue(maxsize=1)
        
        work_list = list(raw_data_iterator)
        if not work_list:
            raise ValueError("Data iterator is empty, cannot initialize pipeline.")
            
        # Start the CPU worker PROCESS
        self.cpu_worker = ctx.Process(
            target=_loader_worker_process,
            args=(self.cpu_queue, work_list, preprocessing_fn, self.batch_size_queue),
            daemon=True
        )
        self.cpu_worker.start()
        
        # THE FIX: Start the GPU transfer PROCESS
        self.gpu_worker = ctx.Process(
            target=_gpu_transfer_process,
            args=(self.cpu_queue, self.gpu_queue, self.device),
            daemon=True
        )
        self.gpu_worker.start()

    def _preallocate_buffers(self, data_iterator, preprocessing_fn, batch_size, buffer_count):
        print("[Loader] Pre-allocating GPU buffers...")
        first_item = next(tensorize_and_collate_iterator(data_iterator, preprocessing_fn))
        template_batch = first_item.expand(batch_size)
        buffer_list = [template_batch] * buffer_count
        on_device_pool = torch.stack(buffer_list, dim=0).to(self.device)
        print(f"[Loader] On-device buffer pool created with shape: {on_device_pool.shape}")
        return on_device_pool

    def get_next_batch(self) -> TensorDict:
        """Called by the main loop to get the next ready batch from the GPU queue."""
        gpu_td = self.gpu_queue.get()
        if gpu_td is None:
            raise StopIteration
        return gpu_td

    def set_batch_size(self, new_bs: int):
        try:
            self.batch_size_queue.get_nowait()
        except mp.queues.Empty:
            pass
        self.batch_size_queue.put(new_bs)

    def __iter__(self):
        # Start the refill thread
        self.refill_thread = mp.Process(target=self._refill_loop, daemon=True)
        self.refill_thread.start()
        return self

    def _refill_loop(self):
        """The background THREAD that moves data from CPU queue to GPU buffers."""
        while True:
            cpu_batch = self.cpu_queue.get()
            if cpu_batch is None:
                self.ready_buffers.put(None) # Propagate the end-of-stream signal
                break
            
            buffer_idx = self.free_buffers.get()
            
            # Update the on-device buffer in-place
            self.buffers[buffer_idx] = cpu_batch.pin_memory().to(
                self.device, non_blocking=True
            )
            self.ready_buffers.put(buffer_idx)
                
    def release_buffer(self, buffer_key: str):
        """Called by the main loop after it's done with a buffer."""
        self.free_buffers.put(buffer_key)

    def close(self):
        """Clean up the worker processes."""
        self.cpu_worker.terminate()
        self.gpu_worker.terminate()
        self.cpu_worker.join()
        self.gpu_worker.join()


def advise_next_batch_size(
    calibration_kwargs: Dict
) -> Tuple[int, Dict]:
    """
    Passively advises the next batch size based on the results of the last loop.
    This is a stateless calculator; all state is passed in via calibration_kwargs.
    """
    # Unpack state from the kwargs dictionary
    history: List[Dict] = calibration_kwargs.get("history", [])
    config: Dict = calibration_kwargs["config"] # VRAM budget, safety margins, etc.
    last_bs: int = calibration_kwargs.get("last_bs", 0)
    start_time: float = calibration_kwargs.get("start_time")

    # --- 1. Calculate results from the PREVIOUS loop ---
    if start_time is not None:
        duration = time.time() - start_time
        throughput = last_bs / duration if duration > 0 else 0
        vram_peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
        torch.cuda.reset_max_memory_allocated() # Reset for next measurement

        history.append({
            "bs": last_bs,
            "duration": duration,
            "throughput": throughput,
            "vram_peak_gb": vram_peak_gb
        })
        print(f"  [Advisor] Last Loop (BS={last_bs}): Throughput={throughput:.2f} items/s, VRAM Peak={vram_peak_gb:.2f}GB")

    # --- 2. Decide the NEXT batch size ---
    if not history: # First run
        next_bs = config.get("initial_batch_size", 1)
    else:
        last_run = history[-1]
        
        # --- VRAM Safety Check ---
        # A simple linear projection for marginal VRAM cost
        if len(history) > 1:
            prev_run = history[-2]
            marginal_vram = (last_run['vram_peak_gb'] - prev_run['vram_peak_gb']) / (last_run['bs'] - prev_run['bs'])
        else: # Estimate from first run
            marginal_vram = last_run['vram_peak_gb'] / last_run['bs']

        # Exponential warmup
        next_bs = last_run['bs'] * 2

        projected_vram = last_run['vram_peak_gb'] + marginal_vram * (next_bs - last_run['bs'])
        if projected_vram > config["vram_safety_budget_gb"]:
            print(f"  [Advisor] Exponential jump to {next_bs} too risky. Switching to linear scan.")
            next_bs = last_run['bs'] + 1 # Switch to linear increments
        
        # --- Throughput Check ---
        # Stop if throughput is degrading
        if len(history) > 1 and last_run['throughput'] < history[-2]['throughput'] * 0.98:
            print(f"  [Advisor] Throughput degraded. Converging to previous best.")
            best_run = max(history[:-1], key=lambda x: x['throughput'])
            next_bs = best_run['bs']
        else:
             # Re-check VRAM for the selected next_bs
            projected_vram = last_run['vram_peak_gb'] + marginal_vram * (next_bs - last_run['bs'])
            if projected_vram > config["vram_safety_budget_gb"]:
                 print(f"  [Advisor] Linear step to {next_bs} too risky. Converging to current BS.")
                 next_bs = last_run['bs']

    # --- 3. Return advice and state for the NEXT loop ---
    calibration_kwargs["history"] = history
    calibration_kwargs["last_bs"] = next_bs
    calibration_kwargs["start_time"] = time.time() # Start the timer for the upcoming loop

    return next_bs, calibration_kwargs

def scale_tensor_synthesizer_v1(**kwargs) -> dict:
    """
    [COMPLIANT] A simple function that now accepts an iterator and processes
    each item, returning a dict of results keyed by work_id.
    """
    data_iterator = kwargs['data_iterator']
    results = {}
    
    for spec in data_iterator:
        work_id = spec['work_id']
        scale_metadata = spec['input_data']['scale_metadata']
        # The core logic is tiny, but it's now wrapped to be compliant.
        results[work_id] = {"scales_tensor": scale_metadata}
        
    return results

def dummy_denoise_input_encoder(**kwargs) -> dict:
    """[COMPLIANT] STUB that now iterates and returns results by work_id."""
    results = {}
    for spec in kwargs['data_iterator']:
        results[spec['work_id']] = {
            "noisy_latent": "stub_latent", 
            "timestep_for_unet": "stub_timestep"
        }
    return results

def dummy_text_encoder(**kwargs) -> dict:
    """[COMPLIANT] STUB that now iterates and returns results by work_id."""
    results = {}
    for spec in kwargs['data_iterator']:
        results[spec['work_id']] = {
            "text_embedding": "stub_text_embed", 
            "pooled_text_embedding": "stub_pooled_embed"
        }
    return results

def dummy_time_id_synthesizer(**kwargs) -> dict:
    """[COMPLIANT] STUB that now iterates and returns results by work_id."""
    results = {}
    for spec in kwargs['data_iterator']:
        results[spec['work_id']] = {"time_embedding": "stub_time_embed"}
    return results

# ==============================================================================
# ==           EXAMPLE PREPROCESSORS (LOGIC BELONGS TO THE USER)            ==
# ==============================================================================

# --- For Image VAE Encoding ---
from PIL import Image
from torchvision import transforms

def sdxl_vae_image_preprocessor(item: Dict) -> Dict[str, torch.Tensor]:
    """Takes a data item, loads the image, and prepares it for the VAE."""
    #image_path = item['image_path']
    image_path = item['input_data']['image']
    image = Image.open(image_path).convert("RGB")
    target_dtype = item.get('dtype', torch.bfloat16)
    # ... (all the necessary resizing, normalization, etc.) ...
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))
    image = image.resize((w, h))
    image_tensor = transforms.ToTensor()(image)
    image_tensor = image_tensor * 2.0 - 1.0
    image_tensor = image_tensor.to(dtype=target_dtype)
    return {"image": image_tensor}

# --- For Fictitious Text Encoding ---
def dummy_prompt_recipe_parser(recipe: str) -> str:
    """Stub that pretends to parse a complex prompt structure."""
    return f"parsed: {recipe}"

def dummy_text_tokenizer(text: str) -> Dict[str, torch.Tensor]:
    """Stub that pretends to tokenize text for a CLIP model."""
    return {
        "input_ids": torch.randint(0, 49408, (77,)),
        "pooled_output": torch.randn(1280)
    }


def real_image_to_latent_encoder(**kwargs) -> dict:
    """
    A REAL asset-producing function that explicitly infers the model's key format,
    applies a named transformation, and then loads the VAE to encode an image.
    """
    # --- Standard setup from kwargs ---
    image_path = kwargs.get('image')
    model_path = kwargs.get('model_path')
    config_path = kwargs.get('config_path')
    device = kwargs.get('device', 'cpu')
    dtype_str = kwargs.get('dtype', 'bfloat16')
    work_iterator = kwargs['data_iterator']
    torch_dtype = torch.float32 if dtype_str == 'float32' else torch.bfloat16
    # ... (input validation) ...

    # --- 1. Load the raw state dictionary and all its keys ---
    raw_vae_keys = {}
    with safetensors.safe_open(model_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith("first_stage_model."):
                raw_vae_keys[key] = f.get_tensor(key)

    # --- 2. Infer format and apply mapping EXPLICITLY ---
    # key we look for is vae-specific so we use an infer_vae... fn
    inferred_format = infer_vae_sdxl_state_dict_format(raw_vae_keys)
    
    # MANDATORY LOG PRINT
    print("\n" + "="*80)
    print("== [MODEL LOADER] Key Format Inference Report")
    print(f"==  - Model Path: {os.path.basename(model_path)}")
    print(f"==  - Inferred Format: '{inferred_format}'")
    
    if inferred_format == "A1111_KOHYA":
        print("==  - Action: Applying 'A1111_KOHYA_TO_DIFFUSERS' key transformation.")
        # --- Apply the NEW, smart translation function ---
        working_state_dict = {
        translate_a1111_to_diffusers_vae(key): tensor 
        for key, tensor in raw_vae_keys.items()
    }
    elif inferred_format == "Diffusers":
        print("==  - Action: No key transformation needed. Loading keys as-is.")
    else:
        print("==  - WARNING: Unknown format. Attempting to load keys as-is. This may fail.")
    print("="*80 + "\n")

    # --- Reshape attention tensors ---
    # Iterate over a copy of the keys to allow modification
    for key in list(working_state_dict.keys()):
        # Target the specific projection weights in the attention blocks
        if "attentions." in key and key.endswith(".weight"):
            tensor = working_state_dict[key]
            if len(tensor.shape) == 4:
                # Squeeze the 4D Conv2d weight into a 2D Linear weight
                new_tensor = tensor.squeeze()
                working_state_dict[key] = new_tensor
                # the only mandatory log prints we got are about the necessity of applying a string remapping at all...
                #print(f"  [RESHAPE] Correcting tensor shape for '{key}': {tensor.shape} -> {new_tensor.shape}")

    # --- 3. Load the model with the corrected state dict ---
    with open(config_path, 'r') as f:
        vae_config = json.load(f)
    vae = AutoencoderKL.from_config(vae_config)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    vae.load_state_dict(working_state_dict) # This will now succeed
    vae.to(device=device, dtype=torch_dtype)
    
    # 2. Initialize the pipeline and advisor tools
    loader = PipelinedTensorDictLoader(
        raw_data_iterator=work_iterator,
        preprocessing_fn=sdxl_vae_image_preprocessor, # Pass the correct preprocessor
        device=device,
        initial_bs=1
    )
    calibration_state = {
        "config": {"vram_safety_budget_gb": 20.0, "initial_batch_size": 1},
        "history": []
    }
    all_results = []

    # 3. Run the internal processing loop
    while True:
        try:
            next_bs, calibration_state = advise_next_batch_size(calibration_state)
            loader.set_batch_size(next_bs)

            # The interface is now even simpler
            on_device_batch = loader.get_next_batch()

            with torch.no_grad():
                # The actual model computation
                posterior = vae.encode(on_device_batch['image']).latent_dist
                latents = posterior.sample() * vae.config.scaling_factor
            
            torch.cuda.synchronize() # Crucial for accurate timing
            all_results.append(latents.cpu())
            #loader.release_buffer(buffer_key)

        except StopIteration:
            print("[VAE Encoder] Finished processing all images.")
            break
            
    loader.close()

    # 4. Collate results and return in the format expected by the DAG
    final_latents = torch.cat(all_results, dim=0)
    print(f"--- [VAE Encoder] Complete. Produced final latent tensor of shape: {final_latents.shape} ---")
    return {"latent": final_latents}
    


def fictitious_sdxl_text_encoder(**kwargs) -> dict:
    """
    A STUBBED asset-producing function demonstrating how the *exact same* pipeline
    can encode text by simply swapping the preprocessor.
    """
    print("\n--- [Text Encoder] Starting high-performance batch encoding. ---")
    # 1. Unpack arguments
    work_iterator = kwargs['data_iterator']
    text_encoder_model = kwargs['text_encoder_model'] # A dummy model
    device = kwargs['device']

    # 2. Initialize the pipeline and advisor tools
    loader = PipelinedTensorDictLoader(
        raw_data_iterator=work_iterator,
        preprocessing_fn=sdxl_text_preprocessor, # SWAP to the text preprocessor
        device=device,
        initial_bs=4 # Text models often handle larger batches
    )
    calibration_state = {
        "config": {"vram_safety_budget_gb": 16.0, "initial_batch_size": 4},
        "history": []
    }
    all_embeds, all_pooled = [], []

    # 3. Run the internal processing loop (code is identical to the VAE one)
    while True:
        try:
            next_bs, calibration_state = advise_next_batch_size(calibration_state)
            loader.set_batch_size(next_bs)

            on_device_batch = loader.get_next_batch()

            with torch.no_grad():
                # Dummy model computation
                embeds, pooled = text_encoder_model(on_device_batch['input_ids'])

            torch.cuda.synchronize()
            all_embeds.append(embeds.cpu())
            all_pooled.append(pooled.cpu())
            #loader.release_buffer(buffer_key)

        except StopIteration:
            print("[Text Encoder] Finished processing all prompts.")
            break
            
    loader.close()

    # 4. Collate and return results
    final_embeds = torch.cat(all_embeds, dim=0)
    final_pooled = torch.cat(all_pooled, dim=0)
    print(f"--- [Text Encoder] Complete. Produced final tensors of shapes: {final_embeds.shape}, {final_pooled.shape} ---")
    return {"text_embedding": final_embeds, "pooled_text_embedding": final_pooled}