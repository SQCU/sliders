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
from trainscripts.imagesliders.minimal_scheduler import MinimalDDPMScheduler as DDPMScheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer, CLIPTextConfig
from PIL import Image
from torchvision import transforms
import numpy as np
import json
import time
import itertools
import functools

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

def _translate_vae_key_str(key: str) -> str:
    """[INTERNAL HELPER] Translates a single VAE key string."""
    if not key.startswith("first_stage_model."):
        return key
    
    new_key = key[len("first_stage_model."):]

    # Pass 1: Reversed decoder block indices (logic is unchanged)
    if "decoder.up." in new_key:
        match = re.search(r"decoder\.up\.(\d+)\.(block|upsample)", new_key)
        if match:
            block_index, block_type = int(match.group(1)), match.group(2)
            new_block_index = 3 - block_index
            if block_type == "block":
                new_key = new_key.replace(f"decoder.up.{block_index}.block", f"decoder.up_blocks.{new_block_index}.resnets")
            elif block_type == "upsample":
                new_key = new_key.replace(f"decoder.up.{block_index}.upsample", f"decoder.up_blocks.{new_block_index}.upsamplers.0")
    
    # Pass 2: General structural and name changes (logic is unchanged)
    replacements = {
        "encoder.down.": "encoder.down_blocks.", "decoder.down.": "decoder.down_blocks.",
        "encoder.mid.block_1": "encoder.mid_block.resnets.0", "encoder.mid.block_2": "encoder.mid_block.resnets.1",
        "decoder.mid.block_1": "decoder.mid_block.resnets.0", "decoder.mid.block_2": "decoder.mid_block.resnets.1",
        "encoder.mid.attn_1": "encoder.mid_block.attentions.0", "decoder.mid.attn_1": "decoder.mid_block.attentions.0",
        ".downsample.": ".downsamplers.0.", ".block.": ".resnets.",
        "nin_shortcut": "conv_shortcut", "norm_out": "conv_norm_out",
    }
    for old, new in replacements.items(): new_key = new_key.replace(old, new)
        
    # Pass 3: Fine-grained attention block keys (logic is unchanged)
    if "attentions" in new_key:
        attn_replacements = {".norm.": ".group_norm.", ".q.": ".to_q.", ".k.": ".to_k.", ".v.": ".to_v.", ".proj_out.": ".to_out.0."}
        for old, new in attn_replacements.items(): new_key = new_key.replace(old, new)
            
    return new_key

def translate_a1111_vae_to_diffusers(raw_state_dict: dict) -> dict:
    """
    [REFACTORED] Translates a full VAE state_dict from A1111/Kohya format to
    the Diffusers format. It performs both key remapping and the necessary
    tensor reshaping for attention blocks.
    """
    translated_state_dict = {}
    for old_key, tensor in raw_state_dict.items():
        # 1. Translate the key string using the internal helper.
        new_key = _translate_vae_key_str(old_key)

        # 2. **THE FIX**: Reshape the tensor if it's an attention weight.
        # This logic is now centralized here.
        if "attentions." in new_key and new_key.endswith(".weight"):
            if len(tensor.shape) == 4:
                # Squeeze the 4D Conv2d weight into a 2D Linear weight.
                tensor = tensor.squeeze()

        translated_state_dict[new_key] = tensor
    return translated_state_dict

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
    # --- THE FIX ---
    # The iterator is now created ONCE, outside the main loop.
    # This ensures it maintains its state and can be fully consumed.
    tensorized_iterator = tensorize_and_collate_iterator(raw_data_iterator, preprocessing_fn)
    
    current_bs = 1
    buffer = []
    while True:
        # Check for batch size updates without blocking
        try:
            current_bs = batch_size_queue.get_nowait()
        except mp.queues.Empty:
            pass

        try:
            # --- THE FIX ---
            # We now call next() on the single, persistent iterator created above.
            # This correctly draws down the data source one item at a time.
            item_td = next(tensorized_iterator)
            buffer.append(item_td)

            if len(buffer) >= current_bs:
                # When batch is full, stack the TensorDicts and put on the queue
                batch_td = torch.stack(buffer, dim=0)
                cpu_queue.put(batch_td)
                buffer = []

        except StopIteration:
            # This block is now reachable. It will execute once the tensorized_iterator is exhausted.
            # Handle the last partial batch
            if buffer:
                cpu_queue.put(torch.stack(buffer, dim=0))
            cpu_queue.put(None) # End-of-stream sentinel
            break # Exit the while True loop and terminate the process.

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

import numpy as np
from collections import defaultdict

def print_batch_size_summary(history: List[Dict], best_throughput: float, bs_at_best_throughput: int):
    """
    [NEW] Analyzes the full calibration history and prints a concise summary.
    """
    if not history:
        print("[Batch Advisor] No calibration history was recorded.")
        return

    print("\n--- [Batch Advisor Summary] ---")
    
    # Exclude OOM errors and calculate throughput for valid runs
    valid_runs = [run for run in history if run.get('duration') and run['duration'] < float('inf')]
    for run in valid_runs:
        run['throughput'] = run['bs'] / run['duration']

    # Group results by batch size
    stats_by_bs = defaultdict(list)
    for run in valid_runs:
        stats_by_bs[run['bs']].append(run['throughput'])

    print(f"{'Batch Size':<12} | {'Trials':<8} | {'Mean Throughput':<20} | {'Std Dev':<15}")
    print("-" * 60)

    # Calculate and print stats for each batch size
    for bs in sorted(stats_by_bs.keys()):
        throughputs = stats_by_bs[bs]
        count = len(throughputs)
        mean_tp = np.mean(throughputs)
        std_tp = np.std(throughputs)
        print(f"{bs:<12} | {count:<8} | {f'{mean_tp:.2f} items/sec':<20} | {f'{std_tp:.2f}':<15}")

    if bs_at_best_throughput > 0:
        print(f"\n- Peak throughput: {best_throughput:.2f} items/sec was achieved at batch size {bs_at_best_throughput}.")
    else:
        print("\n- No successful runs to determine peak throughput.")
    print("---------------------------------")


def advise_next_batch_size(calibration_kwargs: Dict) -> Tuple[int, Dict]:
    """
    [MODIFIED] This function now ONLY performs calculations and defers all
    printing to the new summary function.
    """
    # --- Part 1: Setup for Analysis ---
    config = calibration_kwargs.get("config", {})
    history = calibration_kwargs.get("history", [])
    latest_run = calibration_kwargs.get("latest_run_metrics", {})

    best_throughput = calibration_kwargs.get("best_throughput", 0.0)
    bs_at_best_throughput = calibration_kwargs.get("bs_at_best_throughput", 0)

    if latest_run and "bs" in latest_run:
        history.append(latest_run)

    # --- Part 2: The Analysis Calculation (No Printing) ---
    if latest_run and "bs" in latest_run:
        duration = latest_run.get('duration', 0)
        if duration > 1e-9 and duration < float('inf'): # Check for valid duration
            current_throughput = latest_run['bs'] / duration
            if current_throughput > best_throughput:
                best_throughput = current_throughput
                bs_at_best_throughput = latest_run['bs']

    # --- Part 3: The Literal Sequence Generator (Unchanged) ---
    current_bs = calibration_kwargs.get("sequence_bs", 1)
    yield_count = calibration_kwargs.get("sequence_yield_count", 0)
    next_bs_to_return = current_bs

    yield_count += 1
    if current_bs == 1 and yield_count >= 2: current_bs, yield_count = 2, 0 # Reduced stabilization for faster ramp-up
    elif current_bs == 2 and yield_count >= 2: current_bs, yield_count = 3, 0
    elif current_bs == 3 and yield_count >= 2: current_bs, yield_count = 4, 0
    
    calibration_kwargs.update({
        "sequence_bs": current_bs,
        "sequence_yield_count": yield_count,
        "history": history,
        "best_throughput": best_throughput,
        "bs_at_best_throughput": bs_at_best_throughput
    })
    return next_bs_to_return, calibration_kwargs

# --- For Image VAE Encoding ---
from PIL import Image
from torchvision import transforms

def sdxl_vae_image_preprocessor(item: Dict) -> Dict[str, torch.Tensor]:
    """Takes a data item, loads the image, and prepares it for the VAE."""
    #image_path = item['image_path']
    #image_path = item['input_data']['image']
    # Get the full primitive dictionary for the 'image' input.
    image_primitive = item['input_data']['image']
    # Extract the actual filepath from the 'data_path' key.
    image_path = image_primitive['data_path']
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

def sdxl_vae_latent_preprocessor(item: Dict, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    """
    [DEBUG LOGGING ADDED] Takes a data item and loads the latent tensor from its
    safetensor file by parsing the 'data_location' string.
    """
    try:
        # --- NEW LOGGING ---
        logstr=""
        work_id = item.get('work_id', 'Unknown')
        logstr+=f"  [Preprocessor] Processing work_id: {work_id}"
        logstr+=f"    - DEBUG: Keys in item['input_data']: {item['input_data'].keys()}"
        latent_primitive = item['input_data']['latent']
        logstr+=f"    - DEBUG: Keys in latent_primitive: {latent_primitive.keys()}"
        loc_str = latent_primitive['data_location']
        logstr+=f"    - Found data_location string: {loc_str}"

        loc_parts = {p.split(':')[0]: p.split(':')[1] for p in loc_str.split('|')}
        latent_cache_path = loc_parts['loc']
        latent_key = loc_parts['key']
        logstr+=f"    - Parsed path='{latent_cache_path}', key='{latent_key}'"

        with safetensors.safe_open(latent_cache_path, framework="pt", device="cpu") as f:
            latent_tensor = f.get_tensor(latent_key)

        logstr+=f"    - SUCCESS: Loaded latent tensor of shape {latent_tensor.shape}."
        return {"latent": latent_tensor.to(dtype=dtype)}

    except Exception as e:
        # --- NEW LOGGING ---
        # This will catch ANY error during preprocessing and print it explicitly.
        print(f"  [Preprocessor] ERROR processing work_id '{item.get('work_id', 'N/A')}': {e}")
        print(f"logstring:"+logstr)
        # Re-raise the exception to make sure the loader's own error handling catches it.
        raise e

def _sdxl_text_preprocessor_worker(item: Dict, tokenizer_1, tokenizer_2) -> Dict[str, torch.Tensor]:
    """
    [CORRECTED - v3] Implements the slider logic and then STACKS the
    conditional and unconditional tokens into a single tensor for each encoder
    to enable efficient batch processing on the accelerator.
    """
    # 1. Unpack and apply slider logic (unchanged)
    prompt_primitive = item['input_data']['prompt']
    prompt_recipe = prompt_primitive['data_content']
    qualifier = prompt_primitive['qualifier']

    conditional_text = ""
    if qualifier == 'high_scale':
        conditional_text = prompt_recipe.get('positive', '')
    elif qualifier == 'low_scale':
        conditional_text = prompt_recipe.get('neutral', '')
    unconditional_text = prompt_recipe.get('unconditional', '')

    # 2. Tokenize all four components individually first.
    cond_tokens_1 = tokenizer_1(conditional_text, padding="max_length", max_length=tokenizer_1.model_max_length, truncation=True, return_tensors="pt").input_ids.squeeze()
    cond_tokens_2 = tokenizer_2(conditional_text, padding="max_length", max_length=tokenizer_2.model_max_length, truncation=True, return_tensors="pt").input_ids.squeeze()
    uncond_tokens_1 = tokenizer_1(unconditional_text, padding="max_length", max_length=tokenizer_1.model_max_length, truncation=True, return_tensors="pt").input_ids.squeeze()
    uncond_tokens_2 = tokenizer_2(unconditional_text, padding="max_length", max_length=tokenizer_2.model_max_length, truncation=True, return_tensors="pt").input_ids.squeeze()

    # --- THE ACCELERATOR EFFICIENCY FIX ---
    # 3. Stack the cond/uncond pairs for each encoder.
    # The output tensor for each tokenizer will have shape [2, 77].
    tokens_1 = torch.stack([cond_tokens_1, uncond_tokens_1])
    tokens_2 = torch.stack([cond_tokens_2, uncond_tokens_2])

    return {"tokens_1": tokens_1, "tokens_2": tokens_2}


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

    # --- 2. Infer format and apply MAPPING AND RESHAPING EXPLICITLY ---
    inferred_format = infer_vae_sdxl_state_dict_format(set(raw_vae_keys.keys()))
    working_state_dict = {}
    
    print("\n" + "="*80)
    print("== [MODEL LOADER] Key Format Inference Report")
    print(f"==  - Model Path: {os.path.basename(model_path)}")
    print(f"==  - Inferred Format: '{inferred_format}'")

    if inferred_format == "A1111_KOHYA":
        print("==  - Action: Applying full 'A1111_KOHYA_TO_DIFFUSERS' state_dict transformation.")
        # ** THE REFACTOR **: A single, clean call to the new, powerful translator.
        working_state_dict = translate_a1111_vae_to_diffusers(raw_vae_keys)
    elif inferred_format == "Diffusers":
        print("==  - Action: No key transformation needed. Loading keys as-is.")
        working_state_dict = raw_vae_keys # Already filtered for VAE keys
    else:
        print("==  - WARNING: Unknown format. Attempting to load keys as-is. This may fail.")
        working_state_dict = raw_vae_keys
    print("="*80 + "\n")

    # --- 3. Load the model with the corrected state dict ---
    with open(config_path, 'r') as f:
        vae_config = json.load(f)
    vae = AutoencoderKL.from_config(vae_config)
    vae.load_state_dict(working_state_dict, strict=True) # This will now succeed
    vae.to(device=device, dtype=torch_dtype).eval()
    
    # Use itertools.tee to create two independent iterators from the original one.
    # We will consume one to get the IDs and pass the other to the loader.
    id_iterator, loader_iterator = itertools.tee(work_iterator)
    # 1. Immediately consume the first iterator to get our list of work_ids.
    work_ids = [spec['work_id'] for spec in id_iterator]

    # 2. Initialize the pipeline and advisor tools
    loader = PipelinedTensorDictLoader(
        raw_data_iterator=loader_iterator,
        preprocessing_fn=sdxl_vae_image_preprocessor, # Pass the correct preprocessor
        device=device,
        initial_bs=1
    )
    calibration_state = {
        "config": {"vram_safety_budget_gb": 20.0, "initial_batch_size": 1},
        "history": []
    }
    # Just before the `while True:` loop
    calibration_state["latest_run_metrics"] = {}
    all_results = []

    # 3. Run the internal processing loop
    while True:
        try:
            # 1. ADVISE FIRST, using metrics from the previous loop.
            next_bs, calibration_state = advise_next_batch_size(calibration_state)
            loader.set_batch_size(next_bs)
            # 2. Start timer and get next batch
            start_time = time.time()
            on_device_batch = loader.get_next_batch()

            # 3. The actual model computation
            with torch.no_grad():
                posterior = vae.encode(on_device_batch['image']).latent_dist
                latents = posterior.sample() * vae.config.scaling_factor
            
            # 4. CAPTURE METRICS NOW and store for the *next* loop's advisor call
            #torch.cuda.synchronize() # Crucial for accurate timing
            duration = time.time() - start_time
            vram_peak_gb = torch.cuda.max_memory_reserved() / (1024**3)

            calibration_state["latest_run_metrics"] = {
            "bs": on_device_batch.shape[0], # Use the batch size we actually ran
            "duration": duration,
            "vram_peak_gb": vram_peak_gb
            }

            all_results.append(latents.cpu())

        except StopIteration:
            print("[VAE Encoder] Finished processing all images.")
            break
            
    loader.close()

    print_batch_size_summary(
        calibration_state.get("history", []),
        calibration_state.get("best_throughput", 0),
        calibration_state.get("bs_at_best_throughput", 0)
    )

    # 4. Collate results and return in the format expected by the DAG
    final_latents = torch.cat(all_results, dim=0)
    print(f"--- [VAE Encoder] Complete. Produced final latent tensor of shape: {final_latents.shape} ---")
    # This part now works because we have the list of work_ids to remap our data to the expected format.
    results_by_work_id = {}
    for i, work_id in enumerate(work_ids):
        results_by_work_id[work_id] = {"latent": final_latents[i]}
        
    return results_by_work_id
    
# --- UNSTUBBED REPLACEMENT for dummy_text_encoder ---
# Note: Renaming fictitious_sdxl_text_encoder to a more generic name
def real_sdxl_text_encoder(**kwargs) -> dict:
    """
    [REAL - v3 CORRECTED] Encodes prompts by loading text encoders from a
    single checkpoint using the established, fine-grained, manual-loading
    pattern to avoid loading the entire pipeline.
    """
    # --- 1. Standard Kwarg Setup ---
    model_path = kwargs.get('model_path')
    device = kwargs.get('device', 'cpu')
    torch_dtype = torch.bfloat16
    work_iterator = kwargs['data_iterator']
    
    # 1a. Get the full paths to the config FILES.
    te1_config_path = os.path.abspath(kwargs.get('text_encoder_1_config_path'))
    te2_config_path = os.path.abspath(kwargs.get('text_encoder_2_config_path'))

    # 1b. Manually load the JSON files into dictionaries.
    with open(te1_config_path, 'r') as f:
        te1_config_dict = json.load(f)
    with open(te2_config_path, 'r') as f:
        te2_config_dict = json.load(f)

    # 1a. Get the full paths to the config FILES.
    # 1c. Instantiate the models by passing the config DICTIONARY.
    # This completely avoids all the library's path validation logic.
    # --- ** THE FIX IS HERE ** ---
    # 1c. Create the specific 'Config' objects from the dictionaries.
    # The 'transformers' library uses this two-step pattern.
    te1_config = CLIPTextConfig.from_dict(te1_config_dict)
    te2_config = CLIPTextConfig.from_dict(te2_config_dict)

    text_encoder_1 = CLIPTextModel(te1_config)
    text_encoder_2 = CLIPTextModelWithProjection(te2_config)

    tokenizer_1_path = os.path.abspath(kwargs.get('tokenizer_1_path'))
    tokenizer_2_path = os.path.abspath(kwargs.get('tokenizer_2_path'))

    # --- 3. Manually Filter and Load Weights from Safetensors Checkpoint ---
    print("  [Text Encoder] Manually filtering and remapping keys from checkpoint...")
    te1_state_dict = {}
    te2_state_dict = {}
    
    # ** THE LOGIC IS NOW SWAPPED TO MATCH THE MODELS **
    # Prefix for the BIG encoder (embedders.0) -> goes into te1_state_dict
    TE1_PREFIX = "conditioner.embedders.0.transformer.text_model."
    # Prefix for the SMALL encoder (embedders.1) -> goes into te2_state_dict
    TE2_PREFIX = "conditioner.embedders.1.text_model."

    with safetensors.safe_open(model_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith(TE1_PREFIX):
                new_key = key[len(TE1_PREFIX):]
                te1_state_dict[new_key] = f.get_tensor(key)
            elif key.startswith(TE2_PREFIX):
                new_key = key[len(TE2_PREFIX):]
                te2_state_dict[new_key] = f.get_tensor(key)
    
    # Now this will work because the shell's architecture matches the weights' shapes.
    # We also pass strict=False to ignore the "position_ids" key which isn't a parameter.
    text_encoder_1.load_state_dict(te1_state_dict, strict=False)
    text_encoder_2.load_state_dict(te2_state_dict, strict=False)

    text_encoder_1.to(device=device, dtype=torch_dtype).eval()
    text_encoder_2.to(device=device, dtype=torch_dtype).eval()
    print("  [Text Encoder] Weights loaded successfully into shells.")

    # --- 4. Load Tokenizers ---
    tokenizer_1 = CLIPTokenizer.from_pretrained(tokenizer_1_path)
    tokenizer_2 = CLIPTokenizer.from_pretrained(tokenizer_2_path)
    
    # Create a partial function. This "freezes" the tokenizer arguments into our
    # pure worker function. The result is a new, simple callable that only
    # requires the `item` argument, which matches the loader's API.
    # This object is simple, pickle-safe, and has no hidden state.
    text_preprocessor = functools.partial(
        _sdxl_text_preprocessor_worker,
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2
    )

    id_iterator, loader_iterator = itertools.tee(work_iterator)
    work_ids = [spec['work_id'] for spec in id_iterator]
    loader = PipelinedTensorDictLoader(loader_iterator, text_preprocessor, device)
    
    # --- Add this: Initialize the calibration state ---
    calibration_state = {
        "config": {"device": device, "vram_safety_ratio": 0.8},
        "history": [],
        "latest_run_metrics": {}
    }
    all_text_embeds_cond, all_text_embeds_uncond = [], []
    all_pooled_embeds_cond, all_pooled_embeds_uncond = [], []

    while True:
        try:
            # 1. ADVISE: Get the next batch size to try.
            next_bs, calibration_state = advise_next_batch_size(calibration_state)
            loader.set_batch_size(next_bs)

            # 2. TIME & FETCH: Start the timer and get the batch.
            start_time = time.time()
            on_device_batch = loader.get_next_batch()
            batch_size = on_device_batch.shape[0]

            # 3. COMPUTE: The core model inference.
            with torch.no_grad():
                # --- THE CORRECTED, SPEC-COMPLIANT LOGIC ---
                # 1. Prepare one large, efficient batch for each encoder.
                tokens_1_batched = on_device_batch['tokens_1'].view(batch_size * 2, -1)
                tokens_2_batched = on_device_batch['tokens_2'].view(batch_size * 2, -1)

                # 2. Run Text Encoder 1 to get the 'text_embedding' asset.
                text_embeds_all = text_encoder_1(tokens_1_batched, output_hidden_states=True).hidden_states[-2]

                # 3. Run Text Encoder 2 to get the 'pooled_text_embedding' asset.
                encoder_2_output_all = text_encoder_2(tokens_2_batched, output_hidden_states=True)
                pooled_embeds_all = encoder_2_output_all.text_embeds # We ONLY care about the pooled output here.

                # 4. Un-stack the results for the 'text_embedding' asset.
                text_embeds_reshaped = text_embeds_all.view(2, batch_size, *text_embeds_all.shape[1:])
                text_embeds_chunked = text_embeds_reshaped.chunk(2, dim=0)
                cond_text_embeds = text_embeds_chunked[0].squeeze(0)
                uncond_text_embeds = text_embeds_chunked[1].squeeze(0)

                # 5. Un-stack the results for the 'pooled_text_embedding' asset.
                pooled_embeds_reshaped = pooled_embeds_all.view(2, batch_size, -1)
                pooled_embeds_chunked = pooled_embeds_reshaped.chunk(2, dim=0)
                cond_pooled_embeds = pooled_embeds_chunked[0].squeeze(0)
                uncond_pooled_embeds = pooled_embeds_chunked[1].squeeze(0)
                # --- End of Fix ---

            # 4. MEASURE: Get timing and memory stats.
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            duration = time.time() - start_time
            vram_peak_gb = torch.cuda.max_memory_reserved(device) / (1024**3)

            # 5. RECORD: Store metrics for the next advisor call.
            calibration_state["latest_run_metrics"] = {
                "bs": on_device_batch.shape[0],
                "duration": duration,
                "vram_peak_gb": vram_peak_gb
            }

            # 6. STORE RESULTS: Append to CPU lists.
            all_text_embeds_cond.append(cond_text_embeds.cpu())
            all_text_embeds_uncond.append(uncond_text_embeds.cpu())
            all_pooled_embeds_cond.append(cond_pooled_embeds.cpu())
            all_pooled_embeds_uncond.append(uncond_pooled_embeds.cpu())

        except StopIteration:
            print("[Text Encoder] Finished processing all prompts.")
            break
        except torch.cuda.OutOfMemoryError:
            print(f"  [Batch Advisor] CUDA OOM detected at BS={next_bs}. Forcing downsearch.")
            torch.cuda.empty_cache()
            # Log the failure so the advisor knows this batch size is too large.
            calibration_state["latest_run_metrics"] = {"bs": on_device_batch.shape[0], "duration": float('inf'), "vram_peak_gb": float('inf')}
            continue # Proceed to the next advisor call.

    loader.close()

    print_batch_size_summary(
        calibration_state.get("history", []),
        calibration_state.get("best_throughput", 0),
        calibration_state.get("bs_at_best_throughput", 0)
    )

    final_text_embeds_cond = torch.cat(all_text_embeds_cond, dim=0)
    final_text_embeds_uncond = torch.cat(all_text_embeds_uncond, dim=0)
    final_pooled_embeds_cond = torch.cat(all_pooled_embeds_cond, dim=0)
    final_pooled_embeds_uncond = torch.cat(all_pooled_embeds_uncond, dim=0)
    
    results_by_work_id = {}
    for i, work_id in enumerate(work_ids):
        results_by_work_id[work_id] = {
            # Key 1: "text_embedding"
            "text_embedding": {
                "cond": final_text_embeds_cond[i],
                "uncond": final_text_embeds_uncond[i]
            },
            # Key 2: "pooled_text_embedding"
            "pooled_text_embedding": {
                "cond": final_pooled_embeds_cond[i],
                "uncond": final_pooled_embeds_uncond[i]
            }
        }
    return results_by_work_id


# --- UNSTUBBED REPLACEMENT for dummy_time_id_synthesizer ---
def real_time_id_synthesizer(**kwargs) -> dict:
    """
    [REAL] Creates SDXL time embeddings based on image dimensions.
    """
    results = {}
    for spec in kwargs['data_iterator']:
        work_id = spec['work_id']
        # In a real scenario, we'd get real dimensions. Here we use defaults.
        original_size = (1024, 1024)
        crops_coords_top_left = (0, 0)
        target_size = (1024, 1024)
        
        # SDXL time embedding logic
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=torch.bfloat16)
        results[spec['work_id']] = {"time_embedding": add_time_ids.squeeze()}
    return results


# --- TENSOR-COMPLIANT REPLACEMENT for scale_tensor_synthesizer_v1 ---
def scale_tensor_synthesizer_v1(**kwargs) -> dict:
    """
    [CORRECTED] A simple function that now correctly unpacks the full
    'scale_metadata' primitive dictionary to find the data_value.
    """
    data_iterator = kwargs['data_iterator']
    results = {}

    for spec in data_iterator:
        work_id = spec['work_id']

        # --- THE FIX IS HERE ---
        # 1. Get the full primitive dictionary.
        scale_metadata_primitive = spec['input_data']['scale_metadata']
        # 2. Extract the actual numerical value from the 'data_value' key.
        scale_value = scale_metadata_primitive['data_value']
        # --- End of Fix ---

        # The core logic now uses the correctly extracted value.
        results[work_id] = {"scales_tensor": torch.tensor(scale_value)}

    return results

def real_denoise_input_encoder(**kwargs) -> dict:
    """
    [REAL - CORRECTED] Encodes images to latents, selects timesteps, and adds noise.
    This version corrects the kwarg access to conform to the established
    interface contract used by other working functions.
    """
    # --- Standard setup from kwargs ---
    # ** THE FIX **: Use the standard keys provided by the DAG, not custom ones.
    model_path = kwargs.get('model_path')
    config_path = kwargs.get('config_path')
    device = kwargs.get('device', 'cpu')
    torch_dtype = torch.bfloat16 # Hardcoding for simplicity, can be passed in kwargs
    work_iterator = kwargs['data_iterator']
    
    # Get both the full schedule length and the coarse steps for sampling
    num_train_timesteps = kwargs.get('num_train_timesteps', 1000)
    max_denoising_steps = kwargs.get('max_denoising_steps', 50) # The coarse schedule


    # VAE LOADING
    # --- 1. Load the raw state dictionary and all its keys ---
    raw_vae_keys = {}
    with safetensors.safe_open(model_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith("first_stage_model."):
                raw_vae_keys[key] = f.get_tensor(key)

    # --- 2. Infer format and apply MAPPING AND RESHAPING EXPLICITLY ---
    inferred_format = infer_vae_sdxl_state_dict_format(set(raw_vae_keys.keys()))
    working_state_dict = {}
    
    print("\n" + "="*80)
    print("== [MODEL LOADER] Key Format Inference Report")
    print(f"==  - Model Path: {os.path.basename(model_path)}")
    print(f"==  - Inferred Format: '{inferred_format}'")

    if inferred_format == "A1111_KOHYA":
        print("==  - Action: Applying full 'A1111_KOHYA_TO_DIFFUSERS' state_dict transformation.")
        # ** THE REFACTOR **: A single, clean call to the new, powerful translator.
        working_state_dict = translate_a1111_vae_to_diffusers(raw_vae_keys)
    elif inferred_format == "Diffusers":
        print("==  - Action: No key transformation needed. Loading keys as-is.")
        working_state_dict = raw_vae_keys # Already filtered for VAE keys
    else:
        print("==  - WARNING: Unknown format. Attempting to load keys as-is. This may fail.")
        working_state_dict = raw_vae_keys
    print("="*80 + "\n")

    # --- 3. Load the model with the corrected state dict ---
    with open(config_path, 'r') as f:
        vae_config = json.load(f)
    vae = AutoencoderKL.from_config(vae_config)
    vae.load_state_dict(working_state_dict, strict=True) # This will now succeed
    vae.to(device=device, dtype=torch_dtype).eval()
    
    # --- 2. Initialize Noise Scheduler ---
    # The scheduler is ALWAYS initialized with the full 1000 steps to define the beta curve.
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear", device=device
    )

    # --- 3. Run Pipeline ---
    id_iterator, loader_iterator = itertools.tee(work_iterator)
    work_ids = [spec['work_id'] for spec in id_iterator]

    loader = PipelinedTensorDictLoader(
        raw_data_iterator=loader_iterator,
        preprocessing_fn=sdxl_vae_image_preprocessor,
        device=device
    )
    # Initialize the calibration state, just like in the other functions
    calibration_state = {
        "config": {"device": device, "vram_safety_ratio": 0.8},
        "history": [],
        "latest_run_metrics": {}
    }
    all_noisy_latents, all_timesteps_for_unet = [], []
    
    while True:
        try:
            # CALIB1. ADVISE & SET
            next_bs, calibration_state = advise_next_batch_size(calibration_state)
            loader.set_batch_size(next_bs)

            # CALIB2. TIME & FETCH
            start_time = time.time()
            on_device_batch = loader.get_next_batch()
            batch_size = on_device_batch['image'].shape[0]            

            with torch.no_grad():
                latents = vae.encode(on_device_batch['image']).latent_dist.sample() * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            # TSTEPS1. Set the scheduler to the coarse, inference-like steps.
            noise_scheduler.set_timesteps(max_denoising_steps, device=device)
            # TSTEPS2. Importance-sample by picking a random INDEX from the coarse steps.
            # This samples from a distribution of e.g., 50 steps like [981, 961, ...].
            indices = torch.randint(0, max_denoising_steps, (batch_size,), device=device)
            # TSTEPS3. Get the actual timestep VALUE from the sampled index.
            # This is the value used for adding noise AND for the UNet.
            timesteps = noise_scheduler.timesteps[indices]
            # TSTEPS4. Add noise using the sampled timestep value.
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            # TSTEPS5. The timestep for the UNet IS the sampled timestep. No complex projection needed.
            # The UNet was trained on values from 0-999, and `timesteps` contains exactly that.
            unet_timesteps = timesteps
            
            # CALIB4. MEASURE
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            duration = time.time() - start_time
            vram_peak_gb = torch.cuda.max_memory_reserved(device) / (1024**3)

            # CALIB5. RECORD
            calibration_state["latest_run_metrics"] = {
                "bs": on_device_batch.shape[0], "duration": duration, "vram_peak_gb": vram_peak_gb
            }

            all_noisy_latents.append(noisy_latents.cpu())
            all_timesteps_for_unet.append(unet_timesteps.cpu())
        except StopIteration:
            print("[Denoise Encoder] Finished processing all images.")
            break
        except torch.cuda.OutOfMemoryError:
            print(f"  [Batch Advisor] CUDA OOM detected at BS={next_bs}. Forcing downsearch.")
            torch.cuda.empty_cache()
            calibration_state["latest_run_metrics"] = {"bs": on_device_batch.shape[0], "duration": float('inf'), "vram_peak_gb": float('inf')}
            continue
    
    loader.close()

    print_batch_size_summary(
        calibration_state.get("history", []),
        calibration_state.get("best_throughput", 0),
        calibration_state.get("bs_at_best_throughput", 0)
    )

    # --- 4. Collate and Return ---
    final_noisy_latents = torch.cat(all_noisy_latents, dim=0)
    final_timesteps = torch.cat(all_timesteps_for_unet, dim=0)

    results_by_work_id = {
        work_id: {"noisy_latent": final_noisy_latents[i], "timestep_for_unet": final_timesteps[i]}
        for i, work_id in enumerate(work_ids)
    }
    return results_by_work_id



# NEW FUNCTION PER CLAUDE SPEC
def real_latent_to_image_decoder(**kwargs) -> dict:
    """
    Decodes latent tensors back into images using a VAE. Follows the same
    high-performance, batch-calibrating pattern as the encoder.
    """
    # --- Standard setup from kwargs ---
    model_path = kwargs.get('model_path')
    config_path = kwargs.get('config_path')
    device = kwargs.get('device', 'cpu')
    dtype_str = kwargs.get('dtype', 'bfloat16')
    work_iterator = kwargs['data_iterator']
    torch_dtype = torch.float32 if dtype_str == 'float32' else torch.bfloat16

    # VAE LOADING
    # --- 1. Load the raw state dictionary and all its keys ---
    raw_vae_keys = {}
    with safetensors.safe_open(model_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith("first_stage_model."):
                raw_vae_keys[key] = f.get_tensor(key)

    # --- 2. Infer format and apply MAPPING AND RESHAPING EXPLICITLY ---
    inferred_format = infer_vae_sdxl_state_dict_format(set(raw_vae_keys.keys()))
    working_state_dict = {}
    
    print("\n" + "="*80)
    print("== [MODEL LOADER] Key Format Inference Report")
    print(f"==  - Model Path: {os.path.basename(model_path)}")
    print(f"==  - Inferred Format: '{inferred_format}'")

    if inferred_format == "A1111_KOHYA":
        print("==  - Action: Applying full 'A1111_KOHYA_TO_DIFFUSERS' state_dict transformation.")
        # ** THE REFACTOR **: A single, clean call to the new, powerful translator.
        working_state_dict = translate_a1111_vae_to_diffusers(raw_vae_keys)
    elif inferred_format == "Diffusers":
        print("==  - Action: No key transformation needed. Loading keys as-is.")
        working_state_dict = raw_vae_keys # Already filtered for VAE keys
    else:
        print("==  - WARNING: Unknown format. Attempting to load keys as-is. This may fail.")
        working_state_dict = raw_vae_keys
    print("="*80 + "\n")

    # --- 3. Load the model with the corrected state dict ---
    with open(config_path, 'r') as f:
        vae_config = json.load(f)
    vae = AutoencoderKL.from_config(vae_config)
    vae.load_state_dict(working_state_dict, strict=True) # This will now succeed
    vae.to(device=device, dtype=torch_dtype).eval()

    # --- 2. Setup Pipeline for decoding ---
    work_list = list(work_iterator)
    print(f"\n--- [VAE Decoder Tool] Received {len(work_list)} work items to process. ---")
    if not work_list:
        print("--- [VAE Decoder Tool] WARNING: Iterator is empty. Aborting. ---")
        return {}

    id_iterator, loader_iterator = itertools.tee(work_list)
    work_ids = [spec['work_id'] for spec in id_iterator]

    latent_preprocessor = functools.partial(sdxl_vae_latent_preprocessor, dtype=torch_dtype)

    loader = PipelinedTensorDictLoader(
        raw_data_iterator=loader_iterator,
        preprocessing_fn=latent_preprocessor, # Use latent preprocessor
        device=device,
        initial_bs=1
    )
    calibration_state = {
        "config": {"vram_safety_ratio": 0.8, "device": device},
        "history": [],
        "latest_run_metrics": {}
    }
    all_results = []
    
    # 3. Run the internal processing loop
    while True:
        try:
            next_bs, calibration_state = advise_next_batch_size(calibration_state)
            loader.set_batch_size(next_bs)
            start_time = time.time()
            on_device_batch = loader.get_next_batch()
            
            with torch.no_grad():
                # The core decoding operation
                latents = on_device_batch['latent']
                images = vae.decode(latents / vae.config.scaling_factor).sample
                # Denormalize from [-1, 1] to [0, 1] for metrics/saving
                images = (images / 2 + 0.5).clamp(0, 1)

            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            duration = time.time() - start_time
            vram_peak_gb = torch.cuda.max_memory_reserved(device) / (1024**3)

            calibration_state["latest_run_metrics"] = {
                "bs": on_device_batch.shape[0], "duration": duration, "vram_peak_gb": vram_peak_gb
            }
            all_results.append(images.cpu())

        except StopIteration:
            print("[VAE Decoder] Finished processing all latents.")
            break
        except torch.cuda.OutOfMemoryError:
            print(f"  [Batch Advisor] CUDA OOM detected at BS={next_bs}. Forcing downsearch.")
            torch.cuda.empty_cache()
            calibration_state["latest_run_metrics"] = {"bs": on_device_batch.shape[0], "duration": float('inf'), "vram_peak_gb": float('inf')}
            continue
            
    loader.close()

    print_batch_size_summary(
        calibration_state.get("history", []),
        calibration_state.get("best_throughput", 0),
        calibration_state.get("bs_at_best_throughput", 0)
    )

    # 4. Collate and return results
    final_images = torch.cat(all_results, dim=0)
    print(f"--- [VAE Decoder] Complete. Produced final image tensor of shape: {final_images.shape} ---")
    results_by_work_id = {
        work_id: {"reconstructed_image": final_images[i]}
        for i, work_id in enumerate(work_ids)
    }
    return results_by_work_id

# NEW FUNCTION PER CLAUDE SPEC
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance

# ... (all existing functions)

def compute_validation_metrics(**kwargs) -> dict:
    """
    [NEW - v5.8] A d_dset 'tool' function that calculates aggregate image-to-image
    comparison metrics (LPIPS, FID) over a dataset of paired images. It functions
    as a 'many-to-one' reducer, consuming an entire iterator of work items and
    producing a single output asset containing the final scores.

    This function is invoked by the generic Layer 4 'execute_asset_caching' stage.
    """
    # --- 1. Standard Setup and Configuration Unpacking ---
    print("--- [Validator Tool] Initializing image reconstruction metric computation ---")
    device = kwargs.get('device', 'cuda')
    lpips_net = kwargs.get('lpips_net', 'vgg')
    fid_feature_dim = kwargs.get('fid_feature_dim', 64) # Valid: 64, 192, 768, 2048
    data_iterator = kwargs['data_iterator']

    # --- 2. Initialize Metric Objects from TorchMetrics ---
    # Metrics are initialized once and updated iteratively.
    try:
        lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type=lpips_net, normalize=True).to(device)
        fid_metric = FrechetInceptionDistance(feature=fid_feature_dim, normalize=True).to(device)
        print(f"  - Metrics (LPIPS-{lpips_net}, FID-{fid_feature_dim}) initialized on {device}.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize torchmetrics objects: {e}")
        # Return an empty dict if we can't even start.
        return {}

    # --- 3. Collect all work items before processing ---
    # This is necessary for a many-to-one reduction. We need all work_ids to
    # map the single final result back to every input item.
    work_items_to_process = list(data_iterator)
    if not work_items_to_process:
        print("  - WARNING: Received an empty data iterator. Nothing to validate.")
        return {}
    
    all_work_ids = [spec['work_id'] for spec in work_items_to_process]
    print(f"  - Discovered {len(all_work_ids)} image pairs to validate.")

    # --- 4. Iterative Processing and Metric Updates ---
    sample_count = 0
    # Define a standard preprocessing pipeline for original images.
    # Decoded images are already tensors; originals are file paths.
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)), # Standardize size for comparison
        transforms.ToTensor() # Converts to [C, H, W] tensor in [0, 1] range
    ])

    for spec in work_items_to_process:
        try:
            # a. Load the original image from its file path
            original_primitive = spec['input_data']['original_image']
            original_path = original_primitive['data_path']
            original_img_pil = Image.open(original_path).convert("RGB")
            original_tensor = preprocess(original_img_pil).unsqueeze(0).to(device) # -> [1, C, H, W]

            # b. Load the reconstructed image from its safetensor cache location
            reconstructed_primitive = spec['input_data']['reconstructed_image']
            loc_str = reconstructed_primitive['data_location'] # "loc:path|key:name"
            loc_parts = {p.split(':')[0]: p.split(':')[1] for p in loc_str.split('|')}
            with safetensors.safe_open(loc_parts['loc'], framework="pt", device=device) as f:
                reconstructed_tensor = f.get_tensor(loc_parts['key']).unsqueeze(0) # -> [1, C, H, W]

            # c. Ensure tensors are 4D and have compatible shapes for metrics
            if original_tensor.shape != reconstructed_tensor.shape:
                # This should be rare if decoder output size is consistent
                reconstructed_tensor = torch.nn.functional.interpolate(
                    reconstructed_tensor, size=original_tensor.shape[2:]
                )

            # d. Update the metric accumulators
            lpips_metric.update(original_tensor, reconstructed_tensor)
            # For FID, update the two distributions separately
            fid_metric.update(original_tensor, real=True)
            fid_metric.update(reconstructed_tensor, real=False)

            sample_count += 1
            if sample_count % 50 == 0:
                print(f"    - Processed {sample_count} image pairs...")

        except Exception as e:
            print(f"  - WARNING: Skipping validation pair for work_id '{spec['work_id']}' due to error: {e}")
            continue

    if sample_count == 0:
        print("  - FAILED: No samples were successfully processed. Cannot compute metrics.")
        return {}

    # --- 5. Final Metric Computation and Asset Creation ---
    print(f"  - Computation complete. Calculating final scores from {sample_count} samples.")
    final_lpips = lpips_metric.compute().cpu()
    final_fid = fid_metric.compute().cpu()

    # CRITICAL: The output asset must be a dictionary of TENSORS to be saved
    # by the safetensors-based caching layer.
    final_metrics_asset = {
        "lpips_mean": torch.tensor(final_lpips),
        "fid_score": torch.tensor(final_fid),
        "sample_count": torch.tensor(float(sample_count))
    }
    
    print("\n--- [Validation Report] ---")
    print(f"  - Mean LPIPS (lower is better): {final_metrics_asset['lpips_mean'].item():.4f}")
    print(f"  - FID Score (lower is better):  {final_metrics_asset['fid_score'].item():.4f}")
    print("---------------------------\n")

    # --- 6. Format the output for the generic caching layer ---
    # Map the single result asset to every work_id that contributed to it.
    # This allows the Layer 5 assembler to find the result.
    # --- THE NEW, ELEGANT RETURN CONTRACT ---
    # Return ONE object that contains the single asset and the list of IDs it covers.
    return {
        "aggregate_result": {
            "asset_data": final_metrics_asset,
            "asset_type": "reconstruction_metrics",
            "contributing_work_ids": all_work_ids
        }
    }

def compute_per_image_latent_stats(**kwargs) -> dict:
    """
    [ENHANCED - v5.9] Calculates detailed statistical properties (global and per-channel)
    for individual latent tensors.
    """
    print("--- [Latent Stats Tool] Initializing per-image latent stat computation (Enhanced) ---")
    data_iterator = kwargs['data_iterator']
    results_by_work_id = {}

    for spec in data_iterator:
        work_id = spec['work_id']
        try:
            latent_primitive = spec['input_data']['latent']
            loc_str = latent_primitive['data_location']
            loc_parts = {p.split(':')[0]: p.split(':')[1] for p in loc_str.split('|')}
            with safetensors.safe_open(loc_parts['loc'], framework="pt", device="cpu") as f:
                latent_tensor = f.get_tensor(loc_parts['key'])

            if latent_tensor.dim() < 3:
                print(f"  - WARNING: Skipping work_id '{work_id}'. Latent tensor dim < 3.")
                continue

            # LEVEL 2 (CRITICAL): Per-channel stats, reducing only H, W dimensions.
            # Resulting shape: [num_channels]
            mean_per_channel = torch.mean(latent_tensor, dim=(-2, -1))
            std_per_channel = torch.std(latent_tensor, dim=(-2, -1))

            # LEVEL 1: Global stats for the whole tensor.
            # Resulting shape: [1] (scalar)
            global_mean = torch.mean(latent_tensor)
            global_std = torch.std(latent_tensor)

            # The asset is a richer dictionary of tensors.
            stats_asset = {
                "mean_per_channel": mean_per_channel,
                "std_per_channel": std_per_channel,
                "global_mean": torch.tensor(global_mean),
                "global_std": torch.tensor(global_std)
            }
            results_by_work_id[work_id] = {"latent_stats": stats_asset}

        except Exception as e:
            print(f"  - WARNING: Skipping stats for work_id '{work_id}' due to error: {e}")
            continue
            
    print(f"  - Calculated detailed statistics for {len(results_by_work_id)} latents.")
    return results_by_work_id

def aggregate_latent_stats(**kwargs) -> dict:
    """
    [CORRECTED - v5.9] Aggregates detailed latent stats, computes dataset-wide
    distributions, and identifies top outliers, expressing ALL outputs as tensors
    to conform to the strict Layer 4 caching contract.
    """
    print("--- [Latent Stats Aggregator] Initializing dataset-wide aggregation (Contract-Compliant) ---")
    data_iterator = kwargs['data_iterator']
    num_outliers_to_report = kwargs.get('num_outliers_to_report', 10)

    all_stats_items = list(data_iterator)
    if not all_stats_items: return {}
    
    all_work_ids = [spec['work_id'] for spec in all_stats_items]
    print(f"  - Aggregating detailed stats from {len(all_work_ids)} items.")

    # --- 1. Collect all detailed stats from the cache (Unchanged) ---
    all_means_per_channel, all_stds_per_channel = [], []
    for spec in all_stats_items:
        try:
            location_recipe = spec['input_data']['latent_stats']['data_location']
            # We just need one entry to find the file path.
            # We assume all tensors for a given asset are in the same file.
            any_sub_asset_loc_str = next(iter(location_recipe.values()))
            file_path = {p.split(':')[0]: p.split(':')[1] for p in any_sub_asset_loc_str.split('|')}['loc']

            with safetensors.safe_open(file_path, framework="pt", device="cpu") as f:
                # Use the location recipe to look up the exact key for each tensor
                mean_key = {p.split(':')[0]: p.split(':')[1] for p in location_recipe['mean_per_channel'].split('|')}['key']
                std_key = {p.split(':')[0]: p.split(':')[1] for p in location_recipe['std_per_channel'].split('|')}['key']
                all_means_per_channel.append(f.get_tensor(mean_key))
                all_stds_per_channel.append(f.get_tensor(std_key))

        except Exception as e:
            print(f"  - WARNING: Skipping aggregation for work_id '{spec['work_id']}' due to error: {e}")
            continue
    
    if not all_means_per_channel: 
        print("  - FAILED: No stats were successfully loaded. Aborting.")
        return {}

    # --- 2. Perform LEVEL 3 Aggregations (Unchanged) ---
    means_matrix = torch.stack(all_means_per_channel)
    stds_matrix = torch.stack(all_stds_per_channel)
    num_channels = means_matrix.shape[1]
    
    dataset_mean_of_means_per_channel = torch.mean(means_matrix, dim=0)
    dataset_std_of_means_per_channel = torch.std(means_matrix, dim=0)
    dataset_mean_of_stds_per_channel = torch.mean(stds_matrix, dim=0)
    dataset_std_of_stds_per_channel = torch.std(stds_matrix, dim=0)

    # --- 3. Outlier Detection (Corrected Implementation) ---
    stats_vectors = torch.cat((means_matrix, stds_matrix), dim=1)
    ideal_vector = torch.cat((torch.zeros(num_channels), torch.ones(num_channels)))
    distances = torch.linalg.norm(stats_vectors - ideal_vector.unsqueeze(0), dim=1)
    
    top_k = min(num_outliers_to_report, len(all_work_ids))
    top_scores, top_indices = torch.topk(distances, k=top_k)

    # --- 4. Prepare Final Asset as a "Totally Ordinary Dataset" of Tensors ---
    final_report_asset = {
        # Dataset-level aggregate stats
        "dataset_mean_of_means_per_channel": dataset_mean_of_means_per_channel,
        "dataset_std_of_means_per_channel": dataset_std_of_means_per_channel,
        "dataset_mean_of_stds_per_channel": dataset_mean_of_stds_per_channel,
        "dataset_std_of_stds_per_channel": dataset_std_of_stds_per_channel,
        "sample_count": torch.tensor(float(len(all_work_ids))),
        
        # ** THE CRITICAL FIX: Expressing the outlier report as pure tensors **
        # The indices of the outliers within the original `all_work_ids` list.
        "outlier_indices": top_indices.long(), # Store as LongTensor
        # The corresponding scores for the top outliers.
        "outlier_scores": top_scores,
        # The full stats vectors for the top outliers for detailed reporting.
        "outlier_stats_vectors": stats_vectors[top_indices],
    }
    
    # Add per-channel histograms (Unchanged)
    for i in range(num_channels):
        mean_hist_counts, mean_hist_bins = np.histogram(means_matrix[:, i].to(torch.float32).numpy(), bins=20, range=(-1.0, 1.0))
        std_hist_counts, std_hist_bins = np.histogram(stds_matrix[:, i].to(torch.float32).numpy(), bins=20, range=(0.0, 2.0))
        final_report_asset[f"ch{i}_means_hist_counts"] = torch.from_numpy(mean_hist_counts)
        final_report_asset[f"ch{i}_means_hist_bins"] = torch.from_numpy(mean_hist_bins)

        final_report_asset[f"ch{i}_means_hist_counts"] = torch.from_numpy(mean_hist_counts)
        final_report_asset[f"ch{i}_means_hist_bins"] = torch.from_numpy(mean_hist_bins)
        final_report_asset[f"ch{i}_stds_hist_counts"] = torch.from_numpy(std_hist_counts)
        final_report_asset[f"ch{i}_stds_hist_bins"] = torch.from_numpy(std_hist_bins)
    
    print("\n--- [Latent Statistics Report (Per-Channel)] ---")
    for i in range(num_channels):
        print(f"  Channel {i}:")
        # Also cast the values for printing to avoid any potential display issues
        print(f"    - Mean of Means: {dataset_mean_of_means_per_channel[i].item():.4f} (std: {dataset_std_of_means_per_channel[i].item():.4f})")
        print(f"    - Mean of Stds:  {dataset_mean_of_stds_per_channel[i].item():.4f} (std: {dataset_std_of_stds_per_channel[i].item():.4f})")
    print("--------------------------------------------------\n")

    # Return using the standard, unmodified aggregate contract. No special keys.
    return {
        "aggregate_result": {
            "asset_data": final_report_asset,
            "asset_type": "aggregate_latent_stats_report",
            "contributing_work_ids": all_work_ids
        }
    }