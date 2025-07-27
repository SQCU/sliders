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
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from PIL import Image
from torchvision import transforms
import json
import time
import itertools



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


def advise_next_batch_size(calibration_kwargs: Dict) -> Tuple[int, Dict]:
    """
    [V2.1 - ROBUST] Implements exponential upsearch and binary downsearch.
    This version corrects a KeyError by ensuring history only contains complete
    run metrics, avoiding tentative states.
    """
    config = calibration_kwargs.get("config", {})
    history = calibration_kwargs.get("history", [])
    latest_run = calibration_kwargs.get("latest_run_metrics", {})
    device = config.get("device", "cuda")

    # --- Configuration ---
    if torch.cuda.is_available():
        total_vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    else:
        total_vram_gb = 16.0 # Mock VRAM for CPU/non-cuda testing
    vram_safety_ratio = config.get("vram_safety_ratio", 0.8)
    vram_budget_gb = config.get("vram_budget_gb", total_vram_gb * vram_safety_ratio)
    max_iterations = config.get("max_search_iterations", 20)

    # --- Phase 1: State Update (Process last run's results) ---
    # This block is the core of the fix. It only adds complete records to history.
    if latest_run and "bs" in latest_run:
        bs, duration, vram_peak = latest_run["bs"], latest_run["duration"], latest_run["vram_peak_gb"]
        items_per_second = bs / duration if duration > 0 else float('inf')
        
        phase = "upsearch"
        if history and "downsearch" in history[-1].get("phase", "upsearch"):
             phase = "downsearch"

        new_entry = {
            "bs": bs, "duration": duration, "vram_peak_gb": vram_peak,
            "items_per_second": items_per_second, "phase": phase
        }
        history.append(new_entry)
        print(f"  [Batch Advisor] Logged BS={bs}: {items_per_second:.2f} items/s, VRAM Peak: {vram_peak:.2f}GB")

    # --- Phase 2: Decision Logic (Based on updated history) ---
    if not history:
        print("  [Batch Advisor] First call. Starting with BS=1.")
        return 1, calibration_kwargs

    last_entry = history[-1]

    # Check for terminal conditions
    if last_entry.get("phase") == "converged":
        return last_entry["bs"], calibration_kwargs
    if len(history) >= max_iterations:
        valid_runs = [h for h in history if h["vram_peak_gb"] <= vram_budget_gb]
        if not valid_runs:
             print("[Batch Advisor] WARNING: No successful run within VRAM budget. Defaulting to BS=1.")
             return 1, calibration_kwargs
        final_bs = max(valid_runs, key=lambda h: h["items_per_second"])["bs"]
        print(f"  [Batch Advisor] Max iterations reached. Converged to BS={final_bs}")
        last_entry["phase"] = "converged"
        return final_bs, calibration_kwargs

    # Check for transitions from upsearch to downsearch
    is_upsearching = "downsearch" not in last_entry.get("phase")
    if is_upsearching:
        if last_entry["vram_peak_gb"] > vram_budget_gb:
            print(f"  [Batch Advisor] VRAM limit exceeded at BS={last_entry['bs']}. Starting downsearch.")
            last_entry["phase"] = "downsearch_init"
        else:
            upsearch_history = [h for h in history if "downsearch" not in h.get("phase")]
            if len(upsearch_history) > 1:
                current_throughput = upsearch_history[-1]["items_per_second"]
                worst_throughput_so_far = min(h["items_per_second"] for h in upsearch_history[:-1])
                if current_throughput < worst_throughput_so_far:
                    print(f"  [Batch Advisor] Throughput regression at BS={last_entry['bs']}. Starting downsearch.")
                    last_entry["phase"] = "downsearch_init"

    # --- Phase 3: Determine Next Batch Size ---
    current_phase = last_entry.get("phase")

    if "downsearch" in current_phase:
        good_runs = [h for h in history if h["vram_peak_gb"] <= vram_budget_gb]
        bad_runs = [h for h in history if h["vram_peak_gb"] > vram_budget_gb]
        
        low_bs = max([h["bs"] for h in good_runs]) if good_runs else 0
        high_bs = min([h["bs"] for h in bad_runs]) if bad_runs else last_entry["bs"]

        if high_bs - low_bs <= 1:
            next_bs = low_bs if low_bs > 0 else 1
            print(f"  [Batch Advisor] Downsearch converged. Optimal BS: {next_bs}")
            last_entry["phase"] = "converged"
        else:
            next_bs = low_bs + (high_bs - low_bs) // 2
            if next_bs >= high_bs: next_bs = high_bs - 1
            if next_bs <= low_bs: next_bs = low_bs + 1
            if next_bs == 0: next_bs = 1
            print(f"  [Batch Advisor] Phase: downsearch, testing BS={next_bs} (Range: [{low_bs}, {high_bs}])")
        return next_bs, calibration_kwargs
    else: # Exponential Upsearch
        next_bs = last_entry["bs"] * 2
        print(f"  [Batch Advisor] Phase: upsearch, testing BS={next_bs}")
        return next_bs, calibration_kwargs

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
            torch.cuda.synchronize() # Crucial for accurate timing
            duration = time.time() - start_time
            vram_peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
            torch.cuda.reset_max_memory_allocated() # Reset for next measurement

            calibration_state["latest_run_metrics"] = {
            "bs": next_bs, # Use the batch size we just ran
            "duration": duration,
            "vram_peak_gb": vram_peak_gb
            }

            all_results.append(latents.cpu())

        except StopIteration:
            print("[VAE Encoder] Finished processing all images.")
            break
            
    loader.close()

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
    
    # Paths to configs and tokenizers now provided by the manifest
    te1_config_path = kwargs.get('text_encoder_1_config_path')
    te2_config_path = kwargs.get('text_encoder_2_config_path')
    tokenizer_1_path = kwargs.get('tokenizer_1_path')
    tokenizer_2_path = kwargs.get('tokenizer_2_path')

    # --- 2. Instantiate Empty Models from Local Canon Configs ---
    print("  [Text Encoder] Instantiating model shells from local configs...")
    with open(te1_config_path, 'r') as f: te1_config = json.load(f)
    with open(te2_config_path, 'r') as f: te2_config = json.load(f)
    
    text_encoder_1 = CLIPTextModel.from_config(te1_config)
    text_encoder_2 = CLIPTextModelWithProjection.from_config(te2_config)

    # --- 3. Manually Filter and Load Weights from Safetensors Checkpoint ---
    print("  [Text Encoder] Manually filtering and remapping keys from checkpoint...")
    te1_state_dict = {}
    te2_state_dict = {}
    
    # Define the prefixes used in SDXL checkpoints for the two text encoders
    # These are the equivalent of "first_stage_model." for the VAE.
    TE1_PREFIX = "conditioner.embedders.0.transformer.text_model."
    TE2_PREFIX = "conditioner.embedders.1.text_model."

    with safetensors.safe_open(model_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith(TE1_PREFIX):
                # Remap key by stripping the prefix
                new_key = key[len(TE1_PREFIX):]
                te1_state_dict[new_key] = f.get_tensor(key)
            elif key.startswith(TE2_PREFIX):
                # Remap key by stripping the prefix
                new_key = key[len(TE2_PREFIX):]
                te2_state_dict[new_key] = f.get_tensor(key)
    
    # Load the filtered and remapped state dicts into the model shells
    text_encoder_1.load_state_dict(te1_state_dict)
    text_encoder_2.load_state_dict(te2_state_dict)

    text_encoder_1.to(device=device, dtype=torch_dtype).eval()
    text_encoder_2.to(device=device, dtype=torch_dtype).eval()
    print("  [Text Encoder] Weights loaded successfully into shells.")

    # --- 4. Load Tokenizers ---
    tokenizer_1 = CLIPTokenizer.from_pretrained(tokenizer_1_path)
    tokenizer_2 = CLIPTokenizer.from_pretrained(tokenizer_2_path)
    
    # --- 5. The Rest of the Pipeline (unchanged) ---
    def text_preprocessor(item: Dict) -> Dict[str, torch.Tensor]:
        prompt = item['input_data']['prompt']
        tokens_1 = tokenizer_1(prompt, padding="max_length", max_length=tokenizer_1.model_max_length, truncation=True, return_tensors="pt").input_ids
        tokens_2 = tokenizer_2(prompt, padding="max_length", max_length=tokenizer_2.model_max_length, truncation=True, return_tensors="pt").input_ids
        return {"tokens_1": tokens_1.squeeze(), "tokens_2": tokens_2.squeeze()}

    id_iterator, loader_iterator = itertools.tee(work_iterator)
    work_ids = [spec['work_id'] for spec in id_iterator]
    loader = PipelinedTensorDictLoader(loader_iterator, text_preprocessor, device)
    
    # --- Add this: Initialize the calibration state ---
    calibration_state = {
        "config": {"device": device, "vram_safety_ratio": 0.8},
        "history": [],
        "latest_run_metrics": {}
    }
    all_embeds, all_pooled = [], []
    while True:
        try:
            # 1. ADVISE: Get the next batch size to try.
            next_bs, calibration_state = advise_next_batch_size(calibration_state)
            loader.set_batch_size(next_bs)

            # 2. TIME & FETCH: Start the timer and get the batch.
            start_time = time.time()
            on_device_batch = loader.get_next_batch()

            # 3. COMPUTE: The core model inference.
            with torch.no_grad():
                prompt_embeds_1 = text_encoder_1(on_device_batch['tokens_1'], output_hidden_states=True).hidden_states[-2]
                encoder_2_output = text_encoder_2(on_device_batch['tokens_2'], output_hidden_states=True)
                prompt_embeds_2 = encoder_2_output.hidden_states[-2]
                pooled_prompt_embeds = encoder_2_output.text_embeds
            
            final_prompt_embeds = torch.cat((prompt_embeds_1, prompt_embeds_2), dim=-1)

            # 4. MEASURE: Get timing and memory stats.
            torch.cuda.synchronize(device)
            duration = time.time() - start_time
            vram_peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
            torch.cuda.reset_peak_memory_stats(device)

            # 5. RECORD: Store metrics for the next advisor call.
            calibration_state["latest_run_metrics"] = {
                "bs": next_bs,
                "duration": duration,
                "vram_peak_gb": vram_peak_gb
            }

            # 6. STORE RESULTS: Append to CPU lists.
            all_embeds.append(final_prompt_embeds.cpu())
            all_pooled.append(pooled_prompt_embeds.cpu())

        except StopIteration:
            print("[Text Encoder] Finished processing all prompts.")
            break
        except torch.cuda.OutOfMemoryError:
            print(f"  [Batch Advisor] CUDA OOM detected at BS={next_bs}. Forcing downsearch.")
            torch.cuda.empty_cache()
            # Log the failure so the advisor knows this batch size is too large.
            calibration_state["latest_run_metrics"] = {"bs": next_bs, "duration": float('inf'), "vram_peak_gb": float('inf')}
            continue # Proceed to the next advisor call.

    loader.close()
    
    final_embeds = torch.cat(all_embeds, dim=0)
    final_pooled = torch.cat(all_pooled, dim=0)
    
    results_by_work_id = {
        work_id: {"text_embedding": final_embeds[i], "pooled_text_embedding": final_pooled[i]}
        for i, work_id in enumerate(work_ids)
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
    [COMPLIANT] A simple function that now returns a torch.Tensor.
    """
    data_iterator = kwargs['data_iterator']
    results = {}

    for spec in data_iterator:
        work_id = spec['work_id']
        scale_metadata = spec['input_data']['scale_metadata']
        # The core logic is tiny, but it's now wrapped to be compliant.
        results[work_id] = {"scales_tensor": torch.tensor(scale_metadata)}

    return results


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
    id_iterator, loader_iterator = itertools.tee(work_iterator)
    work_ids = [spec['work_id'] for spec in id_iterator]

    loader = PipelinedTensorDictLoader(
        raw_data_iterator=loader_iterator,
        preprocessing_fn=sdxl_vae_latent_preprocessor, # Use latent preprocessor
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

            torch.cuda.synchronize(device)
            duration = time.time() - start_time
            vram_peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
            torch.cuda.reset_peak_memory_stats(device)

            calibration_state["latest_run_metrics"] = {
                "bs": next_bs, "duration": duration, "vram_peak_gb": vram_peak_gb
            }
            all_results.append(images.cpu())

        except StopIteration:
            print("[VAE Decoder] Finished processing all latents.")
            break
        except torch.cuda.OutOfMemoryError:
            print(f"  [Batch Advisor] CUDA OOM detected at BS={next_bs}. Forcing downsearch.")
            torch.cuda.empty_cache()
            calibration_state["latest_run_metrics"] = {"bs": next_bs, "duration": float('inf'), "vram_peak_gb": float('inf')}
            continue
            
    loader.close()

    # 4. Collate and return results
    final_images = torch.cat(all_results, dim=0)
    print(f"--- [VAE Decoder] Complete. Produced final image tensor of shape: {final_images.shape} ---")
    results_by_work_id = {
        work_id: {"decoded_image": final_images[i]}
        for i, work_id in enumerate(work_ids)
    }
    return results_by_work_id


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
            torch.cuda.synchronize(device)
            duration = time.time() - start_time
            vram_peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
            torch.cuda.reset_peak_memory_stats(device)

            # CALIB5. RECORD
            calibration_state["latest_run_metrics"] = {
                "bs": next_bs, "duration": duration, "vram_peak_gb": vram_peak_gb
            }

            all_noisy_latents.append(noisy_latents.cpu())
            all_timesteps_for_unet.append(unet_timesteps.cpu())
        except StopIteration:
            print("[Denoise Encoder] Finished processing all images.")
            break
        except torch.cuda.OutOfMemoryError:
            print(f"  [Batch Advisor] CUDA OOM detected at BS={next_bs}. Forcing downsearch.")
            torch.cuda.empty_cache()
            calibration_state["latest_run_metrics"] = {"bs": next_bs, "duration": float('inf'), "vram_peak_gb": float('inf')}
            continue
    
    loader.close()

    # --- 4. Collate and Return ---
    final_noisy_latents = torch.cat(all_noisy_latents, dim=0)
    final_timesteps = torch.cat(all_timesteps_for_unet, dim=0)

    results_by_work_id = {
        work_id: {"noisy_latent": final_noisy_latents[i], "timestep_for_unet": final_timesteps[i]}
        for i, work_id in enumerate(work_ids)
    }
    return results_by_work_id

# NEW FUNCTION PER CLAUDE SPEC
def compute_validation_metrics(**kwargs) -> dict:
    """
    Computes LPIPS and FID between original and reconstructed images.
    Consumes two synchronized data streams.
    """
    # --- Config ---
    device = kwargs.get('device', 'cuda')
    lpips_net = kwargs.get('lpips_net', 'vgg') # 'vgg' or 'alex'
    fid_feature_dim = kwargs.get('fid_feature_dim', 64) # 64, 192, 768, 2048
    
    # --- Iterators (passed by the DAG runner) ---
    original_images_iterator = kwargs['original_data_stream']
    reconstructed_data_iterator = kwargs['reconstructed_data_iterator']

    # --- Metric Initialization ---
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type=lpips_net, normalize=True).to(device) # normalize expects [0,1]
    fid_metric = FrechetInceptionDistance(feature=fid_feature_dim, normalize=True).to(device) # normalize expects [0,1]
    print(f"--- [Validator] Metrics (LPIPS-{lpips_net}, FID-{fid_feature_dim}) initialized on {device}. ---")

    # --- Processing Loop ---
    # This assumes the two iterators are aligned, which the DAG must guarantee
    # by using the same source data stream and planning logic.
    sample_count = 0
    for original_spec, recon_spec in zip(original_images_iterator, reconstructed_data_iterator):
        try:
            # 1. Load original image and preprocess
            original_path = original_spec['input_data']['image']
            original_img = Image.open(original_path).convert("RGB")
            # NOTE: Preprocessing MUST match here. We assume a simple ToTensor for [0,1] range.
            preprocess = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
            original_tensor = preprocess(original_img).unsqueeze(0).to(device)

            # 2. Load reconstructed image tensor
            recon_tensor = recon_spec['input_data']['decoded_image'].unsqueeze(0).to(device)
            # Ensure tensors are 4D [B, C, H, W] and compatible shapes
            if original_tensor.shape != recon_tensor.shape:
                recon_tensor = torch.nn.functional.interpolate(recon_tensor, size=original_tensor.shape[2:])

            # 3. Update metrics
            lpips_metric.update(original_tensor, recon_tensor)
            fid_metric.update(original_tensor, real=True)
            fid_metric.update(recon_tensor, real=False)
            
            sample_count += 1
            if sample_count % 50 == 0:
                print(f"  [Validator] Processed {sample_count} image pairs...")

        except Exception as e:
            print(f"  - WARNING: Skipping validation pair due to error: {e}")
            continue

    # --- Final Computation ---
    final_lpips = lpips_metric.compute().item()
    final_fid = fid_metric.compute().item()
    
    print("--- [Validator] Metric Computation Complete ---")
    print(f"  - Total Samples: {sample_count}")
    print(f"  - Mean LPIPS:    {final_lpips:.4f}")
    print(f"  - FID Score:     {final_fid:.4f}")

    # Per spec, return a dictionary suitable for logging
    results = {
        "validation_metrics": {
           "lpips_mean": final_lpips,
           "fid_score": final_fid,
           "sample_count": sample_count,
        }
    }
    return results
