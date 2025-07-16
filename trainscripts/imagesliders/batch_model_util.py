import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DDPMScheduler, SchedulerMixin, DDIMScheduler, LMSDiscreteScheduler, EulerAncestralDiscreteScheduler, UNet2DConditionModel, AutoencoderKL
import torch
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from typing import Literal, Union

AVAILABLE_SCHEDULERS = Literal["ddim", "ddpm", "lms", "euler_a"]

SDXL_TEXT_ENCODER_TYPE = Union[CLIPTextModel, CLIPTextModelWithProjection]

DIFFUSERS_CACHE_DIR = None # if you want to change the cache dir, change this

def load_models(config, device, weight_dtype):
    print(f"Loading models from {config.pretrained_model.name_or_path} to device: {device} with dtype: {weight_dtype}")

    # Load the pipeline from the local .safetensors file
    pipe = StableDiffusionXLPipeline.from_single_file(
        config.pretrained_model.name_or_path,
        torch_dtype=weight_dtype,
        cache_dir=DIFFUSERS_CACHE_DIR,
    )

    unet = pipe.unet
    print(f"UNet time_embedding_dim: {unet.config.time_embedding_dim}")
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
    text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
    if len(text_encoders) == 2:
        text_encoders[1].pad_token_id = 0
    vae = pipe.vae
    del pipe
    #GET RID OF PIPE!!! 
    #you HAVE TO GET RID OF THE PIPE EVERY TIME!!!
    #if you do a 'blah = pipe.blah.to(device, dtype)' you DOUBLE LOAD THE MODEL,

    # Enable gradient checkpointing if configured
    if hasattr(config, 'other') and hasattr(config.other, 'gradient_checkpointing') and config.other.gradient_checkpointing:
        print("Enabling gradient checkpointing for UNet.")
        unet.enable_gradient_checkpointing()

    return vae, unet, tokenizers, text_encoders


def create_noise_scheduler(
    scheduler_name: AVAILABLE_SCHEDULERS = "ddpm",
    prediction_type: Literal["epsilon", "v_prediction"] = "epsilon",
) -> SchedulerMixin:
    # 正直、どれがいいのかわからない。元の実装だとDDIMとDDPMとLMSを選べたのだけど、どれがいいのかわからぬ。

    name = scheduler_name.lower().replace(" ", "_")
    if name == "ddim":
        # https://huggingface.co/docs/diffusers/v0.17.1/en/api/schedulers/ddim
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
            prediction_type=prediction_type,  # これでいいの？
        )
    elif name == "ddpm":
        # https://huggingface.co/docs/diffusers/v0.17.1/en/api/schedulers/ddpm
        scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
            prediction_type=prediction_type,
        )
    elif name == "lms":
        # https://huggingface.co/docs/diffusers/v0.17.1/en/api/schedulers/lms_discrete
        scheduler = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            prediction_type=prediction_type,
        )
    elif name == "euler_a":
        # https://huggingface.co/docs/diffusers/v0.17.1/en/api/schedulers/euler_ancestral
        scheduler = EulerAncestralDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            prediction_type=prediction_type,
        )
    else:
        raise ValueError(f"Unknown scheduler name: {name}")

    return scheduler

def load_diffusers_model_xl(
    pretrained_model_name_or_path: str,
    weight_dtype: torch.dtype = torch.float32,
) -> tuple[list[CLIPTokenizer], list[SDXL_TEXT_ENCODER_TYPE], UNet2DConditionModel,]:
    # returns tokenizer, tokenizer_2, text_encoder, text_encoder_2, unet

    tokenizers = [
        CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer",
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
        ),
        CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
            pad_token_id=0,  # same as open clip
        ),
    ]

    text_encoders = [
        CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
        ),
        CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
        ),
    ]

    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
        torch_dtype=weight_dtype,
        cache_dir=DIFFUSERS_CACHE_DIR,
    )
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
    return tokenizers, text_encoders, unet, vae

#gemini wanted this to be trainscripts/imagesliders/patcher.py
import types

def _to_method(self, *args, **kwargs):
    """
    The method that will be attached to the scheduler instance.
    It iterates over all attributes and moves tensors to the specified device.
    """
    # Iterate through all attributes of the object
    for attr_name in dir(self):
        # Skip private/special methods and the 'to' method itself to avoid recursion
        if attr_name.startswith('_') or attr_name == 'to':
            continue

        attr_value = getattr(self, attr_name)

        # If the attribute is a tensor, move it
        if isinstance(attr_value, torch.Tensor):
            new_tensor = attr_value.to(*args, **kwargs)
            setattr(self, attr_name, new_tensor)
            
    # Return self to allow for chaining, like nn.Module.to()
    return self

def add_to_method_to_instance(instance):
    """
    Monkey-patches a .to() method onto a single object instance.
    This is safer than patching the entire class.

    Args:
        instance: The object instance (e.g., a DDIMScheduler) to patch.
    """
    # Use types.MethodType to bind our function to the instance as a method
    instance.to = types.MethodType(_to_method, instance)
    return instance

# in batch_model_util
# gemini wanted to squirrel these away in an extra file >:(

import torch
import gc
from tqdm import tqdm
import time

class OffloadingOrchestrator:
    """
    Manages device placement, VRAM, and dynamic batching for a sequence of
    computationally heavy steps (e.g., inference, decoding).

    This class is "functionally ignorant." It only knows how to move models,
    find optimal batch sizes for the current hardware, and call the
    functions provided to it in the environment.
    """
    def __init__(self, device: str):
        self.device = device
        self._device_snapshot = {}

    def _snapshot_devices(self, environment: dict):
        """Records the current device of all nn.Module instances."""
        self._device_snapshot = {}
        for key, module in environment.items():
            if isinstance(module, torch.nn.Module):
                try:
                    self._device_snapshot[key] = next(module.parameters()).device
                except StopIteration:
                    pass

    def _restore_devices(self, environment: dict):
        """Restores all models to their original devices."""
        for key, device in self._device_snapshot.items():
            if key in environment and isinstance(environment[key], torch.nn.Module):
                try:
                    environment[key].to(device)
                except Exception as e:
                    print(f"Warning: Could not restore {key} to {device}. Reason: {e}")
        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def execute(self,
                initial_workload: list,
                environment: dict,
                processing_steps: list[dict]):
        
        self._snapshot_devices(environment)
        current_workload = initial_workload
        
        try:
            for step_config in processing_steps:
                fn_key = step_config['key'] + '_fn'
                model_key = step_config['model_key']
                
                model = environment[model_key]
                processing_fn = environment[fn_key]

                # Use the new throughput-based batch finder
                # The "sample_input" is now the fully-formed dictionary

                # Pass the processing_fn and the environment to the finder.
                # The finder now has everything it needs to perform a realistic test.
                batch_finder = ThroughputBatchFinder(
                    model, 
                    current_workload[0], # The sample input dict
                    self.device,
                    processing_fn,       # The function to test
                    environment          # The context for the function
                )
                batch_size = batch_finder.find()
                
                model.to(self.device)
                for key, m in environment.items():
                    if isinstance(m, torch.nn.Module) and m is not model:
                        m.to('cpu')
                gc.collect()
                torch.cuda.empty_cache()

                output_pool = []
                for i in tqdm(range(0, len(current_workload), batch_size), desc=f"Processing {fn_key}"):
                    batch_work = current_workload[i:i + batch_size]
                    output_chunk = processing_fn(model, batch_work, environment)
                    output_pool.extend(output_chunk)
                
                current_workload = output_pool
                
        finally:
            self._restore_devices(environment)
            
        return current_workload

# gemini wanted to squirrel these away in an extra file >:(

from tqdm import tqdm

def _get_pred_xl(unet, latents, timestep, conditioning, guidance_scale):
    """A helper function to get the guided prediction for a single step."""
    # The input `latents` is already the CFG-duplicated tensor
    # The conditioning tensors need to be moved to the device
    encoder_hidden_states = conditioning['encoder_hidden_states'].to(unet.device)
    added_cond_kwargs = {k: v.to(unet.device) for k, v in conditioning['added_cond_kwargs'].items()}
    
    noise_pred = unet(
        latents.to(unet.dtype),
        timestep,
        encoder_hidden_states=encoder_hidden_states,
        added_cond_kwargs=added_cond_kwargs
    ).sample
    
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    guided_target = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )
    return guided_target

def diffusion_fn(model, batch_work, environment):
    """
    Performs the ENTIRE multi-step diffusion loop for a batch.
    This is the "simple function" called by the orchestrator.
    """
    # 1. Unpack components from environment
    scheduler = environment['scheduler'] # MinimalDDPMScheduler is passed in
    device = environment['device']

    # 2. Collate batch data and move initial state to device
    initial_latents = torch.cat([d['initial_latents'] for d in batch_work]).to(device)
    
    # 3. All items in a batch share diffusion parameters and conditioning
    diffusion_params = batch_work[0]['diffusion_params']
    conditioning_uncond = batch_work[0]['conditioning_uncond'] # Assuming an uncond is prepared
    conditioning_cond = batch_work[0]['conditioning']

    guidance_scale = diffusion_params['guidance_scale']
    total_timesteps = diffusion_params['total_timesteps']
    
    # 4. Prepare for the loop
    scheduler.set_timesteps(total_timesteps, device=device)
    latents = initial_latents

    # 5. The Loop (as an implementation detail inside this function)
    for t in scheduler.timesteps:
        # Prepare for CFG: duplicate latents and conditioning
        latent_model_input = torch.cat([latents] * 2)
        
        # This is where the multiple conditioning dicts would be used
        # For simplicity, we assume one cond and one uncond for the whole batch
        combined_conditioning = {
            'encoder_hidden_states': torch.cat([conditioning_uncond['encoder_hidden_states'], conditioning_cond['encoder_hidden_states']]),
            'added_cond_kwargs': {
                'text_embeds': torch.cat([conditioning_uncond['added_cond_kwargs']['text_embeds'], conditioning_cond['added_cond_kwargs']['text_embeds']]),
                'time_ids': torch.cat([conditioning_uncond['added_cond_kwargs']['time_ids'], conditioning_cond['added_cond_kwargs']['time_ids']]),
            }
        }
        
        # Get the model's prediction
        model_pred = _get_pred_xl(model, latent_model_input, t, combined_conditioning, guidance_scale)
        
        # Compute the previous noisy sample x_t -> x_{t-1}
        latents = scheduler.step(model_pred, t, latents)

    # 6. Return the final latents as a list of CPU tensors
    return list(torch.chunk(latents.cpu(), chunks=latents.shape[0], dim=0))

def vae_decoder_fn(model, batch_work, environment):
    """
    A stateless VAE decoding function.

    Args:
        model: The VAE model, already on the correct device.
        batch_work (list): A list of latent tensors.
    
    Returns:
        A list of decoded images on the CPU.
    """
    device = environment['device']
    
    # Combine the list of tensors into a single batch
    latents_batch = torch.cat(batch_work, dim=0).to(device)
    
    # 1 / 0.18215 is the scaling factor for Stable Diffusion's VAE
    images = model.decode(latents_batch / 0.18215).sample
    images = (images / 2 + 0.5).clamp(0, 1) # Denormalize
    
    # Return a list of CPU tensors
    return list(torch.chunk(images.cpu(), chunks=latents_batch.shape[0], dim=0))


def testeruvit_diffusion_fn(model, batch_work, environment):
    """
    A shim diffusion function for the TesterUViT.
    It performs a single forward pass, as there's no multi-step sampling loop.
    It respects the expected input/output data contract.
    
    Args:
        model (nn.Module): The TesterUViT model.
        batch_work (list[dict]): A list of workload dicts.
                                 Expects {'initial_latents': tensor, 'timesteps': tensor}
    
    Returns:
        list[torch.Tensor]: A list of output tensors (predicted noise), on CPU.
    """
    model_device = environment["model_device"]
    # Collate the batch from the list of workload dictionaries
    latents_batch = torch.cat([d['initial_latents'] for d in batch_work]).to(model_device)
    timesteps_batch = torch.cat([d['timesteps'] for d in batch_work]).to(model_device)

    # The TesterUViT's forward pass is simple. It ignores conditioning kwargs.
    # The network is already a FlexibleLoRANetwork, so this just works.
    output_tensors = model(latents_batch, timesteps_batch)
    
    # Return a list of CPU tensors, fulfilling the contract of the orchestrator's functions.
    return list(torch.chunk(output_tensors.cpu(), chunks=output_tensors.shape[0], dim=0))

def testeruvit_decoder_fn(model, batch_work, environment):
    """
    A shim VAE decoder for the TesterUViT.
    Since the model operates in pixel space, this is an identity function.
    It receives "latents" (which are actually images) and returns them.
    
    Returns:
        list[torch.Tensor]: The same list of tensors that was passed in.
    """
    # This function does nothing because the "latents" are already the final images.
    # It exists solely to satisfy the orchestrator's pipeline structure.
    return batch_work

import time
import numpy as np

class ThroughputBatchFinder:
    """
    Finds the optimal batch size by measuring throughput, not by waiting for
    an OOM error. Detects the performance cliff caused by VRAM oversubscription.
    """
    def __init__(self, model, sample_input, device, processing_fn, environment,
    max_batch_size=256, slowdown_threshold=0.95):
        self.model = model
        self.sample_input = sample_input # This is a single dict {'initial_latents':...}
        self.device = device
        self.processing_fn = processing_fn
        self.environment = environment
        self.max_batch_size = max_batch_size
        self.slowdown_threshold = slowdown_threshold
        self.environment["model_device"] = device

    @torch.no_grad()
    def find(self):
        self.model.to(self.device)
        
        best_bs = 1
        best_throughput = 0.0
        slowdown_bs = -1

        # 1. Exponential search to find the performance cliff
        print(f"--- [BatchFinder] Starting exponential search for {self.model.__class__.__name__}... ---")
        bs = 1
        while bs <= self.max_batch_size:
            #batch_input = torch.cat([self.sample_input] * bs, dim=0).to(self.device)
            batch_work = [self.sample_input] * bs
            
            try:
                # Warm-up run
                _ = self.processing_fn(self.model, batch_work, self.environment)
                
                # Timed run
                start_time = time.time()
                _ = self.processing_fn(self.model, batch_work, self.environment)
                torch.cuda.synchronize() # Wait for GPU to finish
                end_time = time.time()
            except torch.cuda.OutOfMemoryError:
                print(f"  - OOM at BS={bs}. Halting search.")
                break

            throughput = bs / ((end_time - start_time)+1e-6)   #a little epsilon for my friends
            print(f"  - Testing BS={bs}: {throughput:.2f} items/sec")

            if throughput > best_throughput:
                best_throughput = throughput
                best_bs = bs
            
            if throughput < best_throughput * self.slowdown_threshold and bs > best_bs:
                print(f"  - Throughput slowdown detected at BS={bs}. Optimal is near {best_bs}.")
                slowdown_bs = bs
                break
                
            bs *= 2
            del batch_work, _ # Free memory before next iteration

        # 2. Binary search to refine the optimal point if a slowdown was found
        if slowdown_bs != -1:
            print("--- [BatchFinder] Starting binary search to refine... ---")
            low = best_bs
            high = slowdown_bs
            
            # We search for the highest batch size that does NOT cause a slowdown
            while low <= high:
                mid = (low + high) // 2
                if mid == 0: break

                batch_work = [self.sample_input] * mid
                
                start_time = time.time()
                _ = self.processing_fn(self.model, batch_work, self.environment)
                torch.cuda.synchronize()
                end_time = time.time()
                
                mid_throughput = mid / ((end_time - start_time)+1e-6)   #a little epsilon for my friends
                print(f"  - Binary search BS={mid}: {mid_throughput:.2f} items/sec")

                if mid_throughput >= best_throughput * self.slowdown_threshold:
                    # This batch size is still good, try for more
                    best_bs = mid
                    low = mid + 1
                else:
                    # This batch size is too slow, reduce
                    high = mid - 1
                del batch_work, _

        self.model.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"--- [BatchFinder] Optimal batch size found: {best_bs} ---")
        return best_bs

def run_evaluation_flow(model_to_test, environment, workload, sampling_config):
    """
    A stateless utility that runs the full generation pipeline for a given workload.
    It uses the OffloadingOrchestrator to manage VRAM and batching.
    """
    # 1. Prepare the environment for this specific run
    eval_env = environment.copy()
    eval_env['inference_network'] = model_to_test
    
    # These functions are assumed to be in the environment already
    # eval_env['diffusion_fn'] = diffusion_fn
    # eval_env['decoder_fn'] = vae_decoder_fn
    
    # 2. Configure the processing pipeline
    processing_steps = [
        {'key': 'diffusion', 'model_key': 'inference_network', 'kwargs': sampling_config},
        {'key': 'decoder', 'model_key': 'vae'}
    ]

    # 3. Execute with the orchestrator
    orchestrator = OffloadingOrchestrator(device=eval_env['device'])
    final_images = orchestrator.execute(
        initial_workload=workload,
        environment=eval_env,
        processing_steps=processing_steps
    )
    return final_images