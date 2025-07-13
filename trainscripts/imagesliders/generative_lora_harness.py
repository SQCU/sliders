# generative_lora_harness.py
#python -m trainscripts.imagesliders.generative_lora_harness
# The next evolution of our testing system, focused on a meaningful generative task.
# This script includes a new scheduler, a synthetic dataset, real evaluation metrics,
# and a new experiment runner to tie them all together.

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os
from PIL import Image, ImageDraw
import numpy as np

# We assume the previously "locked-in" components are available.
# For this script, we'll redefine them for clarity.
from .flexible_lora_system import FlexibleLoRANetwork, LoRAConfigLoader, TesterUViT
from .architecture_search_controller import create_search_space_yaml
#new project imports
#uv pip install torchmetrics[image]
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
#from torchmetrics.image.frechet_inception_distance import FrechetInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance

# ==============================================================================
# SECTION 1: THE NEW, TRANSPARENT DIFFUSION SCHEDULER
# ==============================================================================

class MinimalDDPMScheduler:
    """
    A functional, from-scratch DDPM scheduler that is clear and supports
    multiple prediction targets, as requested for serious research.
    """
    def __init__(self, num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, device='cpu'):
        self.num_train_timesteps = num_train_timesteps
        self.device = device
        
        # The core of the schedule: a linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        
        # Pre-compute the alphas and their cumulative products, which are the
        # actual values used in the diffusion process. This is the "physics".
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # These are the terms used in the q(x_t | x_0) forward process
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Move everything to the correct device once during initialization
        self.to(device)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        return self

    def _gather(self, consts: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Gathers the constants for a given timestep t."""
        c = consts.gather(-1, t)
        return c.reshape(-1, 1, 1, 1) # Reshape for broadcasting to image shape

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """The forward diffusion process q(x_t | x_0)."""
        sqrt_alpha_prod = self._gather(self.sqrt_alphas_cumprod, timesteps)
        sqrt_one_minus_alpha_prod = self._gather(self.sqrt_one_minus_alphas_cumprod, timesteps)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def get_prediction_target(self, original_samples, noise, timesteps, prediction_type="epsilon"):
        """
        As requested, this function provides the correct target for the model's
        prediction based on the desired training objective.
        - 'epsilon': (Default) Predict the noise that was added.
        - 'sample': Predict the original clean image (x_0).
        - 'v_prediction': Predict the "velocity", a reparameterization that can improve stability.
        """
        if prediction_type == "epsilon":
            return noise
        elif prediction_type == "sample":
            return original_samples
        elif prediction_type == "v_prediction":
            sqrt_alpha_prod = self._gather(self.sqrt_alphas_cumprod, timesteps)
            sqrt_one_minus_alpha_prod = self._gather(self.sqrt_one_minus_alphas_cumprod, timesteps)
            # v = sqrt(alpha_prod) * noise - sqrt(1-alpha_prod) * x_0
            return sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * original_samples
        else:
            raise ValueError(f"Unknown prediction type: {prediction_type}")

# ==============================================================================
# SECTION 2: THE "SHAM" SYNTHETIC DATASET
# ==============================================================================

class ShamImageDataset(torch.utils.data.Dataset):
    """
    Creates a synthetic dataset of two textured spheres on a textured background.
    This provides a simple, reproducible, yet non-trivial generative task.
    """
    def __init__(self, num_samples=128, size=64):
        self.num_samples = num_samples
        self.size = size
        self.cache = {}

    def _create_texture(self, size, p1, p2, color1, color2):
        """Creates a checkerboard or striped texture."""
        img = Image.new('RGB', (size, size))
        pixels = img.load()
        for i in range(size):
            for j in range(size):
                if (i // p1) % 2 == (j // p2) % 2:
                    pixels[i, j] = color1
                else:
                    pixels[i, j] = color2
        return np.array(img) / 255.0

    def _create_image(self, index):
        """Generates a single image for the dataset."""
        # Background texture
        bg_color1, bg_color2 = ((50, 50, 50), (60, 60, 60)) if index % 2 == 0 else ((200, 200, 200), (210, 210, 210))
        background = self._create_texture(self.size, 8, 8, bg_color1, bg_color2)

        # Sphere 1
        s1_texture = self._create_texture(self.size, 4, 8, (255, 0, 0), (200, 0, 0)) # Red stripes
        s1_mask = Image.new('L', (self.size, self.size), 0)
        draw = ImageDraw.Draw(s1_mask)
        draw.ellipse((5, 10, 25, 30), fill=255)
        
        # Sphere 2
        s2_texture = self._create_texture(self.size, 5, 5, (0, 0, 255), (0, 0, 200)) # Blue checkerboard
        s2_mask = Image.new('L', (self.size, self.size), 0)
        draw = ImageDraw.Draw(s2_mask)
        draw.ellipse((35, 25, 60, 50), fill=255)

        # Composite the image
        s1_mask_np = np.array(s1_mask)[:, :, None] / 255.0
        s2_mask_np = np.array(s2_mask)[:, :, None] / 255.0
        
        image = background * (1 - s1_mask_np) + s1_texture * s1_mask_np
        image = image * (1 - s2_mask_np) + s2_texture * s2_mask_np

        return torch.from_numpy(image).permute(2, 0, 1).float() # HWC -> CHW

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index]
        image = self._create_image(index)
        self.cache[index] = image
        return image

# ==============================================================================
# SECTION 3: REAL EVALUATION SUITE
# ==============================================================================

class GenerativeEvaluator:
    """Uses real metrics (LPIPS, FID) to evaluate generative model quality."""
    def __init__(self, ground_truth_dataset, device='cpu'):
        self.device = device
        self.ground_truth_dataset = ground_truth_dataset
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
        
        # NOTE: Using a smaller feature size for FID for speed in this test harness.
        self.fid = FrechetInceptionDistance(feature=64).to(device)
        
        # Pre-calculate features for the ground truth dataset once.
        print("--- Evaluator: Calculating features for ground truth dataset... ---")
        gt_loader = torch.utils.data.DataLoader(ground_truth_dataset, batch_size=16)
        for batch in gt_loader:
            # FID expects uint8 images in range [0, 255]
            self.fid.update((batch.to(device) * 255).to(torch.uint8), real=True)
        print("--- Ground truth features cached. ---")

    @torch.no_grad()
    def evaluate(self, model_to_test, num_samples=64):
        """Generates images from the model and computes performance scores."""
        model_to_test.eval()
        
        # Generate a batch of images from the model
        # NOTE: A real generation loop would use the scheduler to denoise from random noise.
        # For this test, we'll just do a single forward pass for simplicity.
        noise = torch.randn(num_samples, 3, self.ground_truth_dataset.size, self.ground_truth_dataset.size, device=self.device)
        generated_images = model_to_test(noise).clamp(0, 1)

        # Get a corresponding batch of ground truth images
        gt_images = torch.stack([self.ground_truth_dataset[i] for i in range(num_samples)]).to(self.device)

        # 1. Calculate LPIPS (lower is better)
        lpips_score = self.lpips(generated_images, gt_images)
        
        # 2. Update FID with generated images and compute score (lower is better)
        self.fid.update((generated_images * 255).to(torch.uint8), real=False)
        fid_score = self.fid.compute()
        self.fid.reset() # Reset for the next evaluation

        # The final score is a combination, we can weight them.
        # We want to MINIMIZE this score.
        performance_score = lpips_score + fid_score * 0.1 # Weight FID less
        
        print(f"Evaluation Complete: LPIPS={lpips_score:.4f}, FID={fid_score:.4f} -> Final Score={performance_score:.4f}")
        return performance_score.item()

# ==============================================================================
# SECTION 4: THE NEW GENERATIVE EXPERIMENT RUNNER
# ==============================================================================

def run_generative_experiment(config_dict, freeze_base_model=True, log_dir="gen_logs"):
    """
    The new experiment runner, focused on the generative task.
    This replaces the simple `run_experiment` and becomes our new "fitness function".
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- 1. Setup Dataset and Scheduler ---
    dataset = ShamImageDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    scheduler = MinimalDDPMScheduler(device=device)
    
    # --- 2. Build the Network (with the "Bold/Stupid" option) ---
    uvit = TesterUViT()
    if freeze_base_model:
        print("--- Freezing base model weights for adapter-only training stress test ---")
        for param in uvit.parameters():
            param.requires_grad = False
    
    # Use the existing config loader
    loader = LoRAConfigLoader(config_dict=config_dict)
    resolved_config = loader.get_resolved_config(uvit)
    network = FlexibleLoRANetwork(uvit, resolved_config).to(device)
    
    optimizer = torch.optim.AdamW(network.prepare_optimizer_params(), lr=1e-3)
    
    # --- 3. The Generative Training Loop ---
    print("--- Starting Generative Training Loop ---")
    for epoch in range(5): # Short training run for a test
        for i, clean_images in enumerate(dataloader):
            optimizer.zero_grad()
            
            # --- The DataMunger Step ---
            clean_images = clean_images.to(device)
            noise = torch.randn_like(clean_images)
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (clean_images.shape[0],), device=device).long()
            
            noisy_images = scheduler.add_noise(clean_images, noise, timesteps)
            target = scheduler.get_prediction_target(clean_images, noise, timesteps, prediction_type="epsilon")
            # --- End DataMunger ---
            
            model_output = network(noisy_images) # The model predicts the target
            loss = F.mse_loss(model_output, target)
            
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Final Batch Loss: {loss.item():.4f}")
        
    # --- 4. Evaluate the trained model ---
    evaluator = GenerativeEvaluator(dataset, device=device)
    final_score = evaluator.evaluate(network)
    
    return final_score


# --- Example of how the search controller would use this ---
if __name__ == "__main__":
    # The search controller would be nearly identical to the previous version,
    # but its main call would be to `run_generative_experiment`.
    
    print("\n--- Example single run of the new Generative Harness ---")
    
    # Create a sample config to test with
    config_dict = {
        'lora_rules': [
            {'name': "All_Attention", 'rank': 8, 'alpha': 1.0, 'init_scheme': 'kaiming_cheald', 'target_name_contains': ['attn']},
            {'name': "All_FFN", 'rank': 16, 'alpha': 2.0, 'target_name_contains': ['ff.net']}
        ]
    }
    
    # Run the experiment.
    # We set freeze_base_model=True to run the "bold/stupid" test.
    score = run_generative_experiment(config_dict, freeze_base_model=True)
    
    print(f"\nExperiment finished with final performance score: {score:.4f} (lower is better)")