# In a new file, e.g., 'search_evaluator.py'

import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.frechet_inception_distance import FrechetInceptionDistance
# Note: FD-DINOv2 would require a custom implementation or finding a library.
# We'll use standard FID here as a stand-in, but the principle is the same.

class SearchEvaluator:
    def __init__(self, reference_model, reference_prompts_and_images, training_prompts_and_images, device='cuda'):
        self.device = device
        self.reference_model = reference_model.to(device).eval()
        self.ref_prompts = [item['prompt'] for item in reference_prompts_and_images]
        self.ref_images = torch.stack([item['image'] for item in reference_prompts_and_images]).to(device)
        self.train_prompts = [item['prompt'] for item in training_prompts_and_images]
        self.train_images = torch.stack([item['image'] for item in training_prompts_and_images]).to(device)
        
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
        self.fid = FrechetInceptionDistance(feature=64).to(device) # Using FID-64 for speed

    @torch.no_grad()
    def evaluate(self, candidate_network):
        """
        Runs the full evaluation on a candidate network and returns a single score.
        """
        candidate_network.to(self.device).eval()
                                                                                                                                                                                                     
        # 1. Fidelity Score (similarity to training data)
        train_gen_images = self._generate_images(candidate_network, self.train_prompts)
        fidelity_score = -self.lpips(self.train_images, train_gen_images) # Negative LPIPS: higher is better

        # 2. Forgetting Score (similarity to reference model's output)
        ref_gen_images = self._generate_images(self.reference_model, self.ref_prompts)
        candidate_gen_images_for_ref = self._generate_images(candidate_network, self.ref_prompts)
        forgetting_score = -self.lpips(ref_gen_images, candidate_gen_images_for_ref)
        
        # 3. Generality/Coherence Score (distributional similarity)
        # Update FID with real (reference) and fake (candidate) images
        self.fid.update(ref_gen_images.byte(), real=True)
        self.fid.update(candidate_gen_images_for_ref.byte(), real=False)
        generality_score = -self.fid.compute() # Negative FID: higher is better

        # Combine scores (weights are hyperparameters of the search)
        # This embodies the "Pareto" idea - a weighted combination of objectives.
        
        # each of these components should be logged independently in addition to their sum reduction.
        # is that a numpy kinda thing? maybe...
        w_fidelity = 1.0
        w_forgetting = 1.0
        w_generality = 0.5
        
        total_score = (w_fidelity * fidelity_score + 
                       w_forgetting * forgetting_score + 
                       w_generality * generality_score)
                       
        return total_score.item()

    def _generate_images(self, model, prompts):
        # Placeholder for your model's actual generation function
        # This would use your diffusion pipeline (e.g., DDIM, Euler)
        # to generate a batch of images from a list of prompts.
        # For now, returns random tensors for structure.

        # this should not be a complicated function under any circumstances; 
        # if necessary we can pass an environment dict into the searchevaluator init.
        # the searchevaluator init can then use the exact generation function our network design needs.
        return torch.randn_like(self.ref_images)