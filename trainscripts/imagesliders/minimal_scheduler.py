# minimal_scheduler.py
# A standalone, from-scratch DDPM scheduler.

import torch
import torch.nn.functional as F

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
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # These are the terms used in the q(x_t | x_0) forward process
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # === NEW: Values used in the REVERSE process (step) ===
        # alphas_cumprod for the PREVIOUS timestep
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # This is the term that gets multiplied by the model's output (predicted noise)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # This is the variance of the posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
        # Move everything to the correct device once during initialization
        self.to(device)

    def to(self, device):
        # Move all pre-computed tensors to the specified device
        for attr_name in dir(self):
            attr_value = getattr(self, attr_name)
            if torch.is_tensor(attr_value):
                setattr(self, attr_name, attr_value.to(device))
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

    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor, prediction_type="epsilon") -> torch.Tensor:
        """
        The core of the sampling loop. Predicts the sample at the previous timestep, x_{t-1}.
        """
        t = torch.tensor([timestep], device=self.device)
        
        # 1. Get the predicted original sample (x_0) from the model's output
        if prediction_type == "epsilon":
            # The formula to derive x_0 from x_t and the predicted noise (epsilon)
            pred_original_sample = self._gather(self.sqrt_recip_alphas_cumprod, t) * sample - \
                                   self._gather(self.sqrt_recipm1_alphas_cumprod, t) * model_output
        else:
            raise NotImplementedError("Only epsilon prediction is implemented for sampling.")

        # Optional: Clamp the predicted x_0 to be in the valid range [-1, 1] or [0, 1]
        # This is a common trick to improve stability. Assuming our data is [0, 1].
        # pred_original_sample = torch.clamp(pred_original_sample, 0.0, 1.0)

        # 2. Compute the coefficients for the posterior mean q(x_{t-1} | x_t, x_0)
        beta_t = self._gather(self.betas, t)
        sqrt_one_minus_alphas_cumprod_t = self._gather(self.sqrt_one_minus_alphas_cumprod, t)
        sqrt_alpha_t_prev = self._gather(torch.sqrt(self.alphas_cumprod_prev), t)
        
        # Equation (7) from DDPM paper
        posterior_mean = (sqrt_alpha_t_prev * beta_t / (1. - self.alphas_cumprod[t])) * pred_original_sample + \
                         (self._gather(torch.sqrt(self.alphas), t) * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])) * sample

        # 3. Add noise to get the final sample for the previous timestep
        variance = self._gather(self.posterior_variance, t)
        noise = torch.randn_like(sample)

        # No noise is added at the final step
        prev_sample = posterior_mean + (variance.sqrt() * noise if timestep > 0 else 0)

        return prev_sample