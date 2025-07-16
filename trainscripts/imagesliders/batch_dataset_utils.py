# In a new file, e.g., batch_dataset_utils.py

from .batch_model_util import OffloadingOrchestrator
from .batch_model_util import run_evaluation_flow # Import the stateless utility
import torch 
from PIL import Image, ImageDraw, ImageFont
import numpy as np

class EvalGroundedT2IDataset:
    """
    Creates an evaluation dataset grounded in the actual training data.
    """
    def __init__(self, config_path: str, environment: dict):
        # ... logic to load a subset of training prompts/images from the config ...
        print("--- [GroundedFactory] Initialized. Will use a subset of training data. ---")
        self.model_inputs = [] # list of totalized conditioning dicts
        self.ground_truth_outputs = [] # list of corresponding ground truth images

    def create(self):
        # ... logic to use JIT_ConditioningProvider and load images ...
        # This is mostly file I/O and pre-processing.
        print("--- [GroundedFactory] Dataset created successfully. ---")
        return {"model_inputs": self.model_inputs, "ground_truth_outputs": self.ground_truth_outputs}

class EvalGroundlessT2IDataset:
    """
    Creates an evaluation dataset 'groundlessly' by using a reference model
    to generate the 'ground truth' for novel prompts. This is the horrible mess.
    """
    def __init__(self, config: dict, environment: dict, reference_model: torch.nn.Module):
        self.config = config
        self.env = environment
        self.reference_model = reference_model
        self.dataset = None
        print("--- [GroundlessFactory] Initialized. Ready to generate reference data. ---")

    def create(self):
        """
        This is where the magic happens. We use our own harness to generate the dataset.
        This is a (model, batchwork, environment) scenario.
        """
        if self.dataset:
            return self.dataset

        print("--- [GroundlessFactory] Beginning generation of reference dataset... ---")
        
        # 1. Load novel prompts from a config file. No ad-hoc runtime definitions.
        with open(self.config['prompt_source_file'], 'r') as f:
            novel_prompts = [line.strip() for line in f.readlines()]

        # 2. Materialize conditioning for the novel prompts.
        cond_provider = JIT_ConditioningProvider(self.env)
        model_inputs = cond_provider.materialize(novel_prompts, self.config)

        # 3. Generate the "ground truth" images using the reference model.
        # We call our stateless utility function to do this. This is the doctrine.
        print("--- [GroundlessFactory] Using the evaluation flow to generate ground truth... ---")
        ground_truth_outputs = run_evaluation_flow(
            model_to_test=self.reference_model,
            environment=self.env,
            workload=model_inputs,
            sampling_config=self.config['sampling_config']
        )

        # (Here is where you would do the "stupid morass" analysis)
        # - Load training prompts, create training embeddings
        # - Compute cosine similarity between novel_embeddings and training_embeddings
        # - Use a feature extractor (DINOv2) on the embeddings themselves
        # - Log all these divergence metrics alongside the dataset.
        print("--- [GroundlessFactory] Analysis of prompt divergence would happen here. ---")

        self.dataset = {"model_inputs": model_inputs, "ground_truth_outputs": ground_truth_outputs}
        print("--- [GroundlessFactory] Groundless dataset created successfully. ---")
        return self.dataset



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

# in batch_datset_utils
class JIT_ConditioningProvider:
    """
    Takes a high-level request (e.g., list of text prompts) and materializes
    the full, "totalized" tensor conditioning required by the UNet.

    It performs its own offloading and bounds testing for the text encoders.
    """
    def __init__(self, environment: dict):
        self.device = environment['device']
        self.text_encoders = environment['text_encoders'] # Expects a list of encoders
        self.tokenizers = environment['tokenizers']
        # Store other necessary components from env if needed
        
    @torch.no_grad()
    def materialize(self, prompt_list: list, config: dict):
        from .batch_train_util import encode_prompts_xl
        """The main function that runs the text-to-tensor pipeline."""
        print("--- [ConditioningProvider] Starting JIT materialization... ---")
        
        # Determine optimal batch size for the text encoders
        # We assume all text encoders can handle the same batch size for simplicity
        sample_text_input = self.tokenizers[0]("sample", return_tensors="pt").input_ids
        # Note: A real implementation would need a dummy forward pass for text encoders
        # For now, we'll hardcode a reasonable batch size.
        text_encoder_batch_size = 16 
        print(f"--- [ConditioningProvider] Using batch size {text_encoder_batch_size} for text encoders. ---")

        # Offload other models from the environment before starting
        # (This would be a more complex device management in a full system)
        
        workload = []
        for i in tqdm(range(0, len(prompt_list), text_encoder_batch_size), desc="Encoding prompts"):
            batch_prompts = prompt_list[i:i + text_encoder_batch_size]
            
            # This is where the actual text encoding logic from your original script lives
            # It should be a stateless function call.
            text_embeddings, pooled_embeds = encode_prompts_xl(batch_prompts)
            
            # Create one workload dictionary for each item in the batch
            for j in range(len(batch_prompts)):
                add_time_ids = get_add_time_ids(image_size[0], image_size[1], False, dtype=torch.float32)
                
                workload.append({
                    "encoder_hidden_states": text_embeddings[j:j+1].cpu(),
                    "added_cond_kwargs": {
                        "text_embeds": pooled_embeds[j:j+1].cpu(),
                        "time_ids": add_time_ids.cpu(),
                    }
                })

        print(f"--- [ConditioningProvider] Materialization complete. Produced {len(workload)} workload items. ---")
        return workload