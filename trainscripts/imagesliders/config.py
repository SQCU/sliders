# Written by Gemini 2.5, under review

from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_resolution: tuple[int, int] = (512, 512)
    batch_size: int = 4
    num_epochs: int = 10
    learning_rate: float = 1e-4
