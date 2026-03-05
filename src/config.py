"""Configuration for Nayhein-8.8B-MDRM"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Base model configuration"""
    reasoning_model: str = "Qwen/Qwen3.5-0.8B-Base"
    diffusion_model: str = "GSAI-ML/LLaDA-V"
    use_reasoning_tags: bool = True
    reasoning_tag_start: str = "<think>"
    reasoning_tag_end: str = "</think>"
    output_tag_start: str = "<output>"
    output_tag_end: str = "</output>"


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 4
    learning_rate: float = 2e-5
    num_epochs: int = 3
    max_steps: Optional[int] = None
    max_length: int = 2048
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    warmup_steps: int = 20
    use_unsloth_low_memory: bool = True


@dataclass
class SyntheticDataConfig:
    """Synthetic data generation configuration"""
    num_samples: int = 100
    provider: str = "lmstudio"  # "lmstudio", "ollama", or "local"
    model_name: Optional[str] = None
    host: str = "http://localhost:1234"  # lmstudio default
    ollama_host: str = "http://localhost:11434"


@dataclass
class DualModeConfig:
    """Dual-mode generation configuration for unified model

    This configures the behavior of the DualModeGenerationModel which has:
    - Auto-regressive mode for sequential reasoning generation
    - Diffusion-style mode for parallel/noise-based output generation
    """
    # Auto-regressive settings
    ar_temperature: float = 0.7
    ar_top_p: float = 0.9
    ar_max_tokens: int = 1024

    # Diffusion settings
    diffusion_steps: int = 10
    noise_schedule: str = "linear"  # "linear", "cosine" - for future continuous noise
    diffusion_max_tokens: int = 512

    # Mode switching tags
    # Format: prompt<reasoning>reasoning</reasoning>output (output directly after end tag)
    reasoning_start_tag: str = "<reasoning>"
    reasoning_end_tag: str = "</reasoning>"
    output_start_tag: str = ""  # Not used - output comes after </reasoning>
    output_end_tag: str = ""  # Not used
