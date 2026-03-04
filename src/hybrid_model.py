"""Hybrid Auto-Regressive Reasoning to Diffusion Model"""

from typing import Optional, Tuple
from dataclasses import dataclass
from src.model import ReasoningModel
from src.diffusion_model import DiffusionOutputModel
from src.config import ModelConfig, TrainingConfig


@dataclass
class HybridOutput:
    """Complete output with reasoning and final response"""
    prompt: str
    reasoning: str
    final_output: str
    used_diffusion: bool
    diffusion_steps: list = None


class HybridReasoningDiffusionModel:
    """
    Nayhein-8.8B-MDRM: Hybrid Auto-Regressive Reasoning to Diffusion Model
    
    Combines:
    - Qwen3.5-0.8B-Base for reasoning in <think></think> tags
    - LLaDA-V for diffusion-based final output generation
    """

    def __init__(
        self,
        reasoning_model_path: str = "Qwen/Qwen3.5-0.8B-Base",
        diffusion_model_path: str = "GSAI-ML/LLaDA-V",
        config: Optional[ModelConfig] = None,
        training_config: Optional[TrainingConfig] = None
    ):
        self.config = config or ModelConfig()
        self.training_config = training_config or TrainingConfig()

        # Initialize reasoning model
        self.reasoning_model = ReasoningModel(
            model_path=reasoning_model_path,
            config=self.config,
            training_config=self.training_config
        )

        # Initialize diffusion model
        self.diffusion_model = DiffusionOutputModel(
            model_path=diffusion_model_path,
            config=self.config
        )

    def generate(
        self,
        prompt: str,
        max_reasoning_tokens: int = 1024,
        max_output_tokens: int = 512,
        temperature: float = 0.7,
        show_diffusion_steps: bool = False
    ) -> HybridOutput:
        """
        Generate complete response with reasoning and final output
        
        Process:
        1. Reasoning model generates thinking in <think></think> tags
        2. Diffusion model generates final output based on reasoning
        """
        # Step 1: Generate reasoning
        reasoning = self.reasoning_model.generate_reasoning(
            prompt=prompt,
            max_new_tokens=max_reasoning_tokens,
            temperature=temperature
        )

        # Extract reasoning content from tags
        if self.config.reasoning_tag_start in reasoning:
            think_start = reasoning.find(self.config.reasoning_tag_start) + len(self.config.reasoning_tag_start)
            think_end = reasoning.find(self.config.reasoning_tag_end)
            reasoning_content = reasoning[think_start:think_end].strip()
        else:
            reasoning_content = reasoning

        # Step 2: Generate final output using diffusion model
        diffusion_steps = []
        if self.diffusion_model.model:
            final_output, diffusion_steps = self.diffusion_model.generate_from_reasoning(
                reasoning=reasoning_content,
                original_prompt=prompt,
                max_new_tokens=max_output_tokens,
                temperature=temperature,
                show_steps=show_diffusion_steps
            )
            used_diffusion = True
        else:
            # Fallback: use reasoning model's direct output
            final_output = reasoning.split(self.config.reasoning_tag_end, 1)[-1].strip() if self.config.reasoning_tag_end in reasoning else reasoning
            used_diffusion = False

        return HybridOutput(
            prompt=prompt,
            reasoning=reasoning_content,
            final_output=final_output,
            used_diffusion=used_diffusion,
            diffusion_steps=diffusion_steps if show_diffusion_steps else None
        )

    def format_full_output(self, output: HybridOutput, show_diffusion: bool = False) -> str:
        """Format complete output with all components"""
        result = f"""{output.prompt}

{self.config.reasoning_tag_start}{output.reasoning}{self.config.reasoning_tag_end}

{self.config.output_tag_start}{output.final_output}{self.config.output_tag_end}"""
        
        if show_diffusion and output.diffusion_steps:
            result += "\n\n=== Diffusion Unmasking Steps ===\n"
            for step in output.diffusion_steps:
                result += f"{step}\n"
        
        return result

    def save(self, output_dir: str):
        """Save both models"""
        self.reasoning_model.save(f"{output_dir}/reasoning_model")
        if self.diffusion_model.model:
            self.diffusion_model.save(f"{output_dir}/diffusion_model")

    def load(self, reasoning_path: str, diffusion_path: Optional[str] = None):
        """Load both models"""
        self.reasoning_model.load(reasoning_path)
        if diffusion_path and self.diffusion_model.model:
            self.diffusion_model.load(diffusion_path)
