"""LLaDA-V Diffusion Model integration"""

import torch
from typing import Optional, Tuple
from transformers import AutoModelForCausalLM, AutoProcessor
from src.config import ModelConfig


class DiffusionOutputModel:
    """LLaDA-V Diffusion model for final output generation"""

    def __init__(
        self,
        model_path: str = "GSAI-ML/LLaDA-V",
        config: Optional[ModelConfig] = None
    ):
        self.config = config or ModelConfig()
        self.model_path = model_path

        # Load diffusion model - using AutoModelVision compatible approach
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
            
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else "cpu"
            )
            print(f"Diffusion model loaded on: {device}")
        except Exception as e:
            print(f"Warning: Could not load {model_path}: {e}")
            print("Using fallback mode - reasoning output will be used directly")
            self.model = None
            self.processor = None

        self.model.eval() if self.model else None

    def generate_from_reasoning(
        self,
        reasoning: str,
        original_prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        show_steps: bool = False,
        steps_to_show: int = 5
    ) -> tuple[Optional[str], list[str]]:
        """Generate final output based on reasoning with optional diffusion steps"""
        if not self.model:
            return None, []

        # Format input with reasoning as context
        formatted_input = f"""Original Prompt: {original_prompt}

Reasoning: {reasoning}

Final Output: {self.config.output_tag_start}"""

        inputs = self.processor(
            formatted_input,
            return_tensors="pt"
        ).to(self.model.device)

        diffusion_steps = []
        
        if show_steps:
            # Generate step by step to show diffusion process
            with torch.no_grad():
                current_ids = inputs['input_ids']
                step_interval = max(1, max_new_tokens // steps_to_show)
                
                for i in range(steps_to_show):
                    # Generate a chunk of tokens
                    chunk_outputs = self.model.generate(
                        current_ids,
                        max_new_tokens=step_interval,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                        min_new_tokens=1
                    )
                    
                    # Decode this step
                    new_text = self.processor.decode(
                        chunk_outputs[0][current_ids.shape[1]:],
                        skip_special_tokens=True
                    )
                    
                    if new_text.strip():
                        diffusion_steps.append(f"Step {i+1}: {new_text.strip()}")
                    
                    current_ids = chunk_outputs
        
        # Full generation for final output
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id
            )

        generated_text = self.processor.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        # Ensure output tags are present
        if self.config.output_tag_end not in generated_text:
            generated_text += self.config.output_tag_end

        return generated_text, diffusion_steps

    def save(self, output_dir: str):
        """Save diffusion model"""
        if self.model:
            self.model.save_pretrained(output_dir)
            self.processor.save_pretrained(output_dir)
            print(f"Diffusion model saved to {output_dir}")

    def load(self, model_path: str):
        """Load diffusion model"""
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print(f"Diffusion model loaded from {model_path}")
