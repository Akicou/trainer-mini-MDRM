"""Dual-Mode Generation Model: Auto-Regressive Reasoning + Diffusion-Style Output

This module implements a unified model architecture that preserves both:
1. Auto-regressive sequential generation for reasoning ( tags)
2. Diffusion-style parallel/noise-based generation for final output (<output> tags)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from src.config import ModelConfig, DualModeConfig


class AutoRegressiveHead(nn.Module):
    """Auto-regressive generation head with KV caching for reasoning"""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        end_tag_id: Optional[int] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.end_tag_id = end_tag_id
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through AR head"""
        return self.lm_head(hidden_states)

    @torch.no_grad()
    def generate(
        self,
        backbone: nn.Module,
        input_ids: torch.Tensor,
        max_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """
        Generate tokens auto-regressively with KV caching

        Args:
            backbone: The shared transformer backbone
            input_ids: Input token IDs [batch_size, seq_len]
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            eos_token_id: End-of-sequence token ID (optional)

        Returns:
            generated_ids: Generated token IDs
            generation_info: List of generation info dictionaries
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        generated = []
        past_key_values = None
        generation_info = []

        for step in range(max_tokens):
            # Only process the last token if we have cached KV
            if past_key_values is not None:
                input_ids = input_ids[:, -1:]

            # Forward pass with optional KV cache
            outputs = backbone(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=True
            )

            # Get logits for next token
            # outputs.hidden_states is a tuple - use last element for final layer
            hidden_states = outputs.hidden_states[-1]  # Last layer
            logits = self.lm_head(hidden_states[:, -1:, :])

            # Apply temperature
            logits = logits / temperature

            # Apply top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated.append(next_token)

            # Update input for next iteration
            input_ids = next_token
            past_key_values = outputs.past_key_values

            # Record generation info
            generation_info.append({
                "step": step,
                "token_id": next_token.item(),
                "top_tokens": torch.topk(logits[0], k=5, dim=-1).indices[0].tolist()
            })

            # Check for EOS
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
            if self.end_tag_id is not None and next_token.item() == self.end_tag_id:
                break

        # Concatenate all generated tokens
        generated_ids = torch.cat(generated, dim=1)

        return generated_ids, generation_info


class DiffusionHead(nn.Module):
    """Diffusion-style generation head with token masking approach"""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_steps: int = 10,
        use_continuous_noise: bool = False,
        mask_token_id: Optional[int] = None,
        pad_token_id: int = 0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_steps = num_steps
        self.use_continuous_noise = use_continuous_noise

        # Special mask token - use pad_token_id (a valid token within vocab)
        # Default to 0 (usually padding token) if not specified
        self.mask_token_id = mask_token_id if mask_token_id is not None else pad_token_id

        if use_continuous_noise:
            raise NotImplementedError("Continuous noise not yet implemented")
        else:
            # Output vocab_size logits (not +1 since we use existing token)
            self.masked_lm_head = nn.Linear(hidden_size, vocab_size)

    def get_reveal_schedule(self, step: int, total_steps: int, num_tokens: int) -> torch.Tensor:
        reveal_ratio = 0.5 * (1 + torch.cos(torch.tensor(torch.pi * step / total_steps)))
        num_to_reveal = int(reveal_ratio * num_tokens)
        reveal_mask = torch.zeros(num_tokens, dtype=torch.bool)
        reveal_mask[:num_to_reveal] = True
        perm = torch.randperm(num_tokens)
        reveal_mask = reveal_mask[perm.argsort()]
        return reveal_mask

    @torch.no_grad()
    def generate(
        self,
        backbone: nn.Module,
        reasoning_hidden_states: torch.Tensor,
        prompt_ids: torch.Tensor,
        max_tokens: int,
        temperature: float = 0.7
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        batch_size = prompt_ids.shape[0]
        device = prompt_ids.device
        masked_ids = torch.full(
            (batch_size, max_tokens),
            self.mask_token_id,
            dtype=torch.long,
            device=device
        )
        generation_info = []
        for step in range(self.num_steps):
            full_ids = torch.cat([prompt_ids, masked_ids], dim=1)
            outputs = backbone(input_ids=full_ids, output_hidden_states=True)
            # Get last layer hidden states
            all_hidden = outputs.hidden_states[-1]
            prompt_len = prompt_ids.shape[1]
            masked_hidden = all_hidden[:, prompt_len:prompt_len + max_tokens, :]
            logits = self.masked_lm_head(masked_hidden)
            logits = logits / temperature
            reveal_mask = self.get_reveal_schedule(step, self.num_steps, max_tokens)
            reveal_mask = reveal_mask.to(device)
            probs = F.softmax(logits, dim=-1)
            predicted_tokens = torch.argmax(probs, dim=-1)
            for i in range(batch_size):
                masked_ids[i] = torch.where(reveal_mask, predicted_tokens[i], masked_ids[i])
            num_revealed = reveal_mask.sum().item()
            generation_info.append({
                "step": step,
                "num_revealed": num_revealed,
                "total_tokens": max_tokens,
                "reveal_ratio": num_revealed / max_tokens
            })
        return masked_ids, generation_info


@dataclass
class DualModeOutput:
    prompt: str
    reasoning: str
    reasoning_tokens: List[int]
    output: str
    output_tokens: List[int]
    ar_generation_info: List[Dict[str, Any]]
    diffusion_generation_info: List[Dict[str, Any]]
    total_time_seconds: float = 0.0


class DualModeGenerationModel(nn.Module):
    def __init__(
        self,
        reasoning_model_path: str = "Qwen/Qwen3.5-0.8B-Base",
        diffusion_model_path: str = "GSAI-ML/LLaDA-V",
        config: Optional[DualModeConfig] = None,
        model_config: Optional[ModelConfig] = None
    ):
        super().__init__()
        self.config = config or DualModeConfig()
        self.model_config = model_config or ModelConfig()
        use_cuda = torch.cuda.is_available()
        self.device = "cuda" if use_cuda else "cpu"
        dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16
        self.dtype = dtype
        print(f"Initializing DualModeGenerationModel on {self.device}")
        self.backbone = AutoModelForCausalLM.from_pretrained(
            reasoning_model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto" if use_cuda else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            reasoning_model_path,
            trust_remote_code=True,
            padding_side="right"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Store full token sequences for multi-token tags
        if self.config.reasoning_start_tag:
            self.reasoning_start_ids = self.tokenizer.encode(
                self.config.reasoning_start_tag, add_special_tokens=False
            )
        else:
            self.reasoning_start_ids = []
        
        if self.config.reasoning_end_tag:
            self.reasoning_end_ids = self.tokenizer.encode(
                self.config.reasoning_end_tag, add_special_tokens=False
            )
        else:
            self.reasoning_end_ids = []
        
        # For backward compatibility
        self.reasoning_start_id = self.reasoning_start_ids[0] if self.reasoning_start_ids else None
        self.reasoning_end_id = self.reasoning_end_ids[0] if self.reasoning_end_ids else None
        self.output_start_id = self.tokenizer.encode(
            self.config.output_start_tag, add_special_tokens=False
        )[0] if self.config.output_start_tag else None
        self.output_end_id = self.tokenizer.encode(
            self.config.output_end_tag, add_special_tokens=False
        )[0] if self.config.output_end_tag else None
        hidden_size = self.backbone.config.hidden_size
        vocab_size = self.backbone.config.vocab_size
        self.ar_head = AutoRegressiveHead(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            end_tag_id=self.reasoning_end_id
        )
        if hasattr(self.backbone, 'lm_head'):
            ar_state_dict = self.backbone.lm_head.state_dict()
            self.ar_head.lm_head.load_state_dict(ar_state_dict)
        # Move ar_head to correct device
        self.ar_head = self.ar_head.to(self.device)

        # Use pad_token_id as mask token (valid token within vocabulary)
        pad_token_id = self.tokenizer.pad_token_id or 0
        self.diffusion_head = DiffusionHead(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_steps=self.config.diffusion_steps,
            use_continuous_noise=False,
            pad_token_id=pad_token_id
        )
        # Move diffusion_head to correct device
        self.diffusion_head = self.diffusion_head.to(self.device)
        print("DualModeGenerationModel initialized")

    def generate(
        self,
        prompt: str,
        max_reasoning_tokens: int = 1024,
        max_output_tokens: int = 512,
        temperature: float = 0.7,
        show_steps: bool = False
    ) -> DualModeOutput:
        import time
        start_time = time.time()
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        reasoning_ids, ar_info = self.ar_head.generate(
            backbone=self.backbone,
            input_ids=input_ids,
            max_tokens=max_reasoning_tokens,
            temperature=temperature,
            eos_token_id=self.reasoning_end_id
        )
        reasoning_tokens = reasoning_ids[0].tolist()
        reasoning_text = self.tokenizer.decode(reasoning_tokens, skip_special_tokens=True)
        with torch.no_grad():
            reasoning_outputs = self.backbone(input_ids=reasoning_ids.to(self.device), output_hidden_states=True)
            reasoning_hidden = reasoning_outputs.hidden_states[-1]
        output_ids, diffusion_info = self.diffusion_head.generate(
            backbone=self.backbone,
            reasoning_hidden_states=reasoning_hidden,
            prompt_ids=reasoning_ids.to(self.device),
            max_tokens=max_output_tokens,
            temperature=temperature
        )
        output_tokens = output_ids[0].tolist()
        # Note: mask_token_id is now a valid token (pad_token), so don't filter
        # The model should naturally generate the correct tokens
        output_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
        elapsed = time.time() - start_time
        return DualModeOutput(
            prompt=prompt,
            reasoning=reasoning_text,
            reasoning_tokens=reasoning_tokens,
            output=output_text,
            output_tokens=output_tokens,
            ar_generation_info=ar_info if show_steps else [],
            diffusion_generation_info=diffusion_info if show_steps else [],
            total_time_seconds=elapsed
        )

    def save(self, output_dir: str):
        self.backbone.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        torch.save(self.ar_head.state_dict(), f"{output_dir}/ar_head.pt")
        torch.save(self.diffusion_head.state_dict(), f"{output_dir}/diffusion_head.pt")
        print(f"Model saved to {output_dir}")

    def load(self, checkpoint_dir: str):
        self.backbone = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
        self.ar_head.load_state_dict(torch.load(f"{checkpoint_dir}/ar_head.pt"))
        self.diffusion_head.load_state_dict(torch.load(f"{checkpoint_dir}/diffusion_head.pt"))
        print(f"Model loaded from {checkpoint_dir}")
