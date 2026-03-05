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
            logits = self.lm_head(outputs.last_hidden_state[:, -1:, :])

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
        mask_token_id: Optional[int] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_steps = num_steps
        self.use_continuous_noise = use_continuous_noise

        # Special mask token (like BERT's </think>)
        if mask_token_id is None:
            self.mask_token_id = vocab_size  # Use vocab_size as special