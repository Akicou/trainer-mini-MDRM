"""Training script for dual-mode generation model

This script trains the unified model on both:
1. Auto-regressive reasoning (standard causal LM loss)
2. Diffusion-style output (masked token prediction loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import Dataset, DataLoader
from trl import SFTTrainer, SFTConfig

from src.unified_model import DualModeGenerationModel
from src.config import ModelConfig, DualModeConfig, TrainingConfig

# Try to import wandb, make it optional
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not installed. Install with: pip install wandb")


@dataclass
class DualModeTrainingConfig:
    """Training configuration for dual-mode model"""
    # Standard training settings
    batch_size: int = 2  # Reduced from 4 to save memory
    learning_rate: float = 2e-5
    num_epochs: int = 3
    max_steps: Optional[int] = None
    max_length: int = 512  # Reduced from 2048 to save memory (1/4 VRAM usage)
    gradient_accumulation_steps: int = 8  # Increased to maintain effective batch size
    warmup_ratio: float = 0.1
    warmup_steps: int = 20

    # Loss weighting
    ar_loss_weight: float = 1.0  # Weight for auto-regressive loss
    diffusion_loss_weight: float = 1.0  # Weight for diffusion loss

    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    # Training modes
    use_unsloth: bool = True  # Use Unsloth for memory efficiency

    # WandB logging
    use_wandb: bool = False
    wandb_project: str = "dual-mode-model"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_log_interval: int = 10  # Log every N steps
    wandb_save_model: bool = True  # Save model to wandb


class DualModeDataset(Dataset):
    """Dataset for dual-mode training

    Each item should contain:
    - prompt: The input prompt
    - reasoning: The expected reasoning output (enclosed in  tags)
    - output: The expected final output (enclosed in <output> tags)
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: AutoTokenizer,
        max_length: int = 512  # Reduced from 2048 to save memory (1/4 VRAM)
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Format the full sequence
        # Format: prompt  reasoning </think> <output> output </output>
        prompt = item.get("prompt", "")
        reasoning = item.get("reasoning", "")
        output = item.get("output", "")

        # Construct full text with tags
        full_text = f"{prompt} {reasoning} </think> <output> {output} </output>"

        # Tokenize
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "labels": encodings["input_ids"].squeeze(0).clone(),
            "prompt": prompt,
            "reasoning": reasoning,
            "output": output
        }


class DualModeTrainer:
    """Trainer for dual-mode generation model with optional WandB logging"""

    def __init__(
        self,
        model: DualModeGenerationModel,
        config: DualModeTrainingConfig,
        tokenizer: AutoTokenizer
    ):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device = model.device

        # Initialize WandB if enabled
        self.wandb_run = None
        if config.use_wandb and HAS_WANDB:
            self._init_wandb()

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )

        # Setup scheduler
        total_steps = config.max_steps if config.max_steps else 1000
        warmup_steps = int(total_steps * config.warmup_ratio)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=1e-6
        )

        # Training state
        self.global_step = 0
        self.epoch = 0

    def _init_wandb(self):
        """Initialize WandB for experiment tracking"""
        run_name = self.config.wandb_run_name or f"dual-mode-{int(time.time())}"

        self.wandb_run = wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            name=run_name,
            config={
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "num_epochs": self.config.num_epochs,
                "max_steps": self.config.max_steps,
                "max_length": self.config.max_length,
                "ar_loss_weight": self.config.ar_loss_weight,
                "diffusion_loss_weight": self.config.diffusion_loss_weight,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            },
            resume="allow"
        )
        print(f"WandB initialized: {self.wandb_run.url}")

    def _log_to_wandb(self, metrics: Dict[str, float], step: int):
        """Log metrics to WandB"""
        if self.config.use_wandb and self.wandb_run is not None:
            wandb.log(metrics, step=step)

    def _finish_wandb(self):
        """Finish WandB run"""
        if self.config.use_wandb and self.wandb_run is not None:
            wandb.finish()
            print("WandB run finished")

    def compute_ar_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        reasoning_end_pos: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute auto-regressive causal language modeling loss

        This is the standard cross-entropy loss for next-token prediction.
        """
        # Forward pass through backbone
        outputs = self.model.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        # Get logits
        logits = outputs.logits  # [batch, seq_len, vocab_size]

        # Shift for causal LM: predict next token
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Compute loss
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        ar_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        return ar_loss

    def compute_diffusion_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        reasoning_end_pos: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute diffusion-style masked token prediction loss

        This uses BERT/T5-style masked language modeling.
        """
        batch_size, seq_len = input_ids.shape

        # Get output section (after <output> tag)
        output_start_id = self.model.output_start_id
        if output_start_id is None:
            return torch.tensor(0.0, device=self.device)

        # Find positions of <output> tags
        output_starts = []
        output_ends = []
        for i in range(batch_size):
            ids = input_ids[i].tolist()
            if output_start_id in ids:
                start = ids.index(output_start_id) + 1  # After <output>
                output_starts.append(start)
                # Find end or use sequence end
                if self.model.output_end_id in ids[start:]:
                    end = ids.index(self.model.output_end_id, start)
                else:
                    end = seq_len
                output_ends.append(end)
            else:
                output_starts.append(seq_len)
                output_ends.append(seq_len)

        # If no output sections, return zero loss
        if all(s >= seq_len for s in output_starts):
            return torch.tensor(0.0, device=self.device)

        # Randomly mask tokens in output sections (15% like BERT)
        masked_input_ids = input_ids.clone()
        mask_token_id = self.model.diffusion_head.mask_token_id

        labels_for_diffusion = torch.full_like(input_ids, -100)

        for i in range(batch_size):
            start, end = output_starts[i], output_ends[i]
            if start < end:
                output_tokens = input_ids[i, start:end]
                num_to_mask = max(1, int(0.15 * (end - start)))

                # Randomly select positions to mask
                mask_positions = torch.randperm(end - start)[:num_to_mask] + start
                masked_input_ids[i, mask_positions] = mask_token_id
                labels_for_diffusion[i, mask_positions] = input_ids[i, mask_positions]

        # Forward pass with masked input
        outputs = self.model.backbone(
            input_ids=masked_input_ids,
            attention_mask=attention_mask
        )

        # Get logits from diffusion head
        logits = self.model.diffusion_head.masked_lm_head(
            outputs.last_hidden_state
        )  # [batch, seq_len, vocab_size+1]

        # Compute loss only on masked positions
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        diffusion_loss = loss_fct(
            logits.view(-1, logits.size(-1)),
            labels_for_diffusion.view(-1)
        )

        return diffusion_loss

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with memory optimization"""
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Set model to training mode
        self.model.train()

        # Compute both losses (use gradient checkpointing implicitly via torch)
        with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):  # Mixed precision
            ar_loss = self.compute_ar_loss(input_ids, attention_mask, labels)
            # Clear cache before second forward pass
            torch.cuda.empty_cache()
            diffusion_loss = self.compute_diffusion_loss(input_ids, attention_mask, labels)

            # Combined loss
            total_loss = (
                self.config.ar_loss_weight * ar_loss +
                self.config.diffusion_loss_weight * diffusion_loss
            )

        # Backward pass
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        return {
            "total_loss": total_loss.item(),
            "ar_loss": ar_loss.item(),
            "diffusion_loss": diffusion_loss.item(),
            "learning_rate": self.scheduler.get_last_lr()[0]
        }

    def train(
        self,
        train_dataset: DualModeDataset,
        output_dir: str = "./checkpoints/dual_mode_model",
        num_epochs: Optional[int] = None,
        max_steps: Optional[int] = None
    ):
        """Full training loop with WandB logging and gradient accumulation"""
        num_epochs = num_epochs or self.config.num_epochs
        max_steps = max_steps or self.config.max_steps

        # Create dataloader
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )

        self.global_step = 0
        self.accumulation_step = 0
        start_time = time.time()

        for epoch in range(num_epochs):
            self.epoch = epoch
            print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

            epoch_start_time = time.time()

            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Training step with gradient accumulation
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                self.model.train()

                # Forward pass with mixed precision
                with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                    ar_loss = self.compute_ar_loss(input_ids, attention_mask, labels)
                    torch.cuda.empty_cache()  # Clear cache between passes
                    diffusion_loss = self.compute_diffusion_loss(input_ids, attention_mask, labels)

                    # Scale loss for gradient accumulation
                    total_loss = (
                        self.config.ar_loss_weight * ar_loss +
                        self.config.diffusion_loss_weight * diffusion_loss
                    ) / self.config.gradient_accumulation_steps

                # Backward pass (accumulate gradients)
                total_loss.backward()

                self.accumulation_step += 1

                # Only step optimizer after accumulating enough gradients
                if self.accumulation_step % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    # Optimizer step
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    # Update global step
                    self.global_step += 1

                    # Console logging
                    if self.global_step % 10 == 0:
                        elapsed = time.time() - start_time
                        steps_per_sec = self.global_step / elapsed if elapsed > 0 else 0
                        print(f"Step {self.global_step}: "
                              f"loss={total_loss.item() * self.config.gradient_accumulation_steps:.4f}, "
                              f"ar_loss={ar_loss.item():.4f}, "
                              f"diffusion_loss={diffusion_loss.item():.4f}, "
                              f"lr={self.scheduler.get_last_lr()[0]:.2e} "
                              f"({steps_per_sec:.2f} steps/s)")

                    # WandB logging
                    if self.global_step % self.config.wandb_log_interval == 0:
                        log_metrics = {
                            "train/total_loss": total_loss.item() * self.config.gradient_accumulation_steps,
                            "train/ar_loss": ar_loss.item(),
                            "train/diffusion_loss": diffusion_loss.item(),
                            "train/learning_rate": self.scheduler.get_last_lr()[0],
                            "train/epoch": epoch,
                            "train/step": self.global_step,
                            "train/samples_per_second": self.config.batch_size * self.config.gradient_accumulation_steps / (time.time() - start_time + 1e-6) * self.global_step,
                        }
                        self._log_to_wandb(log_metrics, self.global_step)

                # Check max steps
                if max_steps and self.global_step >= max_steps:
                    print(f"\nReached max_steps ({max_steps}). Stopping training.")
                    break

            # Epoch-level logging
            epoch_time = time.time() - epoch_start_time
            epoch_metrics = {
                "epoch": epoch,
                "epoch/time": epoch_time,
            }
            self._log_to_wandb(epoch_metrics, self.global_step)
            print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")

            if max_steps and self.global_step >= max_steps:
                break

        # Final training stats
        total_time = time.time() - start_time
        final_metrics = {
            "train/total_time": total_time,
            "train/final_step": self.global_step,
        }
        self._log_to_wandb(final_metrics, self.global_step)
        print(f"\nTraining completed in {total_time:.2f}s ({self.global_step} steps)")

        # Save model
        print(f"\nSaving model to {output_dir}")
        self.save_model(output_dir)

        # Save to WandB if enabled
        if self.config.use_wandb and self.config.wandb_save_model:
            self._save_to_wandb(output_dir)

        # Finish WandB run
        self._finish_wandb()

    def _save_to_wandb(self, output_dir: str):
        """Save model checkpoint to WandB"""
        if not HAS_WANDB or self.wandb_run is None:
            return

        try:
            artifact = wandb.Artifact(
                name=f"dual-mode-model-epoch-{self.epoch}",
                type="model"
            )
            artifact.add_dir(output_dir)
            wandb.log_artifact(artifact)
            print(f"Model saved to WandB as artifact")
        except Exception as e:
            print(f"Warning: Failed to save model to WandB: {e}")

    def save_model(self, output_dir: str):
        """Save model checkpoint"""
        self.model.backbone.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Save AR head
        torch.save(
            self.model.ar_head.state_dict(),
            f"{output_dir}/ar_head.pt"
        )

        # Save diffusion head
        torch.save(
            self.model.diffusion_head.state_dict(),
            f"{output_dir}/diffusion_head.pt"
        )

        print(f"Model saved to {output_dir}")

    def load_model(self, checkpoint_dir: str):
        """Load model checkpoint"""
        # Load backbone
        self.model.backbone = AutoModelForCausalLM.from_pretrained(checkpoint_dir)

        # Load AR head
        self.model.ar_head.load_state_dict(
            torch.load(f"{checkpoint_dir}/ar_head.pt")
        )

        # Load diffusion head
        self.model.diffusion_head.load_state_dict(
            torch.load(f"{checkpoint_dir}/diffusion_head.pt")
        )

        print(f"Model loaded from {checkpoint_dir}")


def train_dual_mode(
    train_data: List[Dict[str, str]],
    reasoning_model_path: str = "Qwen/Qwen3.5-0.8B-Base",
    diffusion_model_path: str = "GSAI-ML/LLaDA-V",
    output_dir: str = "./checkpoints/dual_mode_model",
    config: Optional[DualModeTrainingConfig] = None,
    dual_mode_config: Optional[DualModeConfig] = None,
    model_config: Optional[ModelConfig] = None,
    use_wandb: bool = False,
    wandb_project: str = "dual-mode-model",
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None
) -> DualModeGenerationModel:
    """
    Main training function for dual-mode model

    Args:
        train_data: List of training examples with "prompt", "reasoning", "output"
        reasoning_model_path: Path to reasoning model for backbone
        diffusion_model_path: Path to diffusion model for weight extraction
        output_dir: Directory to save checkpoints
        config: Training configuration
        dual_mode_config: Dual-mode model configuration
        model_config: Base model configuration
        use_wandb: Enable WandB logging
        wandb_project: WandB project name
        wandb_entity: WandB entity (username or team)
        wandb_run_name: Custom WandB run name

    Returns:
        Trained dual-mode model
    """
    config = config or DualModeTrainingConfig()

    # Override wandb settings if provided
    if use_wandb:
        config.use_wandb = True
        config.wandb_project = wandb_project
        config.wandb_entity = wandb_entity
        config.wandb_run_name = wandb_run_name

    dual_mode_config = dual_mode_config or DualModeConfig()
    model_config = model_config or ModelConfig()

    print("Initializing dual-mode model...")
    model = DualModeGenerationModel(
        reasoning_model_path=reasoning_model_path,
        diffusion_model_path=diffusion_model_path,
        config=dual_mode_config,
        model_config=model_config
    )

    print("Creating dataset...")
    dataset = DualModeDataset(
        data=train_data,
        tokenizer=model.tokenizer,
        max_length=config.max_length
    )

    print("Initializing trainer...")
    if config.use_wandb:
        print(f"WandB enabled: project={config.wandb_project}")
        if not HAS_WANDB:
            print("Warning: wandb requested but not installed. Install with: pip install wandb")
            config.use_wandb = False

    trainer = DualModeTrainer(
        model=model,
        config=config,
        tokenizer=model.tokenizer
    )

    print("Starting training...")
    trainer.train(
        train_dataset=dataset,
        output_dir=output_dir,
        num_epochs=config.num_epochs,
        max_steps=config.max_steps
    )

    print("Training complete!")
    return model


# Example usage
if __name__ == "__main__":
    # Example training data
    example_data = [
        {
            "prompt": "What is the capital of France?",
            "reasoning": "The capital of France is Paris. It is located in the north-central part of the country.",
            "output": "The capital of France is Paris."
        },
        {
            "prompt": "Explain photosynthesis.",
            "reasoning": "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen. This process occurs in chloroplasts.",
            "output": "Photosynthesis is the biological process where plants use sunlight to convert water and carbon dioxide into glucose and oxygen, producing energy for growth."
        }
    ]

    # Train the model
    model = train_dual_mode(
        train_data=example_data,
        output_dir="./checkpoints/dual_mode_example"
    )
