#!/usr/bin/env python3
"""
Nayhein-8.8B-MDRM CLI Interface

Usage:
    # Hybrid mode (original)
    python main.py "Explain machine learning"
    python main.py "Build a REST API" --show-reasoning
    python main.py --interactive

    # Dual-mode unified model
    python main.py dual-mode "Explain quantum computing" --show-steps
    python main.py train-dual --dataset nvidia/Nemotron-Cascade-SFT-Stage-1 --output ./checkpoints/dual-mode

    # Post-training for quality
    python main.py post-train --checkpoint ./checkpoints/dual-mode --dataset nvidia/Nemotron-Cascade-SFT-Stage-1
"""

import argparse
import sys
from typing import Optional

# Hybrid model imports (original)
from src.hybrid_model import HybridReasoningDiffusionModel
from src.auto_model_vision import NayheinForVisionText2Text
from src.config import ModelConfig, TrainingConfig

# Dual-mode imports (new)
from src.unified_model import DualModeGenerationModel, DualModeOutput
from src.train_dual_mode import train_dual_mode, DualModeDataset, DualModeTrainingConfig
from src.config import DualModeConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Nayhein-8.8B-MDRM - Hybrid and Dual-Mode Models",
        epilog="For more information, see README.md",
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dual-mode",
        action="store_true",
        help="Use dual-mode unified model (default: hybrid)",
    )
    mode_group.add_argument(
        "--train-dual", action="store_true", help="Train dual-mode model"
    )
    mode_group.add_argument(
        "--post-train",
        action="store_true",
        help="Post-train dual-mode model on quality dataset",
    )

    # Input prompt (for generation modes)
    parser.add_argument("prompt", nargs="?", default=None, help="Input prompt")

    # Generation options
    parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode"
    )
    parser.add_argument(
        "--show-reasoning", action="store_true", help="Show reasoning output"
    )
    parser.add_argument(
        "--show-diffusion-steps",
        action="store_true",
        help="Show diffusion unmasking steps",
    )
    parser.add_argument(
        "--show-steps",
        action="store_true",
        help="Show both AR and diffusion generation steps (dual-mode)",
    )

    # Model paths
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3.5-0.8B-Base",
        help="Base model path (for reasoning/backbone)",
    )
    parser.add_argument(
        "--diffusion-model",
        type=str,
        default="GSAI-ML/LLaDA-V",
        help="Diffusion model path",
    )
    parser.add_argument(
        "--load-from", type=str, default=None, help="Load model from checkpoint"
    )

    # Generation parameters
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature for generation"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1024, help="Maximum output tokens"
    )
    parser.add_argument(
        "--ar-max-tokens",
        type=int,
        default=1024,
        help="Max tokens for AR reasoning (dual-mode)",
    )
    parser.add_argument(
        "--diffusion-max-tokens",
        type=int,
        default=512,
        help="Max tokens for diffusion output (dual-mode)",
    )

    # Training options
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset to train on (e.g., nvidia/Nemotron-Cascade-SFT-Stage-1)",
    )
    parser.add_argument(
        "--dataset-split", type=str, default="train", help="Dataset split to use"
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="general",
        help="Dataset config name (for multi-config datasets like Nemotron)",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=10000,
        help="Number of samples to use from dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./checkpoints/dual-mode",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Maximum training steps"
    )

    # WandB options
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="dual-mode-model",
        help="WandB project name",
    )
    parser.add_argument(
        "--wandb-entity", type=str, default=None, help="WandB entity (username or team)"
    )
    parser.add_argument(
        "--wandb-run-name", type=str, default=None, help="Custom WandB run name"
    )

    return parser.parse_args()


def load_huggingface_dataset(
    dataset_name: str,
    split: str = "train",
    config: Optional[str] = None,
    num_samples: int = 10000,
):
    """Load dataset from HuggingFace and format for dual-mode training"""
    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "Error: datasets library not installed. Install with: pip install datasets"
        )
        sys.exit(1)

    print(f"Loading dataset: {dataset_name}/{config if config else split} ({split})")
    print(f"Using {num_samples} samples...")

    # Load dataset (with config if specified)
    if config:
        dataset = load_dataset(dataset_name, config, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)

    # Take subset if needed
    if num_samples < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(num_samples))

    # Format for dual-mode training
    formatted_data = []
    for item in dataset:
        # Try to extract prompt, reasoning, and output
        # This depends on the dataset structure
        if "prompt" in item and "completion" in item:
            formatted_data.append(
                {
                    "prompt": item["prompt"],
                    "reasoning": "",  # Will be generated by model
                    "output": item["completion"],
                }
            )
        elif "input" in item and "output" in item:
            formatted_data.append(
                {"prompt": item["input"], "reasoning": "", "output": item["output"]}
            )
        elif "messages" in item:
            # Chat format (ShareGPT)
            # The assistant message already contains <thinking> tags with reasoning
            messages = item["messages"]
            if len(messages) >= 2:
                # Find the user and assistant messages
                user_msg = None
                assistant_msg = None
                for msg in messages:
                    if msg["role"] == "user":
                        user_msg = msg["content"]
                    elif msg["role"] == "assistant":
                        assistant_msg = msg["content"]
                        break  # Take the first assistant response

                if user_msg is not None and assistant_msg is not None:
                    # Parse assistant_msg to extract reasoning (inside tags) and output (after tags)
                    import re
                    # Match <think>...</think> pattern
                    match = re.search(r"<think>(.*?)</think>", assistant_msg, re.DOTALL)
                    if match:
                        reasoning = match.group(1).strip()
                        output = assistant_msg[match.end():].strip()
                    else:
                        # Fallback if no tags found
                        reasoning = assistant_msg
                        output = ""
                    
                    formatted_data.append(
                        {
                            "prompt": user_msg,
                            "reasoning": reasoning,
                            "output": output,
                        }
                    )
        else:
            # Fallback: use text field
            text = item.get("text", "")
            if text:
                # Simple split on first newline or use whole text as output
                parts = text.split("\n", 1)
                formatted_data.append(
                    {
                        "prompt": "",
                        "reasoning": "",
                        "output": parts[0] if len(parts) == 1 else parts[1],
                    }
                )

    print(f"Formatted {len(formatted_data)} examples for training")
    return formatted_data


def run_dual_mode_generation(args):
    """Run dual-mode unified model generation"""
    import time

    print("Loading dual-mode unified model...")

    # Initialize dual-mode model
    dual_config = DualModeConfig(
        ar_max_tokens=args.ar_max_tokens, diffusion_max_tokens=args.diffusion_max_tokens
    )

    if args.load_from:
        # Load from trained checkpoint
        print(f"Loading from checkpoint: {args.load_from}")
        model = DualModeGenerationModel(
            reasoning_model_path=args.load_from,
            diffusion_model_path=args.diffusion_model,
            config=dual_config,
        )
        # Load trained heads
        import torch
        model.ar_head.load_state_dict(
            torch.load(f"{args.load_from}/ar_head.pt", map_location=model.device)
        )
        model.diffusion_head.load_state_dict(
            torch.load(f"{args.load_from}/diffusion_head.pt", map_location=model.device)
        )
        print("Loaded trained heads from checkpoint!")
    else:
        model = DualModeGenerationModel(
            reasoning_model_path=args.model,
            diffusion_model_path=args.diffusion_model,
            config=dual_config,
        )

    print("Model loaded successfully!")

    # Interactive mode
    if args.interactive:
        print("\nEnter your prompt (type 'quit' to exit):")
        while True:
            try:
                prompt = input("\nYou: ").strip()
                if prompt.lower() == "quit":
                    break

                start_time = time.time()
                output = model.generate(
                    prompt=prompt,
                    max_reasoning_tokens=args.ar_max_tokens,
                    max_output_tokens=args.diffusion_max_tokens,
                    temperature=args.temperature,
                    show_steps=args.show_steps,
                )
                elapsed = time.time() - start_time

                print(
                    f"\n{model.config.output_tag_start}{output.output}{model.config.output_tag_end}"
                )

                if args.show_reasoning:
                    print(f"\nReasoning:")
                    print(
                        f"{model.config.reasoning_tag_start}{output.reasoning}{model.config.reasoning_tag_end}"
                    )

                if args.show_steps:
                    print(f"\n=== Generation Stats ===")
                    print(f"Total time: {elapsed:.2f}s")
                    print(f"AR tokens: {len(output.reasoning_tokens)}")
                    print(f"Output tokens: {len(output.output_tokens)}")

            except KeyboardInterrupt:
                break
        return

    # Single prompt mode
    if not args.prompt:
        print("Error: No prompt provided. Use --interactive for interactive mode.")
        return

    start_time = time.time()
    output = model.generate(
        prompt=args.prompt,
        max_reasoning_tokens=args.ar_max_tokens,
        max_output_tokens=args.diffusion_max_tokens,
        temperature=args.temperature,
        show_steps=args.show_steps,
    )
    elapsed = time.time() - start_time

    print(
        f"\n{model.config.output_tag_start}{output.output}{model.config.output_tag_end}"
    )

    if args.show_reasoning:
        print(f"\nReasoning:")
        print(
            f"{model.config.reasoning_tag_start}{output.reasoning}{model.config.reasoning_tag_end}"
        )

    if args.show_steps:
        print(f"\n=== Generation Stats ===")
        print(f"Total time: {elapsed:.2f}s")
        print(f"AR tokens: {len(output.reasoning_tokens)}")
        print(f"Output tokens: {len(output.output_tokens)}")


def run_dual_mode_training(args):
    """Train dual-mode model"""
    if not args.dataset:
        print("Error: --dataset is required for training")
        print("Example: --dataset nvidia/Nemotron-Cascade-SFT-Stage-1")
        sys.exit(1)

    # Load dataset
    train_data = load_huggingface_dataset(
        dataset_name=args.dataset,
        split=args.dataset_split,
        config=args.dataset_config,
        num_samples=args.dataset_size,
    )

    # Training config
    train_config = DualModeTrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        max_steps=args.max_steps,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
    )

    # Dual-mode config
    dual_config = DualModeConfig()

    print(f"\nTraining dual-mode model on {args.dataset}")
    print(f"Output directory: {args.output}")
    print(f"Samples: {len(train_data)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    if args.wandb:
        print(f"WandB: enabled (project={args.wandb_project})")

    # Train
    model = train_dual_mode(
        train_data=train_data,
        reasoning_model_path=args.model,
        diffusion_model_path=args.diffusion_model,
        output_dir=args.output,
        config=train_config,
        dual_mode_config=dual_config,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
    )

    print(f"\nTraining complete! Model saved to {args.output}")


def run_post_training(args):
    """Post-train dual-mode model on quality dataset"""
    if not args.dataset:
        print("Error: --dataset is required for post-training")
        print("Example: --dataset nvidia/Nemotron-Cascade-SFT-Stage-1")
        sys.exit(1)

    if not args.load_from:
        print("Error: --load-from is required for post-training")
        print("Specify the checkpoint to continue training from")
        sys.exit(1)

    # Load dataset
    train_data = load_huggingface_dataset(
        dataset_name=args.dataset,
        split=args.dataset_split,
        config=args.dataset_config,
        num_samples=args.dataset_size,
    )

    # Training config (use lower learning rate for post-training)
    train_config = DualModeTrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate * 0.5,  # Lower LR for fine-tuning
        num_epochs=args.epochs,
        max_steps=args.max_steps,
    )

    print(f"\nPost-training dual-mode model from {args.load_from}")
    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {args.output}")

    # Note: For true post-training, we'd need to modify train_dual_mode to support loading checkpoint
    print(
        "Note: Post-training feature requires checkpoint loading in train_dual_mode.py"
    )
    print("For now, you can train from scratch with the full dataset.")


def main():
    args = parse_args()

    # Dual-mode training
    if args.train_dual:
        run_dual_mode_training(args)
        return

    # Post-training
    if args.post_train:
        run_post_training(args)
        return

    # Dual-mode generation
    if args.dual_mode:
        run_dual_mode_generation(args)
        return

    # Original hybrid mode (default)
    print("Loading Nayhein-8.8B-MDRM (Hybrid Mode)...")
    try:
        if args.load_from:
            model = HybridReasoningDiffusionModel(
                reasoning_model_path=args.load_from,
                diffusion_model_path=args.diffusion_model,
            )
            model.load(args.load_from)
        else:
            model = HybridReasoningDiffusionModel(
                reasoning_model_path=args.model,
                diffusion_model_path=args.diffusion_model,
            )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print(
            "Make sure the model is downloaded or use --load-from with fine-tuned checkpoint"
        )
        return

    # Interactive mode
    if args.interactive:
        print("\nEnter your prompt (type 'quit' to exit):")
        while True:
            try:
                prompt = input("\nYou: ").strip()
                if prompt.lower() == "quit":
                    break

                output = model.generate(
                    prompt=prompt,
                    max_reasoning_tokens=args.max_tokens,
                    temperature=args.temperature,
                    show_diffusion_steps=args.show_diffusion_steps,
                )

                print(
                    f"\n{model.config.output_tag_start}{output.final_output}{model.config.output_tag_end}"
                )

                if args.show_reasoning:
                    print(
                        f"\n{model.config.reasoning_tag_start}{output.reasoning}{model.config.reasoning_tag_end}"
                    )

                if args.show_diffusion_steps and output.diffusion_steps:
                    print(f"\n=== Diffusion Unmasking Steps ===")
                    for step in output.diffusion_steps:
                        print(step)

            except KeyboardInterrupt:
                break
        return

    # Single prompt mode
    if not args.prompt:
        from argparse import Namespace

        # Re-create parser to show help
        parser = argparse.ArgumentParser()
        parser.parse_args([])
        parser.print_help()
        return

    output = model.generate(
        prompt=args.prompt,
        max_reasoning_tokens=args.max_tokens,
        temperature=args.temperature,
        show_diffusion_steps=args.show_diffusion_steps,
    )

    print(
        f"\n{model.config.output_tag_start}{output.final_output}{model.config.output_tag_end}"
    )

    if args.show_reasoning:
        print(f"\nReasoning:")
        print(
            f"{model.config.reasoning_tag_start}{output.reasoning}{model.config.reasoning_tag_end}"
        )

    if args.show_diffusion_steps and output.diffusion_steps:
        print(f"\n=== Diffusion Unmasking Steps ===")
        for step in output.diffusion_steps:
            print(step)


if __name__ == "__main__":
    main()
