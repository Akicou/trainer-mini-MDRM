#!/usr/bin/env python3
"""
Finetuning script for Nayhein-8.8B-MDRM

Usage:
    python finetune.py --data-path ./data/train.json
    python finetune.py --generate-synthetic-data --provider lmstudio --num-samples 100
    python finetune.py --generate-synthetic-data --provider ollama --model llama3.2
"""

import argparse
import json
import sys
from pathlib import Path
from datasets import Dataset

# Ensure repository root is on PYTHONPATH so 'src' can be imported
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.config import SyntheticDataConfig, TrainingConfig, ModelConfig
from src.data_generator import SyntheticDataGenerator
from src.model import ReasoningModel


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune Nayhein-8.8B-MDRM")
    
    # Data arguments
    parser.add_argument("--data-path", type=str, default=None, 
                        help="Path to training data JSON file")
    parser.add_argument("--generate-synthetic-data", action="store_true",
                        help="Generate synthetic training data")
    parser.add_argument("--provider", type=str, default="lmstudio",
                        choices=["lmstudio", "ollama"],
                        help="Provider for synthetic data generation")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Model name for data generation")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of synthetic samples to generate")
    parser.add_argument("--output-data-path", type=str, default="./data/synthetic_train.json",
                        help="Output path for synthetic data")
    
    # Training arguments
    parser.add_argument("--output-dir", type=str, default="./checkpoints/nayhein-8b",
                        help="Output directory for fine-tuned model")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Maximum sequence length")
    
    # LoRA arguments
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha")
    
    # Model arguments
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3.5-0.8B-Base",
                        help="Base model to fine-tune")
    
    return parser.parse_args()


def generate_synthetic_data(args):
    """Generate synthetic training data"""
    config = SyntheticDataConfig(
        num_samples=args.num_samples,
        provider=args.provider,
        model_name=args.model_name
    )
    
    generator = SyntheticDataGenerator(config)
    samples = generator.generate_dataset()
    
    # Save data
    output_path = Path(args.output_data_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generator.save_dataset(samples, str(output_path))
    
    print(f"Generated {len(samples)} samples")
    return str(output_path)


def load_dataset(data_path: str) -> Dataset:
    """Load training dataset from JSON"""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Convert to formatted text
    formatted_data = [item["formatted"] if "formatted" in item else item["text"] for item in data]
    
    return Dataset.from_dict({"text": formatted_data})


def finetune(args):
    """Fine-tune the model"""
    # Load or generate data
    if args.generate_synthetic_data:
        data_path = generate_synthetic_data(args)
    elif args.data_path:
        data_path = args.data_path
    else:
        raise ValueError("Must provide --data-path or --generate-synthetic-data")
    
    # Load dataset
    print(f"Loading dataset from {data_path}")
    train_dataset = load_dataset(data_path)
    print(f"Loaded {len(train_dataset)} samples")
    
    # Configure training
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    )
    
    # Initialize model
    print(f"Loading base model: {args.base_model}")
    model = ReasoningModel(
        model_path=args.base_model,
        training_config=training_config,
        use_lora=True
    )
    
    # Fine-tune
    print("Starting fine-tuning...")
    model.train(
        train_dataset=train_dataset,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs
    )
    
    print(f"Fine-tuning complete! Model saved to {args.output_dir}")


def main():
    args = parse_args()
    
    # Create output directories
    Path(args.output_dir).parent.mkdir(parents=True, exist_ok=True)
    
    if args.generate_synthetic_data:
        # Only generate data
        data_path = generate_synthetic_data(args)
        print(f"Data saved to {data_path}")
        print("Run again with --data-path to fine-tune")
    else:
        # Fine-tune
        finetune(args)


if __name__ == "__main__":
    main()
