#!/usr/bin/env python3
"""
Nayhein-8.8B-MDRM CLI Interface

Usage:
    python main.py "Explain machine learning"
    python main.py "Build a REST API" --show-reasoning
    python main.py --interactive
"""

import argparse
import sys
from src.hybrid_model import HybridReasoningDiffusionModel
from src.auto_model_vision import NayheinForVisionText2Text
from src.config import ModelConfig, TrainingConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Nayhein-8.8B-MDRM Hybrid Model")
    
    parser.add_argument("prompt", nargs="?", default=None,
                        help="Input prompt")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--show-reasoning", action="store_true",
                        help="Show reasoning output")
    parser.add_argument("--show-diffusion-steps", action="store_true",
                        help="Show diffusion unmasking steps")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-0.8B-Base",
                        help="Base model path")
    parser.add_argument("--diffusion-model", type=str, default="GSAI-ML/LLaDA-V",
                        help="Diffusion model path")
    parser.add_argument("--load-from", type=str, default=None,
                        help="Load fine-tuned model from checkpoint")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for generation")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Maximum output tokens")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize model
    print("Loading Nayhein-8.8B-MDRM...")
    try:
        if args.load_from:
            model = HybridReasoningDiffusionModel(
                reasoning_model_path=args.load_from,
                diffusion_model_path=args.diffusion_model
            )
            model.load(args.load_from)
        else:
            model = HybridReasoningDiffusionModel(
                reasoning_model_path=args.model,
                diffusion_model_path=args.diffusion_model
            )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure the model is downloaded or use --load-from with fine-tuned checkpoint")
        return
    
    # Interactive mode
    if args.interactive:
        print("\nEnter your prompt (type 'quit' to exit):")
        while True:
            try:
                prompt = input("\nYou: ").strip()
                if prompt.lower() == 'quit':
                    break
                
                output = model.generate(
                    prompt=prompt,
                    max_reasoning_tokens=args.max_tokens,
                    temperature=args.temperature,
                    show_diffusion_steps=args.show_diffusion_steps
                )
                
                print(f"\n{model.config.output_tag_start}{output.final_output}{model.config.output_tag_end}")
                
                if args.show_reasoning:
                    print(f"\n{model.config.reasoning_tag_start}{output.reasoning}{model.config.reasoning_tag_end}")
                
                if args.show_diffusion_steps and output.diffusion_steps:
                    print(f"\n=== Diffusion Unmasking Steps ===")
                    for step in output.diffusion_steps:
                        print(step)
                    
            except KeyboardInterrupt:
                break
        return
    
    # Single prompt mode
    if not args.prompt:
        parser.print_help()
        return
    
    output = model.generate(
        prompt=args.prompt,
        max_reasoning_tokens=args.max_tokens,
        temperature=args.temperature,
        show_diffusion_steps=args.show_diffusion_steps
    )
    
    print(f"\n{model.config.output_tag_start}{output.final_output}{model.config.output_tag_end}")
    
    if args.show_reasoning:
        print(f"\nReasoning:")
        print(f"{model.config.reasoning_tag_start}{output.reasoning}{model.config.reasoning_tag_end}")
    
    if args.show_diffusion_steps and output.diffusion_steps:
        print(f"\n=== Diffusion Unmasking Steps ===")
        for step in output.diffusion_steps:
            print(step)


if __name__ == "__main__":
    main()
