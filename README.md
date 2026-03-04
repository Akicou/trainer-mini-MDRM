# Nayhein-8.8B-MDRM

Hybrid Auto-Regressive Reasoning to Diffusion Model combining Qwen3.5-0.8B-Base for reasoning with LLaDA-V for output generation.

## Features

- **Reasoning in <think></think> tags**: Model thinks step-by-step before generating output
- **Hybrid architecture**: Auto-regressive reasoning + diffusion-based output
- **Synthetic data generation**: Generate training data using lmstudio or ollama
- **LoRA fine-tuning**: Efficient parameter-efficient fine-tuning
- **AutoModelVision compatible**: Load with standard transformers API

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Generate Synthetic Training Data

```bash
# Using lmstudio (default localhost:1234)
python src/finetune.py --generate-synthetic-data --provider lmstudio --num-samples 100

# Using ollama
python src/finetune.py --generate-synthetic-data --provider ollama --model llama3.2 --num-samples 100
```

### Fine-tune Model

```bash
# Fine-tune with existing data
python src/finetune.py --data-path ./data/train.json --output-dir ./checkpoints/nayhein-8b

# Generate data and fine-tune in one command
python src/finetune.py --generate-synthetic-data --provider lmstudio --num-samples 100 \
    --output-dir ./checkpoints/nayhein-8b
```

### Inference

```bash
# Single prompt
python main.py "Explain how to build a REST API"

# Show reasoning
python main.py "Design a database schema" --show-reasoning

# Interactive mode
python main.py --interactive

# Load fine-tuned model
python main.py "Your prompt" --load-from ./checkpoints/nayhein-8b
```

## Programmatic Usage

```python
from src.hybrid_model import HybridReasoningDiffusionModel

# Initialize model
model = HybridReasoningDiffusionModel(
    reasoning_model_path="Qwen/Qwen3.5-0.8B-Base",
    diffusion_model_path="GSAI-ML/LLaDA-V"
)

# Generate response
output = model.generate(prompt="Explain machine learning")

print(f"Reasoning: {output.reasoning}")
print(f"Output: {output.final_output}")
```

### AutoModelVision Compatible

```python
from src.auto_model_vision import NayheinForVisionText2Text

# Load like any transformers model
model = NayheinForVisionText2Text.from_pretrained("./checkpoints/nayhein-8b")
output = model.generate("Your prompt here")
```

## Architecture

1. **Reasoning Model** (Qwen3.5-0.8B-Base): Processes input and generates step-by-step reasoning in <think></think> tags
2. **Diffusion Model** (LLaDA-V): Takes reasoning as context and generates final output in <output></output> tags

## Project Structure

```
mmdrllm/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration classes
│   ├── data_generator.py      # Synthetic data generation
│   ├── model.py               # Reasoning model
│   ├── diffusion_model.py     # Diffusion model
│   ├── hybrid_model.py        # Combined hybrid model
│   ├── auto_model_vision.py   # AutoModelVision wrapper
│   └── finetune.py            # Fine-tuning script
├── main.py                    # CLI interface
├── requirements.txt
└── AGENTS.md
```

## Configuration

All configuration is in `src/config.py`:

- `ModelConfig`: Model paths and tag formats
- `TrainingConfig`: Training hyperparameters
- `SyntheticDataConfig`: Data generation settings

## Output Format

```
User prompt

<think>
Reasoning step by step...
</think>

<output>
Final response
</output>
```

# trainer-mini-MDRM
