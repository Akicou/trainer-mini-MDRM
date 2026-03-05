# Deployment Guide: Dual-Mode Model Training

This guide covers deploying and training the dual-mode generation model on cloud platforms.

## Recommended Platform: PrimeIntellect

PrimeIntellect is recommended over RunPod for this project due to:
- Lower cost per GPU hour
- Better optimization for distributed training
- Seamless HuggingFace dataset streaming
- Simpler setup and configuration

## Quick Start

### 1. Create a PrimeIntellect Instance

```bash
# Create a new instance with A100 40GB
primeintellect instance create \
  --gpu a100-40gb \
  --disk-size 100GB \
  --name dual-mode-training \
  --image pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# SSH into the instance
primeintellect ssh dual-mode-training
```

### 2. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/YOUR-USERNAME/trainer-mini-MDRM.git
cd trainer-mini-MDRM

# Install dependencies
pip install -r requirements.txt
pip install datasets accelerate

# Login to HuggingFace (for dataset access)
huggingface-cli login

# Optional: Login to WandB for experiment tracking
wandb login
```

### 3. Start Training

#### Basic Training (No WandB)

```bash
python main.py train-dual \
  --dataset nvidia/Nemotron-Cascade-SFT-Stage-1 \
  --dataset-split train \
  --dataset-size 10000 \
  --output ./checkpoints/dual-mode-model \
  --model Qwen/Qwen3.5-0.8B-Base \
  --diffusion-model GSAI-ML/LLaDA-V \
  --epochs 3 \
  --batch-size 4 \
  --learning-rate 2e-5
```

#### Training with WandB Tracking

```bash
python main.py train-dual \
  --dataset nvidia/Nemotron-Cascade-SFT-Stage-1 \
  --dataset-size 10000 \
  --output ./checkpoints/dual-mode-model \
  --epochs 3 \
  --batch-size 4 \
  --learning-rate 2e-5 \
  --wandb \
  --wandb-project dual-mode-model \
  --wandb-run-name "dual-mode-nvidia-10k"
```

## GPU Recommendations

| GPU | VRAM | Est. Cost/hr | Recommended For |
|-----|------|--------------|-----------------|
| RTX 4090 | 24GB | ~$0.50 | Testing, small datasets |
| A100 40GB | 40GB | ~$1.00 | **Recommended** for training |
| A100 80GB | 80GB | ~$1.50 | Large batches, faster training |

## Command Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | Required | HuggingFace dataset name |
| `--dataset-split` | `train` | Dataset split to use |
| `--dataset-size` | `10000` | Number of samples |
| `--output` | `./checkpoints/dual-mode` | Output directory |
| `--model` | `Qwen/Qwen3.5-0.8B-Base` | Reasoning model path |
| `--diffusion-model` | `GSAI-ML/LLaDA-V` | Diffusion model path |
| `--epochs` | `3` | Number of training epochs |
| `--batch-size` | `4` | Per-GPU batch size |
| `--learning-rate` | `2e-5` | Learning rate |
| `--max-steps` | `None` | Max training steps |
| `--wandb` | `False` | Enable WandB logging |
| `--wandb-project` | `dual-mode-model` | WandB project name |
| `--wandb-entity` | `None` | WandB entity/team |
| `--wandb-run-name` | `None` | Custom WandB run name |

## Cost Estimation

| Dataset Size | GPU | Time | Est. Cost |
|--------------|-----|------|-----------|
| 1,000 samples | A100 40GB | ~15 min | ~$0.25 |
| 10,000 samples | A100 40GB | ~2-3 hours | ~$3 |
| 50,000 samples | A100 40GB | ~10-15 hours | ~$15 |

## Monitoring Training

```bash
# In another terminal (use tmux for persistent sessions)
watch -n 5 "nvidia-smi"

# Or use tmux to keep training running in background
tmux new -s training
python main.py train-dual ...
# Press Ctrl+B, then D to detach

# Reattach to training session
tmux attach -t training
```

## Quick Test Run

Before full training, test with a small dataset:

```bash
python main.py train-dual \
  --dataset nvidia/Nemotron-Cascade-SFT-Stage-1 \
  --dataset-size 100 \
  --output ./checkpoints/test-run \
  --epochs 1 \
  --batch-size 2 \
  --max-steps 50
```

## Using RunPod (Alternative)

If you prefer RunPod:

```bash
# Create pod via CLI or web UI
runpodctl create pod \
  --name dual-mode-training \
  --image pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime \
  --gpu-type "NVIDIA A100 40GB" \
  --container-disk-size 100

# SSH and run same commands
ssh root@<pod-ip>
# ... same setup as PrimeIntellect
```

## Model Checkpoint Structure

After training, your checkpoint directory will contain:

```
checkpoints/dual-mode-model/
├── config.json                  # Model configuration
├── model.safetensors            # Backbone weights
├── tokenizer.json               # Tokenizer files
├── ar_head.pt                   # Auto-regressive head
├── diffusion_head.pt            # Diffusion head
└── dual_mode_config.json        # Dual-mode settings
```

## Inference with Trained Model

```bash
# Generate with trained dual-mode model
python main.py dual-mode \
  "Explain quantum computing" \
  --load-from ./checkpoints/dual-mode-model \
  --show-steps

# Interactive mode
python main.py dual-mode \
  --load-from ./checkpoints/dual-mode-model \
  --interactive \
  --show-reasoning
```

## Troubleshooting

### Out of Memory Errors

- Reduce `--batch-size` to 2 or 1
- Reduce `--dataset-size` for testing
- Use a GPU with more VRAM

### Dataset Loading Issues

```bash
# Install datasets library
pip install datasets

# Login to HuggingFace
huggingface-cli login
```

### WandB Not Logging

```bash
# Install wandb
pip install wandb

# Login
wandb login
```

## Next Steps

1. Start with a small dataset (1,000-5,000 samples) to validate
2. Scale up to 10,000-50,000 samples for better quality
3. Monitor WandB dashboard for training metrics
4. Experiment with hyperparameters (learning rate, batch size)
5. Use the trained model for inference with `--dual-mode` flag

## Support

- GitHub Issues: https://github.com/YOUR-USERNAME/trainer-mini-MDRM/issues
- WandB Dashboard: https://wandb.ai/
- PrimeIntellect Docs: https://docs.primeintellect.com/
