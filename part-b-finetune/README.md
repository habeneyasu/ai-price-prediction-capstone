# Part B: Fine-tuned Open-Source Model

Fine-tune an open-source model to compete with frontier models in price prediction using QLoRA (Quantized LoRA) for parameter-efficient fine-tuning.

## Overview

This part demonstrates how to fine-tune open-source language models (e.g., Llama 3.2 3B) using QLoRA to predict product prices from descriptions. The fine-tuned model is evaluated against frontier models to demonstrate competitive performance.

## Features

- **QLoRA Fine-tuning**: Parameter-efficient fine-tuning using 4-bit quantization
- **PEFT Integration**: Uses HuggingFace PEFT library for LoRA adapters
- **Data Preparation**: Automated prompt generation and dataset preparation
- **Model Evaluation**: Comprehensive evaluation with comparison to frontier models
- **HuggingFace Integration**: Push models and datasets to HuggingFace Hub

## Requirements

- Python 3.10+
- GPU recommended (for training)
- HuggingFace account and token
- Optional: Weights & Biases account for experiment tracking

## Installation

Install dependencies:

```bash
# From project root
uv pip install -r requirements.txt

# Additional dependencies for fine-tuning
uv pip install transformers peft bitsandbytes trl accelerate
```

## Quick Start

### 1. Prepare Data

Convert your data to fine-tuning format:

```bash
cd part-b-finetune/src

# From CSV/JSON
python prepare_data.py \
    --data-source ../data/train_data.csv \
    --base-model meta-llama/Llama-3.2-3B \
    --output-dataset your-username/price-prediction-prompts

# From HuggingFace dataset
python prepare_data.py \
    --data-source hf:username/dataset-name \
    --base-model meta-llama/Llama-3.2-3B \
    --output-dataset your-username/price-prediction-prompts
```

### 2. Fine-tune Model

Train the model using QLoRA:

```bash
python train.py \
    --base-model meta-llama/Llama-3.2-3B \
    --dataset your-username/price-prediction-prompts \
    --train-size 1000 \
    --epochs 1 \
    --batch-size 4 \
    --output-dir models/finetuned \
    --hub-model your-username/llama3-price-predictor
```

### 3. Evaluate Model

Evaluate the fine-tuned model:

```bash
python evaluate.py \
    --base-model meta-llama/Llama-3.2-3B \
    --adapter your-username/llama3-price-predictor \
    --test-dataset your-username/price-prediction-prompts \
    --max-samples 200
```

### 4. Compare with Frontier Model

Compare fine-tuned model with frontier model:

```bash
python compare_models.py \
    --base-model meta-llama/Llama-3.2-3B \
    --adapter your-username/llama3-price-predictor \
    --test-dataset your-username/price-prediction-prompts \
    --frontier-model openai/gpt-4o-mini \
    --max-samples 100
```

## Configuration

### Environment Variables

Add to `.env`:

```env
# HuggingFace token (required for model/dataset push)
HF_TOKEN=hf_...

# Optional: Weights & Biases for experiment tracking
WANDB_API_KEY=...
```

### Training Parameters

Key parameters for fine-tuning:

- `--base-model`: Base model identifier (default: `meta-llama/Llama-3.2-3B`)
- `--train-size`: Limit training data (for testing, default: use all)
- `--epochs`: Number of training epochs (default: 1)
- `--batch-size`: Batch size per device (default: 4)
- `--learning-rate`: Learning rate (default: 1e-4)
- `--lora-r`: LoRA rank (default: 32)
- `--lora-alpha`: LoRA alpha scaling (default: 64)

### Supported Models

- `meta-llama/Llama-3.2-3B` (recommended for demo)
- `meta-llama/Meta-Llama-3.1-8B` (requires more memory)
- Other compatible causal language models

## Architecture

```
part-b-finetune/
├── src/
│   ├── prepare_data.py    # Data preparation and prompt generation
│   ├── train.py           # QLoRA fine-tuning script
│   ├── evaluate.py        # Model evaluation
│   └── compare_models.py  # Comparison with frontier models
├── models/                # Saved model checkpoints
└── README.md
```

## Data Format

Training data should be in HuggingFace dataset format with:

- `text`: Full prompt + completion (e.g., "What does this cost... Price is $12.00")
- `price`: Ground truth price
- `title`: Product title
- `category`: Product category

## Cost Considerations

- **Training**: Free on local GPU, or use Google Colab (free tier available)
- **Model Storage**: Free on HuggingFace Hub (public repos)
- **Evaluation**: Uses local inference (no API costs)

## Troubleshooting

### Out of Memory

- Reduce `--batch-size` or `--train-size`
- Use smaller base model (e.g., Llama-3.2-3B instead of 8B)
- Enable 4-bit quantization (default)

### Slow Training

- Use GPU (CUDA)
- Increase `--gradient-accumulation` to maintain effective batch size
- Reduce `--max-seq-length` if possible

### Model Not Improving

- Increase training data size
- Adjust learning rate
- Train for more epochs
- Check data quality

## Next Steps

- Experiment with different base models
- Tune hyperparameters (LoRA rank, learning rate)
- Train on larger datasets
- Compare multiple fine-tuned models
