#!/usr/bin/env python3
"""
Fine-tuning script for open-source models using QLoRA.

This script implements parameter-efficient fine-tuning (PEFT) using QLoRA
to fine-tune open-source language models for price prediction.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.price_prediction_utils import setup_logging, settings

# Setup logging
load_dotenv()
setup_logging(level="INFO")
logger = logging.getLogger(__name__)


class FineTuningConfig:
    """Configuration for fine-tuning."""
    
    def __init__(
        self,
        base_model: str = "meta-llama/Llama-3.2-3B",
        dataset_name: Optional[str] = None,
        output_dir: str = "models/finetuned",
        hub_model_name: Optional[str] = None,
        train_size: Optional[int] = None,
        val_split: float = 0.1,
        max_seq_length: int = 182,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        num_epochs: int = 1,
        learning_rate: float = 1e-4,
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.1,
        use_4bit: bool = True,
        use_wandb: bool = False,
    ):
        self.base_model = base_model
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.hub_model_name = hub_model_name
        self.train_size = train_size
        self.val_split = val_split
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_4bit = use_4bit
        self.use_wandb = use_wandb


def load_and_prepare_data(
    dataset_name: str,
    train_size: Optional[int] = None,
    val_split: float = 0.1,
) -> tuple:
    """
    Load and split dataset into train and validation sets.
    
    Args:
        dataset_name: HuggingFace dataset name
        train_size: Optional limit on training data size
        val_split: Fraction of data to use for validation
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    logger.info(f"Loading dataset: {dataset_name}")
    
    dataset = load_dataset(dataset_name)
    train = dataset['train']
    
    if train_size:
        logger.info(f"Limiting training data to {train_size} samples")
        train = train.select(range(min(train_size, len(train))))
    
    total_size = len(train)
    val_size = int(total_size * val_split)
    
    val_data = train.select(range(val_size))
    train_data = train.select(range(val_size, total_size))
    
    logger.info(f"Train samples: {len(train_data):,}")
    logger.info(f"Validation samples: {len(val_data):,}")
    
    return train_data, val_data


def setup_model_and_tokenizer(
    base_model: str,
    use_4bit: bool = True,
):
    """
    Load and configure model and tokenizer with quantization.
    
    Args:
        base_model: Model identifier
        use_4bit: Whether to use 4-bit quantization
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model and tokenizer: {base_model}")
    
    # Configure quantization
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        quant_config = None
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if use_4bit else torch.float16,
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    if use_4bit:
        memory_mb = model.get_memory_footprint() / 1e6
        logger.info(f"Model memory footprint: {memory_mb:.1f} MB")
    
    return model, tokenizer


def create_lora_config(
    r: int = 32,
    alpha: int = 64,
    dropout: float = 0.1,
    target_modules: Optional[list] = None,
) -> LoraConfig:
    """
    Create LoRA configuration for PEFT.
    
    Args:
        r: LoRA rank
        alpha: LoRA alpha scaling
        dropout: LoRA dropout rate
        target_modules: Modules to apply LoRA to (default: attention layers)
        
    Returns:
        LoraConfig instance
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )


def create_training_config(
    config: FineTuningConfig,
    output_dir: str,
    run_name: str,
) -> SFTConfig:
    """
    Create training configuration.
    
    Args:
        config: FineTuningConfig instance
        output_dir: Output directory for checkpoints
        run_name: Name for this training run
        
    Returns:
        SFTConfig instance
    """
    return SFTConfig(
        output_dir=output_dir,
        run_name=run_name,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        
        # Training
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        group_by_length=True,
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=50,
        per_device_eval_batch_size=1,
        
        # Optimization
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="paged_adamw_32bit",
        weight_decay=0.001,
        max_grad_norm=0.3,
        
        # Precision
        fp16=False,
        bf16=True,
        
        # Logging & Saving
        logging_steps=20,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        report_to="wandb" if config.use_wandb else None,
        
        # Hub
        push_to_hub=config.hub_model_name is not None,
        hub_strategy="end",
        hub_model_id=config.hub_model_name,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )


def train_model(config: FineTuningConfig):
    """
    Fine-tune model using QLoRA.
    
    Args:
        config: FineTuningConfig instance
    """
    # Login to HuggingFace
    hf_token = os.getenv("HF_TOKEN") or settings.openrouter_api_key
    if hf_token:
        login(hf_token, add_to_git_credential=True)
    
    # Setup Weights & Biases if enabled
    if config.use_wandb:
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key:
            try:
                import wandb
                os.environ["WANDB_API_KEY"] = wandb_api_key
                wandb.login()
                project_name = "price-prediction-finetune"
                run_name = f"{datetime.now():%Y-%m-%d_%H-%M-%S}"
                os.environ["WANDB_PROJECT"] = project_name
                wandb.init(project=project_name, name=run_name)
            except ImportError:
                logger.warning("wandb not installed, disabling wandb logging")
                config.use_wandb = False
        else:
            logger.warning("WANDB_API_KEY not found, disabling wandb logging")
            config.use_wandb = False
    
    # Load data
    if not config.dataset_name:
        raise ValueError("dataset_name is required")
    
    train_data, val_data = load_and_prepare_data(
        config.dataset_name,
        train_size=config.train_size,
        val_split=config.val_split,
    )
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        config.base_model,
        use_4bit=config.use_4bit,
    )
    
    # Create LoRA config
    lora_config = create_lora_config(
        r=config.lora_r,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
    )
    
    # Create data collator
    response_template = "Price is $"
    collator = DataCollatorForCompletionOnlyLM(
        response_template,
        tokenizer=tokenizer
    )
    
    # Create training config
    run_name = f"finetune-{datetime.now():%Y%m%d-%H%M%S}"
    train_config = create_training_config(
        config,
        output_dir=config.output_dir,
        run_name=run_name,
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=lora_config,
        args=train_config,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    
    # Train
    logger.info("Starting fine-tuning...")
    trainer.train()
    
    # Save final model
    if config.hub_model_name:
        logger.info(f"Pushing model to Hub: {config.hub_model_name}")
        trainer.push_to_hub()
    else:
        logger.info(f"Saving model to: {config.output_dir}")
        trainer.save_model()
    
    if config.use_wandb:
        try:
            import wandb
            wandb.finish()
        except ImportError:
            pass
    
    logger.info("Fine-tuning completed")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune open-source model for price prediction")
    
    # Model configuration
    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Llama-3.2-3B",
        help="Base model identifier (default: meta-llama/Llama-3.2-3B)",
    )
    
    # Dataset configuration
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="HuggingFace dataset name (e.g., username/dataset_name)",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="Limit training data size (for testing, default: use all)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)",
    )
    
    # Training configuration
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size per device (default: 4)",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=182,
        help="Maximum sequence length (default: 182)",
    )
    
    # LoRA configuration
    parser.add_argument(
        "--lora-r",
        type=int,
        default=32,
        help="LoRA rank (default: 32)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=64,
        help="LoRA alpha (default: 64)",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.1,
        help="LoRA dropout (default: 0.1)",
    )
    
    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/finetuned",
        help="Output directory for checkpoints (default: models/finetuned)",
    )
    parser.add_argument(
        "--hub-model",
        type=str,
        default=None,
        help="HuggingFace Hub model name to push to (optional)",
    )
    
    # Options
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization (requires more memory)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    
    args = parser.parse_args()
    
    config = FineTuningConfig(
        base_model=args.base_model,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        hub_model_name=args.hub_model,
        train_size=args.train_size,
        val_split=args.val_split,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_4bit=not args.no_4bit,
        use_wandb=args.wandb,
    )
    
    try:
        train_model(config)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
