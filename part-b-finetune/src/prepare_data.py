#!/usr/bin/env python3
"""
Prepare data for fine-tuning.

Converts items to prompt format and prepares dataset for HuggingFace.
"""

import sys
import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from huggingface_hub import login
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.price_prediction_utils import Item, load_from_csv, load_from_json, load_from_huggingface
from shared.price_prediction_utils import setup_logging

load_dotenv()
setup_logging()
import logging
logger = logging.getLogger(__name__)


PREFIX = "Price is $"
QUESTION = "What does this cost to the nearest dollar?"


def prepare_prompts(
    items: List[Item],
    tokenizer: AutoTokenizer,
    max_tokens: int = 110,
    is_training: bool = True,
) -> List[dict]:
    """
    Prepare prompts for fine-tuning.
    
    Args:
        items: List of Item objects
        tokenizer: Tokenizer for counting tokens
        max_tokens: Maximum tokens in summary
        is_training: Whether this is training data (includes completion)
        
    Returns:
        List of dictionaries with 'text' field for training
    """
    prompts = []
    
    for item in items:
        # Use Item's make_prompts method
        item.make_prompts(tokenizer, max_tokens, do_round=True)
        
        # Create text field combining prompt and completion for training
        if is_training:
            text = item.prompt + item.completion
        else:
            text = item.prompt
        
        prompts.append({
            "text": text,
            "price": item.price,
            "title": item.title,
            "category": item.category,
        })
    
    return prompts


def prepare_dataset(
    train_items: List[Item],
    val_items: List[Item],
    test_items: List[Item],
    base_model: str = "meta-llama/Llama-3.2-3B",
    max_tokens: int = 110,
    dataset_name: Optional[str] = None,
) -> DatasetDict:
    """
    Prepare dataset for fine-tuning and optionally push to HuggingFace.
    
    Args:
        train_items: Training items
        val_items: Validation items
        test_items: Test items
        base_model: Base model for tokenizer
        max_tokens: Maximum tokens in summary
        dataset_name: Optional HuggingFace dataset name to push to
        
    Returns:
        DatasetDict with train/validation/test splits
    """
    logger.info(f"Loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    
    logger.info("Preparing training prompts...")
    train_prompts = prepare_prompts(train_items, tokenizer, max_tokens, is_training=True)
    
    logger.info("Preparing validation prompts...")
    val_prompts = prepare_prompts(val_items, tokenizer, max_tokens, is_training=True)
    
    logger.info("Preparing test prompts...")
    test_prompts = prepare_prompts(test_items, tokenizer, max_tokens, is_training=False)
    
    # Create datasets
    dataset_dict = DatasetDict({
        "train": Dataset.from_list(train_prompts),
        "validation": Dataset.from_list(val_prompts),
        "test": Dataset.from_list(test_prompts),
    })
    
    logger.info(f"Created dataset: {len(train_prompts)} train, {len(val_prompts)} val, {len(test_prompts)} test")
    
    # Push to Hub if specified
    if dataset_name:
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            login(hf_token, add_to_git_credential=True)
            logger.info(f"Pushing dataset to Hub: {dataset_name}")
            dataset_dict.push_to_hub(dataset_name)
            logger.info(f"Dataset available at: https://huggingface.co/datasets/{dataset_name}")
        else:
            logger.warning("HF_TOKEN not found, skipping Hub upload")
    
    return dataset_dict


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare data for fine-tuning")
    
    parser.add_argument(
        "--data-source",
        type=str,
        required=True,
        help="Data source: CSV/JSON file, 'hf:dataset_name', or 'sample'",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Llama-3.2-3B",
        help="Base model for tokenizer (default: meta-llama/Llama-3.2-3B)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=110,
        help="Maximum tokens in summary (default: 110)",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="Limit training data size (default: use all)",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=None,
        help="Validation set size (default: 10%% of training)",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=None,
        help="Test set size (default: use all test data)",
    )
    parser.add_argument(
        "--output-dataset",
        type=str,
        default=None,
        help="HuggingFace dataset name to push to (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Local directory to save dataset (optional)",
    )
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from: {args.data_source}")
    
    # Helper function to split data
    def split_data(items: List[Item], train_ratio: float = 0.8, val_ratio: float = 0.1):
        """Split items into train/val/test sets."""
        train_size = int(len(items) * train_ratio)
        val_size = int(len(items) * val_ratio)
        train_items = items[:train_size]
        val_items = items[train_size:train_size + val_size]
        test_items = items[train_size + val_size:]
        return train_items, val_items, test_items
    
    if args.data_source.startswith("hf:"):
        dataset_name = args.data_source[3:]
        train_items, val_items, test_items = Item.from_hub(dataset_name)
    elif args.data_source.endswith(".csv"):
        all_items = load_from_csv(args.data_source)
        train_items, val_items, test_items = split_data(all_items)
    elif args.data_source.endswith(".json"):
        all_items = load_from_json(args.data_source)
        train_items, val_items, test_items = split_data(all_items)
    elif args.data_source == "sample":
        from shared.price_prediction_utils import create_sample_items
        all_items = create_sample_items(count=100)
        train_items, val_items, test_items = split_data(all_items)
    else:
        raise ValueError(f"Unknown data source: {args.data_source}")
    
    # Limit sizes if specified
    if args.train_size:
        train_items = train_items[:args.train_size]
    if args.val_size:
        val_items = val_items[:args.val_size]
    if args.test_size:
        test_items = test_items[:args.test_size]
    
    logger.info(f"Loaded: {len(train_items)} train, {len(val_items)} val, {len(test_items)} test")
    
    # Prepare dataset
    dataset_dict = prepare_dataset(
        train_items,
        val_items,
        test_items,
        base_model=args.base_model,
        max_tokens=args.max_tokens,
        dataset_name=args.output_dataset,
    )
    
    # Save locally if specified
    if args.output_dir:
        logger.info(f"Saving dataset to: {args.output_dir}")
        dataset_dict.save_to_disk(args.output_dir)
    
    logger.info("Data preparation completed")


if __name__ == "__main__":
    main()
