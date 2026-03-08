"""
Data loading utilities for price prediction.

This module provides functions to load test data from various sources
including CSV files, HuggingFace datasets, and JSON files.
"""

import csv
import json
import logging
from typing import List, Optional
from pathlib import Path

from shared.price_prediction_utils.item import Item

logger = logging.getLogger(__name__)


def load_from_csv(
    file_path: str,
    title_col: str = "title",
    category_col: str = "category",
    price_col: str = "price",
    description_col: Optional[str] = None,
    summary_col: Optional[str] = None,
) -> List[Item]:
    """
    Load items from a CSV file.
    
    Args:
        file_path: Path to CSV file
        title_col: Column name for title
        category_col: Column name for category
        price_col: Column name for price
        description_col: Column name for description (optional)
        summary_col: Column name for summary (optional)
        
    Returns:
        List of Item objects
    """
    items = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            try:
                item = Item(
                    title=row.get(title_col, ""),
                    category=row.get(category_col, "Unknown"),
                    price=float(row.get(price_col, 0)),
                    description=row.get(description_col) if description_col else None,
                    summary=row.get(summary_col) if summary_col else None,
                    item_id=str(idx),
                )
                items.append(item)
            except Exception as e:
                logger.warning(f"Error loading row {idx}: {e}")
                continue
    
    logger.info(f"Loaded {len(items)} items from {file_path}")
    return items


def load_from_json(file_path: str) -> List[Item]:
    """
    Load items from a JSON file.
    
    Expected format: List of dictionaries with Item fields.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        List of Item objects
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    items = []
    for idx, item_data in enumerate(data):
        try:
            if 'item_id' not in item_data:
                item_data['item_id'] = str(idx)
            item = Item(**item_data)
            items.append(item)
        except Exception as e:
            logger.warning(f"Error loading item {idx}: {e}")
            continue
    
    logger.info(f"Loaded {len(items)} items from {file_path}")
    return items


def load_from_huggingface(
    dataset_name: str,
    split: str = "test",
    hf_token: Optional[str] = None,
) -> List[Item]:
    """
    Load items from a HuggingFace dataset.
    
    Args:
        dataset_name: HuggingFace dataset identifier
        split: Dataset split to load (train, validation, test)
        hf_token: HuggingFace token (optional, uses environment if not provided)
        
    Returns:
        List of Item objects
    """
    try:
        from datasets import load_dataset
        from huggingface_hub import login
        
        if hf_token:
            login(hf_token, add_to_git_credential=True)
        
        dataset = load_dataset(dataset_name, split=split)
        
        items = []
        for idx, row in enumerate(dataset):
            try:
                item = Item(
                    title=row.get("title", ""),
                    category=row.get("category", "Unknown"),
                    price=float(row.get("price", 0)),
                    description=row.get("description"),
                    summary=row.get("summary"),
                    item_id=row.get("id", str(idx)),
                )
                items.append(item)
            except Exception as e:
                logger.warning(f"Error loading row {idx}: {e}")
                continue
        
        logger.info(f"Loaded {len(items)} items from HuggingFace dataset {dataset_name}")
        return items
        
    except ImportError:
        logger.error("datasets library not installed. Install with: pip install datasets")
        raise
    except Exception as e:
        logger.error(f"Error loading from HuggingFace: {e}")
        raise


def create_sample_items(count: int = 10) -> List[Item]:
    """
    Create sample items for testing.
    
    Args:
        count: Number of sample items to create
        
    Returns:
        List of sample Item objects
    """
    samples = [
        {
            "title": "Wireless Bluetooth Headphones with Noise Cancellation",
            "category": "Electronics",
            "price": 79.99,
            "summary": "High-quality wireless headphones with active noise cancellation, 30-hour battery life, and premium sound quality.",
        },
        {
            "title": "Stainless Steel Water Bottle 32oz",
            "category": "Home & Kitchen",
            "price": 24.99,
            "summary": "Insulated stainless steel water bottle keeps drinks cold for 24 hours or hot for 12 hours. BPA-free and leak-proof design.",
        },
        {
            "title": "Mechanical Gaming Keyboard RGB",
            "category": "Electronics",
            "price": 129.99,
            "summary": "Full-size mechanical gaming keyboard with RGB backlighting, Cherry MX switches, and programmable keys.",
        },
        {
            "title": "Yoga Mat Premium 6mm Thick",
            "category": "Sports & Outdoors",
            "price": 34.99,
            "summary": "Extra thick non-slip yoga mat with carrying strap. Eco-friendly TPE material, suitable for all yoga practices.",
        },
        {
            "title": "Smart Watch Fitness Tracker",
            "category": "Electronics",
            "price": 199.99,
            "summary": "Advanced smartwatch with heart rate monitor, GPS, sleep tracking, and 7-day battery life. Water-resistant.",
        },
    ]
    
    items = []
    for i in range(count):
        sample = samples[i % len(samples)]
        item = Item(
            title=sample["title"],
            category=sample["category"],
            price=sample["price"],
            summary=sample["summary"],
            item_id=f"sample_{i}",
        )
        items.append(item)
    
    return items
