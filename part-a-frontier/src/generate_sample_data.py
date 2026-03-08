#!/usr/bin/env python3
"""
Generate sample data for testing and demonstration.

This script creates sample product data in CSV and JSON formats.
"""

import sys
from pathlib import Path
import csv
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.price_prediction_utils import create_sample_items

def generate_csv(output_path: str, count: int = 20):
    """Generate CSV file with sample items."""
    items = create_sample_items(count)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['title', 'category', 'price', 'summary', 'item_id'])
        writer.writeheader()
        
        for item in items:
            writer.writerow({
                'title': item.title,
                'category': item.category,
                'price': item.price,
                'summary': item.summary or '',
                'item_id': item.item_id or '',
            })
    
    print(f"Generated {len(items)} items in {output_path}")

def generate_json(output_path: str, count: int = 20):
    """Generate JSON file with sample items."""
    items = create_sample_items(count)
    
    data = [item.model_dump() for item in items]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Generated {len(items)} items in {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate sample data for testing")
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of sample items to generate (default: 20)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "json", "both"],
        default="both",
        help="Output format (default: both)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).parent.parent / "data"),
        help="Output directory (default: part-a-frontier/data)"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.format in ["csv", "both"]:
        csv_path = output_dir / "test_data.csv"
        generate_csv(str(csv_path), args.count)
    
    if args.format in ["json", "both"]:
        json_path = output_dir / "test_data.json"
        generate_json(str(json_path), args.count)
    
    print(f"\nSample data generated in {output_dir}")
