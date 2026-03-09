#!/usr/bin/env python3
"""
Evaluate fine-tuned model.

Loads a fine-tuned model and evaluates it on test data.
"""

import sys
import os
import re
import logging
from pathlib import Path
from typing import List, Callable

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.price_prediction_utils import setup_logging, evaluate, settings

load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)


PREFIX = "Price is $"
QUESTION = "What does this cost to the nearest dollar?"


class FineTunedPredictor:
    """Predictor for fine-tuned models."""
    
    def __init__(
        self,
        base_model: str,
        adapter_path: str,
        use_4bit: bool = True,
    ):
        """
        Initialize fine-tuned model predictor.
        
        Args:
            base_model: Base model identifier
            adapter_path: Path to PEFT adapter (local or HuggingFace)
            use_4bit: Whether to use 4-bit quantization
        """
        logger.info(f"Loading base model: {base_model}")
        
        # Setup quantization
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
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if use_4bit else torch.float16,
        )
        
        # Load PEFT adapter
        logger.info(f"Loading adapter: {adapter_path}")
        self.model = PeftModel.from_pretrained(base_model_obj, adapter_path)
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def predict(self, text: str) -> float:
        """
        Predict price from product description.
        
        Args:
            text: Product description
            
        Returns:
            Predicted price
        """
        # Create prompt
        prompt = f"{QUESTION}\n\n{text}\n\n{PREFIX}"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=182,
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.3,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract price
        price_text = response.split(PREFIX)[-1].strip()
        price = self._extract_price(price_text)
        
        return price
    
    def _extract_price(self, text: str) -> float:
        """Extract numeric price from text."""
        text = text.replace("$", "").replace(",", "").strip()
        match = re.search(r"[-+]?\d*\.?\d+", text)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass
        logger.warning(f"Could not extract price from: {text}")
        return 0.0


def evaluate_finetuned_model(
    base_model: str,
    adapter_path: str,
    test_dataset_name: str,
    max_samples: int = 200,
    use_4bit: bool = True,
):
    """
    Evaluate fine-tuned model on test dataset.
    
    Args:
        base_model: Base model identifier
        adapter_path: Path to PEFT adapter
        test_dataset_name: HuggingFace test dataset name
        max_samples: Maximum samples to evaluate
        use_4bit: Whether to use 4-bit quantization
    """
    # Login to HuggingFace
    hf_token = os.getenv("HF_TOKEN") or settings.openrouter_api_key
    if hf_token:
        login(hf_token, add_to_git_credential=True)
    
    # Load test data
    logger.info(f"Loading test dataset: {test_dataset_name}")
    dataset = load_dataset(test_dataset_name, split="test")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    # Create predictor
    predictor_obj = FineTunedPredictor(
        base_model=base_model,
        adapter_path=adapter_path,
        use_4bit=use_4bit,
    )
    
    # Create prediction function
    def predict_func(item):
        text = item.get("text", "")
        # Extract description (everything before "Price is $")
        if PREFIX in text:
            description = text.split(PREFIX)[0].strip()
            description = description.replace(QUESTION, "").strip()
        else:
            description = text
        return predictor_obj.predict(description)
    
    # Evaluate
    logger.info(f"Evaluating on {len(dataset)} samples...")
    metrics = evaluate(
        predictor=predict_func,
        test_data=list(dataset),
        name=f"Fine-tuned {base_model}",
        max_samples=max_samples,
        plot=True,
    )
    
    logger.info("Evaluation completed")
    return metrics


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Base model identifier (e.g., meta-llama/Llama-3.2-3B)",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        help="Path to PEFT adapter (local path or HuggingFace model name)",
    )
    parser.add_argument(
        "--test-dataset",
        type=str,
        required=True,
        help="HuggingFace test dataset name",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Maximum samples to evaluate (default: 200)",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation report (JSON)",
    )
    
    args = parser.parse_args()
    
    try:
        metrics = evaluate_finetuned_model(
            base_model=args.base_model,
            adapter_path=args.adapter,
            test_dataset_name=args.test_dataset,
            max_samples=args.max_samples,
            use_4bit=not args.no_4bit,
        )
        
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Report saved to {args.output}")
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
