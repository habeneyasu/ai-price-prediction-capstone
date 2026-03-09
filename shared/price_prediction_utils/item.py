"""
Item data model for product price prediction.

This module defines the Item class which represents a product with its
description and price information.
"""

from pydantic import BaseModel, Field
from typing import Optional, Self
from datasets import Dataset, DatasetDict, load_dataset


PREFIX = "Price is $"
QUESTION = "What does this cost to the nearest dollar?"


class Item(BaseModel):
    """
    Represents a product item with price and description information.
    
    Attributes:
        title: Product title
        category: Product category
        price: Actual price in USD
        description: Full product description
        summary: Preprocessed/standardized product summary
        item_id: Unique identifier for the item
        prompt: Prompt text for fine-tuning (optional)
        completion: Completion text for fine-tuning (optional)
    """
    
    title: str = Field(..., description="Product title")
    category: str = Field(..., description="Product category")
    price: float = Field(..., ge=0, description="Product price in USD")
    description: Optional[str] = Field(None, description="Full product description")
    summary: Optional[str] = Field(None, description="Preprocessed product summary")
    item_id: Optional[str] = Field(None, description="Unique item identifier")
    prompt: Optional[str] = Field(None, description="Prompt for fine-tuning")
    completion: Optional[str] = Field(None, description="Completion for fine-tuning")
    full: Optional[str] = Field(None, description="Full text (alias for description)")
    weight: Optional[float] = Field(None, description="Product weight")
    id: Optional[int] = Field(None, description="Numeric ID (alias for item_id)")
    
    def __repr__(self) -> str:
        """String representation of the item."""
        return f"<Item: {self.title[:50]}... = ${self.price:.2f}>"
    
    def get_text_for_prediction(self) -> str:
        """
        Get the text to use for price prediction.
        
        Returns:
            The summary if available, otherwise the description or title.
        """
        if self.summary:
            return self.summary
        elif self.description:
            return self.description
        elif self.full:
            return self.full
        else:
            return self.title
    
    def make_prompt(self, text: str):
        """Create prompt with completion for training."""
        self.prompt = f"{QUESTION}\n\n{text}\n\n{PREFIX}{round(self.price)}.00"
    
    def test_prompt(self) -> str:
        """Create test prompt without completion."""
        if self.prompt:
            return self.prompt.split(PREFIX)[0] + PREFIX
        text = self.get_text_for_prediction()
        return f"{QUESTION}\n\n{text}\n\n{PREFIX}"
    
    def count_tokens(self, tokenizer):
        """Count tokens in the summary."""
        text = self.get_text_for_prediction()
        return len(tokenizer.encode(text, add_special_tokens=False))
    
    def make_prompts(self, tokenizer, max_tokens: int, do_round: bool = True):
        """Make prompts and completions for fine-tuning."""
        text = self.get_text_for_prediction()
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > max_tokens:
            truncated_tokens = tokens[:max_tokens]
            text = tokenizer.decode(truncated_tokens, skip_special_tokens=True).rstrip()
        
        self.prompt = f"{QUESTION}\n\n{text}\n\n{PREFIX}"
        self.completion = f"{round(self.price)}.00" if do_round else str(self.price)
    
    def count_prompt_tokens(self, tokenizer):
        """Count tokens in the full prompt + completion."""
        if not self.prompt or not self.completion:
            return 0
        full = self.prompt + self.completion
        tokens = tokenizer.encode(full, add_special_tokens=False)
        return len(tokens)
    
    def to_datapoint(self) -> dict:
        """Convert to datapoint format for SFT training."""
        return {"prompt": self.prompt, "completion": self.completion}
    
    @staticmethod
    def push_to_hub(dataset_name: str, train: list[Self], val: list[Self], test: list[Self]):
        """Push Item lists to HuggingFace Hub."""
        DatasetDict(
            {
                "train": Dataset.from_list([item.model_dump() for item in train]),
                "validation": Dataset.from_list([item.model_dump() for item in val]),
                "test": Dataset.from_list([item.model_dump() for item in test]),
            }
        ).push_to_hub(dataset_name)
    
    @classmethod
    def from_hub(cls, dataset_name: str) -> tuple[list[Self], list[Self], list[Self]]:
        """Load from HuggingFace Hub and reconstruct Items."""
        ds = load_dataset(dataset_name)
        return (
            [cls.model_validate(row) for row in ds["train"]],
            [cls.model_validate(row) for row in ds["validation"]],
            [cls.model_validate(row) for row in ds["test"]],
        )
    
    @staticmethod
    def push_prompts_to_hub(
        dataset_name: str, train: list[Self], val: list[Self], test: list[Self]
    ):
        """Push Item lists to HuggingFace Hub in prompt-completion format for SFT training."""
        DatasetDict(
            {
                "train": Dataset.from_list([item.to_datapoint() for item in train]),
                "val": Dataset.from_list([item.to_datapoint() for item in val]),
                "test": Dataset.from_list([item.to_datapoint() for item in test]),
            }
        ).push_to_hub(dataset_name)