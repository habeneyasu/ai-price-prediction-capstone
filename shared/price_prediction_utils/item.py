"""
Item data model for product price prediction.

This module defines the Item class which represents a product with its
description and price information.
"""

from pydantic import BaseModel, Field
from typing import Optional


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
    """
    
    title: str = Field(..., description="Product title")
    category: str = Field(..., description="Product category")
    price: float = Field(..., ge=0, description="Product price in USD")
    description: Optional[str] = Field(None, description="Full product description")
    summary: Optional[str] = Field(None, description="Preprocessed product summary")
    item_id: Optional[str] = Field(None, description="Unique item identifier")
    
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
        else:
            return self.title
