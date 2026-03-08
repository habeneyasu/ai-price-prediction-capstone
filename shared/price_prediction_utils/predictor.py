"""
Price prediction service.

This module provides a high-level service interface for predicting product prices
using frontier models with proper error handling and logging.
"""

import logging
from typing import List, Optional, Callable
from functools import wraps
import time

from shared.price_prediction_utils.item import Item
from shared.price_prediction_utils.frontier_models import FrontierModel

logger = logging.getLogger(__name__)


def retry_on_error(max_retries: int = 3, delay: float = 1.0):
    """
    Decorator to retry function calls on errors.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries} attempts failed: {e}")
            raise last_exception
        return wrapper
    return decorator


class PricePredictor:
    """
    High-level service for predicting product prices.
    
    This class provides a clean interface for price prediction with
    error handling, logging, and batch processing capabilities.
    """
    
    def __init__(self, model: FrontierModel):
        """
        Initialize the price predictor.
        
        Args:
            model: Frontier model instance to use for predictions
        """
        self.model = model
        self.predictions_made = 0
        self.errors = 0
    
    @retry_on_error(max_retries=3, delay=1.0)
    def predict(self, item: Item) -> float:
        """
        Predict the price of a single item.
        
        Args:
            item: Item object with description/summary
            
        Returns:
            Predicted price as a float
            
        Raises:
            ValueError: If item has no text for prediction
            RuntimeError: If model prediction fails
        """
        text = item.get_text_for_prediction()
        
        if not text or not text.strip():
            raise ValueError(f"Item {item.item_id} has no text for prediction")
        
        try:
            price = self.model.predict_price(text)
            self.predictions_made += 1
            
            # Ensure non-negative price
            price = max(0.0, price)
            
            logger.debug(
                f"Predicted ${price:.2f} for item: {item.title[:50]}..."
            )
            
            return price
            
        except Exception as e:
            self.errors += 1
            logger.error(f"Failed to predict price for item {item.item_id}: {e}")
            raise
    
    def predict_batch(self, items: List[Item]) -> List[float]:
        """
        Predict prices for a batch of items.
        
        Args:
            items: List of Item objects
            
        Returns:
            List of predicted prices
        """
        predictions = []
        for item in items:
            try:
                price = self.predict(item)
                predictions.append(price)
            except Exception as e:
                logger.error(f"Error predicting for item {item.item_id}: {e}")
                predictions.append(0.0)  # Default to 0 on error
        
        return predictions
    
    def get_stats(self) -> dict:
        """
        Get prediction statistics.
        
        Returns:
            Dictionary with prediction statistics
        """
        model_info = self.model.get_model_info()
        return {
            "predictions_made": self.predictions_made,
            "errors": self.errors,
            "success_rate": (
                (self.predictions_made - self.errors) / self.predictions_made
                if self.predictions_made > 0 else 0.0
            ),
            "model_info": model_info,
        }
