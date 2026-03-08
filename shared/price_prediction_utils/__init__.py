"""
Shared utilities for price prediction across all capstone parts.

This package provides common functionality used by Part A, Part B, and Part C.
"""

from shared.price_prediction_utils.item import Item
from shared.price_prediction_utils.frontier_models import (
    FrontierModel,
    OpenRouterModel,
    OllamaModel,
    create_model,
    ModelProvider,
)
from shared.price_prediction_utils.predictor import PricePredictor
from shared.price_prediction_utils.evaluator import PriceEvaluator, evaluate
from shared.price_prediction_utils.settings import settings, ModelConfig, EvaluationConfig
from shared.price_prediction_utils.data_loader import (
    load_from_csv,
    load_from_json,
    load_from_huggingface,
    create_sample_items,
)
from shared.price_prediction_utils.logging_config import setup_logging

__all__ = [
    "Item",
    "FrontierModel",
    "OpenRouterModel",
    "OllamaModel",
    "create_model",
    "ModelProvider",
    "PricePredictor",
    "PriceEvaluator",
    "evaluate",
    "settings",
    "ModelConfig",
    "EvaluationConfig",
    "load_from_csv",
    "load_from_json",
    "load_from_huggingface",
    "create_sample_items",
    "setup_logging",
]
