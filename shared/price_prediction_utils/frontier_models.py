"""
Frontier Model abstraction layer for price prediction using OpenRouter and Ollama.

This module provides a unified interface for using different LLMs via OpenRouter
and local Ollama models to predict product prices from descriptions.
"""

import re
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from enum import Enum

import requests
from openai import OpenAI

logger = logging.getLogger(__name__)


class ModelProvider(str, Enum):
    """Supported model providers."""
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"


class FrontierModel(ABC):
    """
    Abstract base class for frontier model implementations.
    
    This provides a unified interface for different LLM providers to ensure
    consistent behavior across models.
    """
    
    @abstractmethod
    def predict_price(self, description: str, **kwargs) -> float:
        """
        Predict the price of a product from its description.
        
        Args:
            description: Product description text
            **kwargs: Additional model-specific parameters
            
        Returns:
            Predicted price as a float
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        pass


class OpenRouterModel(FrontierModel):
    """
    OpenRouter model implementation.
    
    OpenRouter provides access to multiple frontier models through a unified API.
    Supports models like GPT-4, Claude, Llama, and many others.
    """
    
    def __init__(
        self,
        model_name: str = "openai/gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 50,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize OpenRouter model.
        
        Args:
            model_name: Model identifier (e.g., 'openai/gpt-4o', 'anthropic/claude-3.5-sonnet')
            api_key: OpenRouter API key (if None, uses environment variable)
            base_url: OpenRouter API base URL
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            system_prompt: Custom system prompt (optional)
        """
        self.model_name = model_name
        self.base_url = base_url or "https://openrouter.ai/api/v1"
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize OpenAI client with OpenRouter endpoint
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
        )
        
        self.system_prompt = system_prompt or (
            "You are a meticulous Data-Driven Market Analyst who estimates "
            "realistic, data-based product prices for online marketplaces. "
            "You base estimates on comparable items and avoid outliers. "
            "Return only the price number in USD format (e.g., $99.99 or 99.99)."
        )
        
        self.total_calls = 0
        self.total_tokens = 0
        self.total_cost = 0.0
    
    def predict_price(self, description: str, **kwargs) -> float:
        """Predict price using OpenRouter."""
        try:
            # Ensure description is not empty
            if not description or not description.strip():
                raise ValueError("Product description cannot be empty")
            
            # Check for placeholder text
            description_lower = description.strip().lower()
            placeholder_patterns = [
                "your product description",
                "product description here",
                "enter product description",
                "describe the product",
            ]
            
            if any(pattern in description_lower for pattern in placeholder_patterns):
                logger.warning(f"Detected placeholder text: '{description}'. Please provide an actual product description.")
                raise ValueError(
                    f"Please provide an actual product description, not placeholder text like '{description}'. "
                    "Example: 'Wireless Bluetooth headphones with noise cancellation, 30-hour battery life'"
                )
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": f"Product description: {description.strip()}\n\nEstimate the price in USD. Return only the number."
                }
            ]
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **kwargs
            )
            
            # Track usage
            self.total_calls += 1
            if hasattr(response, 'usage') and response.usage:
                self.total_tokens += response.usage.total_tokens
            
            # Extract price from response
            content = response.choices[0].message.content.strip()
            price = self._extract_price(content)
            
            return price
            
        except Exception as e:
            logger.error(f"Error predicting price with OpenRouter ({self.model_name}): {e}")
            raise
    
    def _extract_price(self, text: str) -> float:
        """Extract numeric price from model response."""
        # Remove currency symbols and commas
        text = text.replace("$", "").replace(",", "").strip()
        
        # Find first number (integer or float)
        match = re.search(r"[-+]?\d*\.?\d+", text)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass
        
        logger.warning(f"Could not extract price from: {text}")
        return 0.0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenRouter model information."""
        return {
            "provider": ModelProvider.OPENROUTER.value,
            "model": self.model_name,
            "base_url": self.base_url,
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
        }


class OllamaModel(FrontierModel):
    """
    Ollama model implementation for local LLM inference.
    
    Supports running models locally via Ollama API.
    """
    
    def __init__(
        self,
        model_name: str = "llama3.1",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.3,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize Ollama model.
        
        Args:
            model_name: Ollama model name (e.g., 'llama3.1', 'mistral', 'qwen2.5')
            base_url: Ollama API base URL
            temperature: Sampling temperature (0-2)
            system_prompt: Custom system prompt (optional)
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature
        
        self.system_prompt = system_prompt or (
            "You are a meticulous Data-Driven Market Analyst who estimates "
            "realistic, data-based product prices for online marketplaces. "
            "You base estimates on comparable items and avoid outliers. "
            "Return only the price number in USD format (e.g., $99.99 or 99.99)."
        )
        
        self.total_calls = 0
        self.total_tokens = 0
    
    def predict_price(self, description: str, **kwargs) -> float:
        """Predict price using Ollama."""
        try:
            prompt = f"{self.system_prompt}\n\nEstimate the price of this product:\n\n{description}"
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": 50,  # Limit response length
                    },
                    **kwargs
                },
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            content = result.get("response", "").strip()
            
            # Track usage
            self.total_calls += 1
            if "eval_count" in result:
                self.total_tokens += result.get("eval_count", 0)
            
            # Extract price from response
            price = self._extract_price(content)
            
            return price
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to Ollama at {self.base_url}: {e}")
            raise RuntimeError(f"Ollama connection failed. Is Ollama running? {e}")
        except Exception as e:
            logger.error(f"Error predicting price with Ollama ({self.model_name}): {e}")
            raise
    
    def _extract_price(self, text: str) -> float:
        """Extract numeric price from model response."""
        text = text.replace("$", "").replace(",", "").strip()
        match = re.search(r"[-+]?\d*\.?\d+", text)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass
        
        logger.warning(f"Could not extract price from: {text}")
        return 0.0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Ollama model information."""
        return {
            "provider": ModelProvider.OLLAMA.value,
            "model": self.model_name,
            "base_url": self.base_url,
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
        }
    
    def check_availability(self) -> bool:
        """
        Check if Ollama is available and the model exists.
        
        Returns:
            True if Ollama is available and model exists, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            return any(self.model_name in name for name in model_names)
        except Exception as e:
            logger.warning(f"Ollama availability check failed: {e}")
            return False


def create_model(
    provider: str = "openrouter",
    model_name: Optional[str] = None,
    **kwargs
) -> FrontierModel:
    """
    Factory function to create a frontier model instance.
    
    Args:
        provider: Model provider ('openrouter' or 'ollama')
        model_name: Specific model name (optional, uses defaults if not provided)
        **kwargs: Additional model parameters
        
    Returns:
        FrontierModel instance
        
    Examples:
        >>> model = create_model("openrouter", model_name="openai/gpt-4o")
        >>> model = create_model("ollama", model_name="llama3.1")
    """
    provider = provider.lower()
    
    if provider == "openrouter":
        model_name = model_name or "openai/gpt-4o"
        return OpenRouterModel(model_name=model_name, **kwargs)
    
    elif provider == "ollama":
        model_name = model_name or "llama3.1"
        return OllamaModel(model_name=model_name, **kwargs)
    
    else:
        raise ValueError(f"Unsupported provider: {provider}. Use 'openrouter' or 'ollama'")
