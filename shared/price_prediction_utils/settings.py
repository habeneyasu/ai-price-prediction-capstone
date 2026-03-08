"""
Configuration management for the price prediction system.

This module handles environment variables, configuration loading, and
default settings for all three capstone parts.
"""

import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class ModelConfig:
    """Configuration for model settings."""
    provider: str = "openrouter"
    model_name: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 50
    api_key: Optional[str] = None
    base_url: Optional[str] = None


@dataclass
class EvaluationConfig:
    """Configuration for evaluation settings."""
    max_samples: Optional[int] = None
    num_workers: int = 5
    generate_plots: bool = True


@dataclass
class Settings:
    """Main settings class containing all configuration."""
    
    # API Keys
    openrouter_api_key: Optional[str] = None
    openrouter_base_url: Optional[str] = None
    ollama_base_url: Optional[str] = None
    
    # Model configuration
    model: Optional[ModelConfig] = None
    
    # Evaluation configuration
    evaluation: Optional[EvaluationConfig] = None
    
    # Logging
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Initialize default configurations if not provided."""
        # Load API keys from environment
        if self.openrouter_api_key is None:
            self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if self.openrouter_base_url is None:
            self.openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        if self.ollama_base_url is None:
            self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # Default model config
        if self.model is None:
            provider = os.getenv("MODEL_PROVIDER", "openrouter")
            model_name = os.getenv("MODEL_NAME")
            
            self.model = ModelConfig(
                provider=provider,
                model_name=model_name,
                temperature=float(os.getenv("MODEL_TEMPERATURE", "0.3")),
                max_tokens=int(os.getenv("MODEL_MAX_TOKENS", "50")),
                api_key=self.openrouter_api_key,
                base_url=self.openrouter_base_url if provider == "openrouter" else None,
            )
        
        # Default evaluation config
        if self.evaluation is None:
            max_samples_str = os.getenv("EVAL_MAX_SAMPLES", "0")
            max_samples = int(max_samples_str) if max_samples_str else None
            if max_samples == 0:
                max_samples = None
            self.evaluation = EvaluationConfig(
                max_samples=max_samples,
                num_workers=int(os.getenv("EVAL_NUM_WORKERS", "5")),
                generate_plots=os.getenv("EVAL_GENERATE_PLOTS", "true").lower() == "true",
            )
        
        # Logging
        if self.log_level == "INFO":
            self.log_level = os.getenv("LOG_LEVEL", "INFO")


# Global settings instance
settings = Settings()
