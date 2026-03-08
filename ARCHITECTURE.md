# Architecture Overview

## Project Structure

```
ai-price-prediction-capstone/
├── part-a-frontier/          # Part A: Frontier Models (Week 6)
│   ├── README.md
│   ├── data/                 # Data files
│   ├── notebooks/            # Jupyter notebooks
│   └── src/
│       └── main.py          # CLI entry point
│
├── part-b-finetune/          # Part B: Fine-tuned Models (Week 7)
│   ├── README.md
│   ├── data/                 # Training/validation data
│   ├── models/               # Saved model checkpoints
│   └── src/                  # Fine-tuning scripts
│
├── part-c-agent/            # Part C: Autonomous Agents (Week 8)
│   ├── README.md
│   ├── agents/               # Agent implementations
│   ├── notifier/             # Notification services
│   └── src/                  # Orchestration code
│
├── shared/                   # Common utilities
│   └── price_prediction_utils/
│       ├── __init__.py
│       ├── item.py              # Item data model
│       ├── frontier_models.py   # OpenRouter & Ollama models
│       ├── predictor.py         # Price prediction service
│       ├── evaluator.py         # Evaluation framework
│       ├── settings.py          # Configuration management
│       ├── data_loader.py       # Data loading utilities
│       └── logging_config.py    # Logging setup
│
├── requirements.txt          # Global dependencies
├── .env.example             # Environment variables template
└── README.md                # Main documentation
```

## Architecture Principles

### 1. Modular Design
- Each part (A, B, C) is self-contained
- Shared utilities in `shared/` package
- Clear separation of concerns

### 2. Model Abstraction
- Unified interface for different model providers
- Easy to swap between OpenRouter and Ollama
- Extensible for new providers

### 3. Configuration Management
- Environment-based configuration
- Centralized settings in shared package
- Per-part overrides supported

### 4. Evaluation Framework
- Comprehensive metrics (MAE, RMSE, R², MAPE)
- Interactive visualizations
- Reusable across all parts

## Part A: Frontier Models

### Components

1. **Models** (`shared/price_prediction_utils/frontier_models.py`)
   - `OpenRouterModel`: Access to multiple frontier models via OpenRouter API
   - `OllamaModel`: Local inference via Ollama
   - `create_model()`: Factory function for model creation

2. **Predictor** (`shared/price_prediction_utils/predictor.py`)
   - `PricePredictor`: High-level prediction service
   - Error handling and retry logic
   - Batch processing support

3. **Evaluator** (`shared/price_prediction_utils/evaluator.py`)
   - `PriceEvaluator`: Comprehensive evaluation
   - Metrics calculation
   - Visualization generation

### Data Flow

```
Product Description
    ↓
Item Model (Pydantic)
    ↓
Frontier Model (OpenRouter/Ollama)
    ↓
Price Prediction
    ↓
Evaluation & Metrics
    ↓
Visualization & Reports
```

## Part B: Fine-tuned Models (Planned)

### Planned Components

1. **Fine-tuning Pipeline**
   - Data preparation
   - Model training
   - Hyperparameter tuning
   - Model versioning

2. **Model Comparison**
   - Benchmarking framework
   - Performance comparison
   - Cost analysis

## Part C: Autonomous Agents (Planned)

### Planned Components

1. **Agent System**
   - Price monitoring agents
   - Deal detection algorithms
   - Multi-agent coordination

2. **Notification System**
   - Email notifications
   - SMS alerts
   - Webhook integrations

3. **Orchestration**
   - Agent scheduling
   - Task distribution
   - State management

## Shared Utilities

### Item Model
- Pydantic-based data validation
- Flexible text extraction (summary/description/title)
- Type-safe product representation

### Data Loading
- CSV support
- JSON support
- HuggingFace dataset integration
- Sample data generation

### Configuration
- Environment variable management
- Default value handling
- Provider-specific settings

### Logging
- Structured logging
- Configurable log levels
- Console output formatting

## Technology Stack

### Core
- **Python 3.9+**: Main language
- **Pydantic**: Data validation
- **Python-dotenv**: Environment management

### LLM Integration
- **OpenAI SDK**: OpenRouter API client
- **Requests**: Ollama HTTP client

### Data & ML
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations
- **Scikit-learn**: Metrics

### Visualization
- **Plotly**: Interactive charts

### Utilities
- **Tqdm**: Progress bars
- **Datasets**: HuggingFace integration

## Design Patterns

1. **Factory Pattern**: Model creation via `create_model()`
2. **Strategy Pattern**: Interchangeable model providers
3. **Service Layer**: Business logic in `PricePredictor`
4. **Abstract Base Class**: `FrontierModel` interface
5. **Dependency Injection**: Configuration via constructor

## Extension Points

### Adding New Model Providers

1. Create a new class inheriting from `FrontierModel`
2. Implement `predict_price()` and `get_model_info()`
3. Add to `create_model()` factory function
4. Update `ModelProvider` enum

### Adding New Evaluation Metrics

1. Extend `_calculate_metrics()` in `PriceEvaluator`
2. Add to metrics dictionary
3. Update `_print_metrics()` for display

### Adding New Data Sources

1. Add function to `data_loader.py`
2. Update `load_data()` in Part A main script
3. Follow existing patterns (CSV, JSON, HuggingFace)

## Best Practices

1. **Type Hints**: All functions have type annotations
2. **Error Handling**: Comprehensive try/except blocks
3. **Logging**: Structured logging throughout
4. **Documentation**: Docstrings for all public APIs
5. **Configuration**: Environment-based, no hardcoded secrets
6. **Testing**: Modular design enables easy unit testing

## Future Enhancements

1. **Caching**: Cache predictions for repeated queries
2. **Batch Processing**: Optimize batch API calls
3. **Model Versioning**: Track model versions and performance
4. **A/B Testing**: Compare different models in production
5. **Monitoring**: Track prediction errors over time
6. **API Wrapper**: REST API for easy integration
