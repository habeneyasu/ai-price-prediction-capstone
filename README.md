# AI Price Prediction Capstone

A comprehensive three-part capstone project for predicting product prices and identifying deals using modern AI techniques.

## Projects Overview

### Part A: Frontier Models (Week 6) ✅
**Predict product prices from short descriptions using Frontier models**

- Uses OpenRouter API for access to multiple frontier models (GPT-4, Claude, etc.)
- Supports local inference via Ollama
- Comprehensive evaluation framework
- Interactive visualizations

[→ Part A Documentation](./part-a-frontier/README.md)

### Part B: Fine-tuned Models (Week 7) ✅
**Fine-tuned open-source model to compete with Frontier in price prediction**

- QLoRA fine-tuning pipeline for open-source models
- Parameter-efficient fine-tuning with PEFT
- Model comparison and benchmarking
- HuggingFace Hub integration

[→ Part B Documentation](./part-b-finetune/README.md)

### Part C: Autonomous Agent System (Week 8) 🚧
**Autonomous agent system collaborating with models to spot deals and notify you of special bargains**

- Autonomous price monitoring
- Deal detection algorithms
- Multi-model collaboration
- Notification system

[→ Part C Documentation](./part-c-agent/README.md)

## Project Structure

```
ai-price-prediction-capstone/
├── part-a-frontier/          # Part A: Frontier Models
│   ├── README.md
│   ├── data/
│   ├── notebooks/
│   └── src/
├── part-b-finetune/          # Part B: Fine-tuned Models
│   ├── README.md
│   ├── data/
│   ├── models/
│   └── src/
├── part-c-agent/            # Part C: Autonomous Agents
│   ├── README.md
│   ├── agents/
│   ├── notifier/
│   └── src/
├── shared/                   # Common utilities
│   └── price_prediction_utils/
│       ├── item.py
│       ├── frontier_models.py
│       ├── predictor.py
│       ├── evaluator.py
│       ├── settings.py
│       ├── data_loader.py
│       └── logging_config.py
├── requirements.txt          # Global dependencies
├── .env.example             # Environment variables template
└── README.md                # This file
```

## Quick Start

### 1. Installation

```bash
# Clone or navigate to the project
cd ai-price-prediction-capstone

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file in the project root:

```env
# OpenRouter API Key (for Part A)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Ollama Configuration (optional, for local models)
OLLAMA_BASE_URL=http://localhost:11434

# HuggingFace Token (optional, for dataset loading)
HF_TOKEN=your_huggingface_token_here

# Model Configuration (optional)
MODEL_PROVIDER=openrouter
MODEL_NAME=openai/gpt-4o
MODEL_TEMPERATURE=0.3
MODEL_MAX_TOKENS=50

# Evaluation Configuration (optional)
EVAL_MAX_SAMPLES=200
EVAL_NUM_WORKERS=5
EVAL_GENERATE_PLOTS=true

# Logging (optional)
LOG_LEVEL=INFO
```

### 3. Run Part A

```bash
cd part-a-frontier/src
python main.py --provider openrouter --predict "Wireless headphones with noise cancellation"
```

## Shared Utilities

The `shared/price_prediction_utils/` package provides common functionality used across all parts:

- **Item**: Data model for products
- **Frontier Models**: OpenRouter and Ollama model implementations
- **Predictor**: Price prediction service
- **Evaluator**: Comprehensive evaluation framework
- **Data Loader**: CSV, JSON, HuggingFace dataset loading
- **Settings**: Configuration management

## Dependencies

Core dependencies:
- `pydantic`: Data validation
- `openai`: OpenAI-compatible API client (for OpenRouter)
- `requests`: HTTP client (for Ollama)
- `pandas`, `numpy`: Data processing
- `scikit-learn`: Metrics
- `plotly`: Visualizations
- `tqdm`: Progress bars
- `python-dotenv`: Environment variable management

See `requirements.txt` for the complete list.

## API Keys

### OpenRouter

1. Sign up at [OpenRouter.ai](https://openrouter.ai)
2. Get your API key from the dashboard
3. Add to `.env`: `OPENROUTER_API_KEY=your_key`

### Ollama (Local)

1. Install Ollama: https://ollama.ai
2. Pull a model: `ollama pull llama3.1`
3. Start Ollama service (usually runs automatically)
4. Configure in `.env`: `OLLAMA_BASE_URL=http://localhost:11434`

## Usage Examples

### Part A: Frontier Models

```bash
# Quick prediction
cd part-a-frontier/src
python main.py --provider openrouter --predict "Product description here"

# Evaluate on sample data
python main.py --provider openrouter --model openai/gpt-4o --data sample --count 20

# Evaluate with local Ollama
python main.py --provider ollama --model llama3.1 --data sample --count 10
```

## Contributing

This is a capstone project. Each part builds upon the previous one:

1. **Part A** establishes baseline with frontier models
2. **Part B** fine-tunes open-source models to compete
3. **Part C** builds an autonomous system using all models

## License

This project is part of an educational capstone project.

## Acknowledgments

- Based on LLM Engineering course (Weeks 6-8)
- Uses Amazon product data from McAuley-Lab/Amazon-Reviews-2023 dataset
- Built with OpenRouter, Ollama, and modern Python tooling
