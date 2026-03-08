# Part A: Frontier Models

Predict product prices from short descriptions using frontier language models via OpenRouter and Ollama.

## Overview

Price prediction implementation using:
- **OpenRouter**: Unified API for accessing multiple frontier models (GPT-4, Claude, Llama, etc.)
- **Ollama**: Local inference for open-source models

## Installation

1. Install dependencies:
```bash
cd /path/to/ai-price-prediction-capstone
uv pip install -r requirements.txt
```

2. Configure environment variables in `.env` (project root):
```bash
cp ../.env.example .env
# Edit .env and add OPENROUTER_API_KEY
```

Required: `OPENROUTER_API_KEY`  
Optional: `OLLAMA_BASE_URL` (defaults to `http://localhost:11434`)

## Quick Start

### Single Prediction

```bash
cd part-a-frontier/src
uv run python3 main.py --provider openrouter --predict "Wireless Bluetooth headphones with active noise cancellation, 30-hour battery life"
```

### Evaluate on Sample Data

```bash
uv run python3 main.py --provider openrouter --data ../data/test_data.csv --max-samples 20
```

### Use Local Ollama

```bash
ollama pull llama3.1
uv run python3 main.py --provider ollama --model llama3.1 --data sample --count 10
```

## Project Structure

```
part-a-frontier/
├── README.md
├── COST_GUIDE.md
├── data/
│   ├── test_data.csv
│   └── test_data.json
├── notebooks/
│   └── part_a_exploration.ipynb
└── src/
    ├── main.py
    ├── generate_sample_data.py
    ├── example_usage.py
    └── cost_optimization.py
```

## Usage

### Command Line Interface

**Single prediction:**
```bash
uv run python3 main.py --provider openrouter --predict "Product description"
```

**Evaluation:**
```bash
uv run python3 main.py \
  --provider openrouter \
  --model openai/gpt-4o-mini \
  --data ../data/test_data.csv \
  --max-samples 20 \
  --output results.json
```

**Options:**
- `--provider`: `openrouter` or `ollama`
- `--model`: Model identifier (e.g., `openai/gpt-4o-mini`)
- `--data`: Data source (`sample`, CSV/JSON file, or `hf:dataset_name`)
- `--max-samples`: Maximum items to evaluate (default: 20)
- `--temperature`: Sampling temperature (default: 0.3)
- `--no-plot`: Disable visualization generation

### Programmatic Usage

```python
from shared.price_prediction_utils import create_model, PricePredictor, Item

model = create_model(
    provider="openrouter",
    model_name="openai/gpt-4o-mini",
    api_key="your_key"
)
predictor = PricePredictor(model)

item = Item(
    title="Product",
    category="Electronics",
    price=0.0,
    description="Product description..."
)
price = predictor.predict(item)
```

## Supported Models

### OpenRouter
- `openai/gpt-4o`
- `openai/gpt-4o-mini` (default, cost-efficient)
- `openai/gpt-4-turbo`
- `anthropic/claude-3.5-sonnet`
- `meta-llama/llama-3.1-70b-instruct`
- See [OpenRouter.ai/models](https://openrouter.ai/models) for full list

### Ollama
- `llama3.1`
- `mistral`
- `qwen2.5`
- Any model available in local Ollama installation

## Data Sources

- **Sample data**: `--data sample --count N`
- **CSV files**: `--data path/to/file.csv`
- **JSON files**: `--data path/to/file.json`
- **HuggingFace**: `--data hf:username/dataset_name --split test`

Sample data is included in `data/test_data.csv` (20 items).

## Evaluation Metrics

- **MAE**: Mean Absolute Error (average dollar error)
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of determination
- **MAPE**: Mean Absolute Percentage Error
- **Median Error**: Robust to outliers

Visualizations include scatter plots (predicted vs actual) and error trend charts.

## Cost Considerations

Default configuration uses `gpt-4o-mini` for cost efficiency. Estimated costs:
- Single prediction: ~$0.00003
- 20 predictions: ~$0.0006
- 100 predictions: ~$0.003

Use `uv run python3 src/cost_optimization.py --tips` for cost optimization strategies.

For free local inference, use Ollama with `--provider ollama`.

## API Keys

**OpenRouter:**
1. Sign up at [OpenRouter.ai](https://openrouter.ai)
2. Get API key from dashboard
3. Add to `.env`: `OPENROUTER_API_KEY=your_key`

**Ollama:**
1. Install from [ollama.ai](https://ollama.ai)
2. Pull model: `ollama pull llama3.1`
3. Service runs automatically

## Troubleshooting

**API key not found**: Ensure `OPENROUTER_API_KEY` is set in `.env`

**Model not found**: Verify model name at [OpenRouter.ai/models](https://openrouter.ai/models)

**Ollama connection failed**: Ensure Ollama is running (`ollama serve`)

**Import errors**: Install dependencies with `uv pip install -r requirements.txt`

## Examples

**Compare models:**
```bash
uv run python3 main.py --provider openrouter --model openai/gpt-4o --data sample --count 10
uv run python3 main.py --provider openrouter --model anthropic/claude-3.5-sonnet --data sample --count 10
```

**Generate additional sample data:**
```bash
uv run python3 src/generate_sample_data.py --count 50
```

## Documentation

- [Main README](../README.md) - Project overview
- [Architecture](../ARCHITECTURE.md) - System architecture
- [Cost Guide](COST_GUIDE.md) - Cost optimization details
