# Part C: Autonomous Agent System

**Project 8: Capstone Part C** – Autonomous agent system collaborating with models to spot deals and notify you of special bargains.

## Overview

This part implements an autonomous agent system that continuously monitors product deals, evaluates their value using ensemble price prediction models, and sends notifications when significant bargains are detected.

## Features

- **Autonomous Monitoring**: Scans RSS feeds for new product deals
- **Intelligent Filtering**: Uses LLM to identify deals with clear descriptions and prices
- **Ensemble Prediction**: Combines multiple models for accurate price estimation
- **Deal Detection**: Identifies opportunities where estimated value exceeds deal price
- **Notification System**: Alerts when discounts exceed configurable threshold
- **Memory Management**: Tracks processed deals to avoid duplicates

## Architecture

The system consists of four specialized agents:

1. **Scanner Agent**: Monitors RSS feeds and filters promising deals
2. **Ensemble Agent**: Combines multiple price prediction models
3. **Planning Agent**: Orchestrates the workflow and decision-making
4. **Messaging Agent**: Crafts and sends notifications

## Installation

Ensure you have the project dependencies installed:

```bash
# From project root
uv sync
```

## Configuration

Set up your `.env` file with API keys:

```bash
OPENROUTER_API_KEY=your-api-key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OLLAMA_BASE_URL=http://localhost:11434  # Optional, for local inference
```

## Usage

### Command Line

Run a single detection cycle:

```bash
cd part-c-agent/src
python main.py
```

Run multiple cycles:

```bash
python main.py --cycles 5
```

List all opportunities:

```bash
python main.py --list
```

Customize threshold and providers:

```bash
python main.py --threshold 75.0 --scanner-provider ollama --ensemble-primary openrouter
```

### Programmatic Usage

```python
from src.framework import DealAgentFramework

# Initialize framework
framework = DealAgentFramework(
    deal_threshold=50.0,
    scanner_provider="openrouter",
    ensemble_primary="openrouter",
)

# Run detection cycle
opportunity = framework.run()

if opportunity:
    print(f"Found deal: ${opportunity.discount:.2f} discount")
    print(f"URL: {opportunity.deal.url}")
```

### Jupyter Notebook

See `notebooks/part_c_demo.ipynb` for an interactive demonstration.

## Project Structure

```
part-c-agent/
├── README.md              # This file
├── agents/                # Agent implementations
│   ├── agent.py          # Base agent class
│   ├── deals.py          # Deal data models and RSS scraping
│   ├── scanner_agent.py  # RSS feed monitoring and filtering
│   ├── ensemble_agent.py # Multi-model price prediction
│   ├── planning_agent.py # Workflow orchestration
│   └── messaging_agent.py # Notification crafting
├── notifier/             # Notification services
│   └── notifier.py       # Console/file-based notifications
├── src/                  # Main orchestration
│   ├── framework.py      # DealAgentFramework class
│   └── main.py          # CLI entry point
└── notebooks/            # Interactive demos
    └── part_c_demo.ipynb
```

## How It Works

1. **Scanning**: Scanner Agent fetches deals from RSS feeds and uses an LLM to identify the 5 most promising deals with clear descriptions and prices.

2. **Evaluation**: For each deal, the Ensemble Agent estimates the true product value using multiple price prediction models.

3. **Opportunity Detection**: The system calculates the discount (estimated value - deal price) and compares it to the threshold.

4. **Notification**: When a deal exceeds the threshold, the Messaging Agent crafts an engaging notification using an LLM and sends it via the notification system.

5. **Memory**: All processed deals are stored in `memory.json` to avoid reprocessing.

## Configuration Options

- `deal_threshold`: Minimum discount amount to trigger notifications (default: 50.0)
- `scanner_provider`: Model provider for scanner agent ("openrouter" or "ollama")
- `ensemble_primary`: Primary model provider for ensemble agent
- `memory_file`: Path to memory file (default: "memory.json")

## Notification System

Notifications are saved to:
- Console output (with formatted alerts)
- `notifications/notifications.jsonl` (JSON Lines format)

Each notification includes:
- Timestamp
- Crafted message
- Opportunity details (price, estimate, discount, URL)

## Integration with Parts A and B

- Uses shared utilities from `shared/price_prediction_utils/`
- Leverages frontier models from Part A
- Can integrate fine-tuned models from Part B (via ensemble)

## Cost Considerations

- RSS feed scraping is free
- LLM API calls incur costs based on provider and model
- Use cheaper models (e.g., `gpt-4o-mini`) for cost-effective operation
- Consider using Ollama for local inference to reduce API costs

## Limitations

- RSS feed availability depends on external sources
- Price extraction from deal descriptions may have errors
- Model predictions are estimates, not guarantees
- Notification system is file-based (can be extended to email/SMS)

## Future Enhancements

- Email/SMS notification integration
- Webhook support for external integrations
- Vector database for similarity search (RAG)
- Scheduled/continuous monitoring
- Web dashboard for deal visualization
- Integration with fine-tuned models from Part B
