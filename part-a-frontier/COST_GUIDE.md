# Cost Optimization

## Model Recommendations

**Recommended**: `openai/gpt-4o-mini`
- Cost: ~$0.15 per 1M tokens
- Quality: Excellent for price prediction
- Best balance of cost and performance

**Budget**: `meta-llama/llama-3.1-8b-instruct`
- Cost: ~$0.10 per 1M tokens
- Quality: Good
- Suitable for large-scale evaluations

**Free**: Ollama (local)
- Cost: $0
- Quality: Good with proper models
- Requires local hardware

**Premium**: `openai/gpt-4o`
- Cost: ~$5 per 1M tokens
- Quality: Best
- Use only for final evaluation

## Cost Estimates

Per 100 predictions:

| Model | Estimated Cost |
|-------|----------------|
| gpt-4o-mini | ~$0.003 |
| claude-3-haiku | ~$0.005 |
| llama-3.1-8b | ~$0.002 |
| gpt-4o | ~$0.10 |

## Strategies

1. **Use smaller evaluation sets**: Start with `--max-samples 20` for testing
2. **Use cheaper models**: Default to `gpt-4o-mini` for development
3. **Use local Ollama**: Free inference for development and testing
4. **Limit response tokens**: Use `--max-tokens 20` (default is 50)
5. **Cache results**: Save predictions to avoid re-running

## Cost Monitoring

Monitor usage at [OpenRouter.ai/activity](https://openrouter.ai/activity). Set spending limits if needed.

## Estimation Tool

```bash
uv run python3 src/cost_optimization.py --predictions 100 --model openai/gpt-4o-mini
uv run python3 src/cost_optimization.py --compare --predictions 100
```
