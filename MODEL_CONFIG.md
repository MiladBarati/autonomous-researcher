# Model Configuration Guide

## Current Configuration

The default model has been updated to **`llama-3.3-70b-versatile`** (as of November 2024).

## How to Change the Model

Edit `config.py` and update the `MODEL_NAME` setting:

```python
# In config.py, line ~29
MODEL_NAME: str = "your-preferred-model-name"
```

## Available Groq Models (as of Nov 2024)

### Large Models (Best Quality)
- **`llama-3.3-70b-versatile`** ⭐ (Current Default)
  - Best overall performance
  - Good for complex research synthesis
  - ~70B parameters

- **`mixtral-8x7b-32768`**
  - Alternative large model
  - 32K token context window
  - Good for longer documents

### Fast Models (Better Speed)
- **`llama-3.1-8b-instant`**
  - Fastest inference
  - Good for quick research
  - Lower quality but very fast

- **`gemma2-9b-it`**
  - Balanced speed/quality
  - Efficient for most tasks

### Check Current Models

Visit the official Groq documentation for the latest available models:
https://console.groq.com/docs/models

## Troubleshooting

### Error: "model_decommissioned"

This means the model is no longer supported. To fix:

1. Check [Groq Models page](https://console.groq.com/docs/models)
2. Choose a current model
3. Update `config.py`:
   ```python
   MODEL_NAME: str = "new-model-name"
   ```
4. Restart the application

### Error: "rate_limit_exceeded"

Groq has free tier limits. Solutions:
- Wait a few minutes
- Use a faster/smaller model (`llama-3.1-8b-instant`)
- Upgrade your Groq plan

### Error: "invalid_api_key"

Check your `.env` file:
```bash
# View your .env
type .env  # Windows
cat .env   # Linux/Mac
```

Make sure `GROQ_API_KEY` is set correctly.

## Performance Comparison

| Model | Speed | Quality | Context | Best For |
|-------|-------|---------|---------|----------|
| llama-3.3-70b-versatile | Medium | Excellent | 8K | Complex research |
| mixtral-8x7b-32768 | Medium | Very Good | 32K | Long documents |
| llama-3.1-8b-instant | Very Fast | Good | 8K | Quick searches |
| gemma2-9b-it | Fast | Very Good | 8K | Balanced tasks |

## Model Parameters

You can also adjust these in `config.py`:

```python
# Creativity vs. Accuracy (0.0 - 1.0)
MODEL_TEMPERATURE: float = 0.7
# Lower = more factual, Higher = more creative

# Maximum response length
MAX_TOKENS: int = 8192
# Increase for longer reports (model limit permitting)
```

## Examples

### For Fastest Research (Lower Quality)
```python
MODEL_NAME: str = "llama-3.1-8b-instant"
MODEL_TEMPERATURE: float = 0.5
MAX_TOKENS: int = 4096
```

### For Best Quality (Slower)
```python
MODEL_NAME: str = "llama-3.3-70b-versatile"
MODEL_TEMPERATURE: float = 0.7
MAX_TOKENS: int = 8192
```

### For Balanced Performance
```python
MODEL_NAME: str = "gemma2-9b-it"
MODEL_TEMPERATURE: float = 0.6
MAX_TOKENS: int = 6144
```

## Testing Your Configuration

After updating the model, test it:

```bash
# Activate venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Test with a simple topic
python main.py "test topic"

# Or use the web interface
streamlit run app.py
```

If you get errors, check the Groq console for current model names and update accordingly.

