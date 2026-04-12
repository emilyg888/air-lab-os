# air-lab-os

This repo uses a local Ollama server for the planner layer.

## Local LLM setup

Start Ollama and make sure `qwen2.5:14b` is available:

```bash
brew services start ollama
ollama pull qwen2.5:14b
```

The planner defaults are:

```bash
OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=qwen2.5:14b
```

You can override either variable in your shell if needed before running:

```bash
uv run python main.py --loop --dataset <dataset_id>
```
