# air-lab-os

This repo uses LM Studio for the planner layer.

## Local LLM setup

Start the LM Studio local server and load `Qwen2.5-14B-Instruct`.

```bash
Open LM Studio
Start the local inference server
Load Qwen2.5-14B-Instruct
```

The planner defaults are:

```bash
LM_STUDIO_URL=http://127.0.0.1:1234/v1/chat/completions
LM_STUDIO_MODEL=Qwen2.5-14B-Instruct
```

You can override either variable in your shell if needed before running:

```bash
uv run python main.py --loop --dataset <dataset_id>
```
