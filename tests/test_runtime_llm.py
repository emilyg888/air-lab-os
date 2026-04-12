import importlib

import runtime.llm as llm


def _reload_llm():
    return importlib.reload(llm)


def test_llm_defaults_to_local_ollama_qwen_14b(monkeypatch):
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    module = _reload_llm()

    assert module.OLLAMA_URL == "http://localhost:11434/api/generate"
    assert module.QWEN_MODEL == "qwen2.5:14b"


def test_llm_allows_ollama_env_overrides(monkeypatch):
    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:32b")

    module = _reload_llm()

    assert module.OLLAMA_URL == "http://127.0.0.1:11434/api/generate"
    assert module.QWEN_MODEL == "qwen2.5:32b"

    monkeypatch.delenv("OLLAMA_URL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    _reload_llm()
