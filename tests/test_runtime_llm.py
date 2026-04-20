import importlib

import runtime.llm as llm


def _reload_llm():
    return importlib.reload(llm)


def test_llm_defaults_to_local_lm_studio_qwen_14b_instruct(monkeypatch):
    monkeypatch.delenv("LM_STUDIO_URL", raising=False)
    monkeypatch.delenv("LM_STUDIO_MODEL", raising=False)

    module = _reload_llm()

    assert module.LM_STUDIO_URL == "http://127.0.0.1:1234/v1/chat/completions"
    assert module.QWEN_MODEL == "Qwen2.5-14B-Instruct"


def test_llm_allows_lm_studio_env_overrides(monkeypatch):
    monkeypatch.setenv("LM_STUDIO_URL", "http://localhost:1234/v1/chat/completions")
    monkeypatch.setenv("LM_STUDIO_MODEL", "Qwen2.5-32B-Instruct")

    module = _reload_llm()

    assert module.LM_STUDIO_URL == "http://localhost:1234/v1/chat/completions"
    assert module.QWEN_MODEL == "Qwen2.5-32B-Instruct"

    monkeypatch.delenv("LM_STUDIO_URL", raising=False)
    monkeypatch.delenv("LM_STUDIO_MODEL", raising=False)
    _reload_llm()
