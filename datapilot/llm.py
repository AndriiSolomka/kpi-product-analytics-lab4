"""Шар доступу до LLM.

Інструмент свідомо не прив'язаний до одного провайдера: усі модулі
викликають одну функцію `complete_json`, яка повертає вже розпарсений
словник. Підтримуються Anthropic, OpenAI, будь-яка локальна модель з
OpenAI-сумісним API (Ollama, LM Studio, llama.cpp, vLLM) та режим `mock`
(детерміновані відповіді для тестів і запуску без ключа).
"""
from __future__ import annotations

import json
import os
import re
import textwrap
from dataclasses import dataclass


class LLMError(RuntimeError):
    pass


# Типові адреси локальних серверів з OpenAI-сумісним API.
LOCAL_DEFAULT_URLS = {
    "ollama": "http://localhost:11434/v1",
    "lmstudio": "http://localhost:1234/v1",
    "local": "http://localhost:11434/v1",
}


@dataclass
class LLMConfig:
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-5"
    max_tokens: int = 2000
    temperature: float = 0.2
    base_url: str | None = None   # для локальних моделей


def _load_dotenv() -> None:
    """Мінімальний .env-рідер, щоб не тягнути python-dotenv у залежності."""
    for path in (".env", os.path.expanduser("~/.datapilot.env")):
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def resolve_config(provider: str | None = None, model: str | None = None) -> LLMConfig:
    _load_dotenv()
    provider = provider or os.environ.get("DATAPILOT_PROVIDER")
    base_url = os.environ.get("DATAPILOT_BASE_URL")
    if not provider:
        # Своя адреса — найсильніший сигнал: людина свідомо підняла модель локально.
        if base_url:
            provider = "local"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        else:
            provider = "mock"
    if provider in LOCAL_DEFAULT_URLS:
        base_url = base_url or LOCAL_DEFAULT_URLS[provider]
    defaults = {"anthropic": "claude-sonnet-4-5", "openai": "gpt-4o-mini",
                "ollama": "llama3.1", "lmstudio": "local-model", "local": "llama3.1",
                "mock": "mock"}
    model = model or os.environ.get("DATAPILOT_MODEL") or defaults.get(provider, "mock")
    return LLMConfig(provider=provider, model=model, base_url=base_url)


def _extract_json(text: str) -> dict:
    """LLM іноді обгортає JSON у ```json ... ``` або додає текст навколо."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            raise LLMError("Модель повернула не-JSON:\n" + text[:500])
        return json.loads(text[start:end + 1])


def complete_json(system: str, user: str, cfg: LLMConfig | None = None,
                  mock_payload: dict | None = None) -> dict:
    """Один виклик LLM, що гарантовано повертає словник."""
    cfg = cfg or resolve_config()

    if cfg.provider == "mock":
        if mock_payload is None:
            raise LLMError("Режим mock увімкнено, але модуль не передав mock_payload.")
        return mock_payload

    if cfg.provider == "anthropic":
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise LLMError("pip install anthropic") from exc
        client = anthropic.Anthropic()
        # Префіл '{' змушує модель одразу почати з JSON і прибирає преамбули.
        resp = client.messages.create(
            model=cfg.model, max_tokens=cfg.max_tokens, temperature=cfg.temperature,
            system=system,
            messages=[{"role": "user", "content": user},
                      {"role": "assistant", "content": "{"}],
        )
        return _extract_json("{" + resp.content[0].text)

    if cfg.provider in ("openai", "local", "ollama", "lmstudio"):
        return _openai_compatible(system, user, cfg)

    raise LLMError(f"Невідомий провайдер: {cfg.provider}")


def _openai_compatible(system: str, user: str, cfg: LLMConfig) -> dict:
    """OpenAI та будь-який локальний сервер, що повторює її API.

    Локальні збірки по-різному підтримують json-режим: частина приймає
    response_format, частина падає з помилкою. Тому спершу пробуємо
    строгий режим, а на помилці повторюємо без нього — JSON усе одно
    витягується з тексту.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise LLMError("pip install openai") from exc

    is_local = cfg.provider != "openai"
    # Локальні сервери ключ не перевіряють, але SDK вимагає непорожній рядок.
    client = OpenAI(base_url=cfg.base_url,
                    api_key=os.environ.get("OPENAI_API_KEY") or ("local" if is_local else None))
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    kwargs = dict(model=cfg.model, temperature=cfg.temperature, messages=messages)
    try:
        resp = client.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
    except Exception:
        if not is_local:
            raise
        resp = client.chat.completions.create(**kwargs)
    return _extract_json(resp.choices[0].message.content)


def facts_block(facts: dict, title: str = "ФАКТИ") -> str:
    """Факти подаються моделі як JSON у теґах — так їх важче переплутати
    з інструкціями і легше заборонити вихід за їх межі."""
    return textwrap.dedent(f"""\
        <{title.lower()}>
        {json.dumps(facts, ensure_ascii=False, indent=2, default=str)}
        </{title.lower()}>""")
