"""Шар доступу до LLM.

Інструмент свідомо не прив'язаний до одного провайдера: усі модулі
викликають одну функцію `complete_json`, яка повертає вже розпарсений
словник. Підтримуються Anthropic, OpenAI та режим `mock` (детерміновані
відповіді для тестів і запуску без ключа).
"""
from __future__ import annotations

import json
import os
import re
import textwrap
from dataclasses import dataclass


class LLMError(RuntimeError):
    pass


@dataclass
class LLMConfig:
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-5"
    max_tokens: int = 2000
    temperature: float = 0.2


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
    if not provider:
        if os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        else:
            provider = "mock"
    defaults = {"anthropic": "claude-sonnet-4-5", "openai": "gpt-4o-mini", "mock": "mock"}
    model = model or os.environ.get("DATAPILOT_MODEL") or defaults.get(provider, "mock")
    return LLMConfig(provider=provider, model=model)


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

    if cfg.provider == "openai":
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise LLMError("pip install openai") from exc
        client = OpenAI()
        resp = client.chat.completions.create(
            model=cfg.model, temperature=cfg.temperature,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return _extract_json(resp.choices[0].message.content)

    raise LLMError(f"Невідомий провайдер: {cfg.provider}")


def facts_block(facts: dict, title: str = "ФАКТИ") -> str:
    """Факти подаються моделі як JSON у теґах — так їх важче переплутати
    з інструкціями і легше заборонити вихід за їх межі."""
    return textwrap.dedent(f"""\
        <{title.lower()}>
        {json.dumps(facts, ensure_ascii=False, indent=2, default=str)}
        </{title.lower()}>""")
