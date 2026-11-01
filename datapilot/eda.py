"""Функція 1 — автоматичний EDA з текстовою інтерпретацією від LLM."""
from __future__ import annotations

import pandas as pd

from .llm import LLMConfig, complete_json, facts_block
from .profiling import profile_dataframe
from .verify import verify_text

SYSTEM = """Ти — старший продуктовий аналітик. Тобі дають JSON із результатами \
профілювання датасету, яке вже виконано кодом на pandas.

Правила, яких не можна порушувати:
1. Ти НЕ рахуєш нічого сам. Усі числа беруться виключно з наданого JSON.
2. Якщо потрібного числа в JSON немає — пиши про це словами, не вигадуй значення.
3. Пиши українською, по суті, без вступів на кшталт "Звичайно!".
4. Інтерпретуй, а не переказуй: не "колонка age має середнє 31", а що це означає \
для продукту і на що це впливає.
5. Одиниці виміру бери з назви колонки: _min — хвилини, _usd — долари США, _cnt — кількість. \
Не переводь їх в інші одиниці й валюти.
6. Не згадуй метрик, яких немає в JSON (наприклад, коефіцієнт відмов чи утримання).

Відповідай ЛИШЕ валідним JSON такої схеми:
{
  "overview": "2-3 речення: що це за дані, за який період, якого масштабу",
  "data_quality": ["проблема якості з числом з JSON", ...],
  "key_observations": ["спостереження, важливе для бізнесу, з числом", ...],
  "suggested_next_steps": ["що аналітику зробити далі", ...]
}"""

USER_TEMPLATE = """Профіль датасету «{name}»:

{facts}

Дай інтерпретацію за схемою. 3-6 пунктів у кожному списку."""

MOCK = {
    "overview": "Офлайн-режим: LLM не викликалась, нижче лише детерміновані факти.",
    "data_quality": ["(mock) див. quality_flags у JSON"],
    "key_observations": ["(mock) запустіть з API-ключем, щоб отримати інтерпретацію"],
    "suggested_next_steps": ["(mock) —"],
}


def auto_eda(df: pd.DataFrame, name: str = "dataset", cfg: LLMConfig | None = None,
             use_llm: bool = True) -> dict:
    facts = profile_dataframe(df, name)
    result = {"facts": facts, "llm": None, "verification": None}
    if not use_llm:
        return result
    llm = complete_json(SYSTEM, USER_TEMPLATE.format(name=name, facts=facts_block(facts)),
                        cfg, mock_payload=MOCK)
    text = " ".join(
        [llm.get("overview", "")]
        + llm.get("data_quality", []) + llm.get("key_observations", [])
        + llm.get("suggested_next_steps", [])
    )
    result["llm"] = llm
    result["verification"] = verify_text(text, facts)
    return result
