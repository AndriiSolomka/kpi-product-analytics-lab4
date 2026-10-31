"""Функція 2 — генерація гіпотез, обґрунтованих числами з датасету."""
from __future__ import annotations

import pandas as pd

from .insights import build_fact_bundle
from .llm import LLMConfig, complete_json, facts_block
from .verify import verify_items

SYSTEM = """Ти — продуктовий аналітик, який готує список гіпотез для перевірки.

Тобі дають JSON із уже порахованими зрізами: середні метрики по сегментах, \
розкид між найкращою і найгіршою групою, кореляції та денна динаміка з аномаліями.

Жорсткі правила:
1. Кожна гіпотеза ОБОВ'ЯЗКОВО спирається на конкретне число з наданого JSON. \
Гіпотеза без числа — брак.
2. Не вигадуй числа, яких у JSON немає. Краще менше гіпотез, ніж вигадані.
3. Не пиши банальностей ("варто глибше дослідити дані"). Гіпотеза має називати \
конкретний сегмент або дату і конкретну можливу причину.
4. Формулюй так, щоб гіпотезу можна було спростувати перевіркою.
5. Українською.

Відповідай ЛИШЕ валідним JSON:
{
  "hypotheses": [
    {
      "title": "коротке формулювання",
      "evidence": "число(а) з JSON, які на це вказують",
      "why_it_matters": "що це означає для продукту",
      "how_to_check": "конкретна наступна перевірка",
      "priority": "high|medium|low"
    }
  ]
}
Від 5 до 10 гіпотез, відсортованих за пріоритетом."""


def _mock(n: int = 5) -> dict:
    return {"hypotheses": [{
        "title": f"(mock) гіпотеза {i + 1}", "evidence": "0",
        "why_it_matters": "офлайн-режим", "how_to_check": "запустіть з API-ключем",
        "priority": "low"} for i in range(n)]}


def suggest_hypotheses(df: pd.DataFrame, name: str = "dataset",
                       cfg: LLMConfig | None = None, use_llm: bool = True) -> dict:
    facts = build_fact_bundle(df, name)
    result = {"facts": facts, "hypotheses": [], "unverified_count": 0}
    if not use_llm:
        return result
    llm = complete_json(
        SYSTEM,
        f"Зрізи по датасету «{name}»:\n\n{facts_block(facts)}\n\n"
        "Сформулюй гіпотези за схемою.",
        cfg, mock_payload=_mock())
    items = verify_items(llm.get("hypotheses", []), facts, ("title", "evidence"))
    result["hypotheses"] = items
    result["unverified_count"] = sum(1 for i in items if not i["_verification"]["ok"])
    return result
