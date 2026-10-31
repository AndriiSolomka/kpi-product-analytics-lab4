"""Функція 5 — executive summary для стейкхолдера з уже порахованих метрик."""
from __future__ import annotations

from .llm import LLMConfig, complete_json, facts_block
from .verify import verify_text

SYSTEM = """Ти пишеш короткий апдейт для керівника, який не відкриватиме дашборд.

Тобі дають JSON з уже порахованими метриками.

Жорсткі правила:
1. Усі числа — лише з JSON.
2. Без жаргону аналітиків: не "кагорта з churn rate 40%", а "з кожних десяти \
нових користувачів через тиждень лишається шестеро".
3. Структура "що сталось → чому це важливо → що робити". Рекомендація має бути \
дією, а не спостереженням.
4. Не пом'якшуй погані новини і не перебільшуй хороші.
5. Українською, стисло: summary до 4 речень.

Відповідай ЛИШЕ валідним JSON:
{"summary": "...", "highlights": ["..."], "risks": ["..."],
 "recommendations": [{"action": "...", "why": "...", "owner": "команда або роль"}]}"""

MOCK = {"summary": "(mock) офлайн-режим", "highlights": [], "risks": [], "recommendations": []}


def executive_summary(metrics: dict, audience: str = "product manager",
                      cfg: LLMConfig | None = None, use_llm: bool = True) -> dict:
    res = {"facts": metrics, "llm": None, "verification": None}
    if not use_llm:
        return res
    llm = complete_json(
        SYSTEM,
        f"Аудиторія: {audience}.\n\nМетрики:\n\n{facts_block(metrics)}\n\n"
        "Напиши апдейт за схемою.",
        cfg, mock_payload=MOCK)
    text = " ".join([llm.get("summary", "")] + llm.get("highlights", []) + llm.get("risks", [])
                    + [r.get("why", "") for r in llm.get("recommendations", [])])
    res["llm"] = llm
    res["verification"] = verify_text(text, metrics)
    return res
