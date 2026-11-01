"""Функція 4 — пояснення провалу/стрибка метрики через внески сегментів."""
from __future__ import annotations

import pandas as pd

from .llm import LLMConfig, complete_json, facts_block
from .profiling import coerce_dates
from .verify import verify_text

SYSTEM = """Ти пояснюєш зміну продуктової метрики.

Тобі дають JSON: значення метрики у базовому і порівнюваному періодах та \
декомпозицію зміни по сегментах — скільки абсолютної зміни дав кожен сегмент і \
який його внесок у відсотках.

Жорсткі правила:
1. Усі числа — лише з JSON. Нічого не перераховуй і не вигадуй.
2. Спочатку назви сегменти, що дали найбільший внесок, і лише потім — можливу причину.
3. Чітко відокремлюй факт від гіпотези: факт — це число з JSON, причина — твоє \
припущення, і воно має бути позначене як гіпотеза.
4. Якщо зміна рівномірно розмазана по всіх сегментах — так і скажи, це інший \
діагноз, ніж локальна поломка.
5. Окремо назви сегменти зі значним падінням (від'ємна change), навіть якщо загалом \
метрика зросла: загальний ріст часто маскує поломку в окремому сегменті.
6. Усі текстові поля, зокрема supporting_numbers, — українською.

Відповідай ЛИШЕ валідним JSON:
{
  "headline": "одне речення: що сталося з метрикою",
  "main_drivers": ["сегмент + його внесок з JSON", ...],
  "likely_causes": [{"hypothesis": "...", "supporting_numbers": "...", "confidence": "high|medium|low"}],
  "what_to_check_next": ["конкретна дія", ...]
}"""

MOCK = {"headline": "(mock) офлайн-режим", "main_drivers": [],
        "likely_causes": [], "what_to_check_next": []}


def decompose(df: pd.DataFrame, metric: str, date_col: str,
              base: tuple[str, str], compare: tuple[str, str],
              dims: list[str] | None = None, top: int = 6) -> dict:
    """Скільки кожен сегмент додав або забрав від зміни метрики."""
    df = coerce_dates(df)
    d = df[[date_col, metric] + [c for c in df.columns if c not in (date_col, metric)]].copy()
    d[date_col] = pd.to_datetime(d[date_col])

    m_base = d[(d[date_col] >= base[0]) & (d[date_col] <= base[1])]
    m_cmp = d[(d[date_col] >= compare[0]) & (d[date_col] <= compare[1])]
    if m_base.empty or m_cmp.empty:
        raise ValueError("Один із періодів порожній — перевірте дати та колонку дати.")

    total_base = float(m_base[metric].sum())
    total_cmp = float(m_cmp[metric].sum())
    delta = total_cmp - total_base

    if dims is None:
        dims = [c for c in df.columns
                if not pd.api.types.is_numeric_dtype(df[c])
                and not pd.api.types.is_datetime64_any_dtype(df[c])
                and 2 <= df[c].nunique(dropna=True) <= 300]

    out = {
        "metric": metric, "date_col": date_col,
        "base_period": {"from": base[0], "to": base[1], "total": round(total_base, 2),
                        "days": int(m_base[date_col].dt.date.nunique())},
        "compare_period": {"from": compare[0], "to": compare[1], "total": round(total_cmp, 2),
                           "days": int(m_cmp[date_col].dt.date.nunique())},
        "absolute_change": round(delta, 2),
        "relative_change_pct": round(delta / total_base * 100, 2) if total_base else None,
        "by_dimension": {},
    }
    for dim in dims:
        b = m_base.groupby(dim)[metric].sum()
        c = m_cmp.groupby(dim)[metric].sum()
        both = pd.concat([b.rename("base_total"), c.rename("compare_total")], axis=1).fillna(0)
        both["change"] = both["compare_total"] - both["base_total"]
        both["contribution_pct"] = (both["change"] / delta * 100).round(2) if delta else 0.0
        base_nz = both["base_total"].where(both["base_total"] != 0)
        both["change_pct"] = (both["change"] / base_nz * 100).astype(float).round(2)
        both = both.reindex(both["change"].abs().sort_values(ascending=False).index).head(top)
        out["by_dimension"][dim] = [
            {"group": str(idx),
             "base": round(float(row["base_total"]), 2),
             "compare": round(float(row["compare_total"]), 2),
             "change": round(float(row["change"]), 2),
             "change_pct": None if pd.isna(row["change_pct"]) else float(row["change_pct"]),
             "contribution_pct": float(row["contribution_pct"])}
            for idx, row in both.iterrows()
        ]
    return out


def explain_metric_change(df: pd.DataFrame, metric: str, date_col: str,
                          base: tuple[str, str], compare: tuple[str, str],
                          dims: list[str] | None = None,
                          cfg: LLMConfig | None = None, use_llm: bool = True) -> dict:
    facts = decompose(df, metric, date_col, base, compare, dims)
    res = {"facts": facts, "llm": None, "verification": None}
    if not use_llm:
        return res
    llm = complete_json(
        SYSTEM,
        f"Декомпозиція зміни метрики:\n\n{facts_block(facts)}\n\nПоясни за схемою.",
        cfg, mock_payload=MOCK)
    text = " ".join([llm.get("headline", "")] + llm.get("main_drivers", [])
                    + [c.get("supporting_numbers", "") for c in llm.get("likely_causes", [])])
    res["llm"] = llm
    res["verification"] = verify_text(text, facts)
    return res
