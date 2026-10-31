"""Функція 3 — запит природною мовою → код pandas → результат.

Код завжди показується користувачеві: аналітик має бачити логіку, а не
довіряти чорній скриньці.
"""
from __future__ import annotations

import io
import textwrap

import numpy as np
import pandas as pd

from .llm import LLMConfig, complete_json, facts_block
from .profiling import coerce_dates

SYSTEM = """Ти перетворюєш питання аналітика українською на код pandas.

Контекст виконання:
- датафрейм уже завантажений і називається `df`;
- доступні лише `pd` (pandas) та `np` (numpy);
- результат треба покласти у змінну `result` (DataFrame, Series або скаляр).

Жорсткі правила:
1. Жодних імпортів, читання/запису файлів, мережі, os/sys, eval/exec, __ -методів.
2. Користуйся лише тими колонками, які є у наданій схемі. Якщо потрібної колонки \
немає — поверни "code": null і поясни це у "explanation".
3. Якщо рахуєш користувачів — nunique() по id-колонці, а не len().
4. Код має бути коротким і читабельним, без зайвих проміжних змінних.
5. Не друкуй нічого (ніяких print) — тільки присвоєння `result`.

Відповідай ЛИШЕ валідним JSON:
{"code": "рядок з кодом або null", "explanation": "що робить код, українською",
 "assumptions": ["припущення, яке ти зробив", ...]}"""

FORBIDDEN = ("import", "open(", "exec(", "eval(", "__", "os.", "sys.", "subprocess",
             "to_csv", "read_csv", "requests", "socket", "globals(", "locals(")


class UnsafeCode(ValueError):
    pass


def schema_of(df: pd.DataFrame) -> dict:
    out = {"rows": int(len(df)), "columns": []}
    for c in df.columns:
        s = df[c]
        item = {"name": c, "dtype": str(s.dtype)}
        if s.dtype == object or str(s.dtype) in ("str", "string", "category"):
            vals = s.dropna().unique()[:10]
            item["example_values"] = [str(v) for v in vals]
            item["n_unique"] = int(s.nunique(dropna=True))
        elif pd.api.types.is_datetime64_any_dtype(s):
            item["range"] = [str(s.min()), str(s.max())]
        elif pd.api.types.is_numeric_dtype(s):
            item["range"] = [round(float(s.min()), 4), round(float(s.max()), 4)] if s.notna().any() else None
        out["columns"].append(item)
    return out


def check_code(code: str) -> None:
    low = code.lower()
    for bad in FORBIDDEN:
        if bad in low:
            raise UnsafeCode(f"У згенерованому коді є заборонена конструкція: {bad!r}")
    if "result" not in code:
        raise UnsafeCode("Код не присвоює змінну `result`")


def run_code(code: str, df: pd.DataFrame):
    check_code(code)
    env = {"df": df.copy(), "pd": pd, "np": np}
    exec(compile(textwrap.dedent(code), "<datapilot>", "exec"), {"__builtins__": {
        "len": len, "range": range, "min": min, "max": max, "sum": sum,
        "round": round, "sorted": sorted, "abs": abs, "list": list, "dict": dict,
        "set": set, "str": str, "int": int, "float": float, "bool": bool,
        "enumerate": enumerate, "zip": zip, "print": lambda *a, **k: None,
    }}, env)
    return env.get("result")


def ask(df: pd.DataFrame, question: str, cfg: LLMConfig | None = None,
        use_llm: bool = True, max_rows: int = 25) -> dict:
    df = coerce_dates(df)
    schema = schema_of(df)
    out = {"question": question, "schema": schema, "code": None,
           "explanation": None, "assumptions": [], "result": None, "error": None}
    if not use_llm:
        out["error"] = "Режим --no-llm: генерація коду потребує LLM."
        return out

    llm = complete_json(
        SYSTEM,
        f"Схема датафрейму:\n\n{facts_block(schema, 'СХЕМА')}\n\nПитання: {question}",
        cfg,
        mock_payload={"code": "result = df.shape", "explanation": "(mock) розмір таблиці",
                      "assumptions": []})
    out["explanation"] = llm.get("explanation")
    out["assumptions"] = llm.get("assumptions", [])
    code = llm.get("code")
    if not code:
        out["error"] = "Модель не змогла побудувати запит на наявних колонках."
        return out
    out["code"] = code.strip()
    try:
        res = run_code(out["code"], df)
    except UnsafeCode as exc:
        out["error"] = f"Код відхилено перевіркою безпеки: {exc}"
        return out
    except Exception as exc:  # код від LLM може просто не працювати
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out

    if isinstance(res, pd.DataFrame):
        out["result"] = res.head(max_rows)
        out["result_shape"] = list(res.shape)
    elif isinstance(res, pd.Series):
        out["result"] = res.head(max_rows).to_frame()
        out["result_shape"] = [len(res), 1]
    else:
        out["result"] = res
    return out
