"""Детерміноване профілювання датафрейму.

Усі числа рахує pandas. LLM отримує лише готові факти й пише текст —
саме це робить інструмент стійким до галюцинацій.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAX_CATEGORIES = 8


def _kind(s: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    return "categorical"


def coerce_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Колонки з 'date'/'dt' у назві приводимо до datetime — інакше
    будь-який аналіз у часі розсиплеться на рядках."""
    out = df.copy()
    for col in out.columns:
        is_texty = out[col].dtype == object or str(out[col].dtype) in ("str", "string")
        if is_texty and any(t in col.lower() for t in ("date", "_dt", "time")):
            parsed = pd.to_datetime(out[col], errors="coerce")
            if parsed.notna().mean() > 0.9:
                out[col] = parsed
    return out


def column_profile(df: pd.DataFrame) -> list[dict]:
    rows = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        kind = _kind(s)
        item = {
            "column": col,
            "kind": kind,
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing_pct": round(float(s.isna().mean() * 100), 2),
            "unique": int(s.nunique(dropna=True)),
        }
        if kind == "numeric":
            d = s.dropna()
            if len(d):
                q1, q3 = float(d.quantile(0.25)), float(d.quantile(0.75))
                iqr = q3 - q1
                item.update(
                    min=round(float(d.min()), 4), max=round(float(d.max()), 4),
                    mean=round(float(d.mean()), 4), median=round(float(d.median()), 4),
                    std=round(float(d.std()), 4) if len(d) > 1 else 0.0,
                    p25=round(q1, 4), p75=round(q3, 4),
                    zeros_pct=round(float((d == 0).mean() * 100), 2),
                    outliers_iqr=int(((d < q1 - 1.5 * iqr) | (d > q3 + 1.5 * iqr)).sum()),
                )
        elif kind == "datetime":
            d = s.dropna()
            if len(d):
                item.update(min=str(d.min()), max=str(d.max()),
                            span_days=int((d.max() - d.min()).days))
        else:
            vc = s.value_counts(dropna=True).head(MAX_CATEGORIES)
            item["top_values"] = [
                {"value": str(k), "count": int(v), "pct": round(float(v / n * 100), 2)}
                for k, v in vc.items()
            ]
        rows.append(item)
    return rows


def quality_flags(df: pd.DataFrame, profile: list[dict]) -> list[str]:
    """Евристики якості даних. Їх перевіряє код, а не модель."""
    flags = []
    n = len(df)
    dup = int(df.duplicated().sum())
    if dup:
        flags.append(f"Повні дублікати рядків: {dup} ({round(dup / n * 100, 2)}%)")
    for p in profile:
        if p["missing_pct"] > 20:
            flags.append(f"Колонка '{p['column']}': {p['missing_pct']}% пропусків")
        if p["kind"] == "categorical" and p["unique"] == 1:
            flags.append(f"Колонка '{p['column']}' має одне унікальне значення — марна для аналізу")
        if p["kind"] == "numeric" and p.get("zeros_pct", 0) > 60:
            flags.append(f"Колонка '{p['column']}': {p['zeros_pct']}% нулів — перевірте, чи це не пропуски")
        if p["kind"] == "numeric" and p.get("outliers_iqr", 0) > 0:
            share = round(p["outliers_iqr"] / n * 100, 2)
            if share > 5:
                flags.append(f"Колонка '{p['column']}': {p['outliers_iqr']} викидів за IQR ({share}%)")
        if p["kind"] == "numeric" and p.get("min", 0) < 0 and any(
                w in p["column"].lower() for w in ("cnt", "count", "amount", "revenue", "duration", "age")):
            flags.append(f"Колонка '{p['column']}' має від'ємні значення, хоча за змістом не повинна")
    if not flags:
        flags.append("Грубих проблем якості не виявлено")
    return flags


def profile_dataframe(df: pd.DataFrame, name: str = "dataset") -> dict:
    df = coerce_dates(df)
    prof = column_profile(df)
    return {
        "dataset": name,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 ** 2, 2),
        "duplicated_rows": int(df.duplicated().sum()),
        "columns_profile": prof,
        "quality_flags": quality_flags(df, prof),
    }
