"""Детерміновані зрізи, на яких будується генерація гіпотез."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .profiling import coerce_dates

MAX_GROUPS = 12


# Колонки, що зазвичай і є продуктовими метриками. Якщо їх не підняти вгору,
# у зрізи потрапляють перші-ліпші числові колонки (вік, лічильники) і гіпотези
# виходять про що завгодно, тільки не про гроші й утримання.
PRIORITY = ("revenue", "amount", "payer", "payment", "conversion", "retention",
            "duration", "session", "active", "activity", "contact", "new_users", "dau")


def _numeric_cols(df: pd.DataFrame, limit: int | None = None) -> list[str]:
    cols = [c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) >= 2
            and not c.lower().endswith("_id")]

    def rank(name: str) -> int:
        low = name.lower()
        for i, key in enumerate(PRIORITY):
            if key in low:
                return i
        return len(PRIORITY)

    cols.sort(key=lambda c: (rank(c), df.columns.get_loc(c)))
    return cols[:limit] if limit else cols


def _segment_cols(df: pd.DataFrame) -> list[str]:
    out = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        nu = df[c].nunique(dropna=True)
        if 2 <= nu <= MAX_GROUPS:
            out.append(c)
    return out


def segment_cuts(df: pd.DataFrame, metrics: list[str] | None = None) -> list[dict]:
    """Для кожного сегмента — середні значення ключових метрик і розкид
    між найкращою і найгіршою групою."""
    metrics = metrics or _numeric_cols(df, limit=6)
    cuts = []
    for seg in _segment_cols(df):
        g = df.groupby(seg, dropna=True)
        rec = {"segment": seg, "groups": []}
        sizes = g.size()
        for name, size in sizes.items():
            row = {"group": str(name), "rows": int(size),
                   "share_pct": round(float(size / len(df) * 100), 2)}
            for m in metrics:
                row[f"avg_{m}"] = round(float(g[m].mean().loc[name]), 4)
            rec["groups"].append(row)
        for m in metrics:
            means = g[m].mean()
            if means.notna().sum() >= 2 and abs(means.max()) > 1e-9:
                rec.setdefault("spread", {})[m] = {
                    "best_group": str(means.idxmax()), "best_value": round(float(means.max()), 4),
                    "worst_group": str(means.idxmin()), "worst_value": round(float(means.min()), 4),
                    "ratio": round(float(means.max() / means.min()), 2) if means.min() > 0 else None,
                }
        cuts.append(rec)
    return cuts


def correlations(df: pd.DataFrame, top: int = 8) -> list[dict]:
    num = df[_numeric_cols(df)]
    if num.shape[1] < 2:
        return []
    corr = num.corr(numeric_only=True).abs()
    pairs = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            v = corr.loc[a, b]
            if pd.notna(v):
                pairs.append({"a": a, "b": b, "abs_corr": round(float(v), 3)})
    return sorted(pairs, key=lambda p: -p["abs_corr"])[:top]


def time_anomalies(df: pd.DataFrame, date_col: str | None = None,
                   metrics: list[str] | None = None, z: float = 2.5) -> dict:
    """Денна динаміка + точки, що відхиляються більш ніж на z сигм."""
    df = coerce_dates(df)
    if date_col is None:
        cands = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        if not cands:
            return {}
        date_col = cands[0]
    metrics = metrics or _numeric_cols(df, limit=4)
    daily = df.set_index(date_col).resample("D")[metrics].sum(min_count=1)
    out = {"date_col": date_col,
           "period": {"from": str(daily.index.min().date()), "to": str(daily.index.max().date())},
           "series": {}}
    for m in metrics:
        s = daily[m].dropna()
        if len(s) < 5:
            continue
        mu, sd = float(s.mean()), float(s.std())
        spikes = [] if sd == 0 else [
            {"date": str(d.date()), "value": round(float(v), 2),
             "z": round(float((v - mu) / sd), 2)}
            for d, v in s.items() if abs((v - mu) / sd) >= z
        ]
        first, last = float(s.iloc[0]), float(s.iloc[-1])
        out["series"][m] = {
            "mean": round(mu, 2), "std": round(sd, 2),
            "first_day": round(first, 2), "last_day": round(last, 2),
            "change_pct": round((last - first) / first * 100, 2) if first else None,
            "max": {"date": str(s.idxmax().date()), "value": round(float(s.max()), 2)},
            "min": {"date": str(s.idxmin().date()), "value": round(float(s.min()), 2)},
            "anomalies": spikes[:10],
        }
    return out


def build_fact_bundle(df: pd.DataFrame, name: str = "dataset") -> dict:
    df = coerce_dates(df)
    return {
        "dataset": name,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "segment_cuts": segment_cuts(df),
        "correlations": correlations(df),
        "time": time_anomalies(df),
    }
