"""Перевірка чисел, які назвала модель.

LLM упевнено вигадує цифри. Тому кожне число з її відповіді ми
витягуємо регуляркою і шукаємо серед чисел, які реально є у фактах.
Якщо числа немає — воно позначається як неперевірене, і користувач
бачить це у звіті.
"""
from __future__ import annotations

import re

# Числа у тексті моделі бувають із розділювачами розрядів («93 750», «1,234,567»)
# і з комою як десятковою («139,8»). Спочатку ловимо згруповані, потім звичайні.
NUM_RE = re.compile(
    r"-?\d{1,3}(?:[\u00a0\u202f ]\d{3})+(?:[.,]\d+)?"   # 93 750 / 1 234 567,5
    r"|-?\d{1,3}(?:,\d{3})+(?:\.\d+)?"                  # 1,234,567.5
    r"|-?\d+(?:[.,]\d+)?"                                # 139,8 / 4.32
)


def _to_float(token: str) -> float:
    """Нормалізує числовий токен з тексту у float."""
    t = token.replace("\u00a0", "").replace("\u202f", "").replace(" ", "")
    if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?", t):
        t = t.replace(",", "")          # кома як розділювач розрядів
    else:
        t = t.replace(",", ".")         # кома як десяткова
    return float(t)
# Дрібні цілі (роки, номери пунктів, 1-2 рази) не несуть фактичного
# навантаження і лише засмічували б список неперевірених.
IGNORE_SMALL_INTS = set(range(0, 13)) | {24, 100, 2023, 2024, 2025, 2026}


def collect_numbers(obj, acc: set[float] | None = None) -> set[float]:
    """Усі числа, що є у структурі фактів — на будь-якій глибині."""
    acc = set() if acc is None else acc
    if isinstance(obj, bool):
        return acc
    if isinstance(obj, (int, float)):
        acc.add(round(float(obj), 6))
    elif isinstance(obj, dict):
        for v in obj.values():
            collect_numbers(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            collect_numbers(v, acc)
    elif isinstance(obj, str):
        for m in NUM_RE.findall(obj):
            try:
                acc.add(round(_to_float(m), 6))
            except ValueError:
                pass
    return acc


def _matches(value: float, allowed: set[float], tol: float) -> bool:
    # Знак модель передає словами («падіння на 98.98%» для -98.98), тому порівнюємо модулі.
    value = abs(value)
    for a in map(abs, allowed):
        if abs(a - value) <= max(tol * abs(a), 1e-9):
            return True
        # Модель часто округлює: 139793 -> 139.8 тис., 0.7254 -> 72.5%
        for scaled in (a / 1000.0, a * 100.0, a / 100.0, a / 1_000_000.0):
            if abs(scaled - value) <= max(tol * abs(scaled), 1e-9):
                return True
    return False


def verify_text(text: str, facts, tol: float = 0.02) -> dict:
    """Повертає список чисел з тексту та ті з них, яких немає у фактах."""
    allowed = collect_numbers(facts)
    found, unverified = [], []
    for m in NUM_RE.findall(text or ""):
        try:
            v = _to_float(m)
        except ValueError:
            continue
        found.append(v)
        if v.is_integer() and int(v) in IGNORE_SMALL_INTS:
            continue
        if not _matches(v, allowed, tol):
            unverified.append(v)
    return {
        "numbers_found": len(found),
        "unverified": sorted(set(unverified)),
        "ok": not unverified,
    }


def verify_items(items: list[dict], facts, fields: tuple[str, ...], tol: float = 0.02) -> list[dict]:
    """Та сама перевірка, але для списку структурованих елементів."""
    out = []
    for item in items:
        text = " ".join(str(item.get(f, "")) for f in fields)
        check = verify_text(text, facts, tol)
        item = dict(item)
        item["_verification"] = check
        out.append(item)
    return out
