"""CLI: datapilot <команда> ..."""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from . import __version__
from .anomaly import explain_metric_change
from .eda import auto_eda
from .hypotheses import suggest_hypotheses
from .llm import LLMError, resolve_config
from .nl2pandas import ask
from .profiling import coerce_dates
from .summary import executive_summary

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
RED, GREEN, YELLOW = "\033[31m", "\033[32m", "\033[33m"


def h(text: str) -> None:
    print(f"\n{BOLD}{text}{RESET}")


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        print(f"{RED}Файл {path} не містить жодного рядка — аналізувати нічого.{RESET}")
        sys.exit(2)
    return coerce_dates(df)


def show_verification(v: dict | None) -> None:
    if not v:
        return
    if v["ok"]:
        print(f"\n{GREEN}✔ Перевірка чисел: усі {v['numbers_found']} числа з тексту "
              f"знайдено у фактах.{RESET}")
    else:
        print(f"\n{YELLOW}⚠ Перевірка чисел: {len(v['unverified'])} з {v['numbers_found']} "
              f"не знайдено у фактах → {v['unverified']}{RESET}")
        print(f"{DIM}  Ці значення модель, найімовірніше, вигадала — не вставляйте їх у звіт.{RESET}")


def cmd_eda(a) -> None:
    df = load(a.csv)
    res = auto_eda(df, name=a.csv.split("/")[-1], use_llm=not a.no_llm)
    f = res["facts"]
    h("Профіль датасету")
    print(f"Рядків: {f['rows']:,}   Колонок: {f['columns']}   "
          f"Дублікатів: {f['duplicated_rows']:,}   Пам'ять: {f['memory_mb']} МБ")
    h("Колонки")
    rows = [{"колонка": c["column"], "тип": c["kind"], "пропуски, %": c["missing_pct"],
             "унікальних": c["unique"]} for c in f["columns_profile"]]
    print(pd.DataFrame(rows).to_string(index=False))
    h("Прапорці якості даних (пораховано кодом)")
    for flag in f["quality_flags"]:
        print(" •", flag)
    if res["llm"]:
        llm = res["llm"]
        h("Інтерпретація LLM")
        print(llm.get("overview", ""))
        for title, key in (("Якість даних", "data_quality"),
                           ("Ключові спостереження", "key_observations"),
                           ("Що робити далі", "suggested_next_steps")):
            items = llm.get(key) or []
            if items:
                h(title)
                for i in items:
                    print(" •", i)
        show_verification(res["verification"])
    if a.json:
        open(a.json, "w", encoding="utf-8").write(
            json.dumps(res, ensure_ascii=False, indent=2, default=str))
        print(f"\n{DIM}JSON збережено у {a.json}{RESET}")


def cmd_hypotheses(a) -> None:
    df = load(a.csv)
    res = suggest_hypotheses(df, name=a.csv.split("/")[-1], use_llm=not a.no_llm)
    if a.no_llm:
        h("Зрізи, на яких будуються гіпотези")
        print(json.dumps(res["facts"], ensure_ascii=False, indent=2, default=str)[:4000])
        return
    h(f"Гіпотези ({len(res['hypotheses'])})")
    for i, item in enumerate(res["hypotheses"], 1):
        mark = "" if item["_verification"]["ok"] else f" {YELLOW}[число не підтверджено]{RESET}"
        print(f"\n{BOLD}{i}. [{item.get('priority', '?')}] {item.get('title', '')}{RESET}{mark}")
        print(f"   {DIM}Підстава:{RESET} {item.get('evidence', '')}")
        print(f"   {DIM}Чому важливо:{RESET} {item.get('why_it_matters', '')}")
        print(f"   {DIM}Як перевірити:{RESET} {item.get('how_to_check', '')}")
    if res["unverified_count"]:
        print(f"\n{YELLOW}⚠ {res['unverified_count']} гіпотез містять числа, яких немає "
              f"у зрізах — перевірте їх вручну.{RESET}")
    else:
        print(f"\n{GREEN}✔ Усі числа в гіпотезах звірено зі зрізами.{RESET}")
    if a.json:
        open(a.json, "w", encoding="utf-8").write(
            json.dumps(res, ensure_ascii=False, indent=2, default=str))


def cmd_ask(a) -> None:
    df = load(a.csv)
    res = ask(df, a.question, use_llm=not a.no_llm)
    if res["error"] and not res["code"]:
        print(f"{RED}{res['error']}{RESET}")
        sys.exit(1)
    h("Згенерований код")
    print(res["code"])
    if res.get("explanation"):
        h("Пояснення")
        print(res["explanation"])
    for note in res.get("assumptions") or []:
        print(f"{DIM} • припущення: {note}{RESET}")
    if res["error"]:
        print(f"\n{RED}Помилка виконання: {res['error']}{RESET}")
        sys.exit(1)
    h("Результат")
    r = res["result"]
    print(r.to_string() if isinstance(r, pd.DataFrame) else r)
    if res.get("result_shape"):
        print(f"{DIM}(рядків усього: {res['result_shape'][0]}){RESET}")


def cmd_explain(a) -> None:
    df = load(a.csv)
    res = explain_metric_change(df, a.metric, a.date_col,
                                (a.base_from, a.base_to), (a.compare_from, a.compare_to),
                                dims=a.dims.split(",") if a.dims else None,
                                use_llm=not a.no_llm)
    f = res["facts"]
    h(f"Метрика {f['metric']}")
    print(f"Базовий період   {f['base_period']['from']}..{f['base_period']['to']}: "
          f"{f['base_period']['total']:,}")
    print(f"Порівнюваний     {f['compare_period']['from']}..{f['compare_period']['to']}: "
          f"{f['compare_period']['total']:,}")
    print(f"Зміна: {f['absolute_change']:,} ({f['relative_change_pct']}%)")
    for dim, rows in f["by_dimension"].items():
        h(f"Внесок сегментів: {dim}")
        print(pd.DataFrame(rows).to_string(index=False))
    if res["llm"]:
        llm = res["llm"]
        h("Висновок LLM")
        print(llm.get("headline", ""))
        if llm.get("main_drivers"):
            h("Головні драйвери")
            for d in llm["main_drivers"]:
                print(" •", d)
        if llm.get("likely_causes"):
            h("Можливі причини (гіпотези)")
            for c in llm["likely_causes"]:
                print(f" • [{c.get('confidence', '?')}] {c.get('hypothesis', '')}")
                print(f"   {DIM}{c.get('supporting_numbers', '')}{RESET}")
        if llm.get("what_to_check_next"):
            h("Що перевірити далі")
            for s in llm["what_to_check_next"]:
                print(" •", s)
        show_verification(res["verification"])
    if a.json:
        open(a.json, "w", encoding="utf-8").write(
            json.dumps(res, ensure_ascii=False, indent=2, default=str))


def cmd_summary(a) -> None:
    metrics = json.load(open(a.metrics, encoding="utf-8"))
    res = executive_summary(metrics, audience=a.audience, use_llm=not a.no_llm)
    if not res["llm"]:
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return
    llm = res["llm"]
    h("Executive summary")
    print(llm.get("summary", ""))
    for title, key in (("Головне", "highlights"), ("Ризики", "risks")):
        if llm.get(key):
            h(title)
            for i in llm[key]:
                print(" •", i)
    if llm.get("recommendations"):
        h("Рекомендації")
        for r in llm["recommendations"]:
            print(f" • {r.get('action', '')} — {r.get('why', '')} "
                  f"{DIM}[{r.get('owner', '')}]{RESET}")
    show_verification(res["verification"])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="datapilot",
                                description="AI-асистент продуктового аналітика")
    p.add_argument("--version", action="version", version=f"datapilot {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--no-llm", action="store_true",
                        help="лише детерміновані розрахунки, без звернення до моделі")
        sp.add_argument("--json", help="зберегти повний результат у JSON-файл")

    s = sub.add_parser("eda", help="автоматичний EDA з інтерпретацією")
    s.add_argument("csv"); common(s); s.set_defaults(func=cmd_eda)

    s = sub.add_parser("hypotheses", help="гіпотези, обґрунтовані числами")
    s.add_argument("csv"); common(s); s.set_defaults(func=cmd_hypotheses)

    s = sub.add_parser("ask", help="питання природною мовою → код pandas → результат")
    s.add_argument("csv"); s.add_argument("question"); common(s); s.set_defaults(func=cmd_ask)

    s = sub.add_parser("explain", help="пояснення зміни метрики через внески сегментів")
    s.add_argument("csv")
    s.add_argument("--metric", required=True)
    s.add_argument("--date-col", required=True)
    s.add_argument("--base-from", required=True); s.add_argument("--base-to", required=True)
    s.add_argument("--compare-from", required=True); s.add_argument("--compare-to", required=True)
    s.add_argument("--dims", help="перелік розрізів через кому (за замовчуванням — усі придатні)")
    common(s); s.set_defaults(func=cmd_explain)

    s = sub.add_parser("summary", help="executive summary з JSON-метрик")
    s.add_argument("metrics"); s.add_argument("--audience", default="product manager")
    common(s); s.set_defaults(func=cmd_summary)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    cfg = resolve_config()
    if not getattr(args, "no_llm", False):
        print(f"{DIM}LLM: {cfg.provider} / {cfg.model}{RESET}", file=sys.stderr)
        if cfg.provider == "mock":
            print(f"{YELLOW}Локальну модель не знайдено — працюю в режимі mock. "
                  f"Запустіть LM Studio (lms server start) або Ollama, чи задайте API-ключ у .env.{RESET}",
                  file=sys.stderr)
    try:
        args.func(args)
    except LLMError as exc:
        print(f"{RED}Помилка LLM: {exc}{RESET}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
