"""Тести, які перевіряють саме ті місця, де інструмент може збрехати або нашкодити."""
import json
import pathlib
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from datapilot.anomaly import decompose
from datapilot.nl2pandas import UnsafeCode, check_code, run_code
from datapilot.profiling import profile_dataframe
from datapilot.verify import verify_text

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"


def test_verifier_catches_invented_number():
    facts = {"revenue_usd": 93750, "payers": 4206}
    good = verify_text("Виручка склала 93 750 доларів", facts)
    bad = verify_text("Виручка склала 150 000 доларів", facts)
    assert good["ok"]
    assert not bad["ok"] and 150000.0 in bad["unverified"]


def test_verifier_tolerates_rounding_and_units():
    facts = {"new_users": 139793, "conversion": 0.0432}
    v = verify_text("Прийшло 139,8 тис. користувачів, конверсія 4.32%", facts)
    assert v["ok"], v


def test_unsafe_code_is_rejected():
    for bad in ["import os\nresult = 1",
                "result = open('/etc/passwd').read()",
                "result = df.__class__.__mro__"]:
        with pytest.raises(UnsafeCode):
            check_code(bad)


def test_code_without_result_is_rejected():
    with pytest.raises(UnsafeCode):
        check_code("df.head()")


def test_generated_code_runs_in_sandbox():
    df = pd.DataFrame({"country": ["UA", "US", "UA"], "amount": [10, 5, 7]})
    res = run_code("result = df.groupby('country')['amount'].sum()", df)
    assert res.loc["UA"] == 17


def test_sandbox_does_not_mutate_source():
    df = pd.DataFrame({"a": [1, 2, 3]})
    run_code("df['a'] = 0\nresult = df", df)
    assert df["a"].tolist() == [1, 2, 3]


def test_profile_flags_missing_values():
    df = pd.DataFrame({"x": [1, None, None, None, 5]})
    prof = profile_dataframe(df, "t")
    assert prof["columns_profile"][0]["missing"] == 3
    assert any("пропуск" in f for f in prof["quality_flags"])


@pytest.mark.skipif(not (DATA / "daily_metrics.csv").exists(), reason="немає демо-даних")
def test_decompose_finds_ios_collapse():
    df = pd.read_csv(DATA / "daily_metrics.csv")
    facts = decompose(df, "new_users", "date",
                      ("2024-04-05", "2024-04-19"), ("2024-04-20", "2024-05-11"), dims=["app"])
    ios = [g for g in facts["by_dimension"]["app"] if g["group"] == "ios"][0]
    assert ios["change_pct"] < -90          # реєстрації на iOS справді обвалились
    assert facts["absolute_change"] > 0     # при цьому загальний притік виріс
