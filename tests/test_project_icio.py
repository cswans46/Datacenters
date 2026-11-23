"""End-to-end test of project_icio.py on synthetic data (no RData needed)."""
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from icio_projection import synthetic  # noqa: E402
from icio_projection.table import to_wide  # noqa: E402


def test_project_icio_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pyreadr", types.ModuleType("pyreadr"))
    import test_canada_dsut_split as tc
    import project_icio

    hist = synthetic.make_history(seed=2)
    synthetic.write_history(hist, tmp_path / "cp", tmp_path / "pyp")
    data = tmp_path / "data"
    data.mkdir()
    tc.make_sut_csv(data / "sut.csv")
    tc.make_dsut_csv(data / "canada_dsut_2020.csv")
    rows = [{"country": c, "year": y, "dc_capacity_mw": cap0 + 100 * (y - 2026), "price_mn_MW_2022": 2.0,
             "output_mn_usd_per_mw_2022": 2.0 * (cap0 + 100 * (y - 2026))}
            for c, cap0 in (("CAN", 300), ("IND", 200)) for y in range(2026, 2031)]
    pd.DataFrame(rows).to_csv(data / "dc_output_scenarios.csv", index=False)
    ext = tmp_path / "external.csv"
    synthetic.synthetic_external().to_csv(ext, index=False)
    out = tmp_path / "out"

    project_icio.main(["build-panel", "--cp-dir", str(tmp_path / "cp"), "--pyp-dir", str(tmp_path / "pyp"),
                       "--out", str(tmp_path / "panel")])
    project_icio.main(["run", "--panel", str(tmp_path / "panel"), "--external", str(ext), "--out", str(out),
                       "--base-csv", str(tmp_path / "cp" / "2022.csv"), "--backtest-origins", "2014",
                       "--data-dir", str(data), "--output-dir", str(tmp_path / "mout"),
                       "--split-canada-dsut", "--statcan-sut-file", str(data / "sut.csv")])

    cmp_ = pd.read_csv(out / "comparison" / "by_industry.csv", index_col=0)
    assert {"oecd", "arima", "external"} <= set(cmp_.columns)
    assert {"CAN_J62_63_CLD", "CAN_J62_63_DPH", "CAN_J62_63_OTH"} <= set(cmp_.index)
    assert "CAN_J62_63" not in cmp_.index and "IND_J62_63" in cmp_.index
    # data-centre driver: 2030 DC volume = 2022 DC output * parent growth to 2026 + 400 (scenario increment)
    m = pd.read_csv(out / "margins_arima.csv.gz")
    pyp = m[(m.prices == "pyp") & (m["var"] == "x")]
    assert (out / "arima" / "2030_cp.npz").exists()
    ras = pd.read_csv(out / "ras_log_oecd.csv")
    assert (ras["error"] < 1e-4).all()
    bt = pd.read_csv(out / "backtest" / "backtest_by_horizon.csv")
    assert set(bt["method"]) >= {"arima", "oecd_expost", "naive"}


def test_dc_scenario_index(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pyreadr", types.ModuleType("pyreadr"))
    import project_icio
    from types import SimpleNamespace
    pd.DataFrame([{"country": "CAN", "year": y, "dc_capacity_mw": 100 + 50 * (y - 2026), "price_mn_MW_2022": 2.0,
                   "output_mn_usd_per_mw_2022": 0} for y in range(2026, 2031)]).to_csv(tmp_path / "dc_output_scenarios.csv", index=False)
    plan = SimpleNamespace(cloud_key="CAN_J62_63_CLD", hosting_key="CAN_J62_63_DPH")
    base = SimpleNamespace(x=lambda: pd.Series({"CAN_J62_63_CLD": 300.0, "CAN_J62_63_DPH": 700.0}))
    gpar = pd.Series({y: 1.02 ** (y - 2022) for y in range(2023, 2031)})
    idx = project_icio.dc_scenario_index(tmp_path, "CAN", base, plan, gpar)
    assert idx.loc["CAN_J62_63_CLD", 2025] == pytest.approx(1.02 ** 3)
    # 2030: 1000 * 1.02^4 + (300 - 100) * 2 = additional 400 USD mn
    assert idx.loc["CAN_J62_63_CLD", 2030] == pytest.approx((1000 * 1.02 ** 4 + 400) / 1000)
    assert (idx.loc["CAN_J62_63_CLD"] == idx.loc["CAN_J62_63_DPH"]).all()
