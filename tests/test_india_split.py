"""India IOT split of IND_62T63 and RAS re-balancing (synthetic data)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

openpyxl = pytest.importorskip("openpyxl")

import canada_dsut_split as cds  # noqa: E402
import india_iot_split as iis  # noqa: E402
import split_rebalance as sr  # noqa: E402
import test_canada_dsut_split as tc  # noqa: E402

TO_2025 = {"J62_63": "62T63", "D35": "35", "K": "64T66"}


def icio_2025():
    A, X, FD = tc.make_icio()
    keys = [f"{k.split('_', 1)[0]}_{TO_2025.get(k.split('_', 1)[1], k.split('_', 1)[1])}" for k in A.index]
    A.index = A.columns = keys
    X.index = keys
    FD.index = keys
    return A, X, FD


IOT_SECTORS = ["Agriculture and Allied sectors", "Other electricity sources",
               "Financial and Insurance services", "Computer related services", "AI-DC"]


def make_iot(path: Path) -> Path:
    Z = np.array([
        [50, 5, 4, 3, 0.5],
        [10, 20, 6, 8, 2.0],      # electricity
        [5, 4, 30, 9, 1.0],
        [3, 2, 7, 40, 4.0],       # computer services
        [0.5, 0.5, 1.0, 1.0, 0.0],  # AI-DC sales (all intermediate)
    ])
    va = np.array([100, 60, 80, 90, 2.5])
    out = Z.sum(axis=0) + va
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Z RAS Final"
    ws.append(["RAS-balanced IOT"])
    ws.append(["Sr no.", "Sectors", *IOT_SECTORS, "Intermediate Consumption", "Old Total Output"])
    for i, name in enumerate(IOT_SECTORS):
        old = out[i] - (Z[i, -1] if name == "Computer related services" else 0.0)
        ws.append([i + 1, name, *Z[i].tolist(), float(Z[i].sum()), float(old)])
    ws.append([None, "Value Added", *va.tolist()])
    ws.append([None, "Old Output", *out.tolist()])
    ws.append([None, "New Output", *out.tolist()])
    wb.save(path)
    return path


def test_load_and_plan(tmp_path):
    plan = iis.prepare_india_split(make_iot(tmp_path / "iot.xlsx"), parent_sector="62T63")
    assert plan.dc_key == "IND_62T63_DC"
    out_dc = 0.5 + 2 + 1 + 4 + 0 + 2.5
    assert plan.dc_va_share == pytest.approx(2.5 / out_dc)
    assert plan.dc_input_coefficients["Other electricity sources"] == pytest.approx(2 / out_dc)
    assert 0 < plan.weight < 0.1


@pytest.mark.parametrize("rebalance", [True, False])
def test_split_aggregates_and_balances(tmp_path, rebalance):
    A, X, FD = icio_2025()
    plan = iis.prepare_india_split(make_iot(tmp_path / "iot.xlsx"), parent_sector="62T63", dc_share_override=0.05)
    A2, X2, FD2, diag = iis.split_icio_india(A, X, FD, plan, rebalance=rebalance)
    assert {"IND_62T63_DC", "IND_62T63_OTH"} <= set(A2.index)
    assert X2["IND_62T63_DC"] == pytest.approx(0.05 * X["IND_62T63"])
    # Table stays balanced and non-negative.
    summary, negative = sr.balance_report(A2, X2, FD2)
    summary0, _ = sr.balance_report(A, X, FD)
    assert summary["max_row_identity_relative_error"] <= summary0["max_row_identity_relative_error"] + 1e-9
    assert (A2.to_numpy() >= -1e-12).all()
    # Data centres sell nothing to final demand.
    assert FD2.loc["IND_62T63_DC"].abs().sum() == pytest.approx(0.0, abs=1e-9)
    if rebalance:
        target = diag["value_added_target_share"]["IND_62T63_DC"]
        assert diag["value_added_share"]["IND_62T63_DC"] == pytest.approx(target, rel=1e-6)


def test_ras_split_columns_hits_targets():
    rng = np.random.default_rng(0)
    parent = rng.uniform(1, 10, 6)
    seed = np.column_stack([parent * 0.3, parent * 0.7]) * rng.uniform(0.5, 1.5, (6, 2))
    x_sub = np.array([40.0, 60.0])
    va = sr.scaled_va_targets(100 - parent.sum(), x_sub, [0.2, 0.5])
    cols, info = sr.ras_split_columns(seed, parent, x_sub, va)
    np.testing.assert_allclose(cols.sum(axis=1), parent, rtol=1e-8)
    np.testing.assert_allclose(cols.sum(axis=0), x_sub - va, rtol=1e-8)
    assert va.sum() == pytest.approx(100 - parent.sum())


def test_canada_split_value_added_targets(tmp_path):
    A, X, FD = tc.make_icio()
    plan = cds.prepare_canada_split(tc.make_sut_csv(tmp_path / "s.csv"), tc.make_dsut_csv(tmp_path / "d.csv"), 2022)
    A2, X2, FD2, diag = cds.split_icio(A, X, FD, plan)
    shares = diag["subsector_value_added_share"]
    # CLD and DPH share the 51821 value-added share, OTH has its own.
    assert shares[plan.cloud_key] == pytest.approx(shares[plan.hosting_key], rel=1e-6)
    j = A.index.get_loc(plan.parent_key)
    parent_va = X.iloc[j] * (1 - A.iloc[:, j].sum())
    total_va = sum(X2[k] * (1 - A2[k].sum()) for k in plan.sub_keys)
    assert total_va == pytest.approx(parent_va, rel=1e-6)


@pytest.mark.parametrize("passthrough", [False, True])
def test_end_to_end_both_splits(tmp_path, monkeypatch, passthrough):
    import types

    A, X, FD = icio_2025()
    data = tmp_path / "data"
    data.mkdir()
    out = tmp_path / "out"
    tc._write_scenario_inputs(data, A.index.tolist())
    pd.DataFrame([
        {"country": "CAN", "dc_proxy_io_key": "CAN_62T63", "electricity_io_key": "CAN_35"},
        {"country": "IND", "dc_proxy_io_key": "IND_62T63", "electricity_io_key": "IND_35"},
    ]).to_csv(data / "dc_sector_map.csv", index=False)
    tc.make_sut_csv(data / "sut.csv")
    tc.make_dsut_csv(data / "canada_dsut_2020.csv")
    iot = make_iot(tmp_path / "iot.xlsx")

    monkeypatch.setitem(sys.modules, "pyreadr", types.ModuleType("pyreadr"))
    import importlib
    model = importlib.import_module("global_datacentre_icio_model_v2")
    objects = {"A": A, "X": pd.DataFrame([X.to_numpy()], index=["2022"], columns=X.index), "FD": FD}
    monkeypatch.setattr(model, "load_r_object", lambda path, obj, desc, year: objects[Path(path).stem])
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path, **kw: self.to_csv(str(path) + ".csv", index=False))
    argv = [
        "model", "--data-dir", str(data), "--output-dir", str(out),
        "--A-file", "A.RData", "--X-file", "X.RData", "--FD-file", "FD.RData",
        "--emissions-extension", str(data / "direct_emissions.csv"),
        "--split-canada-dsut", "--statcan-sut-file", str(data / "sut.csv"),
        "--split-india-iot", str(iot), "--india-dc-share", "0.05",
        "--it-utilisation", "0.7", "--tiva-ghg-indicators",
    ]
    if passthrough:
        argv.append("--output-includes-power-passthrough")
    monkeypatch.setattr(sys, "argv", argv)
    model.main()

    ext = pd.read_csv(out / "icio_emissions_extension.csv")
    assert {"IND_62T63_DC", "IND_62T63_OTH", "CAN_62T63_CLD"} <= set(ext["io_key"])
    assert "IND_62T63" not in set(ext["io_key"])
    head = pd.read_csv(out / "headline_summary.csv").set_index("shock_destination")
    assert {"CAN", "IND"} <= set(head.index)
    calib = pd.read_csv(out / "dc_output_and_electricity_calibration.csv")
    physical = pd.read_csv(out / "physical_operational_metrics.csv")
    assert physical["facility_electricity_gwh"].iloc[0] == pytest.approx((500) * 0.7)
    if passthrough:
        np.testing.assert_allclose(
            calib["dc_output_usd_mn"], calib["dc_rent_usd_mn"] + calib["power_passthrough_usd_mn"])
        assert (calib["power_passthrough_usd_mn"] > 0).all()
    assert (out / "icio_negative_value_added.csv").exists()
    assert (out / "india_iot_split" / "india_split_diagnostics.json").exists()
