"""Tests for the data-centre country-of-origin override (synthetic data)."""
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import canada_dsut_split as cds  # noqa: E402
import dc_origin_override as dco  # noqa: E402
import test_canada_dsut_split as tc  # noqa: E402


@pytest.fixture()
def split(tmp_path):
    A, X, FD = tc.make_icio()
    plan = cds.prepare_canada_split(tc.make_sut_csv(tmp_path / "s.csv"), tc.make_dsut_csv(tmp_path / "d.csv"), 2022)
    A2, X2, FD2, _ = cds.split_icio(A, X, FD, plan)
    return A2, X2, FD2, plan


def shares_k():
    return pd.DataFrame({"input_sector": ["K", "K", "K"], "origin": ["USA", "IND", "OTHER"],
                         "share": [0.5, 0.3, 0.2], "source": "test"})


def test_override_hits_targets_and_keeps_balance(split):
    A, X, FD, plan = split
    targets = [plan.cloud_key, plan.hosting_key]
    dom_before = (A.loc["CAN_K", targets] * X[targets]).copy()
    imp_before = dco.origin_mix(A, X, plan.cloud_key, "^K$").sum()
    A2, X2, FD2, diag = dco.apply_origin_override(A, X, FD, targets, shares_k())
    mix = dco.origin_mix(A2, X2, plan.cloud_key, "^K$")
    assert mix.sum() == pytest.approx(imp_before)
    assert mix["USA"] / mix.sum() == pytest.approx(0.5)
    assert mix["IND"] / mix.sum() == pytest.approx(0.3)
    assert mix["ROW"] / mix.sum() == pytest.approx(0.2)           # the only unlisted origin
    np.testing.assert_allclose(A2.loc["CAN_K", targets] * X2[targets], dom_before)  # domestic kept
    # table still balanced: rows = output, other columns keep their totals
    Z = A2.to_numpy() * X2.to_numpy()[None, :]
    np.testing.assert_allclose(Z.sum(axis=1) + FD2.to_numpy().sum(axis=1), X2.to_numpy(), rtol=1e-6)
    Z0 = A.to_numpy() * X.to_numpy()[None, :]
    np.testing.assert_allclose(Z.sum(axis=0), Z0.sum(axis=0), rtol=1e-6)
    np.testing.assert_allclose(FD2.sum(axis=0), FD.sum(axis=0), rtol=1e-6)
    assert set(diag["target"]) == set(targets)


def test_load_and_defaults(tmp_path):
    p = dco.write_defaults(tmp_path / "s.csv")
    s = dco.load_shares(p)
    assert s.groupby("input_sector")["share"].sum().iloc[0] == pytest.approx(1.0)
    assert s.set_index("origin").loc["USA", "share"] == pytest.approx(0.25)
    bad = s.copy()
    bad.loc[0, "share"] = 0.9
    bad.to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="sum to one"):
        dco.load_shares(tmp_path / "bad.csv")


def test_shares_from_trade():
    t = pd.DataFrame({"hs_code": ["847150"] * 5, "country": ["United States", "Mexico", "Taiwan", "Re-Imports (Canada)", "Estonia"],
                      "value": [25, 40, 30, 50, 5]})
    s = dco.shares_from_trade(t).set_index("origin")["share"]
    assert s["USA"] == pytest.approx(0.25) and s["MEX"] == pytest.approx(0.40) and s["OTHER"] == pytest.approx(0.05)


def test_model_run_with_override(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pyreadr", types.ModuleType("pyreadr"))
    import importlib
    model = importlib.import_module("global_datacentre_icio_model_v2")
    A, X, FD = tc.make_icio()
    data = tmp_path / "data"
    data.mkdir()
    tc._write_scenario_inputs(data, A.index.tolist())
    tc.make_sut_csv(data / "sut.csv")
    tc.make_dsut_csv(data / "canada_dsut_2020.csv")
    shares_k().to_csv(data / "dc_origin_shares.csv", index=False)
    objects = {"A": A, "X": pd.DataFrame([X.to_numpy()], index=["2022"], columns=X.index), "FD": FD}
    monkeypatch.setattr(model, "load_r_object", lambda path, obj, desc, year: objects[Path(path).stem])
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path, **kw: self.to_csv(str(path) + ".csv", index=False))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "model", "--data-dir", str(data), "--output-dir", str(out),
        "--A-file", "A.RData", "--X-file", "X.RData", "--FD-file", "FD.RData",
        "--emissions-extension", str(data / "direct_emissions.csv"),
        "--split-canada-dsut", "--statcan-sut-file", str(data / "sut.csv"),
        "--dc-origin-shares", str(data / "dc_origin_shares.csv"),
    ])
    model.main()
    d = pd.read_csv(out / "canada_dsut_split" / "dc_origin_override.csv")
    usa = d[(d.target == "CAN_J62_63_CLD") & (d.origin == "USA")]
    assert usa["after_share"].iloc[0] == pytest.approx(0.5)
    assert (out / "upstream_emissions_summary.csv").exists()
