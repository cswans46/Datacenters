"""
Tests for the Canadian DSUT split and TiVA/GHG indicators, on a small
synthetic ICIO and a synthetic StatCan-format SUT extract.

Run: pytest -q tests
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import canada_dsut_split as cds  # noqa: E402
import tiva_ghg_indicators as tgi  # noqa: E402


COUNTRIES = ["CAN", "USA", "IND", "ROW"]
SECTORS = ["01", "D35", "J62_63", "K"]
FD_CATS = ["HFCE", "GFCF"]


# ---------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------

def make_icio(seed: int = 1):
    rng = np.random.default_rng(seed)
    keys = [f"{c}_{s}" for c in COUNTRIES for s in SECTORS]
    n = len(keys)
    Z = rng.uniform(1.0, 10.0, size=(n, n))
    # Home bias.
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            if ki[:3] == kj[:3]:
                Z[i, j] *= 4.0
    fd_cols = [f"{c}_{f}" for c in COUNTRIES for f in FD_CATS]
    FD = rng.uniform(20.0, 60.0, size=(n, len(fd_cols)))
    x = Z.sum(axis=1) + FD.sum(axis=1)
    A = Z / x[None, :]
    return (
        pd.DataFrame(A, index=keys, columns=keys),
        pd.Series(x, index=keys),
        pd.DataFrame(FD, index=keys, columns=fd_cols),
    )


IND_518 = "Data processing, hosting, and related services [BS518000]"
IND_5415 = "Computer systems design and related services [BS541500]"
IND_519 = "Other information services [BS519000]"
IND_ELEC = "Electric power generation, transmission and distribution [BS221100]"
IND_BANK = "Banking and other depository credit intermediation [BS521000]"

PRD_518 = "Data processing, hosting and related services [MPS518]"
PRD_CSD = "Computer systems design services [MPS5415]"
PRD_ELEC = "Electricity [MPG2211]"
PRD_FIN = "Financial services [MPS52]"


def make_sut_csv(path: Path, year: int = 2022) -> Path:
    rows = []

    def add(su, industry, product, value, valuation="Basic price"):
        rows.append({
            "REF_DATE": str(year), "GEO": "Canada", "DGUID": "2016A000011124",
            "Supply and use": su, "Valuation": valuation,
            "Industry": industry, "Product": product, "UOM": "Dollars",
            "SCALAR_FACTOR": "thousands", "VALUE": value,
        })

    # Supply (output by industry and product).
    add("Supply", IND_518, PRD_518, 900)
    add("Supply", IND_518, PRD_CSD, 100)        # 518 output = 1000
    add("Supply", IND_5415, PRD_CSD, 2700)
    add("Supply", IND_5415, PRD_518, 50)        # 5415 output = 2750
    add("Supply", IND_519, PRD_CSD, 250)        # 519 output = 250  -> J = 4000
    add("Supply", IND_ELEC, PRD_ELEC, 5000)
    add("Supply", IND_BANK, PRD_FIN, 8000)
    add("Supply", "Total, all industries", "Total products", 99999)   # must be ignored
    add("Supply", "Imports [IMP]", PRD_518, 300)

    # Use: inputs of 518 are electricity-heavy.
    add("Use", IND_518, PRD_ELEC, 150)
    add("Use", IND_518, PRD_FIN, 50)
    add("Use", IND_518, PRD_CSD, 100)
    add("Use", IND_5415, PRD_ELEC, 30)
    add("Use", IND_5415, PRD_FIN, 300)
    add("Use", IND_5415, PRD_CSD, 400)
    add("Use", IND_519, PRD_ELEC, 5)
    add("Use", IND_519, PRD_FIN, 20)
    # Sales of products: 518 product mostly intermediate and exports.
    add("Use", IND_BANK, PRD_518, 500)
    add("Use", "Household final consumption expenditure [PEC]", PRD_518, 100)
    add("Use", "Exports [EXP]", PRD_518, 400)
    add("Use", IND_BANK, PRD_CSD, 2000)
    add("Use", "Gross fixed capital formation [CAP]", PRD_CSD, 800)
    add("Use", "Exports [EXP]", PRD_CSD, 400)
    add("Use", "Total, all industries", PRD_CSD, 99999)  # ignored
    # Other geography and year rows are filtered out.
    rows.append({**rows[0], "GEO": "Ontario", "VALUE": 1e9})
    rows.append({**rows[0], "REF_DATE": str(year - 1), "VALUE": 1e9})

    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def make_dsut_csv(path: Path, cloud: float = 400.0, total: float = 1000.0, with_sales: bool = False) -> Path:
    cds.write_dsut_template(path)
    df = pd.read_csv(path, dtype=str)
    values = {"cloud_output_industry_51821": cloud, "output_industry_51821": total}
    if with_sales:
        values.update({
            "cloud_use_intermediate": 200, "cloud_use_domestic_final": 20, "cloud_exports": 180,
            "noncloud_hosting_use_intermediate": 400, "noncloud_hosting_use_domestic_final": 100,
            "noncloud_hosting_exports": 100,
        })
    df["value"] = df["measure"].map(values).astype("string")
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def plan(tmp_path):
    sut = make_sut_csv(tmp_path / "sut.csv")
    dsut = make_dsut_csv(tmp_path / "dsut.csv")
    return cds.prepare_canada_split(sut, dsut, sut_year=2022)


# ---------------------------------------------------------------------
# SUT / DSUT
# ---------------------------------------------------------------------

def test_sut_aggregates(tmp_path):
    sut = cds.load_statcan_sut(make_sut_csv(tmp_path / "sut.csv"), year=2022)
    agg = cds.compute_sut_aggregates(sut)
    assert agg.output["518"] == pytest.approx(1000)
    assert agg.output["J"] == pytest.approx(4000)
    assert agg.products_518 == [PRD_518]
    assert PRD_CSD in agg.products_other_j
    # Electricity intensity: 518 = 150/1000; rest of J = 35/3000
    assert agg.input_intensity.loc["electricity", "518"] == pytest.approx(0.15)
    assert agg.input_intensity.loc["electricity", "OTH"] == pytest.approx(35 / 3000)
    assert agg.sales.loc["exports", "518"] == pytest.approx(400)
    assert agg.sales.loc["domestic_final", "OTH"] == pytest.approx(800)
    cats = sut.members.drop_duplicates("column").set_index("column")["category"]
    assert cats["Exports [EXP]"] == "exports"
    assert cats["Household final consumption expenditure [PEC]"] == "domestic_final"
    assert cats["Total, all industries"] == "total"


def test_plan_weights(plan):
    assert plan.share_518 == pytest.approx(0.25)
    assert plan.cloud_share == pytest.approx(0.4)
    np.testing.assert_allclose(plan.weights, [0.1, 0.15, 0.75])
    assert plan.sub_keys == ["CAN_J62_63_CLD", "CAN_J62_63_DPH", "CAN_J62_63_OTH"]
    targets = plan.dc_targets("cloud_and_hosting")
    assert [k for k, _ in targets] == ["CAN_J62_63_CLD", "CAN_J62_63_DPH"]
    assert sum(w for _, w in targets) == pytest.approx(1.0)
    assert targets[0][1] == pytest.approx(0.4)


def test_cloud_share_override(tmp_path):
    p = cds.prepare_canada_split(make_sut_csv(tmp_path / "s.csv"), make_dsut_csv(tmp_path / "d.csv"), 2022, cloud_share_override=0.6)
    np.testing.assert_allclose(p.weights, [0.15, 0.10, 0.75])


def test_dsut_missing_required(tmp_path):
    path = tmp_path / "d.csv"
    cds.write_dsut_template(path)
    with pytest.raises(ValueError, match="cloud_output_industry_51821"):
        cds.load_dsut(path)


def test_dsut_sales_structure(tmp_path):
    p = cds.prepare_canada_split(make_sut_csv(tmp_path / "s.csv"), make_dsut_csv(tmp_path / "d.csv", with_sales=True), 2022)
    # Cloud exports more than other hosting in the DSUT.
    assert p.sales_rho.loc["exports", "CAN_J62_63_CLD"] > p.sales_rho.loc["exports", "CAN_J62_63_DPH"]


# ---------------------------------------------------------------------
# ICIO split
# ---------------------------------------------------------------------

def test_split_aggregates_back(plan):
    A, X, FD = make_icio()
    A2, X2, FD2, diag = cds.split_icio(A, X, FD, plan)
    assert A2.shape == (A.shape[0] + 2, A.shape[1] + 2)
    keys = A.index.tolist()
    j = keys.index("CAN_J62_63")
    assert A2.index.tolist()[j:j + 3] == plan.sub_keys
    assert "IND_J62_63" in A2.index and "USA_J62_63" in A2.index  # others untouched

    Z = A.to_numpy() * X.to_numpy()[None, :]
    Z2 = A2.to_numpy() * X2.to_numpy()[None, :]
    Zb, fdb, xb = cds.aggregate_back(Z2, FD2.to_numpy(), X2.to_numpy(), j, 3)
    np.testing.assert_allclose(Zb, Z, rtol=1e-10)
    np.testing.assert_allclose(fdb, FD.to_numpy(), rtol=1e-10)
    np.testing.assert_allclose(xb, X.to_numpy(), rtol=1e-12)

    # Split rows still balance: X = Z 1 + FD 1 for every subsector.
    np.testing.assert_allclose(Z2.sum(axis=1) + FD2.to_numpy().sum(axis=1), X2.to_numpy(), rtol=1e-8)

    # Data centres buy more electricity per unit output than the rest of J62_63.
    e = diag["subsector_electricity_input_coefficient"]
    assert e["CAN_J62_63_CLD"] == pytest.approx(e["CAN_J62_63_DPH"])
    assert e["CAN_J62_63_CLD"] > e["CAN_J62_63_OTH"]
    # 518 products are export-oriented relative to J62_63.
    q = diag["row_block_shares"]
    assert q["foreign_intermediate"]["CAN_J62_63_CLD"] > plan.weights[0]


def test_split_emissions(plan):
    A, X, FD = make_icio()
    A2, X2, _, _ = cds.split_icio(A, X, FD, plan)
    em = pd.DataFrame({"io_key": A.index, "ghg_tonnes_co2e": np.arange(len(A)) + 1.0, "source": "test"})
    total = em["ghg_tonnes_co2e"].sum()
    for basis in ("output", "electricity_input"):
        out = cds.split_emissions_frame(em, plan, A2, X2, basis=basis)
        assert out["io_key"].tolist() == A2.index.tolist()
        assert out["ghg_tonnes_co2e"].sum() == pytest.approx(total)
    out = cds.split_emissions_frame(em, plan, basis="output")
    parent = float(em.loc[em["io_key"].eq("CAN_J62_63"), "ghg_tonnes_co2e"].iloc[0])
    np.testing.assert_allclose(out.set_index("io_key").loc[plan.sub_keys, "ghg_tonnes_co2e"], parent * plan.weights)


# ---------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------

def _intensity(keys, seed=3):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.uniform(0.0, 5.0, len(keys)), index=keys)


def test_indicator_identities(plan):
    A, X, FD = make_icio()
    A2, X2, FD2, _ = cds.split_icio(A, X, FD, plan)
    f = _intensity(A2.index)
    res = tgi.compute_indicators(A2, X2, FD2, f, focus_keys=plan.sub_keys)
    tiva = res["tiva_country_indicators"]
    np.testing.assert_allclose(tiva["EXGR_DVA_usd_mn"] + tiva["EXGR_FVA_usd_mn"], tiva["EXGR_usd_mn"], rtol=1e-8)
    chk = res["indicator_checks"].iloc[0]
    assert chk["world_va_in_fd_usd_mn"] == pytest.approx(chk["world_fd_usd_mn"], rel=1e-8)
    assert chk["world_demand_ghg_t"] == pytest.approx(chk["world_prod_ghg_t"], rel=1e-8)
    sec = res["can_sector_forward_backward"].set_index("io_key")
    np.testing.assert_allclose(
        sec["backward_va_multiplier_domestic"] + sec["backward_va_multiplier_foreign"], 1.0, rtol=1e-8
    )
    focus = res["can_focus_sectors_by_partner"]
    assert set(focus["focus_io_key"]) == set(plan.sub_keys)
    # Backward partner detail adds to the sector totals.
    back = focus.groupby("focus_io_key")["backward_partner_ghg_in_focus_exports_t"].sum()
    tot = sec["exports_domestic_ghg_t"] + sec["exports_foreign_ghg_t"]
    np.testing.assert_allclose(back.loc[plan.sub_keys], tot.loc[plan.sub_keys], rtol=1e-8)


def test_neutral_split_preserves_country_indicators(plan):
    """With identical subsector structures the split cannot change country totals."""
    plan.input_ratio.loc[:, :] = 1.0
    plan.sales_rho.loc[:, :] = 1.0
    A, X, FD = make_icio()
    A2, X2, FD2, _ = cds.split_icio(A, X, FD, plan)
    f = _intensity(A.index)
    f2 = cds.split_emissions_frame(
        pd.DataFrame({"io_key": f.index, "ghg_tonnes_co2e": f * X, "source": "t"}), plan
    ).set_index("io_key")["ghg_tonnes_co2e"] / X2
    r1 = tgi.compute_indicators(A, X, FD, f, focus_keys=["CAN_J62_63"])
    r2 = tgi.compute_indicators(A2, X2, FD2, f2, focus_keys=plan.sub_keys)
    for name in ("tiva_country_indicators", "ghg_footprint_country_indicators"):
        pd.testing.assert_frame_equal(r1[name], r2[name], rtol=1e-8)


# ---------------------------------------------------------------------
# End-to-end run of global_datacentre_icio_model_v2.py
# ---------------------------------------------------------------------

def _write_scenario_inputs(data_dir: Path, keys):
    years = list(range(2026, 2031))
    rows = []
    for c, cap0 in (("CAN", 100.0), ("IND", 80.0)):
        for t, y in enumerate(years):
            cap = cap0 + 20 * t
            rows.append({"country": c, "year": y, "dc_capacity_mw": cap, "price_mn_MW_2022": 0.5, "output_mn_usd_per_mw_2022": cap * 0.5})
    pd.DataFrame(rows).to_csv(data_dir / "dc_output_scenarios.csv", index=False)

    def per_cy(**cols):
        return pd.DataFrame([{"country": c, "year": y, **{k: (v(c, y) if callable(v) else v) for k, v in cols.items()}} for c in ("CAN", "IND") for y in years])

    per_cy(electricity_gwh=lambda c, y: 500 + 50 * (y - 2026), demand_basis="facility").to_csv(data_dir / "dc_electricity_demand.csv", index=False)
    per_cy(price_2022_usd_mwh=60.0).to_csv(data_dir / "electricity_prices.csv", index=False)
    per_cy(kgco2_per_kwh=0.1, kgco2_per_kwh_natgas=0.45).to_csv(data_dir / "grid_emissions_factors.csv", index=False)
    per_cy(pue=1.3, wue_l_per_kwh_it=0.5).to_csv(data_dir / "water_coefficients.csv", index=False)
    per_cy(us_customer_share=0.6).to_csv(data_dir / "us_customer_shares.csv", index=False)
    pd.DataFrame([
        {"country": "CAN", "dc_proxy_io_key": "CAN_J62_63", "electricity_io_key": "CAN_D35"},
        {"country": "IND", "dc_proxy_io_key": "IND_J62_63", "electricity_io_key": "IND_D35"},
    ]).to_csv(data_dir / "dc_sector_map.csv", index=False)
    pd.DataFrame({"io_key": keys, "ghg_tonnes_co2e": np.linspace(1, 50, len(keys)), "source": "test"}).to_csv(
        data_dir / "direct_emissions.csv", index=False
    )


@pytest.mark.parametrize("target", ["cloud_and_hosting", "cloud"])
def test_end_to_end_model(tmp_path, monkeypatch, target):
    A, X, FD = make_icio()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir = tmp_path / "out"
    _write_scenario_inputs(data_dir, A.index.tolist())
    make_sut_csv(data_dir / "sut.csv")
    make_dsut_csv(data_dir / "canada_dsut_2020.csv")

    # pyreadr/xarray are not needed with the loader stubbed.
    monkeypatch.setitem(sys.modules, "pyreadr", types.ModuleType("pyreadr"))
    import importlib
    model = importlib.import_module("global_datacentre_icio_model_v2")

    objects = {
        "A": A,
        "X": pd.DataFrame([X.to_numpy()], index=["2022"], columns=X.index),
        "FD": FD,
    }
    monkeypatch.setattr(model, "load_r_object", lambda path, obj, desc, year: objects[Path(path).stem])
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path, **kw: self.to_csv(str(path) + ".csv", index=False))

    monkeypatch.setattr(sys, "argv", [
        "model", "--data-dir", str(data_dir), "--output-dir", str(out_dir),
        "--A-file", "A.RData", "--X-file", "X.RData", "--FD-file", "FD.RData",
        "--emissions-extension", str(data_dir / "direct_emissions.csv"),
        "--split-canada-dsut", "--statcan-sut-file", str(data_dir / "sut.csv"),
        "--canada-dc-target", target, "--tiva-ghg-indicators",
    ])
    model.main()

    summary = pd.read_csv(out_dir / "upstream_emissions_summary.csv")
    assert {"CAN", "IND"} == set(summary["shock_destination"])
    ext = pd.read_csv(out_dir / "icio_emissions_extension.csv")
    assert {"CAN_J62_63_CLD", "CAN_J62_63_DPH", "CAN_J62_63_OTH"} <= set(ext["io_key"])
    assert "CAN_J62_63" not in set(ext["io_key"])

    detail = pd.read_csv(out_dir / "upstream_emissions_by_origin_industry.parquet.csv")
    sel = detail[
        detail["shock_destination"].eq("CAN")
        & detail["shock_basis"].eq("2030_additional_output")
        & detail["attribution"].eq("all_expansion")
        & detail["electricity_demand_sensitivity"].eq("icio_endogenous")
        & detail["grid_emissions_sensitivity"].eq("average_grid")
    ].set_index("io_key")
    shock_2030 = 180 * 0.5
    if target == "cloud":
        assert sel.loc["CAN_J62_63_CLD", "output_change_usd_mn"] == pytest.approx(shock_2030)
    else:
        assert sel.loc["CAN_J62_63_CLD", "output_change_usd_mn"] == pytest.approx(shock_2030 * 0.4)
        assert sel.loc["CAN_J62_63_DPH", "output_change_usd_mn"] == pytest.approx(shock_2030 * 0.6)

    assert (out_dir / "tiva_ghg_indicators" / "can_focus_sectors_by_partner.csv").exists()
    assert (out_dir / "canada_dsut_split" / "canada_dsut_split_summary.json").exists()
