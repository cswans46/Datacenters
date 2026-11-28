"""Tests for the upstream/direct/downstream methodology (synthetic ICIO)."""
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def model(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyreadr", types.ModuleType("pyreadr"))
    import importlib
    return importlib.import_module("global_datacentre_icio_model_v2")


SECTORS = ["01", "26", "35", "62T63"]
COUNTRIES = ["CAN", "USA"]


def make_economy():
    keys = [f"{c}_{s}" for c in COUNTRIES for s in SECTORS]
    rng = np.random.default_rng(3)
    A = rng.uniform(0.0, 0.06, (len(keys), len(keys)))
    A = pd.DataFrame(A, index=keys, columns=keys)
    A.loc["CAN_35", "CAN_62T63"] = 0.004   # ICIO-style tiny electricity coefficient
    A.loc["CAN_35", "CAN_01"] = 0.10       # other sectors buy electricity
    A.loc["CAN_01", "CAN_35"] = 0.15       # electricity buys fuel
    X = pd.Series(np.linspace(100, 800, len(keys)), index=keys)
    emissions = pd.DataFrame({
        "io_key": keys,
        "ghg_tonnes_co2e": [X[k] * (2000 if k.endswith("_35") else 50) for k in keys],
        "source": "TEST",
    })
    return A, X, emissions


def make_scenarios(model, A, share_customer=0.0):
    sector_map = pd.DataFrame({
        "io_key": A.index,
        "country": [k.split("_", 1)[0] for k in A.index],
        "sector_code": [k.split("_", 1)[1] for k in A.index],
        "sector_label": [k.split("_", 1)[1] for k in A.index],
        "is_electricity_sector": [k.endswith("_35") for k in A.index],
    })
    dc_map = pd.DataFrame({
        "country": ["CAN"],
        "dc_proxy_io_key": ["CAN_62T63"],
        "electricity_io_key": ["CAN_35"],
        "dc_targets": [[("CAN_62T63", 1.0)]],
        "dc_proxy_label": ["CAN_62T63"],
    })
    years = [2029, 2030]
    out = [100.0, 300.0]
    gwh = [400.0, 1200.0]
    sc = pd.DataFrame({
        "country": "CAN",
        "year": years,
        "dc_output_usd_mn": out,
        "dc_output_yoy_usd_mn": [0.0, 200.0],
        "dc_output_cumulative_usd_mn": [0.0, 200.0],
        "dc_output_shock_usd_mn": [0.0, 300.0],
        "facility_electricity_gwh": gwh,
        "facility_gwh_yoy": [0.0, 800.0],
        "facility_gwh_cumulative": [0.0, 800.0],
        "price_2022_usd_mwh": 60.0,
        "kgco2_per_kwh": 0.04,
        "kgco2_per_kwh_natgas": 0.36,
        "us_customer_share": share_customer,
        "us_parent_operator_share": np.nan,
    })
    sc["us_customer_output_shock_usd_mn"] = sc["dc_output_shock_usd_mn"] * share_customer
    sc["us_parent_output_shock_usd_mn"] = np.nan
    return sector_map, dc_map, sc


def run(model, share_customer=0.0, **kw):
    A, X, emissions = make_economy()
    sector_map, dc_map, sc = make_scenarios(model, A, share_customer)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / "ext.csv"
        emissions.to_csv(tmp, index=False)
        ext, _ = model.load_emissions_extension(tmp, X, sector_map)
    res = model.run_leontief(A, X, sector_map, ext, dc_map, sc, **kw)
    return A, X, ext, res


def pick(df, **f):
    m = pd.Series(True, index=df.index)
    for k, v in f.items():
        m &= df[k] == v
    return df.loc[m]


def test_final_demand_with_extra_equals_mixed_model(model):
    from scipy.linalg import lu_factor
    A, X, _ = make_economy()
    a = A.to_numpy()
    n = len(a)
    s = [A.index.get_loc("CAN_62T63")]
    new_col = a[:, s].copy() * 0.5
    new_col[A.index.get_loc("CAN_35")] = 0.45
    t = np.array([10.0])
    lu, piv = lu_factor(np.eye(n) - a)
    _, dx = model.gross_output_to_final_demand(lu, piv, n, s, t, extra_demand=(new_col - a[:, s]) @ t)
    o = [i for i in range(n) if i not in s]
    expected = np.linalg.solve(np.eye(n - 1) - a[np.ix_(o, o)], new_col[o] @ t)
    np.testing.assert_allclose(dx[o], expected, rtol=1e-10)
    assert dx[s][0] == pytest.approx(10.0)


def test_hybrid_column(model):
    A, _, _ = make_economy()
    a = A.to_numpy()
    s = [A.index.get_loc("CAN_62T63")]
    e = A.index.get_loc("CAN_35")
    cols, f = model.hybrid_target_columns(a, s, e, 0.6, grid_share=0.08)
    assert cols[e, 0] == pytest.approx(0.6 * 0.08)
    assert f[0] == pytest.approx(0.4 / (1 - a[e, s[0]]))
    assert cols[:, 0].sum() <= 1
    with pytest.raises(ValueError, match="between 0% and 100%"):
        model.hybrid_target_columns(a, s, e, 1.2)


def test_components_and_fixes(model):
    A, X, ext, (detail, summary, fd, ddetail, dsummary) = run(model, share_customer=0.0)

    # Attribution share 0 -> no rows (previously electricity alone was shocked).
    assert summary.loc[summary.attribution == "us_customer_demand"].empty

    base = dict(year=2030, shock_basis="cumulative_vs_first_year", attribution="all_expansion")
    phys = pick(summary, electricity_demand_sensitivity="physical_facility_demand", **base)
    assert set(phys.grid_emissions_sensitivity) == {"average_grid", "projected_grid", "btm_natgas"}

    for r in summary.itertuples():
        assert r.direct_ghg_tonnes_co2e + r.upstream_excl_direct_ghg_tonnes_co2e == pytest.approx(r.global_upstream_ghg_tonnes_co2e)

    # Physical: electricity output exceeds the DC's own purchases (indirect use kept).
    det = pick(detail, electricity_demand_sensitivity="physical_facility_demand", grid_emissions_sensitivity="average_grid", **base)
    elec_dx = det.loc[det.io_key == "CAN_35", "output_change_usd_mn"].iloc[0]
    direct_purchase = 800 * 60 / 1000
    assert elec_dx > direct_purchase

    # Projected grid: purchased electricity emissions = GWh x factor.
    proj = pick(phys, grid_emissions_sensitivity="projected_grid").iloc[0]
    assert proj.purchased_electricity_ghg_tonnes_co2e == pytest.approx(800 * 1000 * 0.04)

    # BTM: on-site gas is direct; only 8% of electricity is bought from the grid.
    btm = pick(phys, grid_emissions_sensitivity="btm_natgas").iloc[0]
    assert btm.onsite_btm_generation_ghg_tonnes_co2e == pytest.approx(0.92 * 800 * 1000 * 0.36)
    assert btm.purchased_electricity_ghg_tonnes_co2e == pytest.approx(0.08 * 800 * 1000 * 0.04)
    assert btm.direct_ghg_tonnes_co2e == pytest.approx(btm.direct_datacentre_industry_ghg_tonnes_co2e + btm.onsite_btm_generation_ghg_tonnes_co2e)

    # Downstream: identical across electricity sensitivities (no electricity target).
    d = pick(dsummary, grid_emissions_sensitivity="average_grid", **base)
    vals = d.global_downstream_ghg_tonnes_co2e.to_numpy()
    assert np.allclose(vals, vals[0])
    for r in dsummary.itertuples():
        assert r.downstream_excl_direct_ghg_tonnes_co2e == pytest.approx(r.global_downstream_ghg_tonnes_co2e - r.direct_datacentre_industry_ghg_tonnes_co2e)

    # Value added and trade columns are populated and consistent.
    assert (summary.global_output_change_usd_mn > summary.target_gross_output_change_usd_mn).all()
    assert (summary.cross_border_intermediates_usd_mn >= 0).all()


def test_headline_summary_new_outputs(model, tmp_path):
    import summarize_results as sr
    A, X, ext, (detail, summary, fd, ddetail, dsummary) = run(model)
    summary.to_csv(tmp_path / sr.UPSTREAM_SUMMARY, index=False)
    dsummary.to_csv(tmp_path / sr.DOWNSTREAM_SUMMARY, index=False)
    keys = ["year", "shock_basis", "shock_destination", "grid_emissions_sensitivity",
            "electricity_demand_sensitivity", "attribution", "country", "sector_code",
            "sector_label", "is_datacentre_industry"]
    detail.groupby(keys, as_index=False)[[
        "output_change_usd_mn", "value_added_change_usd_mn", "intermediate_exports_usd_mn",
        "intermediate_exports_to_destination_usd_mn", "ghg_tonnes_co2e",
    ]].sum().to_csv(tmp_path / sr.BY_ORIGIN, index=False)
    physical = pd.DataFrame({
        "country": "CAN", "year": [2029, 2030], "facility_electricity_gwh": [400.0, 1200.0],
        "kgco2_per_kwh": 0.04, "kgco2_per_kwh_natgas": 0.36,
        "operational_water_ml": [10.0, 30.0], "direct_grid_emissions_ktco2": [16.0, 48.0],
    })
    head, origins = sr.write_headline(tmp_path, physical=physical, top_n=1)
    row = head.iloc[0]
    assert row.grid_emissions_sensitivity == "btm_natgas"
    assert not row.legacy_outputs
    assert row.btm_increase_vs_projected_grid_tonnes_co2e == pytest.approx(0.92 * 800 * 1000 * (0.36 - 0.04))
    assert origins.iloc[0].origin_country == "USA"
    assert "Top 1 foreign supplier" in (tmp_path / "headline_summary.md").read_text()
