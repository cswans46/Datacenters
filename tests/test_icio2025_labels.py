"""Canadian split and origin override on ICIO 2025-edition (numeric) labels."""
import sys
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

TO_2025 = {"J62_63": "62T63", "D35": "35", "C26": "26", "J61": "61", "K": "64T66"}


def relabel(key: str) -> str:
    country, sector = key.split("_", 1)
    return f"{country}_{TO_2025.get(sector, sector)}"


def icio_2025():
    A, X, FD = tc.make_icio()
    keys = [relabel(k) for k in A.index]
    A.index = A.columns = keys
    X.index = keys
    FD.index = keys
    return A, X, FD


def test_resolve_parent_sector():
    A, _, _ = icio_2025()
    assert cds.resolve_parent_sector(A.index) == "62T63"
    A0, _, _ = tc.make_icio()
    assert cds.resolve_parent_sector(A0.index) == "J62_63"
    with pytest.raises(ValueError, match="No IT and information services"):
        cds.resolve_parent_sector(["CAN_01"])


def test_split_with_numeric_labels(tmp_path):
    A, X, FD = icio_2025()
    plan = cds.prepare_canada_split(
        tc.make_sut_csv(tmp_path / "s.csv"), tc.make_dsut_csv(tmp_path / "d.csv"), 2022,
        parent_sector=cds.resolve_parent_sector(A.index),
    )
    assert plan.parent_key == "CAN_62T63"
    A2, X2, FD2, _ = cds.split_icio(A, X, FD, plan)
    subs = ["CAN_62T63_CLD", "CAN_62T63_DPH", "CAN_62T63_OTH"]
    assert set(subs) <= set(A2.index) and "CAN_62T63" not in A2.index
    np.testing.assert_allclose(X2[subs].sum(), X["CAN_62T63"])
    # Electricity is recognised as an input group (numeric code 35), so CLD
    # and the rest of the sector get different electricity coefficients.
    assert not np.isclose(A2.loc["CAN_35", "CAN_62T63_CLD"], A2.loc["CAN_35", "CAN_62T63_OTH"])


def test_origin_override_matches_letter_and_numeric_codes(tmp_path):
    A, X, FD = icio_2025()
    plan = cds.prepare_canada_split(
        tc.make_sut_csv(tmp_path / "s.csv"), tc.make_dsut_csv(tmp_path / "d.csv"), 2022,
        parent_sector="62T63",
    )
    A2, X2, FD2, _ = cds.split_icio(A, X, FD, plan)
    targets = [plan.cloud_key, plan.hosting_key]
    origins = [c for c in tc.COUNTRIES if c != "CAN"]
    shares = pd.DataFrame({"input_sector": ["D35"] * 2, "origin": [origins[0], "OTHER"],
                           "share": [0.6, 0.4], "source": "test"})
    A3, X3, FD3, diag = dco.apply_origin_override(A2, X2, FD2, targets, shares)
    mix = dco.origin_mix(A3, X3, plan.cloud_key, "^35$")
    assert mix[origins[0]] / mix.sum() == pytest.approx(0.6)


def test_origin_override_uses_split_blocks_for_empty_country_rows():
    """ICIO 2025: MEX rows are empty; Mexico's data sit in MX1/MX2."""
    import split_rebalance as sr

    keys = ["CAN_26", "CAN_62T63_CLD", "CAN_01", "MEX_26", "MX1_26", "MX2_26", "USA_26", "USA_01"]
    x = pd.Series([50.0, 40.0, 80.0, 0.0, 60.0, 30.0, 90.0, 120.0], index=keys)
    Z = pd.DataFrame(0.0, index=keys, columns=keys)
    Z.loc["CAN_26", ["CAN_62T63_CLD", "CAN_01"]] = [3.0, 2.0]
    Z.loc["MX1_26", ["CAN_62T63_CLD", "CAN_01", "USA_01"]] = [2.0, 1.0, 6.0]
    Z.loc["MX2_26", ["CAN_62T63_CLD", "USA_01"]] = [1.0, 4.0]
    Z.loc["USA_26", ["CAN_62T63_CLD", "CAN_01", "USA_01"]] = [4.0, 2.0, 10.0]
    Z.loc["CAN_01", ["CAN_26", "CAN_01"]] = [5.0, 8.0]
    Z.loc["USA_01", ["USA_26", "MX1_26", "USA_01"]] = [9.0, 4.0, 15.0]
    A = Z / x.where(x > 0, 1.0)
    A.loc[:, "MEX_26"] = 0.0
    fd_total = x - Z.sum(axis=1)
    FD = pd.DataFrame({"CAN": fd_total * 0.5, "USA": fd_total * 0.5})

    shares = pd.DataFrame({"input_sector": ["C26", "C26"], "origin": ["MEX", "USA"],
                           "share": [0.6, 0.4], "source": "test"})
    A2, X2, FD2, diag = dco.apply_origin_override(A, x, FD, ["CAN_62T63_CLD"], shares)

    Z2 = A2 * X2
    assert Z2.loc["MEX_26"].abs().sum() == pytest.approx(0.0)
    imports = Z2.loc[["MX1_26", "MX2_26", "USA_26"], "CAN_62T63_CLD"]
    assert (imports["MX1_26"] + imports["MX2_26"]) / imports.sum() == pytest.approx(0.6)
    assert imports["MX1_26"] / imports["MX2_26"] == pytest.approx(60.0 / 30.0)
    summary, _ = sr.balance_report(A2, X2, FD2)
    assert summary["row_identity_ok"]


@pytest.mark.parametrize("label, sector", [
    ("Banking and other depository credit intermediation [BS521000]", "64T66"),
    ("Data processing, hosting, and related services [BS518000]", "62T63"),
    ("Computer systems design and related services [BS541500]", "62T63"),
    ("Legal services [BS541100]", "69T75"),
    ("Pharmaceutical and medicine manufacturing [BS325400]", "21"),
    ("Iron foundries [BS331511]", "24A"),
    ("Elementary and secondary schools [GS611100]", "85"),
    ("Religious organizations [NP813100]", "94T96"),
    ("Federal government services [GS911000]", "84"),
    ("Household final consumption expenditure [PEC]", None),
])
def test_naics_to_icio(label, sector):
    assert cds.icio_sector_of_sut_industry(label) == sector


@pytest.mark.parametrize("structure", ["sut", "parent"])
def test_dc_sales_follow_sut_purchasers(tmp_path, structure):
    A, X, FD = icio_2025()
    plan = cds.prepare_canada_split(
        tc.make_sut_csv(tmp_path / "s.csv"), tc.make_dsut_csv(tmp_path / "d.csv"), 2022,
        parent_sector="62T63",
    )
    # In the fixture only banks (NAICS 521 -> 64T66) buy the 51821 product.
    assert plan.dc_sales_sector_shares.to_dict() == {"64T66": 1.0}
    A2, X2, FD2, diag = cds.split_icio(A, X, FD, plan, dc_sales_structure=structure)
    Z2 = A2 * X2
    dom_cols = [k for k in A2.columns if k.startswith("CAN_")]
    for key in (plan.cloud_key, plan.hosting_key):
        sales = Z2.loc[key, dom_cols]
        share_fin = sales["CAN_64T66"] / sales.sum()
        if structure == "sut":
            assert share_fin > 0.95
        else:
            parent = (A * X).loc["CAN_62T63", [c for c in A.columns if c.startswith("CAN_")]]
            assert share_fin == pytest.approx(parent["CAN_64T66"] / parent.sum(), rel=1e-6)
    if structure == "sut":
        assert diag["dc_domestic_sales_by_sector"]["achieved_shares"]["64T66"] > 0.95
