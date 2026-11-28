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
