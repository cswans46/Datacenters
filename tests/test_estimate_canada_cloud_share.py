"""Tests for the public-data cloud-share estimate."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import canada_dsut_split as cds  # noqa: E402
import estimate_canada_cloud_share as est  # noqa: E402


def test_us_shares_ordered():
    c = est.us_cloud_share_of_hitip()
    assert 0 < c["low"] < c["central"] < c["high"] < 1
    assert c["central"] == pytest.approx((24.294678 + 48 - 24.294678 - 12.081543 + 15.240362) / (70 + 18.385487))


def test_2020_central_and_dsut_file(tmp_path):
    df = est.estimates()
    row = df[(df.year == 2020) & (df.case == "central")].iloc[0]
    assert row.cloud_share_of_51821_output == pytest.approx(0.636 * est.us_cloud_share_of_hitip()["central"])
    path = est.write_dsut_file(tmp_path / "canada_dsut_2020.csv")
    dsut = cds.load_dsut(path)
    assert dsut.year == 2020
    assert dsut.cloud_share_of_51821_output == pytest.approx(row.cloud_share_of_51821_output, rel=1e-6)
    assert dsut.cloud_sales is None  # StatCan equivalence assumption kept
