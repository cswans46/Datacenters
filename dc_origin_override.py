#!/usr/bin/env python3
"""
Country-of-origin override for the Canadian data-centre sectors.

Why
---
After the DSUT split, CAN_J62_63_CLD and CAN_J62_63_DPH inherit the
country-of-origin mix of CAN_J62_63 for every input. For computers and
electronics (ICIO C26) that mix is an average over all of IT services and,
at the import level, over all Canadian buyers of electronics (phones,
laptops). Canadian data centres buy servers, which come from different
places: in 2024 the US supplied 25% of Canada's server imports (HS 847150)
against 45% for all computers (HS 8471, 2022), and the server importers are
hyperscalers and Taiwanese-owned assemblers (see the sourcing report).

What it does
------------
For each target column (CLD, DPH) and each input sector in the shares file:

1. Domestic purchases (CAN rows) are left unchanged: the 2022 ICIO split
   between domestic and imported supply is kept.
2. Imported purchases (all non-CAN rows of that sector) keep their total,
   which is spread across origins by the target shares. ``OTHER`` is
   spread over the unlisted origins in proportion to their original values.
3. The table is re-balanced with the signed RAS (cswans46/baltic), with
   the CLD and DPH columns held fixed: every row still sums to its gross
   output and every other column to its original total.

Shares file (``dc_origin_shares.csv``)
-------------------------------------
    input_sector, origin, share, source
    C26, USA, 0.25, ...
    C26, MEX, ...
    C26, OTHER, ...

``origin`` is an ICIO country code or OTHER; shares sum to one per input
sector. ``write-defaults`` writes the built-in variant; ``from-trade``
builds a file from Trade Data Online exports (hs_code, country, value).

Command line
------------
    python dc_origin_override.py write-defaults --out data/dc_origin_shares.csv
    python dc_origin_override.py from-trade --trade tdo_c26_2025.csv --out data/dc_origin_shares.csv
The override runs from the main model with --dc-origin-shares <file>.
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from canada_dsut_split import country_of, normalise_sector_code, sector_of
from icio_projection.ras import ras_signed

LOG = logging.getLogger("datacentre_icio.origin_override")

HOME = "CAN"

# Built-in variant "trade2025" for C26 (computers and electronics):
# * USA 25%: Canadian HS 847150 imports 2024, C$1.006 bn of C$3.960 bn
#   (ISED Canadian Importers Database).
# * The remaining 75% follows US imports of HS 8471500150 in 2025:
#   Mexico 49.0%, Taiwan 42.2%, Thailand 4.8%, other 4.0% (SinoBTD from US
#   Census). Replace with the Canadian partner split from Trade Data Online
#   (``from-trade``) when pulled.
DEFAULT_SHARES = [
    ("C26", "USA", 0.25, "ISED CID: Canadian HS 847150 imports 2024, US C$1.006bn of C$3.960bn"),
    ("C26", "MEX", 0.75 * 0.490, "US HS 8471500150 imports 2025, Mexico 49.0% (SinoBTD/US Census), scaled to non-US 75%"),
    ("C26", "TWN", 0.75 * 0.422, "US HS 8471500150 imports 2025, Taiwan 42.2%, scaled to non-US 75%"),
    ("C26", "THA", 0.75 * 0.048, "US HS 8471500150 imports 2025, Thailand 4.8%, scaled to non-US 75%"),
    ("C26", "OTHER", 0.75 * 0.040, "Remainder, spread over other origins in proportion to the 2022 ICIO"),
]

# ICIO 2025 reports China and Mexico as two blocks each (CN1/CN2: domestic
# and processing trade; MX1/MX2: global manufacturing). Their CHN/MEX rows
# exist but are empty (zero output), so shares listed for CHN or MEX are
# shared across the blocks in proportion to their output of the sector.
SPLIT_BLOCKS = {"CHN": ("CN1", "CN2"), "MEX": ("MX1", "MX2")}

# Country names in Trade Data Online / StatCan -> ICIO codes.
COUNTRY_CODES = {
    "united states": "USA", "mexico": "MEX", "taiwan": "TWN", "china": "CHN", "vietnam": "VNM",
    "viet nam": "VNM", "thailand": "THA", "malaysia": "MYS", "korea, south": "KOR", "south korea": "KOR",
    "japan": "JPN", "germany": "DEU", "philippines": "PHL", "singapore": "SGP", "india": "IND",
    "united kingdom": "GBR", "france": "FRA", "netherlands": "NLD", "ireland": "IRL", "israel": "ISR",
    "italy": "ITA", "hungary": "HUN", "czech republic": "CZE", "czechia": "CZE", "poland": "POL",
    "indonesia": "IDN", "hong kong": "HKG", "sweden": "SWE", "switzerland": "CHE", "finland": "FIN",
    "denmark": "DNK", "spain": "ESP", "austria": "AUT", "belgium": "BEL", "australia": "AUS",
    "brazil": "BRA", "costa rica": "CRI", "turkey": "TUR", "türkiye": "TUR",
}


def _code(name: str) -> str:
    n = re.sub(r"\s*\(.*?\)", "", str(name)).strip().lower()
    return COUNTRY_CODES.get(n, "OTHER")


def write_defaults(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(DEFAULT_SHARES, columns=["input_sector", "origin", "share", "source"]).to_csv(path, index=False)
    return path


def shares_from_trade(trade: pd.DataFrame, input_sector: str = "C26", exclude_home: bool = True) -> pd.DataFrame:
    """Value-weighted origin shares from rows (hs_code, country, value).

    Re-imports and Canada itself are dropped (they are domestic supply).
    """
    t = trade.copy()
    t["origin"] = [_code(c) for c in t["country"]]
    drop = t["country"].str.lower().str.contains("re-import|canada", regex=True)
    if exclude_home:
        t = t[~drop]
    s = t.groupby("origin")["value"].sum()
    s = s / s.sum()
    return pd.DataFrame({"input_sector": input_sector, "origin": s.index, "share": s.values,
                         "source": "Trade Data Online, HS " + ",".join(sorted(t["hs_code"].astype(str).unique()))})


def load_shares(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"input_sector", "origin", "share"}
    if not need <= set(df.columns):
        raise ValueError(f"{path} needs columns {sorted(need)}")
    df["origin"] = df["origin"].astype(str).str.strip().str.upper()
    df["input_sector"] = df["input_sector"].astype(str).str.strip()
    if (df["share"] < 0).any():
        raise ValueError("Origin shares must be non-negative.")
    tot = df.groupby("input_sector")["share"].sum()
    bad = tot[(tot - 1).abs() > 1e-6]
    if len(bad):
        raise ValueError(f"Origin shares must sum to one per input sector: {bad.to_dict()}")
    if (df["origin"] == HOME).any():
        raise ValueError("Shares cover imports only; do not list CAN (domestic supply is kept).")
    return df


def origin_mix(A: pd.DataFrame, X: pd.Series, col: str, sector_regex: str) -> pd.Series:
    """Imported purchases of ``col`` from sector rows, by origin country."""
    rows = [k for k in A.index if re.search(sector_regex, sector_of(k)) and country_of(k) != HOME]
    z = A.loc[rows, col] * X[col]
    return z.groupby([country_of(k) for k in rows]).sum()


def apply_origin_override(A: pd.DataFrame, X: pd.Series, FD: pd.DataFrame, target_keys: list[str],
                          shares: pd.DataFrame, ras_kw: dict | None = None):
    """Return A (rebalanced), X, FD and a diagnostics frame (before/after mix)."""
    keys = list(A.index)
    x = X.reindex(keys).to_numpy(float)
    Z = A.to_numpy(float) * x[None, :]
    pos = {k: i for i, k in enumerate(keys)}
    tcols = [pos[k] for k in target_keys if k in pos]
    if len(tcols) != len(target_keys):
        raise ValueError(f"Target keys missing from the ICIO: {sorted(set(target_keys) - set(pos))}")

    diag = []
    for sector, sh in shares.groupby("input_sector"):
        # C26 (2023 ICIO edition) and 26 (2025 edition) are the same sector.
        wanted = normalise_sector_code(sector)
        rows = [
            i for i, k in enumerate(keys)
            if normalise_sector_code(sector_of(k)) == wanted and country_of(k) != HOME
        ]
        if not rows:
            raise ValueError(f"No imported ICIO rows for input sector {sector}")
        row_country = {country_of(keys[i]): i for i in rows}
        listed = {o: float(s) for o, s in zip(sh["origin"], sh["share"]) if o != "OTHER"}
        other = float(sh.loc[sh["origin"] == "OTHER", "share"].sum())
        # Origins reported as split blocks (CN1/CN2, MX1/MX2).
        for o in list(listed):
            if o in row_country and x[row_country[o]] > 0:
                continue
            blocks = [b for b in SPLIT_BLOCKS.get(o, ()) if b in row_country and x[row_country[b]] > 0]
            if blocks:
                weights = np.array([x[row_country[b]] for b in blocks], dtype=float)
                share = listed.pop(o)
                for b, wb in zip(blocks, weights / weights.sum()):
                    listed[b] = listed.get(b, 0.0) + share * wb
                LOG.info("Origin %s %s share assigned to %s by output.", o, sector, blocks)
        missing = [o for o in listed if o not in row_country or x[row_country[o]] <= 0]
        for o in missing:
            LOG.warning("Origin %s has no %s row in the ICIO; its share goes to OTHER.", o, sector)
            other += listed.pop(o)
        # Rows with zero output (e.g. CHN/MEX in ICIO 2025) cannot supply anything.
        rest = [i for c, i in row_country.items() if c not in listed and x[i] > 0]
        for j in tcols:
            before = Z[rows, j].copy()
            total = before.sum()
            if total <= 0:
                continue
            new = np.zeros(len(rows))
            idx = {i: n for n, i in enumerate(rows)}
            for o, s in listed.items():
                new[idx[row_country[o]]] = s * total
            if rest and other > 0:
                w = np.array([Z[i, j] for i in rest])
                if w.sum() <= 0:
                    w = x[rest]
                w = w / w.sum()
                for i, wi in zip(rest, w):
                    new[idx[i]] = other * total * wi
            Z[rows, j] = new
            for n, i in enumerate(rows):
                if before[n] or new[n]:
                    diag.append({"target": keys[j], "input_sector": sector, "origin": country_of(keys[i]),
                                 "before_usd_mn": before[n], "after_usd_mn": new[n],
                                 "before_share": before[n] / total, "after_share": new[n] / total})

    # Rebalance everything except the overridden columns.
    fd = FD.reindex(keys).to_numpy(float)
    free = [j for j in range(len(keys)) if j not in tcols]
    M = np.hstack([Z[:, free], fd])
    row_t = x - Z[:, tcols].sum(axis=1)
    A0 = A.to_numpy(float) * x[None, :]
    col_t = np.concatenate([A0[:, free].sum(axis=0), fd.sum(axis=0)])
    kw = {"max_iter": 10000, "tol": 1e-8, **(ras_kw or {})}
    Mb, info = ras_signed(M, row_t, col_t, label="dc origin override", **kw)
    Z[:, free] = Mb[:, :len(free)]
    fd_new = Mb[:, len(free):]
    LOG.info("Origin override rebalanced: %s iterations, max error %.2e", info["iterations"], info["error"])

    A2 = pd.DataFrame(np.divide(Z, x[None, :], out=np.zeros_like(Z), where=x[None, :] > 0), index=keys, columns=keys)
    FD2 = pd.DataFrame(fd_new, index=keys, columns=FD.columns)
    d = pd.DataFrame(diag)
    d.attrs["ras"] = info
    return A2, X, FD2, d


def _cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write-defaults")
    w.add_argument("--out", type=Path, default=Path("dc_origin_shares.csv"))
    t = sub.add_parser("from-trade")
    t.add_argument("--trade", type=Path, required=True, help="CSV: hs_code, country, value")
    t.add_argument("--input-sector", default="C26")
    t.add_argument("--out", type=Path, default=Path("dc_origin_shares.csv"))
    a = ap.parse_args()
    if a.cmd == "write-defaults":
        print(write_defaults(a.out))
    else:
        shares_from_trade(pd.read_csv(a.trade), a.input_sector).to_csv(a.out, index=False)
        print(a.out)


if __name__ == "__main__":
    _cli()
