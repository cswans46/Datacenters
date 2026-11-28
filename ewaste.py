#!/usr/bin/env python3
"""
End-of-life server e-waste from data-centre capacity.

    servers(MW)          = MW x racks_per_mw x servers_per_rack
    e-waste (tonnes)     = servers x server_kg / 1000
                         = MW x racks_per_mw x servers_per_rack x server_kg / 1000

Defaults describe an NVIDIA GB300 NVL72 AI factory (NVIDIA Enterprise
Reference Architecture, NVL72 AI factory with GB300 NVL72, dual-plane
networking): 7.6 racks per MW of IT capacity, 18 compute trays (servers)
per rack and 130 kg per server, i.e. 17.784 t per MW.
Check: 275 MW -> 2,090 racks -> 37,620 servers -> 4,890.6 t.

Capacity is IT capacity (dc_capacity_mw in dc_output_scenarios.csv).

Reported measures
-----------------
ewaste_stock_tonnes
    Server mass installed in the year: the e-waste generated when all of
    that capacity's servers reach end of life (EOL).
ewaste_new_capacity_tonnes
    The same for capacity added since the first scenario year.
ewaste_annual_tonnes
    Only with a server lifetime L (years): steady-state yearly EOL flow,
    stock / L.

Scope: servers only (no networking, storage, power or cooling equipment,
batteries or packaging); every MW is assumed to be AI capacity at GB300
NVL72 density. Optional per country-year coefficients can be supplied in
ewaste_coefficients.csv (country, year, racks_per_mw, servers_per_rack,
server_kg) in --data-dir.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

RACKS_PER_MW = 7.6
SERVERS_PER_RACK = 18
SERVER_KG = 130.0
COEFFICIENT_FILE = "ewaste_coefficients.csv"
SOURCE = (
    "NVIDIA Enterprise Reference Architecture: NVL72 AI factory with GB300 "
    "NVL72, dual-plane networking (docs.nvidia.com)"
)


def tonnes_per_mw(racks_per_mw=RACKS_PER_MW, servers_per_rack=SERVERS_PER_RACK, server_kg=SERVER_KG):
    return racks_per_mw * servers_per_rack * server_kg / 1000.0


def ewaste_metrics(
    capacity: pd.DataFrame,
    racks_per_mw: float = RACKS_PER_MW,
    servers_per_rack: float = SERVERS_PER_RACK,
    server_kg: float = SERVER_KG,
    server_lifetime_years: Optional[float] = None,
    coefficients: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    capacity: columns country, year, dc_capacity_mw (IT MW).
    coefficients: optional country-year overrides of the three parameters.
    """
    for value, name in [(racks_per_mw, "racks_per_mw"), (servers_per_rack, "servers_per_rack"), (server_kg, "server_kg")]:
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive.")
    if server_lifetime_years is not None and server_lifetime_years <= 0:
        raise ValueError("server_lifetime_years must be positive.")

    df = capacity[["country", "year", "dc_capacity_mw"]].copy()
    df["country"] = df["country"].astype(str).str.strip().str.upper()
    df["year"] = df["year"].astype(int)
    df = df.sort_values(["country", "year"]).reset_index(drop=True)

    df["racks_per_mw"] = float(racks_per_mw)
    df["servers_per_rack"] = float(servers_per_rack)
    df["server_kg"] = float(server_kg)
    df["coefficient_source"] = SOURCE

    if coefficients is not None and not coefficients.empty:
        c = coefficients.copy()
        c["country"] = c["country"].astype(str).str.strip().str.upper()
        c["year"] = c["year"].astype(int)
        if c.duplicated(["country", "year"]).any():
            raise ValueError(f"{COEFFICIENT_FILE} has duplicate country-year rows.")
        cols = [k for k in ("racks_per_mw", "servers_per_rack", "server_kg") if k in c.columns]
        df = df.merge(c[["country", "year", *cols]], on=["country", "year"], how="left", suffixes=("", "_override"))
        for k in cols:
            override = df.pop(f"{k}_override")
            if (override.dropna() <= 0).any():
                raise ValueError(f"{COEFFICIENT_FILE}: {k} must be positive.")
            used = override.notna()
            df.loc[used, k] = override[used]
            df.loc[used, "coefficient_source"] = COEFFICIENT_FILE

    df["tonnes_per_mw"] = df["racks_per_mw"] * df["servers_per_rack"] * df["server_kg"] / 1000.0
    df["racks"] = df["dc_capacity_mw"] * df["racks_per_mw"]
    df["servers"] = df["racks"] * df["servers_per_rack"]
    df["ewaste_stock_tonnes"] = df["dc_capacity_mw"] * df["tonnes_per_mw"]
    df["ewaste_new_capacity_tonnes"] = (
        df["ewaste_stock_tonnes"]
        - df.groupby("country")["ewaste_stock_tonnes"].transform("first")
    )
    df["server_lifetime_years"] = server_lifetime_years if server_lifetime_years else np.nan
    df["ewaste_annual_tonnes"] = (
        df["ewaste_stock_tonnes"] / server_lifetime_years if server_lifetime_years else np.nan
    )
    return df


def load_coefficients(data_dir: Path) -> Optional[pd.DataFrame]:
    path = Path(data_dir) / COEFFICIENT_FILE
    return pd.read_csv(path) if path.exists() else None
