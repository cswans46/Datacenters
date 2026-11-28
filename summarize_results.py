#!/usr/bin/env python3
"""
Headline "summary of a summary" of the data-centre ICIO model results.

For each destination country (one row per country) it reports, for one
headline scenario:

* output gain: the data-centre output change, global / domestic / foreign
  gross output, and value added (new outputs only);
* emissions in three NON-OVERLAPPING components
      direct      data-centre industries' own emissions + on-site
                  behind-the-meter (BTM) gas generation
      upstream    supply-chain emissions excluding direct (incl. purchased
                  grid electricity)
      downstream  income-based (Ghosh) emissions excluding direct, from the
                  supply push of the data-centre industries only
  plus each as a share of 2022 global production emissions;
* the BTM gas comparison (gross on-site emissions and the increase over
  the same electricity drawn from the projected grid);
* operational cooling water;
* end-of-life server e-waste (ewaste.py): installed server mass in the
  headline year, the part from capacity added since the first year, and
  an annual flow if a server lifetime was given;
* the top five foreign supplier countries: value of goods and services
  produced there for the expansion, cross-border intermediates (new
  outputs only), the emissions emitted there, and the products most
  required.

It also writes abstract-ready sentences (headline_summary.md).

Headline scenario (all overridable): last scenario year,
cumulative_vs_first_year, all_expansion, physical_facility_demand, and
btm_natgas where available, else projected_grid, else average_grid.

Legacy outputs
--------------
Outputs written before the methodology fixes (no direct/upstream split
columns) are still summarised:
* direct      = emissions of the data-centre proxy industry (from the
                by-origin file) + 0.92 x GWh x gas factor (BTM rows);
* upstream    = total - direct;
* downstream  = the icio_endogenous Ghosh run (which has no electricity
                target) minus the data-centre industry's own emissions.
Purchased-electricity emissions, value added and cross-border
intermediates are not available for legacy outputs.

Usage
-----
    python summarize_results.py --output-dir datacentre_results
    python summarize_results.py --output-dir out --physical physical_operational_metrics.csv
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import ewaste as ew

LOG = logging.getLogger("datacentre_icio.summary")

UPSTREAM_SUMMARY = "upstream_emissions_summary.csv"
DOWNSTREAM_SUMMARY = "downstream_emissions_summary.csv"
BY_ORIGIN = "upstream_emissions_by_origin_country_sector.csv"
PHYSICAL = "physical_operational_metrics.csv"
CAPEX_SUMMARY = "capex_embodied_emissions_summary.csv"
EWASTE = "ewaste_metrics.csv"
CALIBRATION = "dc_output_and_electricity_calibration.csv"

LEGACY_BTM_SHARE = 0.92

# Data-centre proxy sectors when the by-origin file has no
# is_datacentre_industry flag: J62_63 / 62T63, or its CLD/DPH (Canada) or
# DC (India) split.
DC_SECTOR_PATTERN = re.compile(r"^(J)?62(T|_)63(_CLD|_DPH|_DC)?$")

COUNTRY_NAMES = {
    "USA": "United States", "CHN": "China", "CN1": "China", "CN2": "China",
    "MEX": "Mexico", "MX1": "Mexico", "MX2": "Mexico", "CAN": "Canada",
    "IND": "India", "DEU": "Germany", "GBR": "United Kingdom", "FRA": "France",
    "AUS": "Australia", "JPN": "Japan", "KOR": "Korea", "TWN": "Chinese Taipei",
    "RUS": "Russia", "SAU": "Saudi Arabia", "ARE": "United Arab Emirates",
    "IDN": "Indonesia", "MYS": "Malaysia", "SGP": "Singapore", "THA": "Thailand",
    "VNM": "Viet Nam", "PHL": "Philippines", "BRA": "Brazil", "ITA": "Italy",
    "ESP": "Spain", "NLD": "Netherlands", "IRL": "Ireland", "CHE": "Switzerland",
    "NOR": "Norway", "SWE": "Sweden", "POL": "Poland", "TUR": "Türkiye",
    "ZAF": "South Africa", "KAZ": "Kazakhstan", "ROW": "Rest of the world",
    "NGA": "Nigeria", "EGY": "Egypt", "PAK": "Pakistan", "BGD": "Bangladesh",
    "CHL": "Chile", "ARG": "Argentina", "COL": "Colombia", "PER": "Peru",
    "ISR": "Israel", "BEL": "Belgium", "AUT": "Austria", "DNK": "Denmark",
    "FIN": "Finland", "CZE": "Czechia", "HUN": "Hungary", "ROU": "Romania",
    "UKR": "Ukraine", "NZL": "New Zealand", "HKG": "Hong Kong, China",
}

# Group the split ICIO blocks (processing / non-processing trade).
COUNTRY_GROUPS = {"CN1": "CHN", "CN2": "CHN", "MX1": "MEX", "MX2": "MEX"}

SECTOR_NAMES = {
    "01": "agriculture", "02": "forestry", "03": "fishing",
    "05": "coal mining", "06": "oil and gas extraction",
    "07": "metal ore mining", "08": "other mining and quarrying",
    "09": "mining support services", "10T12": "food products",
    "13T15": "textiles", "16": "wood products", "17T18": "paper and printing",
    "19": "refined petroleum", "20": "chemicals", "21": "pharmaceuticals",
    "22": "rubber and plastics", "23": "non-metallic minerals (cement, glass)",
    "24A": "iron and steel", "24B": "non-ferrous metals",
    "25": "fabricated metals", "26": "computers and electronics",
    "27": "electrical equipment", "28": "machinery", "29": "motor vehicles",
    "301": "ships", "302T309": "other transport equipment",
    "31T33": "other manufacturing and repair",
    "35": "electricity and gas supply", "36T39": "water and waste",
    "41T43": "construction", "45T47": "wholesale and retail trade",
    "49": "land transport", "50": "water transport", "51": "air transport",
    "52": "warehousing", "53": "postal services",
    "55T56": "accommodation and food", "58T60": "publishing and broadcasting",
    "61": "telecommunications", "62T63": "IT and information services",
    "64T66": "finance and insurance", "68": "real estate",
    "69T75": "professional and technical services",
    "77T82": "administrative services", "84": "public administration",
    "85": "education", "86T88": "health", "90T93": "arts and recreation",
    "94T96": "other services", "97T98": "household employers",
}


def country_name(code: str) -> str:
    return COUNTRY_NAMES.get(code, code)


def sector_name(code: str) -> str:
    code = str(code)
    base = code.split("_")[0] if code not in SECTOR_NAMES else code
    return SECTOR_NAMES.get(code, SECTOR_NAMES.get(base, code))


# ---------------------------------------------------------------------
# Scenario selection
# ---------------------------------------------------------------------

def _select(df: pd.DataFrame, **filters) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for column, value in filters.items():
        if value is None or column not in df.columns:
            continue
        mask &= df[column].astype(str).eq(str(value))
    return df.loc[mask]


def headline_keys(
    upstream: pd.DataFrame,
    country: str,
    year: Optional[int],
    shock_basis: str,
    attribution: str,
    electricity: str,
    grid: Optional[str],
) -> Optional[dict]:
    rows = _select(upstream, shock_destination=country)
    if rows.empty:
        return None
    year = int(year if year is not None else rows["year"].max())
    rows = _select(
        rows,
        year=year,
        shock_basis=shock_basis,
        attribution=attribution,
        electricity_demand_sensitivity=electricity,
    )
    if rows.empty:
        return None
    if grid is None:
        available = set(rows["grid_emissions_sensitivity"])
        legacy = "upstream_excl_direct_ghg_tonnes_co2e" not in rows.columns
        # In legacy outputs btm_natgas equals average_grid for countries
        # without a BTM adjustment.
        if "btm_natgas" in available and (
            not legacy or _legacy_has_btm(rows)
        ):
            grid = "btm_natgas"
        elif "projected_grid" in available:
            grid = "projected_grid"
        else:
            grid = "average_grid"
    return {
        "year": year,
        "shock_basis": shock_basis,
        "shock_destination": country,
        "attribution": attribution,
        "electricity_demand_sensitivity": electricity,
        "grid_emissions_sensitivity": grid,
    }


def _legacy_has_btm(rows: pd.DataFrame) -> bool:
    btm = rows.loc[rows["grid_emissions_sensitivity"] == "btm_natgas", "global_upstream_ghg_tonnes_co2e"]
    avg = rows.loc[rows["grid_emissions_sensitivity"] == "average_grid", "global_upstream_ghg_tonnes_co2e"]
    return len(btm) and len(avg) and not np.isclose(btm.iloc[0], avg.iloc[0])


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def load_by_origin(path: Path, keys_list: list[dict]) -> pd.DataFrame:
    """Read only the headline rows of the (large) by-origin file."""
    wanted = {
        (str(k["year"]), k["shock_basis"], k["shock_destination"], k["attribution"],
         k["electricity_demand_sensitivity"], k["grid_emissions_sensitivity"])
        for k in keys_list
    }
    # The average-grid row of each headline is also needed (legacy direct).
    wanted |= {(w[0], w[1], w[2], w[3], w[4], "average_grid") for w in wanted}
    wanted |= {(w[0], w[1], w[2], w[3], "icio_endogenous", "average_grid") for w in wanted}

    parts = []
    for chunk in pd.read_csv(path, chunksize=250_000, dtype={"sector_code": str, "sector_label": str}):
        key = list(zip(
            chunk["year"].astype(str), chunk["shock_basis"], chunk["shock_destination"],
            chunk["attribution"], chunk["electricity_demand_sensitivity"],
            chunk["grid_emissions_sensitivity"],
        ))
        mask = np.fromiter((k in wanted for k in key), dtype=bool, count=len(key))
        if mask.any():
            parts.append(chunk.loc[mask])
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _dc_mask(df: pd.DataFrame, destination: str) -> pd.Series:
    if "is_datacentre_industry" in df.columns:
        return df["is_datacentre_industry"].astype(str).str.lower().isin(["true", "1"])
    return df["country"].eq(destination) & df["sector_code"].astype(str).str.match(DC_SECTOR_PATTERN)


def _physical_for(physical: Optional[pd.DataFrame], country: str, year: int, basis: str) -> dict:
    """Facility GWh, water and grid factors for the headline basis."""
    if physical is None:
        return {}
    p = physical.loc[physical["country"] == country].sort_values("year")
    if p.empty or year not in set(p["year"]):
        return {}
    last = p.loc[p["year"] == year].iloc[0]
    first = p.iloc[0]
    if basis == "cumulative_vs_first_year":
        gwh = last["facility_electricity_gwh"] - first["facility_electricity_gwh"]
        water_change = last["operational_water_ml"] - first["operational_water_ml"]
    elif basis == "annual_increment":
        prev = p.loc[p["year"] == year - 1]
        prev = prev.iloc[0] if len(prev) else last
        gwh = last["facility_electricity_gwh"] - prev["facility_electricity_gwh"]
        water_change = last["operational_water_ml"] - prev["operational_water_ml"]
    else:
        gwh = last["facility_electricity_gwh"]
        water_change = last["operational_water_ml"]
    return {
        "first_year": int(first["year"]),
        "facility_gwh_basis": float(gwh),
        "facility_gwh_level": float(last["facility_electricity_gwh"]),
        "grid_kgco2_per_kwh": float(last["kgco2_per_kwh"]),
        "natgas_kgco2_per_kwh": float(last["kgco2_per_kwh_natgas"]) if pd.notna(last.get("kgco2_per_kwh_natgas")) else np.nan,
        "operational_water_ml_level": float(last["operational_water_ml"]),
        "operational_water_ml_change": float(water_change),
        "direct_grid_emissions_ktco2_level": float(last["direct_grid_emissions_ktco2"]),
    }


def load_ewaste(output_dir: Path) -> Optional[pd.DataFrame]:
    """ewaste_metrics.csv, or rebuilt from capacity with default coefficients."""
    path = output_dir / EWASTE
    if path.exists():
        return pd.read_csv(path)
    calibration = output_dir / CALIBRATION
    if calibration.exists():
        LOG.info("%s not found: e-waste rebuilt from %s with default coefficients.", EWASTE, CALIBRATION)
        return ew.ewaste_metrics(pd.read_csv(calibration))
    return None


def _ewaste_for(table: Optional[pd.DataFrame], country: str, year: int, basis: str) -> dict:
    empty = {k: np.nan for k in (
        "dc_capacity_mw", "servers", "ewaste_stock_tonnes",
        "ewaste_new_capacity_tonnes", "ewaste_annual_tonnes", "ewaste_tonnes_per_mw",
    )}
    if table is None:
        return empty
    t = table.loc[table["country"] == country].sort_values("year")
    row = t.loc[t["year"] == year]
    if row.empty:
        return empty
    row = row.iloc[0]
    if basis == "annual_increment":
        prev = t.loc[t["year"] == year - 1, "ewaste_stock_tonnes"]
        new = row["ewaste_stock_tonnes"] - (prev.iloc[0] if len(prev) else 0.0)
    elif basis == "cumulative_vs_first_year":
        new = row["ewaste_new_capacity_tonnes"]
    else:
        new = row["ewaste_stock_tonnes"]
    return {
        "dc_capacity_mw": float(row["dc_capacity_mw"]),
        "servers": float(row["servers"]),
        "ewaste_stock_tonnes": float(row["ewaste_stock_tonnes"]),
        "ewaste_new_capacity_tonnes": float(new),
        "ewaste_annual_tonnes": float(row["ewaste_annual_tonnes"]),
        "ewaste_tonnes_per_mw": float(row["tonnes_per_mw"]),
    }


# ---------------------------------------------------------------------
# Headline calculation
# ---------------------------------------------------------------------

def summarise(
    output_dir: Path,
    physical: Optional[pd.DataFrame] = None,
    year: Optional[int] = None,
    shock_basis: str = "cumulative_vs_first_year",
    attribution: str = "all_expansion",
    electricity: str = "physical_facility_demand",
    grid: Optional[str] = None,
    top_n: int = 5,
    by_origin_file: Optional[Path] = None,
    upstream_file: Optional[Path] = None,
    downstream_file: Optional[Path] = None,
):
    output_dir = Path(output_dir)
    upstream = pd.read_csv(upstream_file or output_dir / UPSTREAM_SUMMARY)
    downstream = pd.read_csv(downstream_file or output_dir / DOWNSTREAM_SUMMARY)

    if physical is None and (output_dir / PHYSICAL).exists():
        physical = pd.read_csv(output_dir / PHYSICAL)

    ewaste_table = load_ewaste(output_dir)

    legacy = "upstream_excl_direct_ghg_tonnes_co2e" not in upstream.columns
    if legacy:
        LOG.warning("Legacy outputs detected: using the legacy decomposition (see module docstring).")

    countries = list(pd.unique(upstream["shock_destination"]))
    keys_list = []
    for country in countries:
        keys = headline_keys(upstream, country, year, shock_basis, attribution, electricity, grid)
        if keys is None:
            LOG.warning("No headline scenario for %s.", country)
            continue
        keys_list.append(keys)

    by_origin = load_by_origin(by_origin_file or output_dir / BY_ORIGIN, keys_list)

    headline_rows, origin_rows = [], []

    for keys in keys_list:
        country = keys["shock_destination"]
        up = _select(upstream, **keys).iloc[0]
        phys = _physical_for(physical, country, keys["year"], keys["shock_basis"])
        origin = _select(by_origin, **keys)
        avg_keys = {**keys, "grid_emissions_sensitivity": "average_grid"}
        origin_avg = _select(by_origin, **avg_keys)

        baseline = float(up["baseline_global_ghg_tonnes_co2e"])
        destination_baseline = float(up["baseline_destination_country_origin_ghg_tonnes_co2e"])
        total = float(up["global_upstream_ghg_tonnes_co2e"])
        btm = keys["grid_emissions_sensitivity"] == "btm_natgas"

        if legacy:
            dc_industry = float(origin_avg.loc[_dc_mask(origin_avg, country), "ghg_tonnes_co2e"].sum())
            onsite = (
                LEGACY_BTM_SHARE * phys.get("facility_gwh_basis", np.nan) * 1_000.0
                * phys.get("natgas_kgco2_per_kwh", np.nan)
                if btm else 0.0
            )
            direct = dc_industry + onsite
            upstream_excl = total - direct
            purchased = np.nan
            btm_share = LEGACY_BTM_SHARE if btm else 0.0
            down_keys = {**keys, "electricity_demand_sensitivity": "icio_endogenous", "grid_emissions_sensitivity": "average_grid"}
            down = _select(downstream, **down_keys).iloc[0]
            downstream_excl = float(down["global_downstream_ghg_tonnes_co2e"]) - dc_industry
            value_added = np.nan
            dest_value_added = np.nan
            cross_border = np.nan
            imports_to_destination = np.nan
            output_global = float(origin["output_change_usd_mn"].sum())
            output_destination = float(origin.loc[origin["country"] == country, "output_change_usd_mn"].sum())
        else:
            direct = float(up["direct_ghg_tonnes_co2e"])
            dc_industry = float(up["direct_datacentre_industry_ghg_tonnes_co2e"])
            onsite = float(up["onsite_btm_generation_ghg_tonnes_co2e"])
            upstream_excl = float(up["upstream_excl_direct_ghg_tonnes_co2e"])
            purchased = float(up["purchased_electricity_ghg_tonnes_co2e"])
            btm_share = float(up["btm_natgas_share"])
            down = _select(downstream, **keys).iloc[0]
            downstream_excl = float(down["downstream_excl_direct_ghg_tonnes_co2e"])
            value_added = float(up["global_value_added_change_usd_mn"])
            dest_value_added = float(up["destination_value_added_change_usd_mn"])
            cross_border = float(up["cross_border_intermediates_usd_mn"])
            imports_to_destination = float(up["intermediate_imports_to_destination_usd_mn"])
            output_global = float(up["global_output_change_usd_mn"])
            output_destination = float(up["destination_output_change_usd_mn"])

        grid_factor = phys.get("grid_kgco2_per_kwh", np.nan)
        gwh = phys.get("facility_gwh_basis", np.nan)
        onsite_vs_grid = (
            onsite - btm_share * gwh * 1_000.0 * grid_factor if btm else np.nan
        )

        headline_rows.append({
            **keys,
            "first_year": phys.get("first_year", np.nan),
            "legacy_outputs": legacy,
            "dc_output_change_usd_mn": float(up["target_gross_output_change_usd_mn"]),
            "global_output_change_usd_mn": output_global,
            "destination_output_change_usd_mn": output_destination,
            "foreign_output_change_usd_mn": output_global - output_destination,
            "global_value_added_change_usd_mn": value_added,
            "destination_value_added_change_usd_mn": dest_value_added,
            "cross_border_intermediates_usd_mn": cross_border,
            "intermediate_imports_to_destination_usd_mn": imports_to_destination,
            "direct_ghg_tonnes_co2e": direct,
            "direct_datacentre_industry_ghg_tonnes_co2e": dc_industry,
            "onsite_btm_generation_ghg_tonnes_co2e": onsite,
            "btm_increase_vs_projected_grid_tonnes_co2e": onsite_vs_grid,
            "upstream_excl_direct_ghg_tonnes_co2e": upstream_excl,
            "purchased_electricity_ghg_tonnes_co2e": purchased,
            "downstream_excl_direct_ghg_tonnes_co2e": downstream_excl,
            "total_footprint_ghg_tonnes_co2e": direct + upstream_excl,
            "direct_pct_of_global_2022": 100 * direct / baseline,
            "upstream_pct_of_global_2022": 100 * upstream_excl / baseline,
            "downstream_pct_of_global_2022": 100 * downstream_excl / baseline,
            "total_pct_of_destination_2022": 100 * (direct + upstream_excl) / destination_baseline,
            "foreign_origin_ghg_tonnes_co2e": float(up["foreign_origin_ghg_tonnes_co2e"]),
            "facility_gwh_basis": gwh,
            "grid_kgco2_per_kwh": grid_factor,
            "natgas_kgco2_per_kwh": phys.get("natgas_kgco2_per_kwh", np.nan),
            "operational_water_ml_level": phys.get("operational_water_ml_level", np.nan),
            "operational_water_ml_change": phys.get("operational_water_ml_change", np.nan),
            **_ewaste_for(ewaste_table, country, keys["year"], keys["shock_basis"]),
            "baseline_global_ghg_tonnes_co2e": baseline,
        })

        # ---------------- Top foreign supplier countries ----------------
        foreign = origin.loc[origin["country"] != country].copy()
        foreign["origin_group"] = foreign["country"].replace(COUNTRY_GROUPS)
        foreign = foreign.loc[foreign["origin_group"] != country]
        value_cols = [c for c in [
            "output_change_usd_mn", "intermediate_exports_usd_mn",
            "intermediate_exports_to_destination_usd_mn", "ghg_tonnes_co2e",
        ] if c in foreign.columns]
        by_country = foreign.groupby("origin_group")[value_cols].sum()
        # "ROW" is an aggregate of economies, not a country: report it
        # separately and rank the named countries.
        rest_of_world = by_country.loc["ROW"] if "ROW" in by_country.index else None
        ranked = by_country.drop(index="ROW", errors="ignore")
        top = ranked.sort_values("output_change_usd_mn", ascending=False).head(top_n)
        headline_rows[-1].update({
            "rest_of_world_output_change_usd_mn": float(rest_of_world["output_change_usd_mn"]) if rest_of_world is not None else 0.0,
            "rest_of_world_ghg_tonnes_co2e": float(rest_of_world["ghg_tonnes_co2e"]) if rest_of_world is not None else 0.0,
            "top_origins_output_change_usd_mn": float(top["output_change_usd_mn"].sum()),
            "top_origins_ghg_tonnes_co2e": float(top["ghg_tonnes_co2e"].sum()),
            "top_origins_intermediate_exports_usd_mn": float(top["intermediate_exports_usd_mn"].sum()) if "intermediate_exports_usd_mn" in top else np.nan,
        })

        for rank, (code, values) in enumerate(top.iterrows(), start=1):
            sectors = (
                foreign.loc[foreign["origin_group"] == code]
                .assign(sector=lambda d: d["sector_code"].astype(str))
                .groupby("sector")["output_change_usd_mn"].sum()
                .sort_values(ascending=False)
            )
            top_sectors = sectors.head(3)
            origin_rows.append({
                **keys,
                "rank": rank,
                "origin_country": code,
                "origin_country_name": country_name(code),
                "output_change_usd_mn": float(values["output_change_usd_mn"]),
                "intermediate_exports_usd_mn": float(values.get("intermediate_exports_usd_mn", np.nan)),
                "intermediate_exports_to_destination_usd_mn": float(values.get("intermediate_exports_to_destination_usd_mn", np.nan)),
                "ghg_tonnes_co2e": float(values["ghg_tonnes_co2e"]),
                "share_of_foreign_output_pct": 100 * float(values["output_change_usd_mn"]) / by_country["output_change_usd_mn"].sum(),
                "top_products": "; ".join(
                    f"{sector_name(s)} (USD {v:,.1f} mn)" for s, v in top_sectors.items()
                ),
            })

    headline = pd.DataFrame(headline_rows)
    origins = pd.DataFrame(origin_rows)

    capex_path = output_dir / CAPEX_SUMMARY
    capex = pd.read_csv(capex_path) if capex_path.exists() else None

    return headline, origins, capex


# ---------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------

def _mt(t: float) -> str:
    return f"{t / 1e6:,.2f} MtCO2e" if abs(t) >= 1e5 else f"{t / 1e3:,.1f} ktCO2e"


def _bn(usd_mn: float) -> str:
    return f"USD {usd_mn / 1e3:,.2f} bn" if abs(usd_mn) >= 1e3 else f"USD {usd_mn:,.0f} mn"


def headline_text(headline: pd.DataFrame, origins: pd.DataFrame, capex: Optional[pd.DataFrame] = None) -> str:
    lines = ["# Headline results (summary of a summary)", ""]
    for row in headline.itertuples(index=False):
        name = country_name(row.shock_destination)
        period = f"{int(row.first_year)}–{row.year}" if pd.notna(row.first_year) else str(row.year)
        lines.append(f"## {name}")
        lines.append("")
        lines.append(
            f"Scenario: {row.shock_basis}, {row.attribution}, "
            f"{row.electricity_demand_sensitivity}, {row.grid_emissions_sensitivity}"
            + (" (legacy outputs)" if row.legacy_outputs else "") + "."
        )
        lines.append("")
        gain = (
            f"Data-centre output rises by {_bn(row.dc_output_change_usd_mn)} (2022 USD) over {period}; "
            f"with supply chains, global gross output rises by {_bn(row.global_output_change_usd_mn)} "
            f"({_bn(row.destination_output_change_usd_mn)} in {name}, {_bn(row.foreign_output_change_usd_mn)} abroad)"
        )
        if pd.notna(row.global_value_added_change_usd_mn):
            gain += f" and value added by {_bn(row.global_value_added_change_usd_mn)} ({_bn(row.destination_value_added_change_usd_mn)} in {name})"
        lines.append(gain + ".")
        lines.append("")
        emis = (
            f"Direct emissions: {_mt(row.direct_ghg_tonnes_co2e)} "
            f"({row.direct_pct_of_global_2022:.4f}% of 2022 global production emissions)"
        )
        if row.onsite_btm_generation_ghg_tonnes_co2e > 0:
            emis += (
                f", of which {_mt(row.onsite_btm_generation_ghg_tonnes_co2e)} from behind-the-meter gas "
                f"({row.natgas_kgco2_per_kwh:.3f} kg/kWh vs a projected grid of {row.grid_kgco2_per_kwh:.3f} kg/kWh; "
                f"+{_mt(row.btm_increase_vs_projected_grid_tonnes_co2e)} relative to grid supply)"
            )
        emis += (
            f". Upstream supply-chain emissions (excluding direct): {_mt(row.upstream_excl_direct_ghg_tonnes_co2e)} "
            f"({row.upstream_pct_of_global_2022:.4f}%)"
        )
        if pd.notna(row.purchased_electricity_ghg_tonnes_co2e):
            emis += f", of which purchased grid electricity {_mt(row.purchased_electricity_ghg_tonnes_co2e)}"
        emis += (
            f". Downstream income-based emissions (excluding direct): {_mt(row.downstream_excl_direct_ghg_tonnes_co2e)} "
            f"({row.downstream_pct_of_global_2022:.4f}%). Direct plus upstream equals "
            f"{row.total_pct_of_destination_2022:.2f}% of {name}'s 2022 production emissions."
        )
        lines.append(emis)
        lines.append("")
        lines.append(
            f"Operational cooling water: {row.operational_water_ml_level / 1e3:,.1f} billion litres in {row.year} "
            f"({row.operational_water_ml_change / 1e3:+,.1f} billion litres over {period})."
        )
        lines.append("")
        if pd.notna(row.ewaste_stock_tonnes):
            text = (
                f"End-of-life server e-waste: {row.ewaste_stock_tonnes / 1e3:,.1f} kt for the "
                f"{row.dc_capacity_mw:,.0f} MW installed in {row.year} ({row.servers:,.0f} servers, "
                f"{row.ewaste_tonnes_per_mw:.3f} t/MW), of which {row.ewaste_new_capacity_tonnes / 1e3:,.1f} kt "
                f"from capacity added over {period}"
            )
            if pd.notna(row.ewaste_annual_tonnes):
                text += f"; {row.ewaste_annual_tonnes / 1e3:,.1f} kt per year at steady state"
            lines.append(text + ".")
            lines.append("")
        top = origins.loc[origins["shock_destination"] == row.shock_destination].sort_values("rank")
        if len(top):
            lines.append(f"Top {len(top)} foreign supplier countries (value of goods and services produced there for the expansion; emissions emitted there):")
            lines.append("")
            for o in top.itertuples(index=False):
                trade = (
                    f"; cross-border intermediates {_bn(o.intermediate_exports_usd_mn)}"
                    if pd.notna(o.intermediate_exports_usd_mn) else ""
                )
                lines.append(
                    f"{o.rank}. {o.origin_country_name}: {_bn(o.output_change_usd_mn)} "
                    f"({o.share_of_foreign_output_pct:.1f}% of foreign supply){trade}; {_mt(o.ghg_tonnes_co2e)}; "
                    f"mainly {o.top_products}"
                )
            lines.append(
                f"Together: {_bn(row.top_origins_output_change_usd_mn)} and {_mt(row.top_origins_ghg_tonnes_co2e)}"
                + (f"; cross-border intermediates {_bn(row.top_origins_intermediate_exports_usd_mn)}" if pd.notna(row.top_origins_intermediate_exports_usd_mn) else "")
                + f". Rest of the world (aggregate): {_bn(row.rest_of_world_output_change_usd_mn)}, {_mt(row.rest_of_world_ghg_tonnes_co2e)}."
            )
            lines.append("")
        if capex is not None and not capex.empty:
            c = capex.loc[(capex["shock_destination"] == row.shock_destination) & (capex["shock_basis"] == "cumulative_vs_first_year")]
            if len(c):
                c = c.sort_values("year").iloc[-1]
                lines.append(
                    f"Construction and equipment (one-off): {_bn(c.capex_usd_mn)} of capex embodies {_mt(c.embodied_capex_ghg_tonnes_co2e)}."
                )
                lines.append("")
    lines.append(
        "Direct, upstream and downstream are non-overlapping. Downstream is an income-based "
        "allocation (Ghosh), not a causal impact; do not add it to upstream."
    )
    return "\n".join(lines) + "\n"


def write_headline(
    output_dir: Path,
    physical: Optional[pd.DataFrame] = None,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = Path(output_dir)
    headline, origins, capex = summarise(output_dir, physical=physical, **kwargs)
    headline.to_csv(output_dir / "headline_summary.csv", index=False)
    origins.to_csv(output_dir / "headline_top_origin_countries.csv", index=False)
    (output_dir / "headline_summary.md").write_text(headline_text(headline, origins, capex), encoding="utf-8")
    LOG.info("Headline summary written to %s", output_dir / "headline_summary.md")
    return headline, origins


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True, help="Model output directory (also where results are written).")
    parser.add_argument("--physical", type=Path, default=None, help="physical_operational_metrics.csv if not in --output-dir.")
    parser.add_argument("--upstream-file", type=Path, default=None)
    parser.add_argument("--downstream-file", type=Path, default=None)
    parser.add_argument("--by-origin-file", type=Path, default=None)
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--shock-basis", default="cumulative_vs_first_year")
    parser.add_argument("--attribution", default="all_expansion")
    parser.add_argument("--electricity", default="physical_facility_demand")
    parser.add_argument("--grid", default=None, help="Grid sensitivity; default btm_natgas > projected_grid > average_grid.")
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    physical = pd.read_csv(args.physical) if args.physical else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_headline(
        args.output_dir,
        physical=physical,
        year=args.year,
        shock_basis=args.shock_basis,
        attribution=args.attribution,
        electricity=args.electricity,
        grid=args.grid,
        top_n=args.top_n,
        by_origin_file=args.by_origin_file,
        upstream_file=args.upstream_file,
        downstream_file=args.downstream_file,
    )
    print((args.output_dir / "headline_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
