#!/usr/bin/env python3
"""
Split Indian data centres (AI-DC) out of the OECD ICIO using the 2023-24
India input-output table with an AI-DC sector.

Source
------
The workbook (e.g. ``IOT_2023-24_RAS.xlsx``) contains a 70-sector
commodity-by-commodity IOT for 2023-24 (INR crore) in which an AI-DC sector
(data-centre services, NIC 6311 hosting and data processing) has been added
and the table RAS-balanced (sheet "Z RAS Final"). It provides:

* the AI-DC input structure (its column of intermediate inputs and value
  added),
* the AI-DC sales structure (its row: all intermediate, to Indian
  industries),
* AI-DC output relative to "Computer related services".

Only ratios are used, so the currency and price basis of the IOT do not
matter.

Method
------
1. Output weight. ``IND_62T63`` (ICIO "IT and other information services")
   is split into ``IND_62T63_DC`` and ``IND_62T63_OTH`` with

       w_DC = x_AIDC / (x_AIDC + x_CS)

   where x_CS is the output of Computer related services before the AI-DC
   purchases were added (column "Old Total Output"). Override with
   ``--india-dc-share``.

2. Columns. AI-DC input coefficients are mapped from the 70 IOT sectors to
   ICIO 2025 sectors (``IOT_TO_ICIO``; one-to-many mappings are shared by
   Indian ICIO gross output). The IOT records total use (domestic plus
   imported; imports are netted in final demand), so each ICIO input sector's
   coefficient is shared across countries of origin with the 2022 origin mix
   of ``IND_62T63``'s purchases of that sector (or, if it buys none, India's
   economy-wide origin mix). This gives the DC seed column. The n x 2 block
   (DC, OTH) is then RAS-balanced (split_rebalance.ras_split_columns): rows
   add up to the parent column and columns to output minus value-added
   targets (AI-DC and Computer related services value-added shares in the
   IOT, scaled to the parent's value added).

3. Rows. AI-DC sales shares by purchasing IOT sector are mapped to Indian
   ICIO purchasers (sales to Computer related services go to
   ``IND_62T63_OTH``); data centres have no final-demand sales and no
   exports, as in the IOT. The 2 x (purchasers + final demand) row block is
   RAS-balanced: rows add up to subsector output and columns to the parent's
   row cells.

4. Emissions. ``IND_62T63`` direct emissions are split by output (equal
   intensity, default) or by electricity purchases.

The split is exact in aggregate: adding the subsectors back reproduces the
original ICIO rows, columns, output and final demand.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from canada_dsut_split import country_of, normalise_sector_code, sector_of
from icio_projection.ras import ras_signed
import split_rebalance as sr

LOG = logging.getLogger("datacentre_icio.india_split")

HOME_COUNTRY = "IND"
DC_SUFFIX = "DC"
OTH_SUFFIX = "OTH"
DEFAULT_SHEET = "Z RAS Final"
AIDC_LABEL = "AI-DC"
CS_LABEL = "Computer related services"

# 2023-24 India IOT sector -> ICIO 2025 sector code(s).
IOT_TO_ICIO = {
    "Agriculture and Allied sectors": ["01", "02", "03"],
    "Coal and Lignite": ["05"],
    "Natural Gas": ["06"],
    "Crude petroleum": ["06"],
    "Iron ore": ["07"],
    "Manganese ore": ["07"],
    "Bauxite": ["07"],
    "Copper ore": ["07"],
    "Other Metallic minerals": ["07"],
    "Limestone": ["08"],
    "Other non metallic minerals": ["08"],
    "Processed and Other Food Products": ["10T12"],
    "Clothing and Textile Products": ["13T15"],
    "Wood and wood products except furniture": ["16"],
    "Paper, Paper products and newsprint": ["17T18"],
    "Publishing, printing and allied activities": ["17T18", "58T60"],
    "Furniture & Fixtures": ["31T33"],
    "Rubber products": ["22"],
    "Plastic products": ["22"],
    "Petroleum products": ["19"],
    "Coal tar products": ["19"],
    "Inorganic chemicals": ["20"],
    "Organic chemicals": ["20"],
    "Fertilizers and Pesticides": ["20"],
    "Other chemical and pharmaceutical products": ["20", "21"],
    "Cement": ["23"],
    "Non metallic mineral products": ["23"],
    "Iron and steel Ferro alloys": ["24A"],
    "Iron and steel casting and forging": ["24A", "25"],
    "Iron and steel foundries": ["24A"],
    "Non ferrous basic metals (including alloys)": ["24B"],
    "Hand tools, hardware": ["25"],
    "Miscellaneous metal products": ["25"],
    "Other Industrial and Agricultural machinery": ["28"],
    "Other non electrical machinery": ["28"],
    "Electrical industrial machinery": ["27"],
    "Electrical cables, wires": ["27"],
    "Batteries": ["27"],
    "Electrical appliances": ["27"],
    "Communication equipment": ["26"],
    "Other electrical machinery": ["27"],
    "Electronic equipment including T.V": ["26"],
    "Medical precision, optical instrument, watches and clocks": ["26"],
    "Water, Rail and Road equipment": ["29", "301", "302T309"],
    "Miscellaneous products": ["31T33"],
    "Construction": ["41T43"],
    "Gas": ["35"],
    "Water Supply": ["36T39"],
    "Other Utility Services": ["36T39"],
    "Trade": ["45T47"],
    "Repair & Maintenance of Motor Vehicle": ["45T47"],
    "Hotels & Restaurant": ["55T56"],
    "Transport services": ["49", "50", "51"],
    "Storage and warehousing": ["52"],
    "Communication services": ["61", "53"],
    "Financial and Insurance services": ["64T66"],
    "Real estate, ownership of dwellings": ["68"],
    "Renting of machinery & equipment": ["77T82"],
    "Research & Development Services": ["69T75"],
    "Legal services": ["69T75"],
    "Other Business services": ["69T75", "77T82"],
    "Computer related services": ["62T63"],
    "Other services": ["84", "85", "94T96"],
    "Human health and social care services": ["86T88"],
    "Recreation, entertainment and radio & TV broadcasting and other services": ["90T93", "58T60"],
    "Solar energy": ["35"],
    "Wind energy": ["35"],
    "Other electricity sources": ["35"],
    "AI-DC": ["62T63"],
}


def _norm(name) -> str:
    return re.sub(r"\s+", " ", str(name)).strip().lower()


_IOT_TO_ICIO_NORM = {_norm(k): v for k, v in IOT_TO_ICIO.items()}


# ---------------------------------------------------------------------
# Reading the IOT workbook
# ---------------------------------------------------------------------

@dataclass
class IndiaIot:
    sectors: list[str]                  # IOT sector names (incl. AI-DC)
    dc_inputs: pd.Series                # AI-DC purchases by IOT sector
    dc_sales: pd.Series                 # AI-DC sales by purchasing IOT sector
    dc_output: float
    dc_value_added: float
    cs_output: float                    # Computer related services, before AI-DC
    cs_value_added: float
    source: str


def load_india_iot(path: Path, sheet: str = DEFAULT_SHEET) -> IndiaIot:
    """Read the AI-DC column, row, outputs and value added from the IOT sheet."""
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    if sheet not in wb.sheetnames:
        raise ValueError(f"Sheet {sheet!r} not found in {path}; available: {wb.sheetnames[:10]}")
    rows = [list(r) for r in wb[sheet].iter_rows(values_only=True)]

    # Header row: the one containing the AI-DC column label.
    header_idx = next(
        (i for i, r in enumerate(rows) if any(_norm(c) == _norm(AIDC_LABEL) for c in r if c is not None)),
        None,
    )
    if header_idx is None:
        raise ValueError(f"No {AIDC_LABEL!r} column header found in sheet {sheet!r}.")
    header = rows[header_idx]
    col_of = {_norm(c): j for j, c in enumerate(header) if c is not None}

    name_col = next(
        j for j in range(len(header))
        if any(_norm(r[j]) == _norm(AIDC_LABEL) for r in rows[header_idx + 1:] if j < len(r))
    )
    row_of = {}
    for i, r in enumerate(rows[header_idx + 1:], start=header_idx + 1):
        if name_col < len(r) and r[name_col] is not None:
            row_of.setdefault(_norm(r[name_col]), i)

    sectors = [
        str(header[j]).strip() for j in range(len(header))
        if header[j] is not None and _norm(header[j]) in row_of and _norm(header[j]) in _IOT_TO_ICIO_NORM
    ]
    unknown = [
        str(header[j]).strip() for j in range(len(header))
        if header[j] is not None and _norm(header[j]) in row_of and _norm(header[j]) not in _IOT_TO_ICIO_NORM
    ]
    if unknown:
        raise ValueError(f"IOT sectors without an ICIO mapping (add them to IOT_TO_ICIO): {unknown}")

    def value(i, j) -> float:
        v = rows[i][j] if j < len(rows[i]) else None
        return float(v) if isinstance(v, (int, float)) else 0.0

    dc_c = col_of[_norm(AIDC_LABEL)]
    dc_r = row_of[_norm(AIDC_LABEL)]
    cs_c = col_of[_norm(CS_LABEL)]

    dc_inputs = pd.Series({s: value(row_of[_norm(s)], dc_c) for s in sectors})
    dc_sales = pd.Series({s: value(dc_r, col_of[_norm(s)]) for s in sectors})

    va_r = row_of.get(_norm("Value Added"))
    out_r = row_of.get(_norm("New Output")) or row_of.get(_norm("Old Output"))
    if va_r is None or out_r is None:
        raise ValueError("The IOT sheet needs 'Value Added' and 'New Output' rows.")
    dc_output = value(out_r, dc_c)
    dc_va = value(va_r, dc_c)
    cs_output_new = value(out_r, cs_c)
    cs_va = value(va_r, cs_c)

    # Computer related services output before the AI-DC purchases were added.
    old_col = col_of.get(_norm("Old Total Output"))
    cs_row = row_of[_norm(CS_LABEL)]
    cs_output = value(cs_row, old_col) if old_col is not None else cs_output_new - dc_inputs[CS_LABEL]

    if dc_output <= 0 or cs_output <= 0:
        raise ValueError(f"Non-positive outputs in the IOT: AI-DC {dc_output}, computer services {cs_output}.")
    if not np.isclose(dc_inputs.sum() + dc_va, dc_output, rtol=1e-3):
        LOG.warning(
            "AI-DC inputs (%.1f) + value added (%.1f) differ from its output (%.1f).",
            dc_inputs.sum(), dc_va, dc_output,
        )

    return IndiaIot(
        sectors=sectors,
        dc_inputs=dc_inputs,
        dc_sales=dc_sales,
        dc_output=dc_output,
        dc_value_added=dc_va,
        cs_output=cs_output,
        cs_value_added=cs_va,
        source=f"{Path(path).name}:{sheet}",
    )


# ---------------------------------------------------------------------
# Split plan
# ---------------------------------------------------------------------

@dataclass
class IndiaSplitPlan:
    parent_key: str
    dc_key: str
    oth_key: str
    weight: float                       # DC share of parent output
    dc_input_coefficients: pd.Series    # IOT sector -> coefficient (per unit DC output)
    dc_sales_shares: pd.Series          # IOT purchasing sector -> share of DC sales
    dc_va_share: float
    oth_va_share: float
    diagnostics: dict = field(default_factory=dict)

    @property
    def sub_keys(self) -> list[str]:
        return [self.dc_key, self.oth_key]


def prepare_india_split(
    iot_file: Path,
    parent_sector: str = "62T63",
    sheet: str = DEFAULT_SHEET,
    dc_share_override: Optional[float] = None,
) -> IndiaSplitPlan:
    iot = load_india_iot(iot_file, sheet)
    if dc_share_override is not None:
        if not 0 < dc_share_override < 1:
            raise ValueError("--india-dc-share must lie strictly between zero and one.")
        weight = float(dc_share_override)
    else:
        weight = iot.dc_output / (iot.dc_output + iot.cs_output)

    parent_key = f"{HOME_COUNTRY}_{parent_sector}"
    plan = IndiaSplitPlan(
        parent_key=parent_key,
        dc_key=f"{parent_key}_{DC_SUFFIX}",
        oth_key=f"{parent_key}_{OTH_SUFFIX}",
        weight=weight,
        dc_input_coefficients=iot.dc_inputs / iot.dc_output,
        dc_sales_shares=iot.dc_sales / iot.dc_sales.sum(),
        dc_va_share=iot.dc_value_added / iot.dc_output,
        oth_va_share=iot.cs_value_added / (iot.cs_output + iot.dc_inputs.get(CS_LABEL, 0.0)),
        diagnostics={
            "source": iot.source,
            "iot_dc_output": iot.dc_output,
            "iot_computer_services_output": iot.cs_output,
            "dc_share_of_parent_output": weight,
            "dc_share_overridden": dc_share_override is not None,
            "iot_dc_value_added_share": iot.dc_value_added / iot.dc_output,
            "iot_dc_electricity_coefficient": float(
                iot.dc_inputs[[s for s in iot.sectors if IOT_TO_ICIO[s] == ["35"] and s != "Gas"]].sum()
                / iot.dc_output
            ),
        },
    )
    LOG.info(
        "India IOT: AI-DC is %.3f%% of IT services output; AI-DC value added %.1f%% of output, "
        "electricity %.2f%% of output.",
        100 * weight, 100 * plan.dc_va_share, 100 * plan.diagnostics["iot_dc_electricity_coefficient"],
    )
    return plan


def _icio_shares(iot_values: pd.Series, keys_by_sector: dict[str, list[int]], x: np.ndarray) -> dict[str, float]:
    """Map IOT-sector values to ICIO sector codes (shared by Indian output)."""
    out: dict[str, float] = {}
    for iot_sector, v in iot_values.items():
        if v == 0:
            continue
        codes = IOT_TO_ICIO[iot_sector]
        weights = np.array([sum(x[i] for i in keys_by_sector.get(c, [])) for c in codes], dtype=float)
        if weights.sum() <= 0:
            weights = np.ones(len(codes))
        for c, wgt in zip(codes, weights / weights.sum()):
            if wgt > 0:
                out[c] = out.get(c, 0.0) + float(v) * float(wgt)
    return out


# ---------------------------------------------------------------------
# ICIO split
# ---------------------------------------------------------------------

def split_icio_india(
    A: pd.DataFrame,
    X: pd.Series,
    FD: pd.DataFrame,
    plan: IndiaSplitPlan,
    rebalance: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict]:
    keys = A.index.astype(str).tolist()
    if plan.parent_key not in keys:
        raise ValueError(f"{plan.parent_key} is not an ICIO industry.")
    if plan.dc_key in keys:
        raise ValueError("ICIO already contains the Indian data-centre subsector; it has been split already.")

    n = len(keys)
    j = keys.index(plan.parent_key)
    known = {country_of(k) for k in keys}
    country = np.array([country_of(k, known) for k in keys])
    sector = np.array([normalise_sector_code(sector_of(k)) for k in keys])
    home = country == HOME_COUNTRY

    x = X.reindex(keys).to_numpy(dtype=float)
    a = A.loc[keys, keys].to_numpy(dtype=float)
    Z = a * x[None, :]
    fd = FD.reindex(keys).to_numpy(dtype=float)
    fd_cols = FD.columns.astype(str).tolist()

    w = plan.weight
    x_sub = np.array([w * x[j], (1 - w) * x[j]])

    home_by_sector: dict[str, list[int]] = {}
    for i in np.flatnonzero(home):
        home_by_sector.setdefault(sector[i], []).append(i)

    # ---------------- columns ----------------
    coeff_by_icio = _icio_shares(plan.dc_input_coefficients, home_by_sector, x)
    dc_seed = np.zeros(n)
    unmatched = []
    for code, coeff in coeff_by_icio.items():
        rows = np.flatnonzero(sector == code)
        if not len(rows):
            unmatched.append(code)
            continue
        mix = Z[rows, j].clip(min=0)
        if mix.sum() <= 0:
            mix = Z[np.ix_(rows, np.flatnonzero(home))].sum(axis=1).clip(min=0)
        if mix.sum() <= 0:
            mix = (country[rows] == HOME_COUNTRY).astype(float)
        dc_seed[rows] += coeff * x_sub[0] * mix / mix.sum()
    if unmatched:
        raise ValueError(f"ICIO has no industries for sectors {unmatched}.")

    parent_col = Z[:, j]
    oth_seed = parent_col - dc_seed
    seed_cols = np.column_stack([dc_seed, oth_seed])

    parent_va = x[j] - parent_col.sum()
    va_targets = sr.scaled_va_targets(parent_va, x_sub, [plan.dc_va_share, plan.oth_va_share])

    if rebalance:
        new_cols, col_info = sr.ras_split_columns(seed_cols, parent_col, x_sub, va_targets, label="India DC columns")
    else:
        # Without RAS: cap the DC seed at the parent's purchases.
        dc = np.minimum(dc_seed, parent_col.clip(min=0))
        new_cols, col_info = np.column_stack([dc, parent_col - dc]), {}

    Zc = np.concatenate([Z[:, :j], new_cols, Z[:, j + 1:]], axis=1)
    new_keys = keys[:j] + plan.sub_keys + keys[j + 1:]
    col_country = np.concatenate([country[:j], [HOME_COUNTRY, HOME_COUNTRY], country[j + 1:]])
    col_sector = np.concatenate([sector[:j], ["62T63_DC", "62T63_OTH"], sector[j + 1:]])

    # ---------------- rows ----------------
    parent_row = np.concatenate([Zc[j], fd[j]])         # (n+1) + nfd
    ncol = Zc.shape[1]
    col_home = col_country == HOME_COUNTRY
    col_x = np.concatenate([x[:j], x_sub, x[j + 1:]])

    purchasers: dict[str, list[int]] = {}
    for c in np.flatnonzero(col_home):
        purchasers.setdefault(col_sector[c], []).append(int(c))
    # Sales to Computer related services go to the rest of IT services.
    purchasers["62T63"] = [j + 1]
    purchasers.pop("62T63_OTH", None)
    purchasers.pop("62T63_DC", None)

    sales_by_icio = _icio_shares(plan.dc_sales_shares, purchasers, col_x)
    dc_row_seed = np.zeros(parent_row.shape)
    for code, share in sales_by_icio.items():
        cols = purchasers.get(code, [])
        if not cols:
            continue
        weights = parent_row[cols].clip(min=0)
        if weights.sum() <= 0:
            weights = col_x[cols]
        dc_row_seed[cols] += share * x_sub[0] * weights / weights.sum()

    oth_row_seed = parent_row - dc_row_seed
    row_seed = np.vstack([dc_row_seed, oth_row_seed])
    if rebalance:
        seed = np.where(parent_row[None, :] > 0, np.maximum(row_seed, 0.0), row_seed)
        new_rows, row_info = ras_signed(seed, x_sub, parent_row, label="India DC rows")
    else:
        dc = np.minimum(dc_row_seed, parent_row.clip(min=0))
        dc *= x_sub[0] / dc.sum() if dc.sum() > 0 else 0.0
        new_rows, row_info = np.vstack([dc, parent_row - dc]), {}

    new_Z_rows, new_fd_rows = new_rows[:, :ncol], new_rows[:, ncol:]
    Zs = np.concatenate([Zc[:j], new_Z_rows, Zc[j + 1:]], axis=0)
    fds = np.concatenate([fd[:j], new_fd_rows, fd[j + 1:]], axis=0)
    xs = col_x

    As = np.divide(Zs, xs[None, :], out=np.zeros_like(Zs), where=xs[None, :] > 0)
    A2 = pd.DataFrame(As, index=new_keys, columns=new_keys)
    X2 = pd.Series(xs, index=new_keys, name=X.name)
    FD2 = pd.DataFrame(fds, index=new_keys, columns=fd_cols)

    # ---------------- checks ----------------
    back_cols = Zs[:, j] + Zs[:, j + 1]
    back_rows = Zs[j] + Zs[j + 1]
    Z_back = np.delete(np.delete(Zs, j + 1, axis=1), j + 1, axis=0)
    Z_back[:, j] = np.delete(back_cols, j + 1)
    Z_back[j] = np.delete(back_rows, j + 1)
    Z_back[j, j] = Zs[j:j + 2, j:j + 2].sum()
    scale = max(1.0, float(np.abs(Z).max()))
    if not np.allclose(Z_back, Z, rtol=1e-6, atol=1e-6 * scale):
        raise RuntimeError("India split does not aggregate back to the original ICIO.")
    if not np.allclose(fds[j] + fds[j + 1], fd[j], atol=1e-6 * max(1.0, float(np.abs(fd).max()))):
        raise RuntimeError("India split final demand does not aggregate back.")

    row_sum = Zs[j:j + 2].sum(axis=1) + fds[j:j + 2].sum(axis=1)
    va = xs[j:j + 2] - Zs[:, j:j + 2].sum(axis=0)
    elec_rows = [i for i, k in enumerate(new_keys) if col_country[i] == HOME_COUNTRY and col_sector[i] == "35"]
    achieved = {}
    for code in coeff_by_icio:
        rows = [i for i, k in enumerate(new_keys) if col_sector[i] == code]
        achieved[code] = float(Zs[rows, j].sum() / xs[j])

    diag = {
        **plan.diagnostics,
        "parent_output_usd_mn": float(x[j]),
        "subsector_output_usd_mn": dict(zip(plan.sub_keys, xs[j:j + 2].tolist())),
        "row_sum_minus_output": dict(zip(plan.sub_keys, (row_sum - xs[j:j + 2]).tolist())),
        "value_added_share": dict(zip(plan.sub_keys, (va / xs[j:j + 2]).tolist())),
        "value_added_target_share": dict(zip(plan.sub_keys, (va_targets / x_sub).tolist())),
        "dc_electricity_coefficient": float(As[elec_rows, j].sum()) if elec_rows else float("nan"),
        "dc_input_coefficient_target": coeff_by_icio,
        "dc_input_coefficient_achieved": achieved,
        "column_ras": col_info,
        "row_ras": row_info,
    }
    LOG.info(
        "Split %s (output %.1f USD mn) into %s with weights [%.6f, %.6f]; DC value added %.1f%%.",
        plan.parent_key, x[j], plan.sub_keys, w, 1 - w, 100 * va[0] / xs[j],
    )
    return A2, X2, FD2, diag


def write_split_diagnostics(out_dir: Path, plan: IndiaSplitPlan, diag: dict) -> None:
    import json

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "india_split_diagnostics.json").write_text(json.dumps(diag, indent=2, default=float))
    coeff = pd.DataFrame({
        "icio_sector": list(diag["dc_input_coefficient_target"]),
        "iot_coefficient": list(diag["dc_input_coefficient_target"].values()),
        "icio_coefficient_after_ras": [diag["dc_input_coefficient_achieved"].get(k) for k in diag["dc_input_coefficient_target"]],
    })
    coeff.to_csv(out_dir / "india_dc_input_coefficients.csv", index=False)
