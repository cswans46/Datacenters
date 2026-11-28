#!/usr/bin/env python3
"""
Split Canadian data centres out of the OECD ICIO using the Canadian
digital supply-use tables (DSUT).

Methodology (proposal: "How Canadian datacenter buildout drives embodied
emission-intensity of foreign and domestic exports")
-------------------------------------------------------------------------
Only Canada is split. India and every other economy are left unchanged.

1. DSUT step (2020 Canadian DSUT, StatCan, available on request)
   The DSUT carves a "cloud computing services" product row out of the
   conventional SUT product rows. From it we take

       s_cloud = cloud computing services produced by NAICS 51821
                 / total output of NAICS 51821                     (2020)

   which splits NAICS 51821 (computing infrastructure providers, data
   processing, web hosting and related services) into

       CLD  cloud computing services
       DPH  all other infrastructure provision, data processing,
            web hosting and related services.

   StatCan assumes imports and uses of cloud services are split in the
   same proportion as output, so by default CLD and DPH share the 51821
   cost and sales structures. If the DSUT use-table values for the cloud
   product are entered in the template, CLD and DPH get their own sales
   (row) distributions.

   Projection to the SUT year is the user's job (E3MC/MSMR, proprietary):
   pass the projected share with --canada-cloud-share-override. Without
   it the 2020 share is used unchanged and a warning is logged.

2. SUT step (2022 StatCan SUT, table 36-10-0478-01, Canada, basic prices)
   The share of NAICS 51821 in the ICIO sector J62_63 ("IT and other
   information services") is computed from the 2022 detail-level SUT:

       w_518 = output(51821) / output(J62_63 industries)

   where the J62_63 industries are NAICS 5415, 518 and 519 by default.
   The SUT also supplies (a) input intensities of 51821 and of the rest of
   J62_63, grouped into ICIO input sectors, and (b) the destination
   structure (intermediate use, domestic final use, exports) of the
   products each group produces.

3. ICIO split
   CAN_J62_63 is replaced by CAN_J62_63_CLD, CAN_J62_63_DPH and
   CAN_J62_63_OTH with output weights

       w_CLD = w_518 * s_cloud,  w_DPH = w_518 * (1 - s_cloud),
       w_OTH = 1 - w_518.

   Columns (cost structures): each input cell Z[i, J] is allocated to
   the subsectors in proportion to w_k * r_k(g), where r_k(g) is the SUT
   input intensity of subsector k for input group g relative to the
   J62_63 aggregate (electricity, telecoms, hardware, real estate, ...).
   Unmapped inputs are allocated by output weight.

   Rows (sales structures): the J62_63 row is divided into four blocks
   (domestic intermediate, domestic final demand, foreign intermediate,
   foreign final demand). Subsector shares of each block are seeded with
   w_k * rho_k(block), where rho_k is the SUT destination structure of
   the subsector's products relative to J62_63, and then RAS-balanced so
   subsector rows add to subsector output and blocks add to the original
   block totals. Within a block, sales are allocated in proportion to the
   original J62_63 row.

   The split is exact in aggregate: adding the three subsectors back
   together returns the original ICIO rows, columns, output and final
   demand.

4. Emissions extension
   Direct emissions of CAN_J62_63 are split across subsectors in
   proportion to output (equal intensity, default) or to electricity
   purchases ("electricity_input").

Data files
----------
* StatCan SUT: downloaded and cached by ``fetch-sut`` (needs internet
  access to www150.statcan.gc.ca) or supplied as a local CSV/ZIP.
* DSUT: write the template with ``write-templates`` and fill it from the
  2020 DSUT received from StatCan.

Command-line use
----------------
    python canada_dsut_split.py write-templates --out-dir .
    python canada_dsut_split.py fetch-sut --cache-dir statcan_cache --year 2022
    python canada_dsut_split.py inspect-sut --sut-file statcan_cache/... \
        --year 2022 --out-dir canada_split_diagnostics

The split itself runs from global_datacentre_icio_model_v2.py with
--split-canada-dsut.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


LOG = logging.getLogger("datacentre_icio.canada_split")

HOME_COUNTRY = "CAN"
PARENT_SECTOR = "J62_63"
# ICIO sector labels for "IT and other information services": J62_63 in the
# 2023 edition, 62T63 in the 2025 edition (numeric codes).
PARENT_SECTOR_CANDIDATES = ("J62_63", "62T63")


def resolve_parent_sector(keys, home_country: str = "CAN") -> str:
    """The J62_63 / 62T63 sector label used by this ICIO edition."""
    keys = set(map(str, keys))
    for candidate in PARENT_SECTOR_CANDIDATES:
        if f"{home_country}_{candidate}" in keys:
            return candidate
    raise ValueError(
        f"No IT and information services industry for {home_country} in the ICIO; "
        f"tried {[f'{home_country}_{c}' for c in PARENT_SECTOR_CANDIDATES]}."
    )


def normalise_sector_code(code: str) -> str:
    """Drop the ISIC section letter of 2023-edition codes: C26 -> 26, D35 -> 35."""
    return re.sub(r"^[A-U](?=\d)", "", str(code).strip())

SUBSECTOR_SUFFIXES = {
    "CLD": "Cloud computing services (DSUT, part of NAICS 51821)",
    "DPH": "Other NAICS 51821: computing infrastructure, data processing, web hosting",
    "OTH": "Rest of J62_63: NAICS 5415, 519 and other IT/information services",
}

STATCAN_SUT_PID = "36100478"
STATCAN_WDS_URL = (
    "https://www150.statcan.gc.ca/t1/wds/rest/getFullTableDownloadCSV/{pid}/en"
)
STATCAN_DIRECT_URL = "https://www150.statcan.gc.ca/n1/tbl/csv/{pid}-eng.zip"

# Industry members of the StatCan SUT (matched case-insensitively against
# the full member label, which normally ends in an IOIC code such as
# "[BS518000]").
DEFAULT_518_PATTERN = (
    r"data processing, hosting|computing infrastructure provider|\[BS518"
)
DEFAULT_OTHER_J_PATTERN = (
    r"computer systems design|other information services|\[BS5415|\[BS519"
)

# SUT input products grouped into ICIO input sectors. First match wins.
# "icio_sector_regex" is matched against ICIO sector codes: letter codes of
# the 2023 edition (e.g. D35, J61) and numeric codes of the 2025 edition
# (e.g. 35, 61, 41T43).
DEFAULT_PRODUCT_CONCORDANCE = [
    ("electricity", r"electric power|electricity|steam|natural gas distribution", r"^D35$|^D$|^35$"),
    ("telecommunications", r"telecommunicat|wired|wireless|satellite", r"^J61|^61$"),
    ("it_services", r"data processing|hosting|computer systems design|information services|web portal|computer services", r"^J62|^62T63"),
    ("software_publishing", r"software|publish", r"^J58|^58T60$"),
    ("computer_electronic", r"computer|electronic|semiconductor|communications equipment|audio and video equipment", r"^C26|^26$"),
    ("electrical_equipment", r"electrical equipment|electric equipment|wire and cable|batter|transformer|switchgear", r"^C27|^27$"),
    ("machinery", r"machinery|ventilation|heating|air-conditioning|refrigeration", r"^C28|^28$"),
    ("construction", r"construction|repair of (non-)?residential", r"^F|^41T43$"),
    ("real_estate", r"real estate|rental|leasing|rent\b|lessors", r"^L|^68$"),
    ("finance_insurance", r"financ|insurance|bank|deposit|credit|interest|securities", r"^K|^64T66$"),
    ("professional", r"legal|accounting|consult|engineering|architect|advertising|research|design services|professional", r"^M|^69T75$"),
    ("administrative", r"administrative|employment|security|cleaning|janitorial|travel|office support|waste", r"^N|^77T82$"),
    ("trade_margins", r"wholesale|retail|margin", r"^G|^45T47$"),
    ("transport", r"transport|courier|postal|warehous|trucking|pipeline", r"^H|^49$|^50$|^51$|^52$|^53$"),
]

SUT_BLOCKS = ("intermediate", "domestic_final", "exports")
ICIO_BLOCKS = ("domestic_intermediate", "domestic_final", "foreign_intermediate", "foreign_final")
ICIO_TO_SUT_BLOCK = {
    "domestic_intermediate": "intermediate",
    "domestic_final": "domestic_final",
    "foreign_intermediate": "exports",
    "foreign_final": "exports",
}

DSUT_TEMPLATE_ROWS = [
    # measure, required, description
    ("cloud_output_industry_51821", True,
     "Output of the DSUT product 'cloud computing services' produced by industry NAICS 51821 "
     "(computing infrastructure providers, data processing, web hosting and related services), basic prices."),
    ("output_industry_51821", True,
     "Total output of industry NAICS 51821 in the DSUT, all products, basic prices."),
    ("cloud_output_all_industries", False,
     "Output of cloud computing services by all industries (information only; cloud produced by 5415/5112 is not used)."),
    ("cloud_imports", False, "Imports of cloud computing services (information only)."),
    ("cloud_use_intermediate", False,
     "DSUT use table, cloud computing services row: intermediate use by Canadian industries."),
    ("cloud_use_domestic_final", False,
     "DSUT use table, cloud computing services row: domestic final use (households, NPISH, government, GFCF, inventories)."),
    ("cloud_exports", False, "DSUT use table, cloud computing services row: exports."),
    ("noncloud_hosting_use_intermediate", False,
     "DSUT use table, the data processing/hosting product row left after cloud is carved out: intermediate use."),
    ("noncloud_hosting_use_domestic_final", False,
     "Same product row: domestic final use."),
    ("noncloud_hosting_exports", False, "Same product row: exports."),
]


# ---------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------

def country_of(label: str, known_countries: Optional[set[str]] = None) -> str:
    """Country code of an ICIO industry or final-demand label."""
    label = str(label)

    if label.startswith("EU27_2020"):
        return "EU27_2020"

    if known_countries:
        for part in (label.split("_", 1)[0], label.rsplit("_", 1)[-1]):
            if part in known_countries:
                return part
        if label in known_countries:
            return label

    return label.split("_", 1)[0]


def sector_of(io_key: str) -> str:
    io_key = str(io_key)
    if io_key.startswith("EU27_2020_"):
        return io_key[len("EU27_2020_"):]
    return io_key.split("_", 1)[1] if "_" in io_key else io_key


def subsector_key(parent_key: str, suffix: str) -> str:
    return f"{parent_key}_{suffix}"


# ---------------------------------------------------------------------
# StatCan SUT: download, cache and parse
# ---------------------------------------------------------------------

def _download(url: str, dest: Path, timeout: int = 600) -> None:
    LOG.info("Downloading %s", url)
    request = urllib.request.Request(url, headers={"User-Agent": "datacenters-icio/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response, open(dest, "wb") as fh:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)


def fetch_statcan_sut(
    cache_dir: Path,
    year: int,
    pid: str = STATCAN_SUT_PID,
    geo: str = "Canada",
    force: bool = False,
) -> Path:
    """
    Download StatCan table 36-10-0478-01 and cache the Canada rows for one
    year as a CSV. Returns the path of the filtered CSV.

    The full table is large (all provinces and years), so it is read in
    chunks and only the requested geography and year are kept.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    filtered = cache_dir / f"statcan_{pid}_{geo.lower()}_{year}.csv"
    if filtered.exists() and not force:
        LOG.info("Using cached StatCan SUT extract: %s", filtered)
        return filtered

    zip_path = cache_dir / f"{pid}-eng.zip"
    if force or not zip_path.exists():
        url = STATCAN_DIRECT_URL.format(pid=pid)
        try:
            with urllib.request.urlopen(STATCAN_WDS_URL.format(pid=pid), timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("status") == "SUCCESS" and payload.get("object"):
                url = payload["object"]
        except Exception as exc:  # noqa: BLE001 - fall back to the direct URL
            LOG.warning("StatCan WDS lookup failed (%s); using %s", exc, url)
        _download(url, zip_path)

    return extract_statcan_sut(zip_path, filtered, year=year, geo=geo)


def extract_statcan_sut(source: Path, dest: Path, year: int, geo: str = "Canada") -> Path:
    """Filter a StatCan full-table ZIP or CSV to one geography and year."""
    source = Path(source)

    def chunks():
        if source.suffix.lower() == ".zip":
            with zipfile.ZipFile(source) as zf:
                names = [n for n in zf.namelist() if n.lower().endswith(".csv") and "metadata" not in n.lower()]
                if len(names) != 1:
                    raise ValueError(f"Expected one data CSV in {source}, found {names}")
                with zf.open(names[0]) as raw:
                    text = io.TextIOWrapper(raw, encoding="utf-8-sig")
                    yield from pd.read_csv(text, dtype=str, chunksize=500_000)
        else:
            yield from pd.read_csv(source, dtype=str, chunksize=500_000, encoding="utf-8-sig")

    kept = []
    for chunk in chunks():
        cols = _resolve_columns(chunk)
        mask = chunk[cols["year"]].astype(str).str.startswith(str(year))
        mask &= chunk[cols["geo"]].astype(str).str.strip().eq(geo)
        if mask.any():
            kept.append(chunk.loc[mask])

    if not kept:
        raise ValueError(f"No rows for GEO={geo!r}, year={year} in {source}")

    result = pd.concat(kept, ignore_index=True)
    result.to_csv(dest, index=False)
    LOG.info("Wrote %s rows to %s", len(result), dest)
    return dest


_COLUMN_ALIASES = {
    "year": ["ref_date", "year"],
    "geo": ["geo", "geography"],
    "su": ["supply and use", "supply_and_use", "supply/use", "supply and use category"],
    "valuation": ["valuation"],
    "industry": ["industry", "industries"],
    "product": ["product", "products"],
    "value": ["value"],
}


def _resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    resolved = {}
    for key, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lookup:
                resolved[key] = lookup[alias]
                break
    missing = {"year", "geo"} - set(resolved)
    if missing:
        raise ValueError(f"StatCan table is missing columns {sorted(missing)}. Columns: {list(df.columns)}")
    return resolved


def member_code(label: str) -> str:
    match = re.search(r"\[([^\]]+)\]\s*$", str(label))
    return match.group(1).strip() if match else ""


_FINAL_PATTERN = re.compile(
    r"final consumption|household|government|non-profit|npish|capital formation|fixed capital|"
    r"inventor|investment|final demand|final use",
    re.IGNORECASE,
)
_OTHER_PATTERN = re.compile(r"margin|taxes|subsid|value added|compensation|surplus|mixed income", re.IGNORECASE)


def classify_column_member(su: str, industry: str) -> str:
    """
    Classify a StatCan SUT column member (the "Industry" dimension, which
    in the use table also carries final-demand categories, imports and
    exports).
    """
    name = str(industry).strip()
    lowered = name.lower()
    code = member_code(name)

    if lowered.startswith("total"):
        return "total"
    if "export" in lowered:
        return "exports"
    if "import" in lowered:
        return "imports"
    if _FINAL_PATTERN.search(lowered):
        return "domestic_final"
    if code[:2].upper() in {"BS", "NP", "GS"}:
        return "industry"
    if _OTHER_PATTERN.search(lowered):
        return "other"
    return "unclassified"


@dataclass
class SutData:
    year: int
    supply: pd.DataFrame       # columns: product, column, category, value
    use: pd.DataFrame          # columns: product, column, category, value
    members: pd.DataFrame      # diagnostics: column member classification


def load_statcan_sut(
    path: Path,
    year: int,
    geo: str = "Canada",
    member_overrides: Optional[Path] = None,
) -> SutData:
    """
    Read a StatCan SUT extract (36-10-0478-01 layout) into long supply and
    use tables for one year and geography.

    member_overrides: optional CSV with columns ``member`` and ``category``
    (industry, domestic_final, exports, imports, total, other) to correct
    the automatic classification. Check ``sut_column_members.csv`` from
    ``inspect-sut`` first.
    """
    path = Path(path)
    if path.suffix.lower() == ".zip":
        cached = path.with_name(f"statcan_{path.stem}_{geo.lower()}_{year}.csv")
        path = extract_statcan_sut(path, cached, year=year, geo=geo)

    raw = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    cols = _resolve_columns(raw)
    for key in ("su", "industry", "product", "value"):
        if key not in cols:
            raise ValueError(
                f"StatCan SUT file {path} has no '{key}' column. Columns: {list(raw.columns)}"
            )

    df = raw.loc[
        raw[cols["year"]].astype(str).str.startswith(str(year))
        & raw[cols["geo"]].astype(str).str.strip().eq(geo)
    ].copy()
    if df.empty:
        raise ValueError(f"No rows for {geo} {year} in {path}")

    df["su"] = df[cols["su"]].astype(str).str.strip()
    df["column"] = df[cols["industry"]].astype(str).str.strip()
    df["product"] = df[cols["product"]].astype(str).str.strip()
    df["valuation"] = df[cols["valuation"]].astype(str).str.strip() if "valuation" in cols else ""
    df["value"] = pd.to_numeric(df[cols["value"]], errors="coerce").fillna(0.0)

    side = df["su"].str.lower()
    df["side"] = np.where(side.str.startswith("supply"), "supply", np.where(side.str.startswith("use"), "use", ""))
    if not df["side"].isin(["supply", "use"]).any():
        raise ValueError(f"Could not identify supply and use rows from values {df['su'].unique()[:10]}")

    members = (
        df[["side", "su", "column"]].drop_duplicates().reset_index(drop=True)
    )
    members["category"] = [classify_column_member(s, c) for s, c in zip(members["su"], members["column"])]

    if member_overrides is not None:
        overrides = pd.read_csv(member_overrides, dtype=str)
        overrides = dict(zip(overrides["member"].str.strip(), overrides["category"].str.strip()))
        members["category"] = [overrides.get(c, cat) for c, cat in zip(members["column"], members["category"])]

    unclassified = members.loc[members["category"].eq("unclassified"), "column"].unique().tolist()
    if unclassified:
        LOG.warning(
            "StatCan SUT column members not classified (ignored): %s. "
            "Supply a member override file if any are industries or final uses.",
            unclassified[:20],
        )

    df = df.merge(members[["side", "su", "column", "category"]], on=["side", "su", "column"], how="left")
    df = df.loc[~df["product"].str.lower().str.startswith("total")]

    def pick_valuation(frame: pd.DataFrame, preferred: tuple[str, ...]) -> pd.DataFrame:
        vals = frame["valuation"].str.lower()
        for wanted in preferred:
            sel = vals.str.contains(wanted, regex=False)
            if sel.any():
                return frame.loc[sel]
        return frame

    supply = pick_valuation(df.loc[df["side"].eq("supply")], ("basic",))
    use = pick_valuation(df.loc[df["side"].eq("use")], ("basic", "purchaser"))

    if supply.empty or use.empty:
        raise ValueError("StatCan SUT extract must contain both supply and use rows.")

    keep = ["product", "column", "category", "value"]
    return SutData(year=year, supply=supply[keep].reset_index(drop=True), use=use[keep].reset_index(drop=True), members=members)


# ---------------------------------------------------------------------
# SUT aggregates for J62_63 and NAICS 51821
# ---------------------------------------------------------------------

@dataclass
class SutAggregates:
    year: int
    industries_518: list[str]
    industries_other_j: list[str]
    output: pd.Series                 # index 518, OTH, J
    input_intensity: pd.DataFrame     # index input group, columns 518, OTH, J (u/x)
    sales: pd.DataFrame               # index SUT block, columns 518, OTH, J (values)
    products_518: list[str]
    products_other_j: list[str]
    diagnostics: dict = field(default_factory=dict)


def _group_of_product(name: str, concordance: list[tuple[str, str, str]]) -> str:
    lowered = str(name).lower()
    for group, pattern, _ in concordance:
        if re.search(pattern, lowered):
            return group
    return "other"


def compute_sut_aggregates(
    sut: SutData,
    pattern_518: str = DEFAULT_518_PATTERN,
    pattern_other_j: str = DEFAULT_OTHER_J_PATTERN,
    concordance: list[tuple[str, str, str]] = DEFAULT_PRODUCT_CONCORDANCE,
    product_share_threshold: float = 0.5,
) -> SutAggregates:
    supply_ind = sut.supply.loc[sut.supply["category"].eq("industry")]
    industries = supply_ind["column"].unique()

    ind_518 = [c for c in industries if re.search(pattern_518, c, re.IGNORECASE)]
    ind_oth = [c for c in industries if re.search(pattern_other_j, c, re.IGNORECASE) and c not in ind_518]

    if not ind_518:
        raise ValueError(
            "No StatCan SUT industry matched the NAICS 51821 pattern "
            f"{pattern_518!r}. Industries include: {sorted(industries)[:40]}"
        )
    if not ind_oth:
        raise ValueError(
            "No StatCan SUT industry matched the rest-of-J62_63 pattern "
            f"{pattern_other_j!r}."
        )

    LOG.info("SUT industries for NAICS 51821: %s", ind_518)
    LOG.info("SUT industries for the rest of J62_63: %s", ind_oth)

    by_industry = supply_ind.groupby("column")["value"].sum()
    x_518 = float(by_industry.reindex(ind_518).sum())
    x_oth = float(by_industry.reindex(ind_oth).sum())
    if x_518 <= 0 or x_oth <= 0:
        raise ValueError(f"Non-positive SUT output: 51821={x_518}, rest of J62_63={x_oth}")
    output = pd.Series({"518": x_518, "OTH": x_oth, "J": x_518 + x_oth})

    # Products made mainly by each group (domestic output share).
    prod_by = supply_ind.assign(
        grp=np.where(supply_ind["column"].isin(ind_518), "518", np.where(supply_ind["column"].isin(ind_oth), "OTH", "rest"))
    ).pivot_table(index="product", columns="grp", values="value", aggfunc="sum", fill_value=0.0)
    for grp in ("518", "OTH", "rest"):
        if grp not in prod_by.columns:
            prod_by[grp] = 0.0
    totals = prod_by.sum(axis=1).replace(0, np.nan)
    products_518 = prod_by.index[(prod_by["518"] / totals) >= product_share_threshold].tolist()
    products_oth = prod_by.index[(prod_by["OTH"] / totals) >= product_share_threshold].tolist()

    if not products_518:
        products_518 = [p for p in prod_by.index if re.search(pattern_518, p, re.IGNORECASE)]
        LOG.warning("No product is mostly produced by NAICS 51821; matched by name instead: %s", products_518)
    if not products_518:
        raise ValueError("Could not identify the NAICS 51821 product(s) in the SUT.")
    if not products_oth:
        raise ValueError("Could not identify products of the rest of J62_63 in the SUT.")

    # Input intensities by ICIO input group.
    use = sut.use
    inputs = use.loc[use["category"].eq("industry") & use["column"].isin(ind_518 + ind_oth)].copy()
    inputs["group"] = [_group_of_product(p, concordance) for p in inputs["product"]]
    inputs["grp"] = np.where(inputs["column"].isin(ind_518), "518", "OTH")
    u = inputs.pivot_table(index="group", columns="grp", values="value", aggfunc="sum", fill_value=0.0)
    for grp in ("518", "OTH"):
        if grp not in u.columns:
            u[grp] = 0.0
    u["J"] = u["518"] + u["OTH"]
    intensity = u[["518", "OTH", "J"]].div(output[["518", "OTH", "J"]], axis=1)

    # Destination structure of each product group.
    def sales_for(products: list[str]) -> pd.Series:
        rows = use.loc[use["product"].isin(products)]
        s = rows.groupby("category")["value"].sum()
        return pd.Series({
            "intermediate": float(s.get("industry", 0.0)),
            "domestic_final": float(s.get("domestic_final", 0.0)),
            "exports": float(s.get("exports", 0.0)),
        })

    sales = pd.DataFrame({"518": sales_for(products_518), "OTH": sales_for(products_oth)})
    sales["J"] = sales["518"] + sales["OTH"]
    if (sales.sum() <= 0).any():
        raise ValueError(f"SUT product uses are zero for a group:\n{sales}")

    diagnostics = {
        "sut_year": sut.year,
        "output_518": x_518,
        "output_rest_of_j62_63": x_oth,
        "share_518_in_j62_63": x_518 / (x_518 + x_oth),
        "industries_518": ind_518,
        "industries_rest_of_j62_63": ind_oth,
        "products_518": products_518,
        "products_rest_of_j62_63": products_oth,
    }

    return SutAggregates(
        year=sut.year,
        industries_518=ind_518,
        industries_other_j=ind_oth,
        output=output,
        input_intensity=intensity,
        sales=sales,
        products_518=products_518,
        products_other_j=products_oth,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------
# DSUT
# ---------------------------------------------------------------------

@dataclass
class DsutCloud:
    year: int
    cloud_share_of_51821_output: float
    cloud_sales: Optional[pd.Series] = None      # SUT blocks
    noncloud_sales: Optional[pd.Series] = None   # SUT blocks
    source: str = ""


def write_dsut_template(path: Path, year: int = 2020) -> Path:
    rows = [
        {
            "dsut_year": year,
            "measure": m,
            "value": "",
            "unit": "CAD thousands (any consistent unit; only ratios are used)",
            "required": "yes" if req else "no",
            "description": desc,
            "source_note": "",
        }
        for m, req, desc in DSUT_TEMPLATE_ROWS
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def load_dsut(path: Path) -> DsutCloud:
    df = pd.read_csv(path, dtype=str)
    for col in ("measure", "value"):
        if col not in df.columns:
            raise ValueError(f"DSUT file {path} must have a '{col}' column (see write-templates).")
    df["measure"] = df["measure"].str.strip()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    values = df.dropna(subset=["value"]).set_index("measure")["value"]
    if values.index.has_duplicates:
        raise ValueError(f"DSUT file has duplicate measures: {values.index[values.index.duplicated()].tolist()}")

    year = int(pd.to_numeric(df.get("dsut_year"), errors="coerce").dropna().iloc[0]) if "dsut_year" in df else 2020

    for measure in ("cloud_output_industry_51821", "output_industry_51821"):
        if measure not in values:
            raise ValueError(f"DSUT file {path} is missing the required measure '{measure}'.")

    cloud = float(values["cloud_output_industry_51821"])
    total = float(values["output_industry_51821"])
    if total <= 0 or cloud < 0 or cloud > total:
        raise ValueError(f"Invalid DSUT values: cloud={cloud}, total 51821 output={total}")

    def block(prefix: str, names: tuple[str, str, str]) -> Optional[pd.Series]:
        keys = [f"{prefix}{n}" for n in names]
        present = [k in values for k in keys]
        if not any(present):
            return None
        if not all(present):
            raise ValueError(f"DSUT file: give all or none of {keys}.")
        series = pd.Series(dict(zip(SUT_BLOCKS, [float(values[k]) for k in keys])))
        if (series < 0).any() or series.sum() <= 0:
            raise ValueError(f"DSUT file: invalid values for {keys}: {series.to_dict()}")
        return series

    cloud_sales = block("cloud_", ("use_intermediate", "use_domestic_final", "exports"))
    noncloud_sales = block("noncloud_hosting_", ("use_intermediate", "use_domestic_final", "exports"))
    if (cloud_sales is None) != (noncloud_sales is None):
        raise ValueError("DSUT file: give the cloud and non-cloud hosting use rows together, or neither.")

    LOG.info(
        "DSUT %s: cloud computing share of NAICS 51821 output = %.4f", year, cloud / total
    )
    return DsutCloud(
        year=year,
        cloud_share_of_51821_output=cloud / total,
        cloud_sales=cloud_sales,
        noncloud_sales=noncloud_sales,
        source=str(path),
    )


# ---------------------------------------------------------------------
# Split plan
# ---------------------------------------------------------------------

@dataclass
class CanadaSplitPlan:
    parent_key: str
    sub_keys: list[str]                 # CLD, DPH, OTH order
    weights: np.ndarray                 # output weights, sum to one
    input_ratio: pd.DataFrame           # index input group, columns sub_keys
    sales_rho: pd.DataFrame             # index SUT block, columns sub_keys
    concordance: list[tuple[str, str, str]]
    cloud_share: float
    share_518: float
    diagnostics: dict

    @property
    def cloud_key(self) -> str:
        return self.sub_keys[0]

    @property
    def hosting_key(self) -> str:
        return self.sub_keys[1]

    @property
    def other_key(self) -> str:
        return self.sub_keys[2]

    def dc_targets(self, mode: str) -> list[tuple[str, float]]:
        """Data-centre shock targets and weights for the Canadian proxy."""
        if mode == "cloud":
            return [(self.cloud_key, 1.0)]
        if mode == "cloud_and_hosting":
            w = self.weights[:2] / self.weights[:2].sum()
            return [(self.cloud_key, float(w[0])), (self.hosting_key, float(w[1]))]
        raise ValueError(f"Unknown Canadian data-centre target mode: {mode}")


def build_split_plan(
    sut_agg: SutAggregates,
    dsut: DsutCloud,
    home_country: str = HOME_COUNTRY,
    parent_sector: str = PARENT_SECTOR,
    cloud_share_override: Optional[float] = None,
    concordance: list[tuple[str, str, str]] = DEFAULT_PRODUCT_CONCORDANCE,
) -> CanadaSplitPlan:
    if cloud_share_override is not None:
        if not 0.0 <= cloud_share_override <= 1.0:
            raise ValueError("--canada-cloud-share-override must lie between zero and one.")
        s_cloud = float(cloud_share_override)
        LOG.info(
            "Using projected cloud share of NAICS 51821 output for %s: %.4f (DSUT %s value %.4f).",
            sut_agg.year, s_cloud, dsut.year, dsut.cloud_share_of_51821_output,
        )
    else:
        s_cloud = dsut.cloud_share_of_51821_output
        if dsut.year != sut_agg.year:
            LOG.warning(
                "Applying the %s DSUT cloud share (%.4f) to the %s SUT without projection. "
                "Pass --canada-cloud-share-override with the E3MC-projected share to replace it.",
                dsut.year, s_cloud, sut_agg.year,
            )

    share_518 = float(sut_agg.output["518"] / sut_agg.output["J"])
    weights = np.array([share_518 * s_cloud, share_518 * (1.0 - s_cloud), 1.0 - share_518])

    parent_key = f"{home_country}_{parent_sector}"
    sub_keys = [subsector_key(parent_key, s) for s in SUBSECTOR_SUFFIXES]

    # Input ratios: CLD and DPH share the NAICS 51821 cost structure.
    intensity = sut_agg.input_intensity
    base = intensity["J"].replace(0, np.nan)
    r518 = (intensity["518"] / base).fillna(1.0)
    roth = (intensity["OTH"] / base).fillna(1.0)
    input_ratio = pd.DataFrame({sub_keys[0]: r518, sub_keys[1]: r518, sub_keys[2]: roth})
    if "other" not in input_ratio.index:
        input_ratio.loc["other"] = 1.0

    # Sales destination ratios relative to J62_63.
    dist = sut_agg.sales.div(sut_agg.sales.sum(), axis=1)
    rho518 = (dist["518"] / dist["J"].replace(0, np.nan)).fillna(1.0)
    rhooth = (dist["OTH"] / dist["J"].replace(0, np.nan)).fillna(1.0)
    rho_cld = rho518.copy()
    rho_dph = rho518.copy()

    if dsut.cloud_sales is not None:
        c = dsut.cloud_sales / dsut.cloud_sales.sum()
        nc = dsut.noncloud_sales / dsut.noncloud_sales.sum()
        both = dsut.cloud_sales + dsut.noncloud_sales
        both = both / both.sum()
        rho_cld = rho518 * (c / both.replace(0, np.nan)).fillna(1.0)
        rho_dph = rho518 * (nc / both.replace(0, np.nan)).fillna(1.0)
        LOG.info("Using DSUT cloud-specific sales structure for %s.", sub_keys[0])
    else:
        LOG.info(
            "No DSUT cloud use rows supplied: cloud and other hosting share the NAICS 51821 "
            "sales structure (StatCan's equivalence assumption)."
        )

    sales_rho = pd.DataFrame({sub_keys[0]: rho_cld, sub_keys[1]: rho_dph, sub_keys[2]: rhooth}).reindex(list(SUT_BLOCKS)).fillna(1.0)

    diagnostics = dict(sut_agg.diagnostics)
    diagnostics.update({
        "dsut_year": dsut.year,
        "dsut_source": dsut.source,
        "dsut_cloud_share_of_51821_output": dsut.cloud_share_of_51821_output,
        "cloud_share_applied": s_cloud,
        "cloud_share_overridden": cloud_share_override is not None,
        "weights": dict(zip(sub_keys, weights.tolist())),
    })

    return CanadaSplitPlan(
        parent_key=parent_key,
        sub_keys=sub_keys,
        weights=weights,
        input_ratio=input_ratio,
        sales_rho=sales_rho,
        concordance=concordance,
        cloud_share=s_cloud,
        share_518=share_518,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------
# ICIO split
# ---------------------------------------------------------------------

def ras(seed: np.ndarray, row_targets: np.ndarray, col_targets: np.ndarray, tol: float = 1e-12, max_iter: int = 10_000) -> np.ndarray:
    m = np.array(seed, dtype=float)
    if (m < 0).any():
        raise ValueError("RAS seed must be non-negative.")
    if not np.isclose(row_targets.sum(), col_targets.sum(), rtol=1e-9, atol=1e-9):
        raise ValueError("RAS row and column targets must have the same total.")
    for _ in range(max_iter):
        rs = m.sum(axis=1)
        m *= np.divide(row_targets, rs, out=np.zeros_like(rs), where=rs > 0)[:, None]
        cs = m.sum(axis=0)
        m *= np.divide(col_targets, cs, out=np.zeros_like(cs), where=cs > 0)[None, :]
        if np.allclose(m.sum(axis=1), row_targets, rtol=tol, atol=tol * max(1.0, row_targets.max())):
            return m
    raise RuntimeError("RAS did not converge when splitting the J62_63 row.")


def aggregate_back(
    Zs: np.ndarray, fds: np.ndarray, xs: np.ndarray, j: int, K: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse K split subsectors starting at position j back into one."""
    def collapse(m: np.ndarray, axis: int) -> np.ndarray:
        block = m.take(range(j, j + K), axis=axis).sum(axis=axis, keepdims=True)
        before = m.take(range(0, j), axis=axis)
        after = m.take(range(j + K, m.shape[axis]), axis=axis)
        return np.concatenate([before, block, after], axis=axis)

    Z_back = collapse(collapse(Zs, 0), 1)
    fd_back = collapse(fds, 0)
    x_back = collapse(xs[:, None], 0)[:, 0]
    return Z_back, fd_back, x_back


def input_group_of_icio_sector(sector_code: str, concordance: list[tuple[str, str, str]]) -> str:
    for group, _, sector_regex in concordance:
        if re.search(sector_regex, sector_code):
            return group
    return "other"


def split_icio(
    A: pd.DataFrame,
    X: pd.Series,
    FD: pd.DataFrame,
    plan: CanadaSplitPlan,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict]:
    """
    Replace plan.parent_key with its subsectors in A, X and FD.

    Returns the split A, X, FD and a diagnostics dictionary including
    the row-block shares used.
    """
    keys = A.index.astype(str).tolist()
    if plan.parent_key not in keys:
        raise ValueError(
            f"{plan.parent_key} is not an ICIO industry. Build the plan with "
            "parent_sector=resolve_parent_sector(A.index) (J62_63 in the 2023 "
            "ICIO edition, 62T63 in the 2025 edition)."
        )
    if any(k in keys for k in plan.sub_keys):
        raise ValueError("ICIO already contains the Canadian subsector keys; it has been split already.")

    n = len(keys)
    j = keys.index(plan.parent_key)
    K = len(plan.sub_keys)
    w = np.asarray(plan.weights, dtype=float)
    if not np.isclose(w.sum(), 1.0) or (w < 0).any():
        raise ValueError(f"Subsector weights must be non-negative and sum to one: {w}")

    known = {country_of(k) for k in keys}
    home = country_of(plan.parent_key)
    x = X.reindex(keys).to_numpy(dtype=float)
    a = A.loc[keys, keys].to_numpy(dtype=float)
    Z = a * x[None, :]

    # ---- columns ----------------------------------------------------
    groups = np.array([input_group_of_icio_sector(sector_of(k), plan.concordance) for k in keys])
    ratio = plan.input_ratio.reindex(columns=plan.sub_keys)
    missing_groups = sorted(set(groups) - set(ratio.index))
    for g in missing_groups:
        ratio.loc[g] = 1.0
    r = ratio.loc[groups].to_numpy(dtype=float)            # n x K
    shares = w[None, :] * r
    shares /= shares.sum(axis=1, keepdims=True)
    new_cols = Z[:, [j]] * shares                           # n x K

    Zc = np.concatenate([Z[:, :j], new_cols, Z[:, j + 1:]], axis=1)   # n x (n+K-1)
    new_keys = keys[:j] + plan.sub_keys + keys[j + 1:]
    col_home = np.array([country_of(k, known) == home for k in new_keys])

    # ---- rows -------------------------------------------------------
    fd = FD.reindex(keys).to_numpy(dtype=float)
    fd_cols = FD.columns.astype(str).tolist()
    fd_home = np.array([country_of(c, known) == home for c in fd_cols])

    row = Zc[j]
    fd_row = fd[j]
    R = {
        "domestic_intermediate": row[col_home].sum(),
        "foreign_intermediate": row[~col_home].sum(),
        "domestic_final": fd_row[fd_home].sum(),
        "foreign_final": fd_row[~fd_home].sum(),
    }
    blocks = [b for b in ICIO_BLOCKS if R[b] > 0]
    Rb = np.array([R[b] for b in blocks])
    rho = np.array([[plan.sales_rho.loc[ICIO_TO_SUT_BLOCK[b], k] for b in blocks] for k in plan.sub_keys])
    seed = w[:, None] * rho * Rb[None, :]
    q_values = ras(seed, w * Rb.sum(), Rb)
    q = pd.DataFrame(q_values / Rb[None, :], index=plan.sub_keys, columns=blocks)
    for b in ICIO_BLOCKS:
        if b not in q.columns:
            q[b] = w
    q = q[list(ICIO_BLOCKS)]

    col_q = np.where(col_home[None, :], q["domestic_intermediate"].to_numpy()[:, None], q["foreign_intermediate"].to_numpy()[:, None])
    new_rows = row[None, :] * col_q                         # K x (n+K-1)
    fd_q = np.where(fd_home[None, :], q["domestic_final"].to_numpy()[:, None], q["foreign_final"].to_numpy()[:, None])
    new_fd_rows = fd_row[None, :] * fd_q                    # K x nfd

    Zs = np.concatenate([Zc[:j], new_rows, Zc[j + 1:]], axis=0)
    fds = np.concatenate([fd[:j], new_fd_rows, fd[j + 1:]], axis=0)
    xs = np.concatenate([x[:j], w * x[j], x[j + 1:]])

    As = np.divide(Zs, xs[None, :], out=np.zeros_like(Zs), where=xs[None, :] > 0)

    A2 = pd.DataFrame(As, index=new_keys, columns=new_keys)
    X2 = pd.Series(xs, index=new_keys, name=X.name)
    FD2 = pd.DataFrame(fds, index=new_keys, columns=FD.columns)

    # ---- checks: the split must aggregate back to the original -------
    Z_back, fd_back, x_back = aggregate_back(Zs, fds, xs, j, K)
    scale = max(1.0, float(np.abs(Z).max()))
    if not np.allclose(Z_back, Z, rtol=1e-9, atol=1e-9 * scale):
        raise RuntimeError("Split intermediate matrix does not aggregate back to the original ICIO.")
    if not np.allclose(fd_back, fd, rtol=1e-9, atol=1e-9 * max(1.0, float(np.abs(fd).max()))):
        raise RuntimeError("Split final demand does not aggregate back to the original ICIO.")
    if not np.allclose(x_back, x, rtol=1e-12):
        raise RuntimeError("Split gross output does not aggregate back to the original ICIO.")

    va_share = 1.0 - As[:, j:j + K].sum(axis=0)
    if (va_share < 0).any():
        LOG.warning("Negative value-added share in split subsectors: %s", dict(zip(plan.sub_keys, va_share)))

    elec_rows = [
        i for i, k in enumerate(new_keys)
        if country_of(k, known) == home
        and input_group_of_icio_sector(sector_of(k), plan.concordance) == "electricity"
    ]
    parent_elec = float(a[[keys.index(new_keys[i]) for i in elec_rows], j].sum()) if elec_rows else float("nan")

    diag = {
        "row_block_totals_usd_mn": {b: float(v) for b, v in R.items()},
        "row_block_shares": q.to_dict(),
        "subsector_output_usd_mn": dict(zip(plan.sub_keys, (w * x[j]).tolist())),
        "subsector_value_added_share": dict(zip(plan.sub_keys, va_share.tolist())),
        "parent_electricity_input_coefficient": parent_elec,
        "subsector_electricity_input_coefficient": {
            k: float(As[elec_rows, j + m].sum()) for m, k in enumerate(plan.sub_keys)
        },
        "parent_output_usd_mn": float(x[j]),
    }

    LOG.info(
        "Split %s (output %.1f USD mn) into %s with weights %s.",
        plan.parent_key, x[j], plan.sub_keys, np.round(w, 6).tolist(),
    )
    return A2, X2, FD2, diag


def split_emissions_frame(
    emissions: pd.DataFrame,
    plan: CanadaSplitPlan,
    A_split: Optional[pd.DataFrame] = None,
    X_split: Optional[pd.Series] = None,
    basis: str = "output",
) -> pd.DataFrame:
    """
    Split the hybrid emissions extension row for the parent key.

    basis="output": equal emissions intensity across subsectors.
    basis="electricity_input": in proportion to purchases from the home
    country's electricity sector (requires the split A and X).
    """
    df = emissions.copy().reset_index(drop=True)
    df["io_key"] = df["io_key"].astype(str).str.strip()
    parent = df.loc[df["io_key"].eq(plan.parent_key)]
    if len(parent) != 1:
        if parent.empty and df["io_key"].isin(plan.sub_keys).sum() == len(plan.sub_keys):
            return df  # already split
        raise ValueError(f"Emissions extension must contain exactly one row for {plan.parent_key}.")

    if basis == "output":
        shares = plan.weights
    elif basis == "electricity_input":
        if A_split is None or X_split is None:
            raise ValueError("electricity_input basis requires the split A and X.")
        home = country_of(plan.parent_key)
        elec_rows = [
            k for k in A_split.index
            if country_of(k) == home
            and input_group_of_icio_sector(sector_of(k), plan.concordance) == "electricity"
        ]
        spend = (A_split.loc[elec_rows, plan.sub_keys].sum(axis=0) * X_split[plan.sub_keys]).to_numpy(dtype=float)
        if spend.sum() <= 0:
            raise ValueError("No electricity purchases in the split subsectors; use basis='output'.")
        shares = spend / spend.sum()
    else:
        raise ValueError(f"Unknown emissions split basis: {basis}")

    row = parent.iloc[0]
    new_rows = []
    for key, share in zip(plan.sub_keys, shares):
        new = row.copy()
        new["io_key"] = key
        new["ghg_tonnes_co2e"] = float(row["ghg_tonnes_co2e"]) * float(share)
        if "source" in new.index:
            new["source"] = f"{row['source']}; split_canada_dsut_{basis}"
        new_rows.append(new)

    position = int(parent.index[0])
    return pd.concat(
        [df.iloc[:position], pd.DataFrame(new_rows), df.iloc[position + 1:]],
        ignore_index=True,
    )


def write_split_diagnostics(out_dir: Path, plan: CanadaSplitPlan, split_diag: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {**plan.diagnostics, **split_diag, "share_518_applied": plan.share_518}
    with open(out_dir / "canada_dsut_split_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    plan.input_ratio.rename_axis("input_group").to_csv(out_dir / "canada_split_input_ratios.csv")
    plan.sales_rho.rename_axis("sut_block").to_csv(out_dir / "canada_split_sales_ratios.csv")
    pd.DataFrame(split_diag["row_block_shares"]).rename_axis("subsector").to_csv(out_dir / "canada_split_row_block_shares.csv")


def prepare_canada_split(
    sut_file: Path,
    dsut_file: Path,
    sut_year: int,
    cloud_share_override: Optional[float] = None,
    member_overrides: Optional[Path] = None,
    pattern_518: str = DEFAULT_518_PATTERN,
    pattern_other_j: str = DEFAULT_OTHER_J_PATTERN,
    parent_sector: str = PARENT_SECTOR,
) -> CanadaSplitPlan:
    sut = load_statcan_sut(sut_file, year=sut_year, member_overrides=member_overrides)
    agg = compute_sut_aggregates(sut, pattern_518=pattern_518, pattern_other_j=pattern_other_j)
    dsut = load_dsut(dsut_file)
    return build_split_plan(
        agg, dsut, parent_sector=parent_sector, cloud_share_override=cloud_share_override,
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("write-templates", help="Write the DSUT input template.")
    p.add_argument("--out-dir", type=Path, default=Path("."))
    p.add_argument("--dsut-year", type=int, default=2020)

    p = sub.add_parser("fetch-sut", help="Download and cache StatCan table 36-10-0478-01 for Canada.")
    p.add_argument("--cache-dir", type=Path, default=Path("statcan_cache"))
    p.add_argument("--year", type=int, default=2022)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("inspect-sut", help="Classify SUT members and report the 51821 / J62_63 aggregates.")
    p.add_argument("--sut-file", type=Path, required=True)
    p.add_argument("--year", type=int, default=2022)
    p.add_argument("--out-dir", type=Path, default=Path("canada_split_diagnostics"))
    p.add_argument("--member-overrides", type=Path, default=None)
    p.add_argument("--pattern-518", default=DEFAULT_518_PATTERN)
    p.add_argument("--pattern-other-j", default=DEFAULT_OTHER_J_PATTERN)

    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s: %(message)s")

    if args.command == "write-templates":
        path = write_dsut_template(args.out_dir / f"canada_dsut_{args.dsut_year}.csv", year=args.dsut_year)
        print(f"Wrote {path}. Fill the 'value' column from the {args.dsut_year} Canadian DSUT.")
    elif args.command == "fetch-sut":
        print(fetch_statcan_sut(args.cache_dir, args.year, force=args.force))
    elif args.command == "inspect-sut":
        args.out_dir.mkdir(parents=True, exist_ok=True)
        sut = load_statcan_sut(args.sut_file, year=args.year, member_overrides=args.member_overrides)
        sut.members.to_csv(args.out_dir / "sut_column_members.csv", index=False)
        products = pd.concat([sut.supply["product"], sut.use["product"]]).drop_duplicates().to_frame()
        products["input_group"] = [_group_of_product(p, DEFAULT_PRODUCT_CONCORDANCE) for p in products["product"]]
        products.to_csv(args.out_dir / "sut_products_input_groups.csv", index=False)
        agg = compute_sut_aggregates(sut, pattern_518=args.pattern_518, pattern_other_j=args.pattern_other_j)
        agg.input_intensity.rename_axis("input_group").to_csv(args.out_dir / "sut_input_intensity.csv")
        agg.sales.rename_axis("sut_block").to_csv(args.out_dir / "sut_sales_by_block.csv")
        with open(args.out_dir / "sut_aggregates.json", "w", encoding="utf-8") as fh:
            json.dump(agg.diagnostics, fh, indent=2, default=str)
        print(json.dumps(agg.diagnostics, indent=2, default=str))


if __name__ == "__main__":
    _cli()
