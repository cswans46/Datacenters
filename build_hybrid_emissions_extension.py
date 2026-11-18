#!/usr/bin/env python
"""
Build a hybrid AEA / GTAP 12 production-emissions extension for OECD ICIO.

Source selection is COUNTRY-WIDE:
  * Use OECD AEA for a country if usable resident-industry GHG data are present.
  * Otherwise use GTAP 12 for that country.
  * Never add AEA and GTAP emissions for the same country.

Accounting boundary:
  * AEA: resident-industry GHG emissions only (ACTIVITY_SCOPE=RES).
  * GTAP CO2: MDF + MMF, i.e. fossil-fuel combustion by producing industries
    from domestic and imported fuels. Both are attributed to the GTAP region's
    producing industry. Exclude MDP/MMP and other final-demand accounts.
  * GTAP complementary GHG: EMI_IO + EMI_ENDW + EMI_QO + EMI_IOP, converted
    with the selected GWP assessment. Exclude EMI_HH (households) and EMI_LU
    (land use; exported separately).
  * Do not reallocate electricity-generation emissions to electricity users.
    This is a direct production-emissions extension, not a Scope 1+2 footprint.

GTAP-to-ICIO concordance:
  GTAP's 65 activities do not share the AEA activity classification. The first
  run creates gtap_activity_to_aea.csv as a template. Populate it with a
  concordance from each GTAP activity to one or more AEA activity codes.
  Multiple AEA codes per GTAP activity are allowed if allocation_weight sums
  to one. The script then uses aea_crosswalk.csv and country-specific ICIO
  gross-output shares to allocate to exactly the same ICIO sector codes used
  for AEA.

Required files in --data-dir:
  OECD_AEA.csv
  aea_crosswalk.csv                 columns: aea_activity_code, icio_sector_code
  icio_output.csv                   columns: io_key, output_usd_mn (2022)
  sector_map.csv                    columns: io_key, country, sector_code
  GDX12_23/GSDFEMISS.GDX
  GDX12_23/GSDFNCO2.GDX

Optional:
  gtap_activity_to_aea.csv          generated as a template on first run

Install:
  pip install pandas numpy
  The GAMS Python API must be available in the active environment:
  pip install gamsapi[core]   (matching the local GAMS 51 installation)
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LOG = logging.getLogger("hybrid_emissions")

# Broad AEA activity aggregates. If a broad aggregate and more detailed
# activities from the same section are both reported for a country, retain the
# detailed records and exclude the broad total to avoid double counting.
PARENT_ACTIVITY_CODES = {
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
    "M_N", "O_Q", "R_S", "S", "T", "U",
}

# Intermediate AEA aggregates and their detailed components.
# Exclude an aggregate only when at least one mapped detailed component
# is reported for the same country.
AEA_INTERMEDIATE_AGGREGATES = {
    "C22_23": ("C22", "C23"),
    "C24_25": ("C24", "C25"),
    "C29_30": ("C29", "C30"),
    "C31T33": ("C31_32", "C33"),
    "M69T71": ("M69_70", "M71"),
}

# GTAP GHG driver accounts that are attributed to producing industries.
GTAP_PRODUCTION_GHG_SYMBOLS = ("EMI_IO", "EMI_ENDW", "EMI_QO", "EMI_IOP")


# ICIO industries verified to be structural zeros in the OECD 2025
# ICIO 2022 X, Z and FD arrays.
#
# These are not claims that the underlying economies have no activity.
# They indicate that the OECD ICIO source contains no recorded production,
# intermediate transactions or final-demand sales for these industries.
#
# Do not allocate their GTAP emissions to unrelated ICIO industries.
# Independently verified structural-zero industries in the 2022 OECD ICIO.
#
# Verification: gross output (X), intermediate sales (Z), intermediate
# purchases (Z), and final-demand sales (FD) are all zero.
#
# BRN_29: zero throughout the 1995–2022 ICIO series.
# CHL_29: positive through 2012; zero from 2013 through 2022.
#
# These are zeros in the ICIO accounting representation, not proof that
# the corresponding real-world economic activities were absent.
CONFIRMED_STRUCTURAL_ZERO_IO_KEYS = {
    # Existing verified Singapore structural zeros
    "SGP_02",
    "SGP_05",
    "SGP_06",
    "SGP_07",
    "SGP_08",
    "SGP_09",
# BRN_29:
#   Zero gross output, intermediate sales, intermediate purchases, and
#   final-demand sales throughout the OECD ICIO 1995–2022 series.
#   Classified as a confirmed structural zero in the 2022 ICIO representation.

# CHL_29:
#   Positive gross output and transactions through 2012.
#   Zero gross output, intermediate sales, intermediate purchases, and
#   final-demand sales from 2013 through 2022.
#   Classified as a confirmed structural zero in the 2022 ICIO representation.
#   The disappearance coincides with the period of limited national
#   source-table coverage; the precise cause is not established.

# Neither classification implies that the corresponding real-world
# economic activity was necessarily absent.
    "BRN_29",
    "CHL_29",
}

# The unsplit CHN and MEX national blocks are empty in the 2022 ICIO.
# Their production is represented by CN1/CN2 and MX1/MX2, respectively.
# Verified against X, Z, and FD for all 50 industries in each block.

CHN_MEX_STRUCTURAL_ZERO_SECTORS = (
    "01", "02", "03", "05", "06", "07", "08", "09",
    "10T12", "13T15", "16", "17T18", "19", "20", "21",
    "22", "23", "24A", "24B", "25", "26", "27", "28",
    "29", "301", "302T309", "31T33", "35", "36T39",
    "41T43", "45T47", "49", "50", "51", "52", "53",
    "55T56", "58T60", "61", "62T63", "64T66", "68",
    "69T75", "77T82", "84", "85", "86T88", "90T93",
    "94T96", "97T98",
)

CONFIRMED_STRUCTURAL_ZERO_IO_KEYS.update(
    f"{country}_{sector}"
    for country in ("CHN", "MEX")
    for sector in CHN_MEX_STRUCTURAL_ZERO_SECTORS
)

GTAP_TO_ICIO_REGION_MAP = {
    # GTAP national regions mapped to multiple ICIO regions
    "CHN": ["CN1", "CN2"],
    "MEX": ["MX1", "MX2"],

    # GTAP aggregate containing Myanmar
    "XSE": ["MMR"],

    # GTAP regions assigned to the ICIO Rest of World residual
    # (populate only after verifying the regional composition)
}

GTAP_MIXED_REGIONS = {
    # Add GTAP composite regions here when their membership includes
    # economies separately represented in OECD ICIO.
}

def complete_gtap_to_icio_region_map(
    gtap_regions: set[str],
    io_countries: set[str],
) -> dict[str, list[str]]:
    """
    Map every GTAP source region to its ICIO destination.

    Direct country-code matches are retained.
    Explicit split-region mappings are retained.
    All other unmatched GTAP regions are provisionally mapped to ROW.

    Composite regions requiring country-membership review must be
    listed in GTAP_MIXED_REGIONS and are not automatically assigned.
    """
    mapping = {
        region: list(destinations)
        for region, destinations in GTAP_TO_ICIO_REGION_MAP.items()
    }

    for region in sorted(gtap_regions):
        if region in mapping:
            continue

        if region in io_countries:
            mapping[region] = [region]
            continue

        if region in GTAP_MIXED_REGIONS:
            continue

        if "ROW" in io_countries:
            mapping[region] = ["ROW"]

    return mapping

def classify_gtap_unallocated(
    unallocated: pd.DataFrame,
    crosswalk: pd.DataFrame,
    io: pd.DataFrame,
) -> pd.DataFrame:
    """
    Classify GTAP emissions that could not be allocated to ICIO industries.

    Categories:
      CONFIRMED_STRUCTURAL_ZERO:
          Every mapped ICIO target is present and belongs to the
          independently verified structural-zero list.

      ZERO_OUTPUT_TARGET_UNVERIFIED:
          Mapped ICIO targets exist but all have zero gross output.
          The X/Z/FD accounting checks have not confirmed these targets.

      POSITIVE_OUTPUT_TARGET_NO_WEIGHT:
          Positive-output targets exist, but no allocation weight was found.

      NO_AEA_CONCORDANCE:
          No AEA activity mapping was available for the GTAP record.

    No emissions are redistributed by this function.
    """
    result = unallocated.copy()

    if result.empty:
        result["unallocated_reason"] = pd.Series(dtype="object")
        result["target_icio_sectors"] = pd.Series(dtype="object")
        result["target_io_keys"] = pd.Series(dtype="object")
        result["target_output_usd_mn"] = pd.Series(dtype="float64")
        result["icio_accounting_status"] = pd.Series(dtype="object")
        return result

    crosswalk_pairs = (
        crosswalk[
            ["aea_activity_code", "icio_sector_code"]
        ]
        .drop_duplicates()
        .copy()
    )
    crosswalk_pairs["aea_activity_code"] = normalize_codes(
        crosswalk_pairs["aea_activity_code"]
    )
    crosswalk_pairs["icio_sector_code"] = normalize_codes(
        crosswalk_pairs["icio_sector_code"]
    )

    io_lookup = io[
        ["io_key", "country", "sector_code", "output_usd_mn"]
    ].copy()
    io_lookup["country"] = normalize_codes(io_lookup["country"])
    io_lookup["sector_code"] = normalize_codes(io_lookup["sector_code"])
    io_lookup["io_key"] = io_lookup["io_key"].astype(str).str.strip()
    io_lookup["output_usd_mn"] = pd.to_numeric(
        io_lookup["output_usd_mn"], errors="raise"
    )

    classifications = []

    for row in result.itertuples(index=False):
        country = str(getattr(row, "country", "")).strip().upper()

        activity_value = getattr(row, "aea_activity_code", None)
        activity = (
            str(activity_value).strip().upper()
            if pd.notna(activity_value)
            else ""
        )

        if not activity:
            classifications.append({
                "unallocated_reason": "NO_AEA_CONCORDANCE",
                "target_icio_sectors": "",
                "target_io_keys": "",
                "target_output_usd_mn": np.nan,
                "icio_accounting_status": "NOT_APPLICABLE",
            })
            continue

        targets = sorted(
            crosswalk_pairs.loc[
                crosswalk_pairs["aea_activity_code"].eq(activity),
                "icio_sector_code",
            ].unique().tolist()
        )

        if not targets:
            classifications.append({
                "unallocated_reason": "NO_AEA_CONCORDANCE",
                "target_icio_sectors": "",
                "target_io_keys": "",
                "target_output_usd_mn": np.nan,
                "icio_accounting_status": "NO_CROSSWALK_TARGET",
            })
            continue

        icio_countries = GTAP_TO_ICIO_REGION_MAP.get(country, [country])

        target_io = io_lookup.loc[
            io_lookup["country"].isin(icio_countries)
            & io_lookup["sector_code"].isin(targets)
        ].copy()

        target_keys = sorted(target_io["io_key"].unique().tolist())
        observed_sectors = set(target_io["sector_code"].unique())
        target_output = float(target_io["output_usd_mn"].sum())

        # Require every crosswalk target to be present before classifying
        # the entire activity as a confirmed structural-zero target.
        all_targets_present = observed_sectors == set(targets)

        confirmed_keys = set(target_keys).issubset(
            CONFIRMED_STRUCTURAL_ZERO_IO_KEYS
        )

        if (
            all_targets_present
            and len(target_keys) > 0
            and confirmed_keys
            and np.isclose(target_output, 0.0)
        ):
            reason = "CONFIRMED_STRUCTURAL_ZERO"
            accounting_status = "X_Z_FD_VERIFIED_ZERO"

        elif (
            len(target_io) > 0
            and np.isclose(target_output, 0.0)
        ):
            reason = "ZERO_OUTPUT_TARGET_UNVERIFIED"
            accounting_status = "REQUIRES_X_Z_FD_VERIFICATION"

        elif target_output > 0:
            reason = "POSITIVE_OUTPUT_TARGET_NO_WEIGHT"
            accounting_status = "POSITIVE_OUTPUT_BUT_NO_WEIGHT"

        else:
            reason = "NO_MATCHING_COUNTRY_SECTOR"
            accounting_status = "NO_MATCHING_IO_TARGET"

        classifications.append({
            "unallocated_reason": reason,
            "target_icio_sectors": ";".join(targets),
            "target_io_keys": ";".join(target_keys),
            "target_output_usd_mn": target_output,
            "icio_accounting_status": accounting_status,
        })

    classification_df = pd.DataFrame(
        classifications,
        index=result.index,
    )

    return pd.concat(
        [result, classification_df],
        axis=1,
    )



def read_csv(path: Path, required: Sequence[str], name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")
    df = pd.read_csv(path)
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing columns {missing}. Found: {list(df.columns)}")
    return df


def first_existing_column(df: pd.DataFrame, choices: Sequence[str], label: str) -> str:
    for col in choices:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not identify {label}. Expected one of {list(choices)}; "
        f"available columns are {list(df.columns)}"
    )


def normalize_codes(s) -> pd.Series:
    """
    Normalize country, region, activity, and sector codes.
    Accepts pandas Series, lists, tuples, and other array-like inputs.
    """
    if not isinstance(s, pd.Series):
        s = pd.Series(s)

    return s.astype(str).str.strip().str.upper()

def unit_to_tonnes_factor(unit: str) -> float:
    """Convert mass or CO2e units to tonnes."""
    key = str(unit).strip().upper().replace(" ", "")
    factors = {
        "T": 1.0, "TONNE": 1.0, "TONNES": 1.0,
        "T_CO2E": 1.0, "TONNE_CO2E": 1.0, "TONNES_CO2E": 1.0,
        "KT": 1.0e3, "KT_CO2E": 1.0e3,
        "MT": 1.0e6, "MT_CO2E": 1.0e6,
        "GG": 1.0e3,
    }
    if key not in factors:
        raise ValueError(
            f"Unsupported AEA unit {unit!r}. Add an explicit conversion to "
            "unit_to_tonnes_factor(); do not assume its scale."
        )
    return factors[key]



def load_icio_inputs(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    crosswalk_path = data_dir / "aea_crosswalk.csv"
    output_path = data_dir / "icio_output.csv"
    sector_map_path = data_dir / "sector_map.csv"

    crosswalk = read_csv(
        crosswalk_path,
        ["aea_activity_code", "icio_sector_code"],
        "aea_crosswalk.csv",
    )
    output = read_csv(
        output_path,
        ["io_key", "output_usd_mn"],
        "icio_output.csv",
    )
    sector_map = read_csv(
        sector_map_path,
        ["io_key", "country", "sector_code"],
        "sector_map.csv",
    )

    print("\n=== ICIO INPUT FILE DIAGNOSTICS ===")
    for name, path, df in [
        ("aea_crosswalk.csv", crosswalk_path, crosswalk),
        ("icio_output.csv", output_path, output),
        ("sector_map.csv", sector_map_path, sector_map),
    ]:
        print(f"\n{name}")
        print("Path:", path.resolve())
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print(df.head(3).to_string(index=False))

    if sector_map.empty:
        raise ValueError(
            f"{sector_map_path.resolve()} contains column headers but no data rows. "
            "The ICIO sector map must contain one row per industry. "
            "Rebuild sector_map.csv before running the hybrid extension."
        )

    if output.empty:
        raise ValueError(
            f"{output_path.resolve()} contains column headers but no data rows. "
            "It must contain 2022 ICIO gross output for each industry."
        )

    # Normalize classification codes without converting missing values to "NAN".
    for col in ["aea_activity_code", "icio_sector_code"]:
        crosswalk[col] = (
            crosswalk[col].astype("string").str.strip().str.upper()
        )

    for col in ["io_key", "country", "sector_code"]:
        sector_map[col] = (
            sector_map[col].astype("string").str.strip()
        )

    sector_map["country"] = sector_map["country"].str.upper()
    sector_map["sector_code"] = sector_map["sector_code"].str.upper()

    output["io_key"] = output["io_key"].astype("string").str.strip()
    output["output_usd_mn"] = pd.to_numeric(
        output["output_usd_mn"], errors="coerce"
    )

    # Reject missing or malformed keys.
    for name, df, columns in [
        ("sector_map.csv", sector_map, ["io_key", "country", "sector_code"]),
        ("icio_output.csv", output, ["io_key"]),
        ("aea_crosswalk.csv", crosswalk, ["aea_activity_code", "icio_sector_code"]),
    ]:
        missing = df[columns].isna().any(axis=1)
        if missing.any():
            raise ValueError(
                f"{name} contains {int(missing.sum())} rows with missing "
                f"required keys. Examples:\n"
                f"{df.loc[missing, columns].head(10).to_string(index=False)}"
            )

    if sector_map["io_key"].duplicated().any():
        dup = sector_map.loc[
            sector_map["io_key"].duplicated(keep=False), "io_key"
        ].head(20).tolist()
        raise ValueError(f"Duplicate io_key values in sector_map.csv: {dup}")

    if output["io_key"].duplicated().any():
        dup = output.loc[
            output["io_key"].duplicated(keep=False), "io_key"
        ].head(20).tolist()
        raise ValueError(f"Duplicate io_key values in icio_output.csv: {dup}")

    if output["output_usd_mn"].isna().any():
        raise ValueError("icio_output.csv contains non-numeric or missing output.")

    if (output["output_usd_mn"] < 0).any():
        raise ValueError("ICIO gross output cannot be negative.")

    # Check that the two files describe the same set of industries.
    map_keys = set(sector_map["io_key"])
    output_keys = set(output["io_key"])

    missing_output = sorted(map_keys - output_keys)
    missing_map = sorted(output_keys - map_keys)

    print("\nSector-map industries:", len(map_keys))
    print("Output industries:", len(output_keys))
    print("Sector-map keys missing from output:", len(missing_output))
    print("Output keys missing from sector map:", len(missing_map))

    if missing_output:
        raise ValueError(
            "Some sector_map.csv industries have no gross-output observation. "
            f"Examples: {missing_output[:20]}"
        )

    if missing_map:
        raise ValueError(
            "Some icio_output.csv industries have no sector-map entry. "
            f"Examples: {missing_map[:20]}"
        )

    crosswalk = crosswalk.drop_duplicates()

    if "include" in crosswalk.columns:
        include = crosswalk["include"].astype(str).str.strip().str.lower()
        crosswalk = crosswalk.loc[
            include.isin({"1", "true", "yes", "y"})
        ].copy()

    io = sector_map.merge(
        output,
        on="io_key",
        how="inner",
        validate="one_to_one",
    )

    if io.empty:
        raise ValueError(
            "The ICIO sector map and output files have no matching io_key values."
        )

    if io["country"].nunique() == 0:
        raise ValueError("No ICIO countries were identified.")

    print("\n=== MERGED ICIO DATA ===")
    print("Shape:", io.shape)
    print("Countries:", io["country"].nunique())
    print("Industries:", io["io_key"].nunique())
    print("Total gross output (USD million):", io["output_usd_mn"].sum())
    print(io.head(10).to_string(index=False))
    print("=== END ICIO INPUT DIAGNOSTICS ===\n")

    return crosswalk, io, sector_map




def make_country_activity_weights(
    countries: Iterable[str],
    activity_codes: Iterable[str],
    crosswalk: pd.DataFrame,
    io: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct country-specific ICIO output shares for each AEA activity.

    Zero-output industries are retained with weight zero when at least
    one other industry mapped to the same AEA activity has positive output.

    This preserves the crosswalk while ensuring that allocation weights
    sum to one across the positive-output industries.
    """
    pairs = crosswalk[
        ["aea_activity_code", "icio_sector_code"]
    ].drop_duplicates().copy()

    pairs["aea_activity_code"] = normalize_codes(
        pairs["aea_activity_code"]
    )
    pairs["icio_sector_code"] = normalize_codes(
        pairs["icio_sector_code"]
    )

    rows: List[dict] = []

    io_lookup = io.copy()
    io_lookup["country"] = normalize_codes(io_lookup["country"])
    io_lookup["sector_code"] = normalize_codes(io_lookup["sector_code"])
    io_lookup["output_usd_mn"] = pd.to_numeric(
        io_lookup["output_usd_mn"], errors="raise"
    )

    for country in sorted({
        str(c).strip().upper() for c in countries
    }):
        # Select the corresponding ICIO region(s).
        icio_countries = GTAP_TO_ICIO_REGION_MAP.get(country, [country])

        country_io = io_lookup.loc[
            io_lookup["country"].isin(icio_countries)
        ].copy()

        if country_io.empty:
            continue

        for activity in sorted({
            str(a).strip().upper() for a in activity_codes
        }):
            target = pairs.loc[
                pairs["aea_activity_code"].eq(activity),
                "icio_sector_code",
            ].unique()

            if len(target) == 0:
                continue

            group = country_io.loc[
                country_io["sector_code"].isin(target)
            ].copy()

            if group.empty:
                continue

            # Include zero-output sectors, but calculate the denominator
            # using only positive gross output.
            group["output_usd_mn"] = group["output_usd_mn"].clip(
                lower=0.0
            )

            total = float(group["output_usd_mn"].sum())

            # No allocation is possible if every mapped industry has
            # zero output in this country.
            if total <= 0:
                continue

            group["weight"] = (
                group["output_usd_mn"] / total
            )

            for r in group.itertuples(index=False):
                rows.append({
                    # GTAP national region, used for merging emissions.
                    "country": country,

                    # Actual ICIO region, retained for identification.
                    "icio_country": r.country,

                    "aea_activity_code": activity,
                    "icio_sector_code": r.sector_code,
                    "io_key": r.io_key,
                    "output_usd_mn": float(r.output_usd_mn),
                    "weight": float(r.weight),
                })

    weights = pd.DataFrame(rows)

    if weights.empty:
        raise ValueError(
            "Could not construct any country/activity ICIO allocation weights."
        )

    sums = (
        weights.groupby(
            ["country", "aea_activity_code"]
        )["weight"].sum()
    )

    bad = sums.loc[
        ~np.isclose(sums, 1.0, atol=1e-10)
    ]

    if not bad.empty:
        raise ValueError(
            "ICIO allocation weights do not sum to one: "
            f"{bad.head(20).to_dict()}"
        )

    return weights






def select_nonoverlapping_aea_activities(
    aea: pd.DataFrame,
    mapped_activities: set[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove broad AEA section totals when more detailed activities from
    the same section are also reported.

    Works with both:
      - Direct AEA GHG data: ghg_tonnes_co2e
      - Pollutant-specific AEA data: pollutant, emissions_tonnes

    Only country and activity are required for overlap detection.
    """

    required_columns = {"country", "activity"}
    missing_columns = required_columns.difference(aea.columns)

    if missing_columns:
        raise ValueError(
            "Missing columns in AEA dataframe: "
            f"{sorted(missing_columns)}"
        )

    aea = aea.copy()

    aea["country"] = normalize_codes(aea["country"])
    aea["activity"] = normalize_codes(aea["activity"])

    mapped_activities = {
        str(code).strip().upper()
        for code in mapped_activities
    }

    excluded_groups = []
    keep_groups = []

    special_parent_prefixes = {
        "M_N": ("M", "N"),
        "O_Q": ("O", "P", "Q"),
        "R_S": ("R", "S"),
    }

    for country, group in aea.groupby("country", sort=False):

        present = set(group["activity"].dropna())
        drop = set()

        # L68A is an "of which" memorandum item within L.
        # Exclude it from allocation, but do not let it trigger
        # exclusion of the broader L emissions observation.
        if "L68A" in present:
            drop.add("L68A")
            
        # Handle intermediate aggregates explicitly. Their component codes
        # do not necessarily share the aggregate's string prefix.
        for parent, children in AEA_INTERMEDIATE_AGGREGATES.items():
            if parent not in present:
                continue

            has_children = any(
                child in present and child in mapped_activities
                for child in children
            )

            if has_children:
                drop.add(parent)

        for parent in PARENT_ACTIVITY_CODES:

            if parent not in present:
                continue

            if parent in special_parent_prefixes:
                prefixes = special_parent_prefixes[parent]

                has_children = any(
                    code != parent
                    and code.startswith(prefixes)
                    for code in present
                    if code in mapped_activities
                )

            else:
                has_children = any(
                    code != parent
                    and code != "L68A"
                    and code.startswith(parent)
                    for code in present
                    if code in mapped_activities
                )

            if has_children:
                drop.add(parent)

        # Preserve excluded observations for auditing.
        excluded_group = group.loc[
            group["activity"].isin(drop)
        ].copy()

        if not excluded_group.empty:
            excluded_group = excluded_group.rename(
                columns={"activity": "aea_activity_code"}
            )

            excluded_group["reason"] = (
                "Broad aggregate excluded because detailed "
                "activities are present"
            )

            # Give the memorandum item its own explicit audit reason.
            excluded_group.loc[
                excluded_group["aea_activity_code"].eq("L68A"),
                "reason",
            ] = "Memorandum item excluded; included within L"

            excluded_groups.append(excluded_group)

        kept_group = group.loc[
            ~group["activity"].isin(drop)
        ].copy()

        if not kept_group.empty:
            keep_groups.append(kept_group)

    # Retain the original dataframe structure.
    if keep_groups:
        kept = pd.concat(keep_groups, ignore_index=True)
    else:
        kept = aea.iloc[0:0].copy()

    # Construct an audit table using whichever emissions columns
    # are available in the input dataframe.
    audit_columns = [
        "country",
        "aea_activity_code",
        "pollutant",
        "emissions_tonnes",
        "ghg_tonnes_co2e",
        "reason",
    ]

    if excluded_groups:
        excluded_df = pd.concat(
            excluded_groups,
            ignore_index=True,
        )

        available_columns = [
            col for col in audit_columns
            if col in excluded_df.columns
        ]

        excluded_df = excluded_df[available_columns].copy()

    else:
        available_columns = [
            col for col in audit_columns
            if col in aea.columns or col in {
                "aea_activity_code", "reason"
            }
        ]

        excluded_df = pd.DataFrame(
            columns=available_columns
        )

    return kept, excluded_df





def load_aea(
    path: Path,
    year: int,
    scope: str,
    methodology: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
        Load directly reported OECD AEA GHG emissions.

        Selects GHG observations using the requested year, activity scope,
        methodology, emissions measure and T_CO2E unit.

        GHG_BIO is not substituted for GHG, and no subtraction involving
        CO2_BIO is performed.
    """
    raw = pd.read_csv(path, low_memory=False)
    cols = {
        "country": first_existing_column(raw, ["REF_AREA"], "AEA country"),
        "year": first_existing_column(raw, ["TIME_PERIOD"], "AEA year"),
        "activity": first_existing_column(raw, ["ACTIVITY"], "AEA activity"),
        "pollutant": first_existing_column(raw, ["POLLUTANT"], "AEA pollutant"),
        "unit": first_existing_column(raw, ["UNIT_MEASURE"], "AEA unit"),
        "scope": first_existing_column(raw, ["ACTIVITY_SCOPE"], "AEA activity scope"),
        "measure": first_existing_column(raw, ["MEASURE"], "AEA measure"),
        "value": first_existing_column(raw, ["OBS_VALUE"], "AEA observation value"),
    }
    mult_col = "UNIT_MULT" if "UNIT_MULT" in raw.columns else None
    method_col = "METHODOLOGY" if "METHODOLOGY" in raw.columns else None

    x = pd.DataFrame({
        "country": normalize_codes(raw[cols["country"]]),
        "year": pd.to_numeric(raw[cols["year"]], errors="coerce"),
        "activity": normalize_codes(raw[cols["activity"]]),
        "pollutant": normalize_codes(raw[cols["pollutant"]]),
        "unit": raw[cols["unit"]].astype(str).str.strip().str.upper(),
        "scope": normalize_codes(raw[cols["scope"]]),
        "value": pd.to_numeric(raw[cols["value"]], errors="coerce"),
        "measure": normalize_codes(raw[cols["measure"]]),
    })
    x["unit_mult"] = (
        pd.to_numeric(raw[mult_col], errors="coerce").fillna(0)
        if mult_col else 0
    )
    if method_col:
        x["methodology"] = normalize_codes(raw[method_col])
    else:
        x["methodology"] = ""

    x = x.loc[
        x["year"].eq(year)
        & x["scope"].eq(scope.upper())
        & x["measure"].eq("EMISSIONS")
        & x["unit"].eq("T_CO2E")
    ].copy()

    if methodology and method_col:
        x = x.loc[x["methodology"].eq(methodology.upper())].copy()

    # Only these three pollutants are needed to construct a non-biogenic GHG
    # aggregate. Filter before unit conversion because the raw AEA file also
    # contains non-GHG pollutants measured in NO2e, SO2e, etc.
    x = x.loc[x["pollutant"].isin({"GHG", "GHG_BIO", "CO2_BIO"})].copy()
    x = x.dropna(subset=["value"])
    x["value_tonnes"] = x.apply(
        lambda r: float(r["value"])
        * unit_to_tonnes_factor(r["unit"])
        * (10.0 ** float(r["unit_mult"])),
        axis=1,
    )

    keys = ["country", "activity"]

    # Preferred measure: an explicitly reported GHG aggregate.
    selected = x.loc[
        x["pollutant"].eq("GHG"),
        keys + ["value_tonnes"],
    ].copy()

    if selected.duplicated(keys).any():
        dup = selected.loc[
            selected.duplicated(keys, keep=False), keys
        ].head(20)

        raise ValueError(
            "Multiple GHG observations exist for the same country/activity "
            "after filtering. Check the AEA dimensions. Examples: "
            f"{dup.to_dict('records')}"
        )

    selected = selected.rename(
        columns={"value_tonnes": "ghg_tonnes_co2e"}
    )
    selected["aea_ghg_method"] = "GHG"

    # Do not silently substitute GHG_BIO or subtract ordinary CO2.
    # GHG_BIO can only be converted to the desired non-biogenic measure
    # when the corresponding CO2_BIO observation is available and the
    # relationship has been verified for that country.
    bio_countries = set(
        x.loc[x["pollutant"].eq("GHG_BIO"), "country"].unique()
    )

    co2bio_countries = set(
        x.loc[x["pollutant"].eq("CO2_BIO"), "country"].unique()
    )

    needs_review = bio_countries - co2bio_countries

    if needs_review:
        LOG.warning(
            "Countries have GHG_BIO but no CO2_BIO observations. "
            "They cannot be converted to non-biogenic GHG using the "
            "available records: %s",
            sorted(needs_review),
        )

    if selected.empty:
        raise ValueError(
            "No directly reported GHG observations are available after "
            "the year, scope, methodology and MEASURE filters. "
            "Do not automatically substitute GHG_BIO or CO2."
        )

    selected["year"] = year
    selected["source"] = "AEA"
    selected["country"] = normalize_codes(selected["country"])
    selected["activity"] = normalize_codes(selected["activity"])
    selected = selected.loc[
        selected["ghg_tonnes_co2e"].notna()
    ].copy()

    return selected, x



def load_aea_pollutants(
    path: Path,
    year: int,
    scope: str,
    methodology: Optional[str],
) -> pd.DataFrame:
    """
    Load pollutant-specific AEA emissions for constructing GTAP allocation
    shares. Aggregate GHG observations are deliberately excluded.

    Uses physical tonnes (UNIT_MEASURE=T), the SEEA methodology, and
    resident-activity scope. Duplicate T_CO2E representations of CO2
    are excluded.

    The resulting values are pollutant masses in tonnes. For a given
    pollutant, conversion to CO2e would multiply all activities by the
    same GWP and therefore would not change within-GTAP-activity
    allocation shares.
    """
    raw = pd.read_csv(path, low_memory=False)

    required = [
        "REF_AREA",
        "TIME_PERIOD",
        "ACTIVITY",
        "POLLUTANT",
        "UNIT_MEASURE",
        "ACTIVITY_SCOPE",
        "MEASURE",
        "METHODOLOGY",
        "OBS_VALUE",
    ]

    missing = sorted(set(required) - set(raw.columns))
    if missing:
        raise ValueError(f"AEA file is missing columns: {missing}")

    x = pd.DataFrame({
        "country": normalize_codes(raw["REF_AREA"]),
        "year": pd.to_numeric(raw["TIME_PERIOD"], errors="coerce"),
        "activity": normalize_codes(raw["ACTIVITY"]),
        "pollutant": normalize_codes(raw["POLLUTANT"]),
        "unit": normalize_codes(raw["UNIT_MEASURE"]),
        "scope": normalize_codes(raw["ACTIVITY_SCOPE"]),
        "value": pd.to_numeric(raw["OBS_VALUE"], errors="coerce"),
        "measure": normalize_codes(raw["MEASURE"]),
        "methodology": normalize_codes(raw["METHODOLOGY"]),
    })

    x["unit_mult"] = (
        pd.to_numeric(raw["UNIT_MULT"], errors="coerce").fillna(0)
        if "UNIT_MULT" in raw.columns
        else 0
    )

    # Select one consistent AEA accounting basis.
    selected_methodology = (
        methodology.upper()
        if methodology
        else "EMISSIONS_SEEA"
    )

    x = x.loc[
        x["year"].eq(year)
        & x["scope"].eq(scope.upper())
        & x["measure"].eq("EMISSIONS")
        & x["methodology"].eq(selected_methodology)
        & x["unit"].eq("T")
        & x["value"].notna()
    ].copy()

    # Do not use aggregate indicators as pollutant-specific allocation keys.
    aggregate_pollutants = {
        "GHG",
        "GHG_BIO",
        "CO2E",
        "CO2_BIO",
    }

    x = x.loc[
        ~x["pollutant"].isin(aggregate_pollutants)
    ].copy()

    if x.empty:
        raise ValueError(
            "No pollutant-specific AEA observations remain after filtering. "
            "Check the year, scope, methodology, unit and pollutant codes."
        )

    # Convert reported values to physical tonnes.
    x["emissions_tonnes"] = x.apply(
        lambda r: (
            float(r["value"])
            * unit_to_tonnes_factor(r["unit"])
            * (10.0 ** float(r["unit_mult"]))
        ),
        axis=1,
    )

    x["emissions_tonnes"] = pd.to_numeric(
        x["emissions_tonnes"],
        errors="raise",
    )

    # Negative values cannot be used as positive allocation weights.
    # Retain the source values here; the allocation-share function should
    # decide how to handle non-positive observations.
    keys = ["country", "activity", "pollutant"]

    duplicate_keys = (
        x.loc[x.duplicated(keys, keep=False), keys]
        .drop_duplicates()
    )

    if not duplicate_keys.empty:
        raw_diag = raw.copy()

        raw_diag["_country_key"] = normalize_codes(raw_diag["REF_AREA"])
        raw_diag["_activity_key"] = normalize_codes(raw_diag["ACTIVITY"])
        raw_diag["_pollutant_key"] = normalize_codes(raw_diag["POLLUTANT"])

        diagnostic = raw_diag.merge(
            duplicate_keys,
            left_on=["_country_key", "_activity_key", "_pollutant_key"],
            right_on=keys,
            how="inner",
        )

        diagnostic_path = (
            path.parent / "aea_pollutant_duplicate_diagnostics.csv"
        )
        diagnostic.to_csv(diagnostic_path, index=False)

        raise ValueError(
            "Duplicate AEA pollutant observations remain after filtering "
            f"to {selected_methodology}, {scope.upper()}, and unit T. "
            f"See {diagnostic_path}"
        )

    LOG.info(
        "Loaded %s pollutant-specific AEA observations for %s "
        "(methodology=%s, scope=%s, unit=T).",
        len(x),
        year,
        selected_methodology,
        scope.upper(),
    )

    return x[
        ["country", "activity", "pollutant", "emissions_tonnes"]
    ].copy()




def build_gtap_variable_shares(
    aea_pollutants: pd.DataFrame,
    gtap_mapping: pd.DataFrame,
    crosswalk: pd.DataFrame,
    io: pd.DataFrame,
    aea_countries: set[str],
) -> pd.DataFrame:
    """
    Build pooled AEA-derived GTAP-activity-to-AEA-activity shares.

    Allocation shares are estimated using all countries with usable
    pollutant-specific AEA observations and ICIO output data.

    This allocation sample is deliberately independent of aea_countries:
    aea_countries controls which source supplies final country emissions,
    whereas the broader AEA sample supplies the allocation proportions.

    For each GTAP activity and pollutant:
      1. Sum AEA pollutant emissions across countries with usable
         pollutant-specific AEA observations.
      2. Normalize across the AEA activities mapped to that GTAP activity.
      3. Use pooled ICIO gross-output shares if pollutant-specific AEA
         observations are unavailable.

    The fixed allocation_weight in gtap_activity_to_aea.csv is not used
    as an emissions allocation key.
    """

    mapping = gtap_mapping[
        ["gtap_activity_code", "aea_activity_code"]
    ].drop_duplicates().copy()

    mapping["gtap_activity_code"] = normalize_codes(
        mapping["gtap_activity_code"]
    )
    mapping["aea_activity_code"] = normalize_codes(
        mapping["aea_activity_code"]
    )

    # Normalize ICIO and AEA keys.
    aea = aea_pollutants.copy()
    aea["country"] = normalize_codes(aea["country"])
    aea["activity"] = normalize_codes(aea["activity"])
    aea["pollutant"] = normalize_codes(aea["pollutant"])

    io_a = io.copy()
    io_a["country"] = normalize_codes(io_a["country"])
    io_a["sector_code"] = normalize_codes(io_a["sector_code"])

    # Countries used to estimate allocation shares.
    # This is independent of the final AEA country-source selection.
    io_countries = set(io_a["country"].dropna().unique())

    allocation_countries = (
        set(aea["country"].dropna().unique()) & io_countries
    )

    if not allocation_countries:
        raise ValueError(
            "No countries have both pollutant-specific AEA observations "
            "and ICIO output data. Cannot construct allocation shares."
        )

    LOG.info(
        "Countries used to estimate AEA allocation shares: %s",
        len(allocation_countries),
    )

    aea = aea.loc[
        aea["country"].isin(allocation_countries)
    ].copy()

    # Exclude broad aggregate activities when detailed activities are
    # available, separately for each country and pollutant.
    nonoverlap = []

    for (country, pollutant), group in aea.groupby(
        ["country", "pollutant"],
        sort=False,
    ):
        kept, _ = select_nonoverlapping_aea_activities(
            group,
            set(group["activity"]),
        )
        nonoverlap.append(kept)

    if nonoverlap:
        aea = pd.concat(nonoverlap, ignore_index=True)

    # Pool pollutant emissions across the allocation sample.
    aea_totals = (
        aea.groupby(["activity", "pollutant"], as_index=False)
        ["emissions_tonnes"].sum()
        .rename(columns={"activity": "aea_activity_code"})
    )

    # Pooled ICIO output fallback across the same allocation countries.
    io_a = io_a.loc[
        io_a["country"].isin(allocation_countries)
    ].copy()

    crosswalk_pairs = crosswalk[
        ["aea_activity_code", "icio_sector_code"]
    ].drop_duplicates().copy()

    crosswalk_pairs["aea_activity_code"] = normalize_codes(
        crosswalk_pairs["aea_activity_code"]
    )
    crosswalk_pairs["icio_sector_code"] = normalize_codes(
        crosswalk_pairs["icio_sector_code"]
    )

    output_by_activity = (
        crosswalk_pairs.merge(
            io_a,
            left_on="icio_sector_code",
            right_on="sector_code",
            how="left",
        )
        .groupby("aea_activity_code", as_index=False)["output_usd_mn"]
        .sum()
        .rename(columns={"output_usd_mn": "pooled_output"})
    )

    rows = []

    for gtap_activity, group in mapping.groupby(
        "gtap_activity_code",
        sort=False,
    ):
        activities = group["aea_activity_code"].drop_duplicates().tolist()

        if not activities:
            continue

        # Pollutant-specific AEA allocation.
        pollutants = sorted(aea_totals["pollutant"].unique())

        for pollutant in pollutants:
            values = (
                aea_totals.loc[
                    aea_totals["aea_activity_code"].isin(activities)
                    & aea_totals["pollutant"].eq(pollutant)
                ]
                .set_index("aea_activity_code")["emissions_tonnes"]
                .reindex(activities)
                .fillna(0.0)
                .clip(lower=0.0)
            )

            total = float(values.sum())

            if total <= 0:
                continue

            shares = values / total

            for activity, share in shares.items():
                if share > 0:
                    rows.append({
                        "gtap_activity_code": gtap_activity,
                        "emission_variable": pollutant,
                        "aea_activity_code": activity,
                        "allocation_share": float(share),
                        "allocation_method": "AEA_POLLUTANT",
                    })

        # Pooled output fallback for activities without usable
        # pollutant-specific AEA allocation data.
        output_values = (
            output_by_activity.set_index("aea_activity_code")
            ["pooled_output"]
            .reindex(activities)
            .fillna(0.0)
            .clip(lower=0.0)
        )

        output_total = float(output_values.sum())

        if output_total > 0:
            for activity, share in (
                output_values / output_total
            ).items():
                if share > 0:
                    rows.append({
                        "gtap_activity_code": gtap_activity,
                        "emission_variable": "__OUTPUT_FALLBACK__",
                        "aea_activity_code": activity,
                        "allocation_share": float(share),
                        "allocation_method": "POOLED_ICIO_OUTPUT",
                    })

    result = pd.DataFrame(rows)

    if result.empty:
        raise ValueError(
            "Could not construct any GTAP-to-AEA variable allocation shares. "
            "Check the GTAP mapping, AEA activity concordance and ICIO "
            "sector crosswalk."
        )

    # Validate pollutant-specific shares.
    sums = result.loc[
        result["emission_variable"].ne("__OUTPUT_FALLBACK__")
    ].groupby(
        ["gtap_activity_code", "emission_variable"]
    )["allocation_share"].sum()

    bad = sums.loc[
        ~np.isclose(sums, 1.0, atol=1e-8)
    ]

    if not bad.empty:
        raise ValueError(
            "AEA pollutant allocation shares do not sum to one: "
            f"{bad.to_dict()}"
        )

    # Validate pooled-output fallback shares.
    fallback_sums = result.loc[
        result["emission_variable"].eq("__OUTPUT_FALLBACK__")
    ].groupby("gtap_activity_code")["allocation_share"].sum()

    bad_fallback = fallback_sums.loc[
        ~np.isclose(fallback_sums, 1.0, atol=1e-8)
    ]

    if not bad_fallback.empty:
        raise ValueError(
            "Pooled output fallback shares do not sum to one: "
            f"{bad_fallback.to_dict()}"
        )

    LOG.info(
        "Constructed %s GTAP-to-AEA allocation-share records.",
        len(result),
    )

    LOG.info(
        "Allocation methods: %s",
        result["allocation_method"].value_counts().to_dict(),
    )

    return result


def gams_read_database(path: Path, system_directory: str):
    try:
        from gams import GamsWorkspace
    except ImportError as exc:
        raise ImportError(
            "The GAMS Python API is not installed in this environment. "
            "Install the API matching your local GAMS 51 installation."
        ) from exc
    ws = GamsWorkspace(system_directory=system_directory)
    return ws.add_database_from_gdx(str(path.resolve()))


def symbol_records(db, symbol_name: str, dimensions: int) -> pd.DataFrame:
    if symbol_name not in {s.name for s in db}:
        raise KeyError(f"GDX does not contain symbol {symbol_name!r}.")
    symbol = db[symbol_name]
    rows = []
    for rec in symbol:
        keys = list(rec.keys)
        if len(keys) != dimensions:
            raise ValueError(
                f"{symbol_name} expected {dimensions} dimensions but got {len(keys)} "
                f"for keys {keys}"
            )
        row = {f"d{i+1}": str(k).strip() for i, k in enumerate(keys)}
        row["value"] = float(rec.value)
        rows.append(row)
    return pd.DataFrame(rows, columns=[*(f"d{i+1}" for i in range(dimensions)), "value"])


def set_members(db, set_name: str) -> set[str]:
    if set_name not in {s.name for s in db}:
        raise KeyError(f"GDX does not contain set {set_name!r}.")
    return {str(rec.keys[0]).strip() for rec in db[set_name]}


def create_gtap_mapping_template(co2_gdx: Path, system_directory: str, output_path: Path) -> None:
    db = gams_read_database(co2_gdx, system_directory)
    activities = sorted(set_members(db, "ACTS"))
    template = pd.DataFrame({
        "gtap_activity_code": activities,
        "aea_activity_code": "",
        "allocation_weight": 1.0,
        "notes": "",
    })
    template.to_csv(output_path, index=False)
    LOG.warning(
        "Created %s with %s GTAP activities. Populate aea_activity_code using "
        "the AEA activity codes in aea_crosswalk.csv. For one-to-many mappings, "
        "add rows and make allocation_weight sum to one for each GTAP activity.",
        output_path, len(activities),
    )



def load_gtap_mapping(path: Path) -> pd.DataFrame:
    mapping = read_csv(
        path,
        ["gtap_activity_code", "aea_activity_code", "allocation_weight"],
        "gtap_activity_to_aea.csv",
    )

    mapping["gtap_activity_code"] = normalize_codes(
        mapping["gtap_activity_code"]
    )
    mapping["aea_activity_code"] = normalize_codes(
        mapping["aea_activity_code"]
    )
    mapping["allocation_weight"] = pd.to_numeric(
        mapping["allocation_weight"], errors="raise"
    )

    mapping = mapping.dropna(
        subset=[
            "gtap_activity_code",
            "aea_activity_code",
            "allocation_weight",
        ]
    )

    mapping = mapping.loc[
        mapping["gtap_activity_code"].ne("")
        & mapping["aea_activity_code"].ne("")
    ].copy()

    if mapping.empty:
        raise ValueError(
            "gtap_activity_to_aea.csv contains no completed mappings."
        )

    # Each GTAP-to-AEA activity pair must appear only once.
    if mapping.duplicated(
        ["gtap_activity_code", "aea_activity_code"]
    ).any():
        dup = mapping.loc[
            mapping.duplicated(
                ["gtap_activity_code", "aea_activity_code"],
                keep=False,
            )
        ]
        raise ValueError(
            "Duplicate GTAP-to-AEA activity pairs would duplicate emissions: "
            f"{dup.head(20).to_dict('records')}"
        )

    # Mapping weights must be positive and finite.
    if not np.isfinite(mapping["allocation_weight"]).all():
        raise ValueError(
            "GTAP-to-AEA allocation weights must be finite."
        )

    if (mapping["allocation_weight"] <= 0).any():
        raise ValueError(
            "GTAP-to-AEA allocation weights must be strictly positive."
        )

    # Normalize mapping weights within each GTAP activity.
    # These weights are NOT the emissions allocation shares.
    # Actual emissions allocation is calculated using pooled,
    # pollutant-specific AEA emissions in build_gtap_variable_shares().
    mapping_totals = mapping.groupby(
        "gtap_activity_code"
    )["allocation_weight"].transform("sum")

    mapping["allocation_weight"] = (
        mapping["allocation_weight"] / mapping_totals
    )

    # Confirm normalized mapping weights sum to one.
    sums = mapping.groupby(
        "gtap_activity_code"
    )["allocation_weight"].sum()

    bad = sums.loc[~np.isclose(sums, 1.0, atol=1e-8)]

    if not bad.empty:
        raise ValueError(
            "Normalized GTAP-to-AEA mapping weights do not sum to one: "
            f"{bad.to_dict()}"
        )

    return mapping






def extract_gtap_variable_emissions(
    co2_path: Path,
    nco2_path: Path,
    system_directory: str,
    gwp_assessment: str,
) -> pd.DataFrame:
    """
    Extract GTAP 12 production emissions by region, activity, component,
    and emissions variable.

    Output:
        region
        activity
        emission_variable
        gtap_component
        emissions_tonnes_co2e

    Accounting treatment
    --------------------
    CO2 combustion:
        MDF + MMF

    Complementary GHG:
        EMI_IO
        EMI_ENDW
        EMI_QO
        EMI_IOP

    GWP:
        GWP values for the requested assessment (e.g. AR6).

    CO2 is excluded from EMI_IO because MDF/MMF already account for
    fossil-fuel combustion CO2. CO2 in EMI_IOP and EMI_QO is retained.

    EMI_HH and EMI_LU are deliberately excluded.
    """

    # ---------------------------------------------------------------
    # 1. Read GTAP databases
    # ---------------------------------------------------------------
    co2_db = gams_read_database(co2_path, system_directory)
    nco2_db = gams_read_database(nco2_path, system_directory)

    ghg_set = {
        code.upper()
        for code in set_members(nco2_db, "GHG")
    }

    activities = {
        code.upper()
        for code in set_members(co2_db, "ACTS")
    }

    if not ghg_set:
        raise ValueError("The GTAP NCO2 GDX contains an empty GHG set.")

    if not activities:
        raise ValueError("The GTAP CO2 GDX contains an empty ACTS set.")

    LOG.info("GTAP GHG gases: %s", len(ghg_set))
    LOG.info("GTAP production activities: %s", len(activities))

    # Retain unaggregated records for diagnostic purposes.
    source_frames = []
    frames = []

    # ---------------------------------------------------------------
    # 2. Load GWP values for the selected assessment
    # ---------------------------------------------------------------
    gwp = symbol_records(nco2_db, "GWP", 3)

    gwp["gas"] = gwp["d1"].str.upper()
    gwp["region"] = gwp["d2"].str.upper()
    gwp["assessment"] = gwp["d3"].str.upper()

    assessment = gwp_assessment.strip().upper()

    gwp = gwp.loc[
        gwp["assessment"].eq(assessment)
        & gwp["gas"].isin(ghg_set)
    ].copy()

    if gwp.empty:
        raise ValueError(
            f"No GWP observations found for assessment {assessment}."
        )

    if gwp.duplicated(["gas", "region"]).any():
        duplicates = gwp.loc[
            gwp.duplicated(["gas", "region"], keep=False),
            ["gas", "region", "assessment"],
        ].head(20)

        raise ValueError(
            "GWP contains duplicate gas/region observations for "
            f"{assessment}: {duplicates.to_dict('records')}"
        )

    gtap_regions = {
        code.upper()
        for code in set_members(nco2_db, "REG")
    }

    gwp_regions = set(gwp["region"].unique())

    missing_gwp_regions = gtap_regions - gwp_regions

    if missing_gwp_regions:
        raise ValueError(
            f"Missing {assessment} GWP coverage for GTAP regions: "
            f"{sorted(missing_gwp_regions)}"
        )

    gwp_lookup = (
        gwp.set_index(["gas", "region"])["value"].to_dict()
    )

    # ---------------------------------------------------------------
    # 3. Extract fossil-fuel combustion CO2
    # ---------------------------------------------------------------
    # MDF/MMF dimensions:
    #   fuel x activity x region
    #
    # Values are in million tonnes (Mt).
    # ---------------------------------------------------------------
    for symbol in ("MDF", "MMF"):

        df = symbol_records(co2_db, symbol, 3)

        df["activity"] = df["d2"].str.upper()
        df["region"] = df["d3"].str.upper()

        unknown_activities = (
            set(df["activity"].unique()) - activities
        )

        if unknown_activities:
            raise ValueError(
                f"{symbol} contains activity codes outside ACTS: "
                f"{sorted(unknown_activities)[:20]}"
            )

        df["emission_variable"] = "CO2"
        df["gtap_component"] = symbol
        df["gwp"] = 1.0
        df["source_value"] = df["value"]

        # Convert Mt CO2 to tonnes CO2e.
        df["emissions_tonnes_co2e"] = (
            df["value"] * 1.0e6
        )

        # Save unaggregated source records.
        source_frames.append(
            df[
                [
                    "region",
                    "activity",
                    "emission_variable",
                    "gtap_component",
                    "source_value",
                    "gwp",
                    "emissions_tonnes_co2e",
                ]
            ].copy()
        )

        aggregated = (
            df.groupby(
                [
                    "region",
                    "activity",
                    "emission_variable",
                    "gtap_component",
                ],
                as_index=False,
            )["emissions_tonnes_co2e"].sum()
        )

        frames.append(aggregated)

    # ---------------------------------------------------------------
    # 4. Extract complementary GHG production emissions
    # ---------------------------------------------------------------
    # EMI_IO:   gas x input/source x activity x region
    # EMI_ENDW: gas x endowment x activity x region
    # EMI_IOP:  gas x intermediate-input category x activity x region
    # EMI_QO:   gas x activity x region
    # ---------------------------------------------------------------
    for symbol in GTAP_PRODUCTION_GHG_SYMBOLS:

        dimensions = 3 if symbol == "EMI_QO" else 4

        df = symbol_records(
            nco2_db,
            symbol,
            dimensions,
        )

        df["gas"] = df["d1"].str.upper()

        # Keep only gases in the GTAP GHG set.
        df = df.loc[
            df["gas"].isin(ghg_set)
        ].copy()

        if df.empty:
            LOG.warning(
                "No GHG observations found for %s.",
                symbol,
            )
            continue

        if symbol == "EMI_QO":

            # Dimensions: EM x ACTS x REG
            df["activity"] = df["d2"].str.upper()
            df["region"] = df["d3"].str.upper()

        else:

            # Four-dimensional symbols:
            # gas x input/endowment category x activity x region
            df["activity"] = df["d3"].str.upper()
            df["region"] = df["d4"].str.upper()

        unknown_activities = (
            set(df["activity"].unique()) - activities
        )

        if unknown_activities:
            raise ValueError(
                f"{symbol} contains activity codes outside ACTS: "
                f"{sorted(unknown_activities)[:20]}"
            )

        # EMI_IO contains fossil-fuel combustion emissions.
        # MDF and MMF already account for combustion CO2.
        # Exclude only CO2 from EMI_IO.
        if symbol == "EMI_IO":
            df = df.loc[
                df["gas"].ne("CO2")
            ].copy()

        if df.empty:
            continue

        # -----------------------------------------------------------
        # 5. Apply region-specific GWP factors
        # -----------------------------------------------------------
        df["gwp"] = [
            gwp_lookup.get((gas, region), np.nan)
            for gas, region in zip(
                df["gas"],
                df["region"],
            )
        ]

        missing_gwp = df.loc[
            df["gwp"].isna(),
            ["gas", "region"],
        ].drop_duplicates()

        if not missing_gwp.empty:
            raise ValueError(
                f"Missing {assessment} GWP values for {symbol}: "
                f"{missing_gwp.head(20).to_dict('records')}"
            )

        # GTAP complementary GHG values are in Mt of the gas.
        # Convert to tonnes CO2e using the selected GWP.
        df["emissions_tonnes_co2e"] = (
            df["value"]
            * 1.0e6
            * df["gwp"]
        )

        df["emission_variable"] = df["gas"]
        df["gtap_component"] = symbol
        df["source_value"] = df["value"]

        # Save unaggregated source records.
        source_frames.append(
            df[
                [
                    "region",
                    "activity",
                    "emission_variable",
                    "gtap_component",
                    "source_value",
                    "gwp",
                    "emissions_tonnes_co2e",
                ]
            ].copy()
        )

        aggregated = (
            df.groupby(
                [
                    "region",
                    "activity",
                    "emission_variable",
                    "gtap_component",
                ],
                as_index=False,
            )["emissions_tonnes_co2e"].sum()
        )

        frames.append(aggregated)

    # ---------------------------------------------------------------
    # 6. Combine all components
    # ---------------------------------------------------------------
    if not frames:
        raise ValueError(
            "No GTAP production-emissions records were extracted."
        )

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    result["region"] = normalize_codes(result["region"])
    result["activity"] = normalize_codes(result["activity"])
    result["emission_variable"] = normalize_codes(
        result["emission_variable"]
    )

    source_records = pd.concat(
        source_frames,
        ignore_index=True,
    )

    source_records["region"] = normalize_codes(
        source_records["region"]
    )
    source_records["activity"] = normalize_codes(
        source_records["activity"]
    )
    source_records["emission_variable"] = normalize_codes(
        source_records["emission_variable"]
    )

    # ---------------------------------------------------------------
    # 7. Validate output
    # ---------------------------------------------------------------
    if not np.isfinite(
        result["emissions_tonnes_co2e"]
    ).all():
        raise ValueError(
            "GTAP extraction produced non-finite emissions."
        )

    negative_tolerance = 1e-8

    negative_records = result.loc[
        result["emissions_tonnes_co2e"] < -negative_tolerance
    ].copy()

    
    if not negative_records.empty:

        print("\n" + "=" * 100)
        print("WARNING: NEGATIVE GTAP EMISSIONS")
        print("=" * 100)

        print(
            f"\nNegative aggregated records: "
            f"{len(negative_records):,}"
        )

        print(
            "Total negative emissions (tonnes CO2e): "
            f"{negative_records['emissions_tonnes_co2e'].sum():,.6f}"
        )

        # -----------------------------------------------------------
        # 7a. By component and gas
        # -----------------------------------------------------------
        component_summary = (
            negative_records
            .groupby(
                ["gtap_component", "emission_variable"],
                as_index=False,
            )
            .agg(
                records=("emissions_tonnes_co2e", "size"),
                total_ghg_tonnes_co2e=(
                    "emissions_tonnes_co2e", "sum"
                ),
                minimum_ghg_tonnes_co2e=(
                    "emissions_tonnes_co2e", "min"
                ),
            )
            .sort_values("total_ghg_tonnes_co2e")
        )

        print("\n--- NEGATIVE EMISSIONS BY COMPONENT AND GAS ---")
        print(component_summary.to_string(index=False))

        # -----------------------------------------------------------
        # 7b. By region
        # -----------------------------------------------------------
        region_summary = (
            negative_records
            .groupby("region", as_index=False)
            .agg(
                records=("emissions_tonnes_co2e", "size"),
                total_ghg_tonnes_co2e=(
                    "emissions_tonnes_co2e", "sum"
                ),
            )
            .sort_values("total_ghg_tonnes_co2e")
        )

        print("\n--- NEGATIVE EMISSIONS BY REGION ---")
        print(region_summary.to_string(index=False))

        # -----------------------------------------------------------
        # 7c. By activity
        # -----------------------------------------------------------
        activity_summary = (
            negative_records
            .groupby("activity", as_index=False)
            .agg(
                records=("emissions_tonnes_co2e", "size"),
                total_ghg_tonnes_co2e=(
                    "emissions_tonnes_co2e", "sum"
                ),
            )
            .sort_values("total_ghg_tonnes_co2e")
        )

        print("\n--- NEGATIVE EMISSIONS BY ACTIVITY ---")
        print(activity_summary.to_string(index=False))

        # -----------------------------------------------------------
        # 7d. Underlying GDX records
        # -----------------------------------------------------------
        keys = [
            "region",
            "activity",
            "emission_variable",
            "gtap_component",
        ]

        negative_source_records = source_records.merge(
            negative_records[keys],
            on=keys,
            how="inner",
        )

        negative_source_records = (
            negative_source_records
            .sort_values("emissions_tonnes_co2e")
        )

        print("\n--- 100 MOST NEGATIVE UNDERLYING GDX RECORDS ---")
        print(
            negative_source_records
            .head(100)
            .to_string(index=False)
        )

        # -----------------------------------------------------------
        # 7e. Save diagnostics
        # -----------------------------------------------------------
        negative_records.to_csv(
            "gtap_negative_emissions_aggregated.csv",
            index=False,
            encoding="utf-8-sig",
        )

        negative_source_records.to_csv(
            "gtap_negative_emissions_source_records.csv",
            index=False,
            encoding="utf-8-sig",
        )

        component_summary.to_csv(
            "gtap_negative_emissions_by_component.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print("\nDiagnostic files saved:")
        print("  gtap_negative_emissions_aggregated.csv")
        print("  gtap_negative_emissions_source_records.csv")
        print("  gtap_negative_emissions_by_component.csv")

        print("\n" + "=" * 100)

        # Preserve negative source values and continue extraction.
        LOG.warning(
            "Retaining %s negative GTAP emissions records for "
            "further investigation.",
            len(negative_records),
        )


    # ---------------------------------------------------------------
    # 8. Check duplicate records
    # ---------------------------------------------------------------
    keys = [
        "region",
        "activity",
        "emission_variable",
        "gtap_component",
    ]

    if result.duplicated(keys).any():
        raise ValueError(
            "GTAP extraction produced duplicate region/activity/"
            "variable/component records."
        )

    result = result[
        [
            "region",
            "activity",
            "emission_variable",
            "gtap_component",
            "emissions_tonnes_co2e",
        ]
    ].sort_values(
        [
            "region",
            "activity",
            "emission_variable",
            "gtap_component",
        ]
    ).reset_index(drop=True)

    # ---------------------------------------------------------------
    # 9. Diagnostic summary
    # ---------------------------------------------------------------
    summary = (
        result.groupby(
            ["gtap_component", "emission_variable"],
            as_index=False,
        )["emissions_tonnes_co2e"].sum()
    )

    LOG.info(
        "Extracted %s GTAP production-emissions records.",
        len(result),
    )

    LOG.info(
        "GTAP production emissions by component and gas:\n%s",
        summary.to_string(index=False),
    )

    return result



def allocate_activity_emissions(
    activity_emissions: pd.DataFrame,
    mapping: pd.DataFrame,
    weights: pd.DataFrame,
    country_col: str,
    activity_col: str,
    value_col: str,
    source: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Map source activities to AEA activity codes, then to ICIO industries."""
    mapped = activity_emissions.merge(
        mapping,
        left_on=activity_col,
        right_on="gtap_activity_code" if activity_col == "activity" and "gtap_activity_code" in mapping.columns else "aea_activity_code",
        how="left",
    )
    if "allocation_weight" not in mapped:
        mapped["allocation_weight"] = 1.0
    mapped["allocation_weight"] = pd.to_numeric(mapped["allocation_weight"], errors="coerce")
    unmapped = mapped.loc[mapped["aea_activity_code"].isna()].copy()

    mapped = mapped.loc[mapped["aea_activity_code"].notna()].copy()
    mapped["allocated_activity_emissions"] = (
        mapped[value_col] * mapped["allocation_weight"]
    )
    mapped = mapped.merge(
        weights,
        left_on=[country_col, "aea_activity_code"],
        right_on=["country", "aea_activity_code"],
        how="left",
        validate="many_to_many",
    )
    mapped = mapped.loc[mapped["io_key"].notna()].copy()
    mapped["ghg_tonnes_co2e"] = (
        mapped["allocated_activity_emissions"] * mapped["weight"]
    )
    mapped["source"] = source
    return mapped, unmapped




def allocate_aea(
    aea: pd.DataFrame,
    crosswalk: pd.DataFrame,
    weights: pd.DataFrame,
    icio_countries: set[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Map AEA emissions to ICIO industries using country-specific output shares.

    Countries outside ICIO are excluded and audited.

    Zero-emission observations are retained in the audit but do not
    require allocation weights.

    Zero-output ICIO industries are retained with zero allocation weight.
    Their emissions are allocated to other positive-output industries
    mapped to the same AEA activity.

    Returns:
        mapped: allocated AEA emissions
        excluded_parents: broad AEA aggregates excluded to avoid overlap
        allocation_audit: out-of-ICIO observations, zero-weight mappings,
                          and zero-emission observations
    """

    aea = aea.copy()
    crosswalk = crosswalk.copy()
    weights = weights.copy()

    # Normalize keys consistently before merging.
    aea["country"] = normalize_codes(aea["country"])
    aea["activity"] = normalize_codes(aea["activity"])

    crosswalk["aea_activity_code"] = normalize_codes(
        crosswalk["aea_activity_code"]
    )
    crosswalk["icio_sector_code"] = normalize_codes(
        crosswalk["icio_sector_code"]
    )

    weights["country"] = normalize_codes(weights["country"])
    weights["aea_activity_code"] = normalize_codes(
        weights["aea_activity_code"]
    )
    weights["icio_sector_code"] = normalize_codes(
        weights["icio_sector_code"]
    )

    icio_countries = {
        str(c).strip().upper() for c in icio_countries
    }

    mapped_activities = set(
        crosswalk["aea_activity_code"].dropna()
    )

    # First select non-overlapping observations from the full AEA dataset.
    # Do not filter to crosswalk activities first: an aggregate observation
    # may be the only available observation for an industry group.
    aea, excluded_parents = select_nonoverlapping_aea_activities(
        aea,
        mapped_activities,
    )

    # ---------------------------------------------------------------
    # 1. Exclude AEA countries not represented in ICIO
    # ---------------------------------------------------------------

    outside_icio = aea.loc[
        ~aea["country"].isin(icio_countries)
    ].copy()

    if not outside_icio.empty:
        outside_icio["allocation_status"] = (
            "EXCLUDED_COUNTRY_OUTSIDE_ICIO"
        )
        outside_icio["allocation_reason"] = (
            "Country has AEA observations but is not represented in ICIO"
        )

    aea = aea.loc[
        aea["country"].isin(icio_countries)
    ].copy()
    
    
    # ---------------------------------------------------------------
    # Special allocation: NZL C29_30 aggregate
    # ---------------------------------------------------------------

    # New Zealand reports C29_30 but does not separately report C29
    # or C30. Allocate the aggregate across the corresponding ICIO
    # industries using their relative gross output.
    nzl_c29_30_mask = (
        aea["country"].eq("NZL")
        & aea["activity"].eq("C29_30")
        & aea["ghg_tonnes_co2e"].fillna(0).ne(0)
    )

    nzl_c29_30_aggregate = aea.loc[nzl_c29_30_mask].copy()

    # Remove the aggregate from the ordinary mapping/unmapped process.
    aea = aea.loc[~nzl_c29_30_mask].copy()

    # Retained observations without crosswalk entries are audited,
    # not allocated to detailed industries.
    unmapped_aea = aea.loc[
        ~aea["activity"].isin(mapped_activities)
    ].copy()

    if not unmapped_aea.empty:
        unmapped_aea["allocation_status"] = "UNMAPPED_AEA_ACTIVITY"
        unmapped_aea["allocation_reason"] = (
            "AEA activity has no AEA-to-ICIO crosswalk entry; "
            "emissions retained as unallocated rather than assigned "
            "to detailed ICIO industries"
        )
        unmapped_aea["source"] = "AEA"
        unmapped_aea["source_activity"] = unmapped_aea["activity"]

    # Only crosswalk activities proceed to ordinary ICIO allocation.
    aea = aea.loc[
        aea["activity"].isin(mapped_activities)
    ].copy()


    # ---------------------------------------------------------------
    # 2. Separate zero-emission observations
    # ---------------------------------------------------------------

    # These records do not need allocation weights. Retain them in
    # the audit so that the source observations remain traceable.
    zero_emissions = aea.loc[
        aea["ghg_tonnes_co2e"].fillna(0).eq(0)
    ].copy()

    if not zero_emissions.empty:
        zero_emissions["allocation_status"] = "ZERO_EMISSIONS"
        zero_emissions["allocation_reason"] = (
            "AEA reports zero GHG emissions; no ICIO allocation required"
        )

    # Only nonzero emissions proceed to the allocation process.
    aea = aea.loc[
        aea["ghg_tonnes_co2e"].fillna(0).ne(0)
    ].copy()

    # ---------------------------------------------------------------
    # 3. Map AEA activities to ICIO sectors
    # ---------------------------------------------------------------

    activity_map = crosswalk[
        ["aea_activity_code", "icio_sector_code"]
    ].drop_duplicates()

    mapped = aea.merge(
        activity_map,
        left_on="activity",
        right_on="aea_activity_code",
        how="inner",
        validate="many_to_many",
    )

    # ---------------------------------------------------------------
    # 4. Attach country-specific output weights
    # ---------------------------------------------------------------

    mapped = mapped.merge(
        weights,
        on=[
            "country",
            "aea_activity_code",
            "icio_sector_code",
        ],
        how="left",
        validate="many_to_many",
        indicator=True,
    )

    missing_weights = mapped.loc[
        mapped["_merge"].eq("left_only")
    ].copy()

    # A missing weight here means that no country/activity allocation
    # weights were constructed for a positive-emission observation.
    if not missing_weights.empty:
        examples = missing_weights[
            ["country", "activity", "icio_sector_code"]
        ].drop_duplicates().head(20).to_dict("records")

        raise RuntimeError(
            "Some positive-emission AEA country/activity combinations "
            "have no ICIO allocation weights. This usually means that "
            "all industries mapped to an AEA activity have zero output, "
            "or that the crosswalk and ICIO sector map do not match. "
            f"Examples: {examples}"
        )

    mapped = mapped.drop(columns=["_merge"])
    
    
    # ---------------------------------------------------------------
    # 4b. Allocate NZL C29_30 using relative gross output
    # ---------------------------------------------------------------

    nzl_aggregate_mapped = pd.DataFrame()

    if not nzl_c29_30_aggregate.empty:

        target_sectors = {"29", "301", "302T309"}

        target_weights = weights.loc[
            weights["country"].eq("NZL")
            & weights["aea_activity_code"].eq("C")
            & weights["icio_sector_code"].isin(target_sectors)
        ].copy()

        target_weights = target_weights.drop_duplicates(
            ["country", "icio_sector_code"]
        )

        found_sectors = set(
            target_weights["icio_sector_code"].astype(str)
        )

        if found_sectors != target_sectors:
            raise RuntimeError(
                "Cannot allocate NZL C29_30: expected gross-output "
                "weights for ICIO sectors 29, 301 and 302T309. "
                f"Found: {sorted(found_sectors)}"
            )

        target_weights["output_usd_mn"] = pd.to_numeric(
            target_weights["output_usd_mn"],
            errors="coerce",
        )

        if (
            target_weights["output_usd_mn"].isna().any()
            or target_weights["output_usd_mn"].le(0).any()
        ):
            raise RuntimeError(
                "NZL C29_30 allocation requires positive gross output "
                "for all three target industries."
            )

        group_output = target_weights["output_usd_mn"].sum()

        target_weights["weight"] = (
            target_weights["output_usd_mn"] / group_output
        )

        target_weights["mapped_output_total"] = group_output
        target_weights["allocation_method"] = (
            "aggregate_C29_30_relative_gross_output"
        )

        nzl_aggregate_mapped = nzl_c29_30_aggregate.merge(
            target_weights,
            on="country",
            how="inner",
            validate="many_to_many",
        )

        if (
            nzl_aggregate_mapped.groupby("activity")["weight"].sum()
            .sub(1.0).abs().gt(1e-10).any()
        ):
            raise RuntimeError(
                "NZL C29_30 gross-output allocation weights do not "
                "sum to one."
            )

        # Preserve the original AEA aggregate as the source activity.
        nzl_aggregate_mapped["source_activity"] = "C29_30"

        # Check that the allocation conserves the original aggregate.
        allocated_total = (
            nzl_aggregate_mapped["ghg_tonnes_co2e"]
            * nzl_aggregate_mapped["weight"]
        ).sum()

        original_total = nzl_c29_30_aggregate[
            "ghg_tonnes_co2e"
        ].sum()

        if not np.isclose(
            allocated_total,
            original_total,
            rtol=1e-10,
            atol=1e-6,
        ):
            raise RuntimeError(
                "NZL C29_30 allocation does not conserve emissions: "
                f"original={original_total}, allocated={allocated_total}"
            )

        mapped = pd.concat(
            [mapped, nzl_aggregate_mapped],
            ignore_index=True,
            sort=False,
        )


    # ---------------------------------------------------------------
    # 5. Audit zero-output mapped industries
    # ---------------------------------------------------------------

    zero_weight = mapped.loc[
        mapped["weight"].eq(0)
    ].copy()

    if not zero_weight.empty:
        zero_weight["allocation_status"] = "ZERO_OUTPUT_INDUSTRY"
        zero_weight["allocation_reason"] = (
            "ICIO industry has zero gross output; its allocation weight "
            "is zero and the activity emissions are allocated to the "
            "other positive-output industries mapped to this activity"
        )

    # ---------------------------------------------------------------
    # 6. Allocate emissions
    # ---------------------------------------------------------------

    mapped["ghg_tonnes_co2e"] = (
        mapped["ghg_tonnes_co2e"] * mapped["weight"]
    )

    mapped["source"] = "AEA"
    mapped["source_activity"] = mapped["activity"]

    # ---------------------------------------------------------------
    # 7. Combine allocation audit records
    # ---------------------------------------------------------------

    audit_frames = []

    if not outside_icio.empty:
        audit_frames.append(outside_icio)

    if not unmapped_aea.empty:
        audit_frames.append(unmapped_aea)

    if not nzl_c29_30_aggregate.empty:
        aggregate_audit = nzl_c29_30_aggregate.copy()
        aggregate_audit["allocation_status"] = (
            "AGGREGATE_ALLOCATED_BY_GROSS_OUTPUT"
        )
        aggregate_audit["allocation_reason"] = (
            "NZL C29_30 allocated across ICIO sectors 29, 301 "
            "and 302T309 using relative gross output"
        )
        aggregate_audit["source"] = "AEA"
        aggregate_audit["source_activity"] = "C29_30"
        audit_frames.append(aggregate_audit)

    if not zero_weight.empty:
        audit_frames.append(zero_weight)

    if not zero_emissions.empty:
        audit_frames.append(zero_emissions)

    if audit_frames:
        allocation_audit = pd.concat(
            audit_frames,
            ignore_index=True,
            sort=False,
        )
    else:
        allocation_audit = pd.DataFrame(
            columns=[
                "country",
                "activity",
                "allocation_status",
                "allocation_reason",
            ]
        )

    return mapped, excluded_parents, allocation_audit




def allocate_gtap_variable_emissions(
    gtap: pd.DataFrame,
    variable_shares: pd.DataFrame,
    weights: pd.DataFrame,
    aea_countries: set[str],
    icio_countries: set[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Allocate GTAP production emissions to ICIO regions.

    GTAP source regions are mapped through GTAP_TO_ICIO_REGION_MAP.
    Direct country-code matches are used where no explicit mapping exists.

    Allocation:
        1. GTAP emissions by source region, activity and pollutant.
        2. Pollutant-specific pooled AEA allocation shares.
        3. Country-specific ICIO gross-output weights.

    The output country is the destination ICIO country.
    The original GTAP source region is retained as gtap_region.

    GTAP emissions are excluded for countries covered by AEA.
    Unmapped GTAP regions and activities without allocation weights
    are returned separately for diagnostics.
    """
    gtap = gtap.copy()

    gtap["country"] = normalize_codes(gtap["region"])
    gtap["activity"] = normalize_codes(gtap["activity"])
    gtap["emission_variable"] = normalize_codes(
        gtap["emission_variable"]
    )

    aea_countries = set(normalize_codes(list(aea_countries)))
    icio_countries = set(normalize_codes(list(icio_countries)))

    # ---------------------------------------------------------------
    # Identify eligible GTAP source regions
    # ---------------------------------------------------------------

    selected_regions = set()

    for region in set(gtap["country"].dropna()):

        # Explicit mapping takes precedence.
        destinations = GTAP_TO_ICIO_REGION_MAP.get(
            region,
            [region] if region in icio_countries else [],
        )

        destinations = set(
            normalize_codes(destinations)
        )

        # Preserve the country-wide AEA / GTAP source-selection rule.
        if region in aea_countries:
            continue

        # A GTAP aggregate cannot be allocated partly to AEA-covered
        # countries and partly to GTAP-covered countries without a
        # defensible disaggregation.
        overlapping_destinations = destinations & aea_countries

        if (
            overlapping_destinations
            and destinations - aea_countries
        ):
            raise ValueError(
                f"GTAP region {region} maps to both AEA-covered and "
                f"GTAP-only ICIO destinations: {sorted(destinations)}. "
                "Resolve this mixed-region mapping before allocation."
            )

        # If every destination is AEA-covered, do not use GTAP.
        if destinations and destinations.issubset(aea_countries):
            continue

        # Keep unresolved regions in the selection so they appear
        # in the unallocated diagnostics rather than disappearing.
        selected_regions.add(region)

    gtap = gtap.loc[
        gtap["country"].isin(selected_regions)
    ].copy()

    # Preserve the original GTAP source-region identifier.
    gtap = gtap.rename(
        columns={"country": "gtap_region"}
    )

    # ---------------------------------------------------------------
    # Prepare allocation shares
    # ---------------------------------------------------------------

    shares = variable_shares.copy()

    shares["gtap_activity_code"] = normalize_codes(
        shares["gtap_activity_code"]
    )
    shares["aea_activity_code"] = normalize_codes(
        shares["aea_activity_code"]
    )
    shares["emission_variable"] = normalize_codes(
        shares["emission_variable"]
    )

    fallback = shares.loc[
        shares["emission_variable"].eq("__OUTPUT_FALLBACK__")
    ].copy()

    pollutant_shares = shares.loc[
        ~shares["emission_variable"].eq("__OUTPUT_FALLBACK__")
    ].copy()

    # ---------------------------------------------------------------
    # First merge exact pollutant-specific shares
    # ---------------------------------------------------------------

    exact = gtap.merge(
        pollutant_shares,
        left_on=["activity", "emission_variable"],
        right_on=["gtap_activity_code", "emission_variable"],
        how="left",
    )

    missing_share = exact["allocation_share"].isna()

    # Use pooled output fallback when no pollutant-specific share exists.
    if missing_share.any():

        fallback_rows = exact.loc[missing_share].drop(
            columns=[
                "gtap_activity_code",
                "aea_activity_code",
                "allocation_share",
                "allocation_method",
            ],
            errors="ignore",
        ).merge(
            fallback[
                [
                    "gtap_activity_code",
                    "aea_activity_code",
                    "allocation_share",
                    "allocation_method",
                ]
            ],
            left_on="activity",
            right_on="gtap_activity_code",
            how="left",
        )

        exact = pd.concat(
            [
                exact.loc[~missing_share],
                fallback_rows,
            ],
            ignore_index=True,
            sort=False,
        )

    # ---------------------------------------------------------------
    # Separate unmapped activities
    # ---------------------------------------------------------------

    unmapped = exact.loc[
        exact["aea_activity_code"].isna()
    ].copy()

    mapped = exact.loc[
        exact["aea_activity_code"].notna()
    ].copy()

    mapped["allocated_activity_emissions"] = (
        mapped["emissions_tonnes_co2e"]
        * mapped["allocation_share"]
    )

    # ---------------------------------------------------------------
    # Join GTAP source regions to their ICIO allocation weights
    # ---------------------------------------------------------------

    # The weights table stores:
    #   country      = original GTAP source region
    #   icio_country = actual ICIO destination
    #   io_key       = destination ICIO industry
    #   weight       = destination gross-output allocation share

    weights_for_merge = weights.copy()

    weights_for_merge["country"] = normalize_codes(
        weights_for_merge["country"]
    )

    weights_for_merge["country"] = (
        weights_for_merge["country"].rename("gtap_region")
    )

    mapped = mapped.merge(
        weights_for_merge.rename(
            columns={"country": "gtap_region"}
        ),
        on=["gtap_region", "aea_activity_code"],
        how="left",
        validate="many_to_many",
    )

    # Preserve records without an ICIO allocation weight.
    no_icio_weight = mapped.loc[
        mapped["io_key"].isna()
    ].copy()

    # Restore the source-region identifier as country for the
    # unallocated diagnostics and structural-zero classification.
    no_icio_weight["country"] = (
        no_icio_weight["gtap_region"]
    )

    mapped = mapped.loc[
        mapped["io_key"].notna()
    ].copy()

    # ---------------------------------------------------------------
    # Assign final ICIO destination country
    # ---------------------------------------------------------------

    mapped["country"] = normalize_codes(
        mapped["icio_country"]
    )

    mapped["ghg_tonnes_co2e"] = (
        mapped["allocated_activity_emissions"]
        * mapped["weight"]
    )

    mapped["source"] = "GTAP12"

    # Retain only valid ICIO destinations.
    invalid_destinations = mapped.loc[
        ~mapped["country"].isin(icio_countries)
    ]

    if not invalid_destinations.empty:
        raise ValueError(
            "GTAP allocation produced destinations absent from ICIO: "
            f"{sorted(invalid_destinations['country'].unique())}"
        )

    return mapped, pd.concat(
        [unmapped, no_icio_weight],
        ignore_index=True,
        sort=False,
    )





def add_nzl_c27_inferred_zero(aea, aea_allocation_audit):
    """Infer NZL C27 = 0 only if the 2022 AEA manufacturing total reconciles."""

    codes = [
        "C10T12", "C13T15", "C16", "C17", "C18", "C19",
        "C20", "C21", "C22", "C23", "C24", "C25", "C26",
        "C28", "C29_30", "C31_32", "C33",
    ]

    required = {"country", "activity", "ghg_tonnes_co2e"}
    missing_columns = required.difference(aea.columns)

    if missing_columns:
        raise RuntimeError(
            f"Cannot infer NZL C27 zero; AEA DataFrame is missing "
            f"columns: {sorted(missing_columns)}"
        )

    d = aea.loc[aea["country"].eq("NZL")].copy()

    if d["activity"].duplicated().any():
        duplicates = sorted(
            d.loc[d["activity"].duplicated(keep=False), "activity"].unique()
        )
        raise RuntimeError(
            f"Duplicate NZL AEA activity records; "
            f"cannot infer C27 zero safely: {duplicates}"
        )

    values = d.set_index("activity")["ghg_tonnes_co2e"]

    required_activities = ["C", *codes]
    missing = [
        code for code in required_activities
        if code not in values.index or pd.isna(values.loc[code])
    ]

    if missing:
        raise RuntimeError(
            f"Cannot infer NZL C27 zero; missing AEA activities: {missing}"
        )

    total = float(values.loc["C"])
    detail_sum = float(values.loc[codes].sum())
    residual = total - detail_sum

    if not np.isclose(residual, 0.0, rtol=0.0, atol=1e-4):
        raise RuntimeError(
            f"NZL manufacturing reconciliation failed: "
            f"C={total:.12f}, detail sum={detail_sum:.12f}, "
            f"residual={residual:.12f}"
        )

    if (
        aea_allocation_audit["country"].eq("NZL")
        & aea_allocation_audit["activity"].eq("C27")
    ).any():
        raise RuntimeError(
            "NZL C27 already exists in the AEA allocation audit."
        )

    inferred = pd.DataFrame([{
        "country": "NZL",
        "activity": "C27",
        "ghg_tonnes_co2e": 0.0,
        "allocation_status": "INFERRED_ZERO_FROM_AGGREGATE_RECONCILIATION",
        "allocation_reason": (
            "Inferred zero: reported 2022 manufacturing total C reconciles "
            "to all other non-overlapping reported manufacturing activities. "
            f"Residual={residual:.12g} tonnes CO2e."
        ),
    }])

    aea_allocation_audit = pd.concat(
        [aea_allocation_audit, inferred],
        ignore_index=True,
    )

    print(
        "NZL C27 inferred zero validated: "
        f"C={total:.6f}, detail sum={detail_sum:.6f}, "
        f"residual={residual:.12g} tonnes CO2e"
    )

    return aea_allocation_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(r"D:\Research\Datacenters"))
    parser.add_argument("--output-dir", type=Path, default=Path("hybrid_emissions"))
    parser.add_argument("--aea-file", type=str, default="OECD_AEA.csv")
    parser.add_argument("--aea-crosswalk", type=str, default="aea_crosswalk.csv")
    parser.add_argument("--icio-output", type=str, default="icio_output.csv")
    parser.add_argument("--sector-map", type=str, default="sector_map.csv")
    parser.add_argument("--gtap-mapping", type=str, default="gtap_activity_to_aea.csv")
    parser.add_argument("--gtap-co2", type=str, default=r"GDX12_23\GSDFEMISS.GDX")
    parser.add_argument("--gtap-nco2", type=str, default=r"GDX12_23\GSDFNCO2.GDX")
    parser.add_argument("--gams-system-dir", type=str, default=r"C:\GAMS\51")
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--aea-scope", type=str, default="RES")
    parser.add_argument("--aea-methodology", type=str, default="EMISSIONS_SEEA")
    parser.add_argument("--gwp-assessment", type=str, default="AR6")
    parser.add_argument("--allow-uncovered", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Load ICIO inputs
    # ---------------------------------------------------------------

    crosswalk, io, sector_map = load_icio_inputs(data_dir)

    io_countries = set(
        normalize_codes(sector_map["country"]).dropna()
    )

    print("\n=== ICIO COUNTRY DIAGNOSTICS ===")
    print("ICIO countries:")
    print(sorted(io_countries))
    print("\nNumber of ICIO countries:", len(io_countries))
    print("Number of ICIO industries:", len(io))
    print("\n=== END ICIO COUNTRY DIAGNOSTICS ===\n")

    co2_path = (data_dir / args.gtap_co2).resolve()
    nco2_path = (data_dir / args.gtap_nco2).resolve()
    mapping_path = (data_dir / args.gtap_mapping).resolve()

    # ---------------------------------------------------------------
    # Load GTAP-to-AEA activity mapping
    # ---------------------------------------------------------------

    if not mapping_path.exists():
        create_gtap_mapping_template(
            co2_path,
            args.gams_system_dir,
            mapping_path,
        )

        raise SystemExit(
            "\nGTAP-to-AEA mapping template created. Fill "
            "gtap_activity_to_aea.csv and rerun. Map GTAP activities "
            "to the AEA activity codes present in aea_crosswalk.csv. "
            "The allocation weights for each GTAP activity must sum to one."
        )

    gtap_mapping = load_gtap_mapping(mapping_path)

    valid_aea_codes = set(crosswalk["aea_activity_code"])

    unknown_aea_codes = sorted(
        set(gtap_mapping["aea_activity_code"]) - valid_aea_codes
    )

    if unknown_aea_codes:
        raise ValueError(
            "gtap_activity_to_aea.csv contains AEA activity codes absent "
            f"from aea_crosswalk.csv: {unknown_aea_codes[:30]}"
        )

    # ---------------------------------------------------------------
    # Extract GTAP 12 production emissions
    # ---------------------------------------------------------------

    LOG.info("Extracting GTAP 12 production-emissions components.")

    gtap_variables = extract_gtap_variable_emissions(
        co2_path,
        nco2_path,
        args.gams_system_dir,
        args.gwp_assessment,
    )

    gtap_variables["region"] = normalize_codes(
        gtap_variables["region"]
    )

    gtap_regions = set(
        gtap_variables["region"].dropna().unique()
    )

    gtap_variables.to_csv(
        output_dir / "gtap_production_emissions_by_variable.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Complete GTAP-to-ICIO regional mapping
    # ---------------------------------------------------------------

    GTAP_TO_ICIO_REGION_MAP.update(
        complete_gtap_to_icio_region_map(
            gtap_regions=gtap_regions,
            io_countries=io_countries,
        )
    )

    # Only GTAP regions actually present in the extracted emissions
    # are required in this run.
    unresolved_gtap_regions = sorted(
        region
        for region in gtap_regions
        if region not in GTAP_TO_ICIO_REGION_MAP
    )

    region_mapping = pd.DataFrame(
        [
            {
                "gtap_region": region,
                "icio_region": destination,
            }
            for region in sorted(gtap_regions)
            for destination in GTAP_TO_ICIO_REGION_MAP.get(region, [])
        ]
    )

    region_mapping.to_csv(
        output_dir / "gtap_to_icio_region_mapping.csv",
        index=False,
    )

    print("\n=== GTAP-TO-ICIO REGION MAPPING ===")
    print(region_mapping.to_string(index=False))

    if unresolved_gtap_regions:
        raise RuntimeError(
            "GTAP regions have no ICIO destination. Review the regional "
            "concordance before allocation: "
            f"{unresolved_gtap_regions}"
        )

    # Validate that every mapped destination exists in ICIO.
    invalid_destinations = sorted(
        {
            destination
            for region in gtap_regions
            for destination in GTAP_TO_ICIO_REGION_MAP[region]
            if destination not in io_countries
        }
    )

    if invalid_destinations:
        raise ValueError(
            "GTAP regional mapping contains destinations absent from "
            f"the ICIO country list: {invalid_destinations}"
        )

    print("\n=== END GTAP-TO-ICIO REGION MAPPING ===\n")

    # ---------------------------------------------------------------
    # Load OECD AEA emissions
    # ---------------------------------------------------------------

    LOG.info("Reading AEA observations for %s.", args.year)

    aea, aea_filtered_raw = load_aea(
        data_dir / args.aea_file,
        year=args.year,
        scope=args.aea_scope,
        methodology=args.aea_methodology or None,
    )

    mapped_activities = set(crosswalk["aea_activity_code"].dropna())

    # Use mapped observations only to determine country-wide AEA eligibility.
    # Preserve the full AEA dataset for allocation and unallocated auditing.
    aea_mapped = aea.loc[
        aea["activity"].isin(mapped_activities)
    ].copy()

    if aea_mapped.empty:
        raise ValueError(
            "No AEA observations matched the AEA-to-ICIO crosswalk after "
            "pollutant, year, scope and methodology filters. Inspect the "
            "AEA activity/pollutant filters before using GTAP."
        )

    # Country-wide source selection: AEA is retained for countries
    # with usable mapped resident-industry GHG observations.
    # Country-wide source selection: AEA is retained for countries
    # with usable mapped resident-industry GHG observations.
    aea_countries = set(aea_mapped["country"].unique()) & io_countries

    # Colombia: use GTAP 12 for the entire country because the AEA
    # extract does not report Section S, leaving ICIO sector 94T96
    # uncovered. Do not combine Colombian AEA and GTAP emissions.
    aea_countries.discard("COL")

    LOG.info("Usable AEA countries in ICIO: %s", len(aea_countries))
    LOG.info("Countries explicitly assigned to GTAP 12: %s", ["COL"])

    # ---------------------------------------------------------------
    # Load pollutant-level AEA observations
    # ---------------------------------------------------------------

    aea_pollutants = load_aea_pollutants(
        data_dir / args.aea_file,
        year=args.year,
        scope=args.aea_scope,
        methodology=args.aea_methodology or None,
    )

    # ---------------------------------------------------------------
    # Diagnose AEA / ICIO country-key overlap
    # ---------------------------------------------------------------

    print("\n=== AEA / ICIO COUNTRY-KEY DIAGNOSTICS ===")

    print("\nAEA pollutant country column:")
    print("Unique countries:", aea_pollutants["country"].nunique())
    print(
        "Sample:",
        sorted(aea_pollutants["country"].dropna().unique())[:30],
    )

    print("\nICIO country column:")
    print("Unique countries:", io["country"].nunique())
    print(
        "Sample:",
        sorted(io["country"].dropna().unique())[:30],
    )

    aea_country_keys = set(
        normalize_codes(aea_pollutants["country"]).dropna().unique()
    )

    io_country_keys = set(
        normalize_codes(io["country"]).dropna().unique()
    )

    print("\nNormalized AEA country keys:", len(aea_country_keys))
    print("Normalized ICIO country keys:", len(io_country_keys))
    print("Intersection:", len(aea_country_keys & io_country_keys))

    print("\nAEA countries not found in ICIO:")
    print(sorted(aea_country_keys - io_country_keys)[:50])

    print("\nICIO countries not found in AEA:")
    print(sorted(io_country_keys - aea_country_keys)[:50])

    print("\n=== END COUNTRY-KEY DIAGNOSTICS ===\n")

    # ---------------------------------------------------------------
    # Construct GTAP variable-specific AEA allocation shares
    # ---------------------------------------------------------------

    variable_shares = build_gtap_variable_shares(
        aea_pollutants=aea_pollutants,
        gtap_mapping=gtap_mapping,
        crosswalk=crosswalk,
        io=io,
        aea_countries=aea_countries,
    )

    variable_shares.to_csv(
        output_dir / "gtap_aea_variable_allocation_shares.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Construct country/activity allocation weights
    # ---------------------------------------------------------------

    # Include all ICIO countries and all GTAP source regions.
    # make_country_activity_weights() uses the global mapping to
    # determine each source region's actual ICIO destinations.
    all_countries = io_countries | gtap_regions

    all_activities = (
        mapped_activities
        | set(gtap_mapping["aea_activity_code"])
    )

    weights = make_country_activity_weights(
        all_countries,
        all_activities,
        crosswalk,
        io,
    )

    print("\n=== CHINA AND MEXICO REGIONAL ALLOCATION WEIGHTS ===")

    print(
        weights.loc[
            weights["country"].isin(["CHN", "MEX"])
        ]
        .groupby(
            ["country", "aea_activity_code", "icio_country"],
            as_index=False,
        )
        .agg(
            output_usd_mn=("output_usd_mn", "sum"),
            weight=("weight", "sum"),
        )
        .to_string(index=False)
    )

    print("\n=== CHINA AND MEXICO WEIGHT SUMS ===")

    print(
        weights.loc[
            weights["country"].isin(["CHN", "MEX"])
        ]
        .groupby(
            ["country", "aea_activity_code"]
        )["weight"]
        .sum()
        .groupby(level=0)
        .agg(["count", "min", "max"])
    )

    # ---------------------------------------------------------------
    # Diagnose missing country/activity weights
    # ---------------------------------------------------------------

    print("\n=== MISSING ICIO WEIGHT DIAGNOSTICS ===")

    for country in ["CHN", "MEX", "SGP", "BRN", "CHL", "MMR", "ROW"]:

        country_io_regions = set(
            GTAP_TO_ICIO_REGION_MAP.get(country, [country])
        )

        country_io = io.loc[
            normalize_codes(io["country"]).isin(country_io_regions)
        ].copy()

        country_weights = weights.loc[
            weights["country"].eq(country)
        ].copy()

        print(f"\n--- {country} ---")
        print("ICIO destination regions:", sorted(country_io_regions))
        print("ICIO industries:", len(country_io))

        print(
            "Total gross output:",
            country_io["output_usd_mn"].sum(),
        )

        print(
            "Positive-output industries:",
            (country_io["output_usd_mn"] > 0).sum(),
        )

        print("Constructed activity weights:", len(country_weights))

        available = set(country_weights["aea_activity_code"])
        missing = sorted(set(all_activities) - available)

        print("\nActivities without weights:", missing)

        if missing:
            print("\nCrosswalk targets for missing activities:")

            targets = (
                crosswalk.loc[
                    crosswalk["aea_activity_code"].isin(missing),
                    ["aea_activity_code", "icio_sector_code"],
                ]
                .drop_duplicates()
                .sort_values(
                    ["aea_activity_code", "icio_sector_code"]
                )
            )

            print(targets.to_string(index=False))

    print("\n=== END MISSING ICIO WEIGHT DIAGNOSTICS ===\n")

    # ---------------------------------------------------------------
    # Additional GTAP-only weight diagnostics
    # ---------------------------------------------------------------

    print("\n=== GTAP-ONLY ICIO WEIGHT DIAGNOSTICS ===")

    print("Weights shape:", weights.shape)

    print("\nWeights for problematic activities:")

    print(
        weights.loc[
            weights["aea_activity_code"].isin(
                ["C29", "55T56", "H51", "C21", "C10T12"]
            )
            & weights["country"].isin(["BRN", "CHL", "CHN", "MMR", "ROW"])
        ].to_string(index=False)
    )

    print("\nCountry coverage:")

    print(
        weights.groupby("country")["aea_activity_code"]
        .nunique()
        .to_string()
    )

    print("\n=== END WEIGHT DIAGNOSTICS ===")

    weights.to_csv(
        output_dir / "icio_activity_allocation_weights.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Diagnose selected AEA emissions associated with zero-output
    # ICIO industries
    # ---------------------------------------------------------------

    diagnostic_targets = {
        ("MLT", "A02"),
        ("LUX", "C19"),
        ("LUX", "A03"),
    }

    aea_diagnostic = aea.loc[
        [
            (country, activity) in diagnostic_targets
            for country, activity in zip(
                aea["country"],
                aea["activity"],
            )
        ]
    ].copy()

    print("\n=== AEA EMISSIONS FOR ZERO-OUTPUT INDUSTRIES ===")

    if aea_diagnostic.empty:
        print("No matching AEA observations.")

    else:
        print(
            aea_diagnostic[
                [
                    "country",
                    "activity",
                    "ghg_tonnes_co2e",
                    "aea_ghg_method",
                ]
            ].to_string(index=False)
        )

        print("\nTotals by country and activity:")

        print(
            aea_diagnostic.groupby(
                ["country", "activity"],
                as_index=False,
            )["ghg_tonnes_co2e"]
            .sum()
            .to_string(index=False)
        )

    # ---------------------------------------------------------------
    # Allocate AEA emissions
    # ---------------------------------------------------------------

    aea_allocated, excluded_aea_parents, aea_allocation_audit = allocate_aea(
        aea=aea,
        crosswalk=crosswalk,
        weights=weights,
        icio_countries=io_countries,
    )
    
    aea_allocation_audit = add_nzl_c27_inferred_zero(
        aea,  # use the name of your raw OECD AEA DataFrame here
        aea_allocation_audit,
    )

    # ---------------------------------------------------------------
    # Validate the original New Zealand C29_30 allocation
    # ---------------------------------------------------------------

    # allocate_aea() already allocates the retained C29_30 aggregate
    # across sectors 29, 301 and 302T309. Do not append another allocation.

    nz_aggregate = (
        aea.loc[
            aea["country"].eq("NZL")
            & aea["activity"].eq("C29_30")
        ]
        .groupby(
            ["country", "activity"],
            as_index=False,
        )
        .agg(
            ghg_tonnes_co2e=("ghg_tonnes_co2e", "sum"),
        )
    )

    if not nz_aggregate.empty:

        nz_existing = aea_allocated.loc[
            aea_allocated["country"].eq("NZL")
            & aea_allocated["activity"].eq("C29_30")
        ].copy()

        expected_nz_sectors = {"29", "301", "302T309"}

        actual_nz_sectors = set(
            nz_existing["icio_sector_code"].astype(str)
        )

        allocated_nz_total = nz_existing["ghg_tonnes_co2e"].sum()

        reported_nz_total = float(
            nz_aggregate["ghg_tonnes_co2e"].iloc[0]
        )

        print("\n=== ORIGINAL NZL C29_30 ALLOCATION ===")

        print(
            nz_existing[
                [
                    "country",
                    "activity",
                    "source_activity",
                    "aea_activity_code",
                    "icio_sector_code",
                    "io_key",
                    "ghg_tonnes_co2e",
                ]
            ].to_string(index=False)
        )

        print("\nReported NZL C29_30 emissions:", reported_nz_total)
        print("Allocated NZL C29_30 emissions:", allocated_nz_total)
        print("Allocated sectors:", sorted(actual_nz_sectors))

        if actual_nz_sectors != expected_nz_sectors:
            raise RuntimeError(
                "The original NZL C29_30 allocation does not cover "
                "exactly sectors 29, 301 and 302T309. "
                f"Found: {sorted(actual_nz_sectors)}"
            )

        if not np.isclose(
            allocated_nz_total,
            reported_nz_total,
            rtol=1e-12,
            atol=1e-6,
        ):
            raise RuntimeError(
                "The original NZL C29_30 allocation does not conserve "
                "the reported aggregate emissions. "
                f"Reported={reported_nz_total}; "
                f"allocated={allocated_nz_total}."
            )

        LOG.info(
            "NZL C29_30 allocation validated: %.6f tonnes CO2e "
            "across sectors 29, 301 and 302T309.",
            allocated_nz_total,
        )

    # Save AEA allocation diagnostics.
    excluded_aea_parents.to_csv(
        output_dir / "aea_aggregate_activity_exclusions.csv",
        index=False,
    )

    aea_allocation_audit.to_csv(
        output_dir / "aea_allocation_audit.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Determine GTAP source coverage using mapped ICIO destinations
    # ---------------------------------------------------------------

    gtap_region_destinations = {
        region: {
            str(destination).strip().upper()
            for destination in GTAP_TO_ICIO_REGION_MAP.get(region, [])
            if pd.notna(destination)
        }
        for region in gtap_regions
    }

    gtap_icio_countries = set()

    for region, destinations in gtap_region_destinations.items():
        gtap_icio_countries.update(destinations)

    # AEA takes precedence at the country level. Do not allow an
    # aggregate GTAP region to overlap AEA coverage.
    gtap_only_destinations = gtap_icio_countries - aea_countries

    missing_gtap_countries = (
        io_countries - aea_countries - gtap_icio_countries
    )

    print("\n=== GTAP / ICIO REGION OVERLAP ===")

    print("GTAP regions:", sorted(gtap_regions))
    print("ICIO regions:", sorted(io_countries))

    print("\nGTAP-to-ICIO destinations:")

    for region in sorted(gtap_region_destinations):
        print(
            f"{region}: {sorted(gtap_region_destinations[region])}"
        )

    print("\nGTAP-covered ICIO destinations:", sorted(gtap_icio_countries))
    print("GTAP-only ICIO destinations:", sorted(gtap_only_destinations))
    print("AEA-covered ICIO destinations:", sorted(aea_countries))
    print("ICIO destinations without AEA or GTAP:", sorted(missing_gtap_countries))

    print("=== END REGION OVERLAP ===\n")

    if missing_gtap_countries:
        LOG.warning(
            "ICIO countries with neither AEA nor GTAP regional "
            "coverage: %s",
            sorted(missing_gtap_countries),
        )

    # ---------------------------------------------------------------
    # Allocate GTAP emissions
    # ---------------------------------------------------------------

    gtap_allocated, gtap_unmapped = allocate_gtap_variable_emissions(
        gtap=gtap_variables,
        variable_shares=variable_shares,
        weights=weights,
        aea_countries=aea_countries,
        icio_countries=io_countries,
    )

    # Add ICIO target and accounting-status diagnostics.
    gtap_unmapped = classify_gtap_unallocated(
        unallocated=gtap_unmapped,
        crosswalk=crosswalk,
        io=io,
    )

    # ---------------------------------------------------------------
    # Diagnose GTAP emissions that could not be allocated to ICIO
    # ---------------------------------------------------------------

    print("\n=== GTAP EMISSIONS WITHOUT ICIO ALLOCATION ===")

    print("Number of unallocated records:", len(gtap_unmapped))

    if not gtap_unmapped.empty:

        positive_unmapped = gtap_unmapped.loc[
            pd.to_numeric(
                gtap_unmapped["emissions_tonnes_co2e"],
                errors="coerce",
            ).fillna(0).abs() > 1e-6
        ].copy()

        # -----------------------------------------------------------
        # 1. Confirmed structural-zero emissions
        # -----------------------------------------------------------

        confirmed_structural_zeros = positive_unmapped.loc[
            positive_unmapped["unallocated_reason"].eq(
                "CONFIRMED_STRUCTURAL_ZERO"
            )
        ]

        # -----------------------------------------------------------
        # 2. Explicitly separate DWE activity
        #
        # DWE is intentionally excluded from the AEA production
        # concordance. It must remain unallocated and must not be
        # redistributed or silently discarded.
        # -----------------------------------------------------------

        intentionally_separate = positive_unmapped.loc[
            positive_unmapped["activity"].eq("DWE")
            & positive_unmapped["unallocated_reason"].eq(
                "NO_AEA_CONCORDANCE"
            )
        ]

        # -----------------------------------------------------------
        # 3. Everything else remains a genuine unresolved error
        # -----------------------------------------------------------

        accounted_for = (
            positive_unmapped["unallocated_reason"].eq(
                "CONFIRMED_STRUCTURAL_ZERO"
            )
            |
            (
                positive_unmapped["activity"].eq("DWE")
                & positive_unmapped["unallocated_reason"].eq(
                    "NO_AEA_CONCORDANCE"
                )
            )
        )

        unresolved_unallocated = positive_unmapped.loc[
            ~accounted_for
        ]

        if not confirmed_structural_zeros.empty:
            LOG.warning(
                "%s positive GTAP emissions records remain explicitly "
                "unallocated because their ICIO targets are confirmed "
                "structural zeros. Total emissions: %.6f tonnes CO2e. "
                "See gtap_unallocated_emissions_diagnostics.csv.",
                len(confirmed_structural_zeros),
                confirmed_structural_zeros[
                    "emissions_tonnes_co2e"
                ].sum(),
            )

        if not intentionally_separate.empty:
            LOG.warning(
                "%s positive GTAP emissions records remain explicitly "
                "unallocated because DWE is intentionally excluded from "
                "the production-sector AEA concordance. Total emissions: "
                "%.6f tonnes CO2e. These emissions are retained in the "
                "unallocated diagnostics and are not redistributed.",
                len(intentionally_separate),
                intentionally_separate[
                    "emissions_tonnes_co2e"
                ].sum(),
            )

        if not unresolved_unallocated.empty:

            print("\n=== UNRESOLVED POSITIVE GTAP EMISSIONS ===")

            print(
                unresolved_unallocated.groupby(
                    [
                        "country",
                        "activity",
                        "unallocated_reason",
                    ],
                    dropna=False,
                    as_index=False,
                )
                .agg(
                    records=("emissions_tonnes_co2e", "size"),
                    emissions_tonnes_co2e=(
                        "emissions_tonnes_co2e",
                        "sum",
                    ),
                )
                .to_string(index=False)
            )

            raise RuntimeError(
                "GTAP regions or activities have positive emissions that "
                "could not be allocated and are not confirmed structural-zero "
                "or explicitly separate DWE cases. Investigate the regional "
                "concordance, GTAP-to-AEA mapping, ICIO targets and allocation "
                "weights. See "
                f"{output_dir / 'gtap_unallocated_emissions_diagnostics.csv'}."
            )

    # ---------------------------------------------------------------
    # Combine AEA and GTAP emissions
    # ---------------------------------------------------------------

    aea_allocated = aea_allocated.loc[
        aea_allocated["country"].isin(aea_countries)
    ].copy()

    # GTAP allocations are already assigned to destination ICIO countries.
    gtap_allocated = gtap_allocated.loc[
        ~gtap_allocated["country"].isin(aea_countries)
    ].copy()

    components = pd.concat(
        [aea_allocated, gtap_allocated],
        ignore_index=True,
        sort=False,
    )

    source_counts = components.groupby("country")["source"].nunique()
    mixed = source_counts[source_counts > 1]

    if not mixed.empty:
        raise RuntimeError(
            "Source selection failed; ICIO countries have multiple sources: "
            f"{mixed.index.tolist()}"
        )

    # ---------------------------------------------------------------
    # Aggregate emissions by ICIO industry
    # ---------------------------------------------------------------

    emissions = (
        components.groupby(
            [
                "io_key",
                "country",
                "icio_sector_code",
                "source",
            ],
            as_index=False,
        )["ghg_tonnes_co2e"]
        .sum()
    )

    emissions.to_csv(
        output_dir / "direct_emissions_components.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Construct complete ICIO coverage audit
    # ---------------------------------------------------------------

    total = (
        components.groupby(
            "io_key",
            as_index=False,
        )["ghg_tonnes_co2e"]
        .sum()
    )

    complete = sector_map[
        ["io_key", "country", "sector_code"]
    ].merge(
        total,
        on="io_key",
        how="left",
        validate="one_to_one",
    )

    complete = complete.merge(
        emissions[
            ["io_key", "source"]
        ].drop_duplicates("io_key"),
        on="io_key",
        how="left",
        validate="one_to_one",
    )

    # ---------------------------------------------------------------
    # Recognize explicitly reported zero-emissions AEA industries
    # ---------------------------------------------------------------

    zero_statuses = {
        "ZERO_EMISSIONS",
        "INFERRED_ZERO_FROM_AGGREGATE_RECONCILIATION",
    }

    zero_audit = aea_allocation_audit.loc[
        aea_allocation_audit["allocation_status"].isin(zero_statuses),
        ["country", "activity"],
    ].drop_duplicates()

    # Only use detailed AEA activities (hierarchy levels 1–3).
    # In particular, do not use residual section aggregates such as C_X.
    detailed_crosswalk = crosswalk.loc[
        pd.to_numeric(
            crosswalk["hierarchy_level"],
            errors="coerce",
        ).le(3),
        ["aea_activity_code", "icio_sector_code"],
    ].drop_duplicates()

    zero_targets = zero_audit.merge(
        weights[
            [
                "country",
                "aea_activity_code",
                "icio_sector_code",
                "io_key",
                "weight",
                "output_usd_mn",
            ]
        ],
        left_on=["country", "activity"],
        right_on=["country", "aea_activity_code"],
        how="inner",
        validate="many_to_many",
    )

    zero_targets = zero_targets.merge(
        detailed_crosswalk,
        on=["aea_activity_code", "icio_sector_code"],
        how="inner",
        validate="many_to_many",
    )

    zero_targets["weight"] = pd.to_numeric(
        zero_targets["weight"],
        errors="coerce",
    )
    zero_targets["output_usd_mn"] = pd.to_numeric(
        zero_targets["output_usd_mn"],
        errors="coerce",
    )

    zero_targets = zero_targets.loc[
        zero_targets["weight"].gt(0)
        & zero_targets["output_usd_mn"].gt(0)
    ].copy()

    zero_io_keys = set(
        zero_targets["io_key"].astype(str).str.strip()
    )

    # Direct, one-to-one AEA zero observations establish coverage
    # even when the corresponding ICIO industry has zero gross output
    # and therefore no positive allocation weight.

    direct_crosswalk = crosswalk.loc[
        crosswalk["mapping_type"].eq("direct"),
        ["aea_activity_code", "icio_sector_code"],
    ].drop_duplicates()

    direct_zero_targets = zero_audit.merge(
        direct_crosswalk,
        left_on="activity",
        right_on="aea_activity_code",
        how="inner",
        validate="many_to_many",
    )

    # Keep only activities with exactly one direct ICIO destination.
    direct_target_counts = (
        direct_zero_targets.groupby(
            ["country", "activity"]
        )["icio_sector_code"]
        .nunique()
    )

    unique_direct_targets = set(
        direct_target_counts[
            direct_target_counts.eq(1)
        ].index
    )

    direct_zero_targets = direct_zero_targets.loc[
        [
            (country, activity) in unique_direct_targets
            for country, activity in zip(
                direct_zero_targets["country"],
                direct_zero_targets["activity"],
            )
        ]
    ].copy()

    direct_zero_targets = direct_zero_targets.merge(
        sector_map[
            ["io_key", "country", "sector_code"]
        ],
        left_on=["country", "icio_sector_code"],
        right_on=["country", "sector_code"],
        how="inner",
        validate="many_to_one",
    )

    direct_zero_keys = set(
        direct_zero_targets.loc[
            direct_zero_targets["sector_code"].ne("97T98"),
            "io_key",
        ].astype(str).str.strip()
    )

    zero_io_keys.update(direct_zero_keys)

    # ---------------------------------------------------------------
    # Infer Colombian zeros from matching reported aggregate totals
    # ---------------------------------------------------------------

    # These are explicit, country-specific inferences:
    # C24_25 = C24 implies no residual emissions for C25.
    # C29_30 = C29 implies no residual emissions for C30.
    #
    # C30 maps to ICIO sectors 301 and 302T309.
    #
    # These are inferred zeros, not directly reported zero observations.

    col_aggregate_zero_rules = [
        {
            "aggregate_activity": "C24_25",
            "detailed_activity": "C24",
            "target_sectors": ["25"],
        },
        {
            "aggregate_activity": "C29_30",
            "detailed_activity": "C29",
            "target_sectors": ["301", "302T309"],
        },
    ]

    col_aea_totals = (
        aea.loc[
            aea["country"].eq("COL")
            & aea["activity"].isin(
                {
                    "C24_25",
                    "C24",
                    "C29_30",
                    "C29",
                }
            )
        ]
        .groupby(
            ["country", "activity"],
            as_index=False,
        )["ghg_tonnes_co2e"]
        .sum()
    )

    inferred_zero_records = []

    for rule in col_aggregate_zero_rules:

        aggregate_activity = rule["aggregate_activity"]
        detailed_activity = rule["detailed_activity"]

        aggregate_rows = col_aea_totals.loc[
            col_aea_totals["activity"].eq(aggregate_activity)
        ]

        detailed_rows = col_aea_totals.loc[
            col_aea_totals["activity"].eq(detailed_activity)
        ]

        if aggregate_rows.empty or detailed_rows.empty:
            raise RuntimeError(
                "Cannot establish Colombian inferred-zero rule: "
                f"missing {aggregate_activity} or {detailed_activity}."
            )

        aggregate_value = float(
            aggregate_rows["ghg_tonnes_co2e"].iloc[0]
        )

        detailed_value = float(
            detailed_rows["ghg_tonnes_co2e"].iloc[0]
        )

        residual = aggregate_value - detailed_value

        if not np.isclose(
            aggregate_value,
            detailed_value,
            rtol=1e-10,
            atol=1e-4,
        ):
            raise RuntimeError(
                "Colombian aggregate does not equal its detailed "
                "observation; inferred zero is not justified. "
                f"{aggregate_activity}={aggregate_value}; "
                f"{detailed_activity}={detailed_value}; "
                f"residual={residual}"
            )

        for sector in rule["target_sectors"]:
            inferred_zero_records.append(
                {
                    "country": "COL",
                    "aggregate_activity": aggregate_activity,
                    "detailed_activity": detailed_activity,
                    "icio_sector_code": sector,
                    "aggregate_emissions_tonnes": aggregate_value,
                    "detailed_emissions_tonnes": detailed_value,
                    "residual_tonnes": residual,
                    "inference_reason": (
                        "Reported aggregate equals the corresponding "
                        "detailed AEA observation; zero residual "
                        "emissions inferred for the remaining activity"
                    ),
                }
            )

    inferred_zero_audit = pd.DataFrame(inferred_zero_records)

    inferred_zero_targets = inferred_zero_audit.merge(
        sector_map[
            ["io_key", "country", "sector_code"]
        ],
        left_on=["country", "icio_sector_code"],
        right_on=["country", "sector_code"],
        how="left",
        validate="many_to_one",
    )

    if inferred_zero_targets["io_key"].isna().any():
        raise RuntimeError(
            "A Colombian inferred-zero target is absent from sector_map."
        )

    inferred_zero_targets["inferred_emissions_tonnes"] = 0.0

    inferred_zero_io_keys = set(
        inferred_zero_targets["io_key"].astype(str).str.strip()
    )

    inferred_zero_targets.to_csv(
        output_dir / "aea_inferred_zero_coverage.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Apply explicit and inferred zero coverage
    # ---------------------------------------------------------------

    known_io_keys = set(
        sector_map["io_key"].astype(str).str.strip()
    )

    invalid_zero_keys = sorted(zero_io_keys - known_io_keys)

    if invalid_zero_keys:
        raise RuntimeError(
            "AEA zero-emissions observations map to ICIO keys "
            f"absent from sector_map: {invalid_zero_keys}"
        )

    invalid_inferred_keys = sorted(
        inferred_zero_io_keys - known_io_keys
    )

    if invalid_inferred_keys:
        raise RuntimeError(
            "Inferred Colombian zero targets map to ICIO keys "
            f"absent from sector_map: {invalid_inferred_keys}"
        )

    # Explicit zero observations establish coverage only where no
    # nonzero emissions component has already been allocated.
    zero_coverage_mask = (
        complete["io_key"].astype(str).str.strip().isin(zero_io_keys)
        & complete["ghg_tonnes_co2e"].isna()
    )

    complete.loc[
        zero_coverage_mask,
        "ghg_tonnes_co2e",
    ] = 0.0

    complete.loc[
        zero_coverage_mask,
        "source",
    ] = "AEA"

    # Apply inferred Colombian zeros separately from directly reported zeros.
    inferred_zero_mask = (
        complete["io_key"].astype(str).str.strip().isin(
            inferred_zero_io_keys
        )
        & complete["ghg_tonnes_co2e"].isna()
    )

    complete.loc[
        inferred_zero_mask,
        "ghg_tonnes_co2e",
    ] = 0.0

    complete.loc[
        inferred_zero_mask,
        "source",
    ] = "AEA"

    # Preserve distinct coverage statuses.
    complete["coverage_status"] = np.where(
        complete["ghg_tonnes_co2e"].notna(),
        "covered",
        "uncovered",
    )

    complete.loc[
        zero_coverage_mask,
        "coverage_status",
    ] = "zero_emissions"

    complete.loc[
        inferred_zero_mask,
        "coverage_status",
    ] = "inferred_zero_from_aggregate"

    complete["io_key"] = (
        complete["io_key"]
        .astype(str)
        .str.strip()
    )

    structural_zero_mask = complete["io_key"].isin(
        CONFIRMED_STRUCTURAL_ZERO_IO_KEYS
    )

    complete.loc[
        structural_zero_mask
        & complete["coverage_status"].eq("uncovered"),
        "coverage_status",
    ] = "structural_zero_unallocated"

    complete.loc[
        structural_zero_mask
        & complete["coverage_status"].eq("structural_zero_unallocated"),
        "source",
    ] = "GTAP12_UNALLOCATED_STRUCTURAL_ZERO"

    # Household employers and own-use production are outside the
    # production-industry emissions extension.
    out_of_scope_mask = (
        complete["sector_code"].astype(str).str.strip().eq("97T98")
    )

    complete.loc[
        out_of_scope_mask,
        "coverage_status",
    ] = "out_of_scope"

    complete.loc[
        out_of_scope_mask,
        "source",
    ] = "OUT_OF_SCOPE"

    gaps = complete.loc[
        complete["coverage_status"].eq("uncovered")
    ].copy()

    complete.to_csv(
        output_dir / "direct_emissions_coverage_audit.csv",
        index=False,
    )

    gaps.to_csv(
        output_dir / "uncovered_icio_industries.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Record source selection by ICIO country / region
    # ---------------------------------------------------------------

    country_source = pd.DataFrame(
        {"country": sorted(io_countries)}
    )

    country_source["source"] = np.where(
        country_source["country"].isin(aea_countries),
        "AEA",
        np.where(
            country_source["country"].isin(gtap_icio_countries),
            "GTAP12",
            "UNCOVERED",
        ),
    )

    country_source.to_csv(
        output_dir / "emissions_source_by_country.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Check uncovered industries
    # ---------------------------------------------------------------

    if not gaps.empty and not args.allow_uncovered:
        raise RuntimeError(
            f"{len(gaps)} ICIO industries have no emissions source. "
            f"See {output_dir / 'uncovered_icio_industries.csv'}. "
            "Review the GTAP-to-AEA concordance and country coverage. "
            "Use --allow-uncovered only if you explicitly accept zero placeholders."
        )

    complete["ghg_tonnes_co2e"] = (
        complete["ghg_tonnes_co2e"].fillna(0.0)
    )

    complete[
        [
            "io_key",
            "ghg_tonnes_co2e",
            "source",
        ]
    ].to_csv(
        output_dir / "direct_emissions.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Country-level emissions summary
    # ---------------------------------------------------------------

    country_summary = (
        components.groupby(
            ["country", "source"],
            as_index=False,
        )
        .agg(
            mapped_icio_industries=("io_key", "nunique"),
            ghg_tonnes_co2e=("ghg_tonnes_co2e", "sum"),
        )
    )

    country_summary.to_csv(
        output_dir / "emissions_source_summary.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Final log
    # ---------------------------------------------------------------

    LOG.info("AEA countries: %s", len(aea_countries))
    LOG.info("GTAP source regions: %s", len(gtap_regions))
    LOG.info(
        "GTAP-covered ICIO destinations: %s",
        len(gtap_icio_countries),
    )
    LOG.info(
        "ICIO countries without AEA or GTAP coverage: %s",
        len(missing_gtap_countries),
    )
    LOG.info(
        "Covered ICIO industries: %s / %s",
        complete["coverage_status"].isin(
            ["covered", "zero_emissions", "inferred_zero_from_aggregate"]
        ).sum(),
        len(complete),
    )
    LOG.info(
        "Explicitly reported zero-emissions industries: %s",
        complete["coverage_status"].eq("zero_emissions").sum(),
    )
    LOG.info(
        "Inferred-zero industries: %s",
        complete["coverage_status"].eq(
            "inferred_zero_from_aggregate"
        ).sum(),
    )
    LOG.info(
        "Structural-zero unallocated industries: %s",
        complete["coverage_status"].eq(
            "structural_zero_unallocated"
        ).sum(),
    )
    LOG.info("Hybrid extension written to %s", output_dir)
    LOG.info(
        "Model-ready file: %s",
        output_dir / "direct_emissions.csv",
    )


if __name__ == "__main__":
    main()

