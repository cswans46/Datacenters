
#!/usr/bin/env python
"""
Diagnose OECD ICIO output construction and sector mapping.

Required inputs:
    icio_output.csv
    sector_map.csv
    aea_crosswalk.csv

Optional:
    A raw ICIO output extract, with one observation per country-sector.
    Supply either a raw io_key column or raw country and sector columns.

The script does not allocate emissions or modify any input files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# These are the cases identified in the GTAP/AEA allocation diagnostics.
TARGET_CASES = [
    ("BRN", "MVH", "C29"),
    ("CHL", "MVH", "C29"),
    ("SGP", "OXT", "B"),
    ("SGP", "OIL", "B"),
    ("SGP", "COA", "B"),
    ("SGP", "GAS", "B"),
    ("SGP", "FRS", "A02"),
]


def normalize_code(series: pd.Series) -> pd.Series:
    """Normalize sector codes without converting them to integers."""
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.upper()
    )


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")

    # utf-8-sig handles files saved with a Windows/Excel BOM.
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    print(f"Loaded {label}: {path} ({len(df):,} rows)")
    return df


def require_columns(
    df: pd.DataFrame,
    required: list[str],
    label: str,
) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def save(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose OECD ICIO output construction and sector mapping."
    )
    parser.add_argument(
        "--icio-output",
        type=Path,
        default=Path("icio_output.csv"),
    )
    parser.add_argument(
        "--sector-map",
        type=Path,
        default=Path("sector_map.csv"),
    )
    parser.add_argument(
        "--crosswalk",
        type=Path,
        default=Path("aea_crosswalk.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("icio_output_diagnostics"),
    )

    # Optional raw-source comparison.
    parser.add_argument("--raw-icio", type=Path, default=None)
    parser.add_argument("--raw-output-column", default=None)
    parser.add_argument("--raw-key-column", default=None)
    parser.add_argument("--raw-country-column", default=None)
    parser.add_argument("--raw-sector-column", default=None)

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 90)
    print("OECD ICIO OUTPUT CONSTRUCTION DIAGNOSTICS")
    print("=" * 90)

    # ------------------------------------------------------------------
    # 1. Read and validate the three constructed input files.
    # ------------------------------------------------------------------
    output = read_csv(args.icio_output, "ICIO output")
    sector_map = read_csv(args.sector_map, "sector map")
    crosswalk = read_csv(args.crosswalk, "AEA-ICIO crosswalk")

    require_columns(
        output,
        ["io_key", "output_usd_mn"],
        "ICIO output",
    )
    require_columns(
        sector_map,
        ["io_key", "country", "sector_code", "sector_label"],
        "Sector map",
    )
    require_columns(
        crosswalk,
        ["aea_activity_code", "icio_sector_code"],
        "AEA-ICIO crosswalk",
    )

    for df, columns in [
        (output, ["io_key"]),
        (sector_map, ["io_key", "country", "sector_code"]),
        (crosswalk, ["aea_activity_code", "icio_sector_code"]),
    ]:
        for col in columns:
            df[col] = normalize_code(df[col])

    output["output_usd_mn"] = pd.to_numeric(
        output["output_usd_mn"], errors="coerce"
    )

    print("\n--- BASIC DATA VALIDATION ---")
    print(f"ICIO output rows: {len(output):,}")
    print(f"Sector map rows:  {len(sector_map):,}")
    print(f"Crosswalk rows:   {len(crosswalk):,}")

    print("\nMissing output values:", output["output_usd_mn"].isna().sum())
    print("Negative output values:", (output["output_usd_mn"] < 0).sum())
    print("Exactly zero output values:", (output["output_usd_mn"] == 0).sum())
    print(
        "Nonzero output values:",
        (output["output_usd_mn"].notna() & (output["output_usd_mn"] != 0)).sum(),
    )

    # Duplicate checks before any merges.
    duplicate_output = output[
        output.duplicated("io_key", keep=False)
    ].sort_values("io_key")

    duplicate_map = sector_map[
        sector_map.duplicated("io_key", keep=False)
    ].sort_values("io_key")

    print("\nDuplicate io_key records:")
    print(f"  ICIO output: {len(duplicate_output):,}")
    print(f"  Sector map:  {len(duplicate_map):,}")

    save(duplicate_output, args.output_dir / "duplicate_output_keys.csv")
    save(duplicate_map, args.output_dir / "duplicate_sector_map_keys.csv")

    if not duplicate_output.empty:
        print("\nWARNING: ICIO output contains duplicate io_key values.")
    if not duplicate_map.empty:
        print("\nWARNING: sector_map contains duplicate io_key values.")

    # Do not allow a many-to-many merge to silently multiply output.
    if not duplicate_output.empty or not duplicate_map.empty:
        raise ValueError(
            "Resolve duplicate io_key records before interpreting the "
            "country/sector diagnostics. Duplicate records were exported."
        )

    # ------------------------------------------------------------------
    # 2. Check whether output keys and sector-map keys agree.
    # ------------------------------------------------------------------
    output_keys = set(output["io_key"].dropna())
    map_keys = set(sector_map["io_key"].dropna())

    output_without_map = sorted(output_keys - map_keys)
    map_without_output = sorted(map_keys - output_keys)

    print("\n--- KEY CONSISTENCY ---")
    print(f"Output keys absent from sector map: {len(output_without_map):,}")
    print(f"Sector-map keys absent from output: {len(map_without_output):,}")

    save(
        pd.DataFrame({"io_key": output_without_map}),
        args.output_dir / "output_keys_without_sector_map.csv",
    )
    save(
        pd.DataFrame({"io_key": map_without_output}),
        args.output_dir / "sector_map_keys_without_output.csv",
    )

    merged = sector_map.merge(
        output,
        on="io_key",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    merged["output_usd_mn"] = pd.to_numeric(
        merged["output_usd_mn"], errors="coerce"
    )

    # Check that io_key is consistent with country + sector_code.
    merged["expected_io_key"] = (
        merged["country"].astype("string")
        + "_"
        + merged["sector_code"].astype("string")
    )
    merged["io_key_matches_country_sector"] = (
        merged["io_key"] == merged["expected_io_key"]
    )

    inconsistent_keys = merged[
        ~merged["io_key_matches_country_sector"]
    ].copy()

    print(
        "io_key inconsistent with country + sector_code:",
        len(inconsistent_keys),
    )
    save(
        inconsistent_keys,
        args.output_dir / "inconsistent_country_sector_keys.csv",
    )

    # ------------------------------------------------------------------
    # 3. Country-level coverage and output totals.
    # ------------------------------------------------------------------
    country_summary = (
        merged.groupby("country", dropna=False)
        .agg(
            n_sectors=("io_key", "size"),
            n_output_records=("_merge", lambda x: (x == "both").sum()),
            n_missing_output=("output_usd_mn", lambda x: x.isna().sum()),
            n_zero_output=("output_usd_mn", lambda x: (x == 0).sum()),
            n_positive_output=("output_usd_mn", lambda x: (x > 0).sum()),
            n_negative_output=("output_usd_mn", lambda x: (x < 0).sum()),
            total_output_usd_mn=("output_usd_mn", "sum"),
        )
        .reset_index()
        .sort_values("country")
    )

    print("\n--- COUNTRY-LEVEL OUTPUT SUMMARY ---")
    print(
        country_summary[
            country_summary["country"].isin(["BRN", "CHL", "SGP"])
        ].to_string(index=False)
    )

    save(
        country_summary,
        args.output_dir / "country_output_summary.csv",
    )

    # ------------------------------------------------------------------
    # 4. Full sector-level output for BRN, CHL, and SGP.
    # ------------------------------------------------------------------
    focus = merged[merged["country"].isin(["BRN", "CHL", "SGP"])].copy()

    focus = focus.sort_values(
        ["country", "output_usd_mn"],
        ascending=[True, False],
        na_position="last",
    )

    print("\n--- ALL SECTORS: BRN, CHL, SGP ---")
    for country in ["BRN", "CHL", "SGP"]:
        subset = focus[focus["country"] == country]
        print(f"\n{country}: {len(subset)} mapped sectors")
        if subset.empty:
            print("  No sectors found.")
        else:
            print(
                subset[
                    ["io_key", "sector_code", "sector_label", "output_usd_mn"]
                ].to_string(index=False)
            )

    save(focus, args.output_dir / "focus_country_sector_output.csv")

    # ------------------------------------------------------------------
    # 5. Crosswalk-to-output diagnostics for the seven known cases.
    # ------------------------------------------------------------------
    cases = pd.DataFrame(
        TARGET_CASES,
        columns=["country", "gtap_activity", "aea_activity_code"],
    )

    targets = cases.merge(
        crosswalk[["aea_activity_code", "icio_sector_code"]].drop_duplicates(),
        on="aea_activity_code",
        how="left",
        validate="many_to_many",
    )

    targets = targets.merge(
        merged[
            [
                "country",
                "sector_code",
                "io_key",
                "sector_label",
                "output_usd_mn",
            ]
        ],
        left_on=["country", "icio_sector_code"],
        right_on=["country", "sector_code"],
        how="left",
        validate="many_to_many",
    )

    targets["target_status"] = np.select(
        [
            targets["io_key"].isna(),
            targets["output_usd_mn"].isna(),
            targets["output_usd_mn"].eq(0),
            targets["output_usd_mn"].gt(0),
            targets["output_usd_mn"].lt(0),
        ],
        [
            "CROSSWALK_TARGET_NOT_IN_SECTOR_MAP",
            "OUTPUT_MISSING",
            "ICIO_ZERO_OUTPUT",
            "ICIO_POSITIVE_OUTPUT",
            "ICIO_NEGATIVE_OUTPUT",
        ],
        default="OTHER",
    )

    print("\n--- TARGET CROSSWALK DIAGNOSTICS ---")
    print(
        targets[
            [
                "country",
                "gtap_activity",
                "aea_activity_code",
                "icio_sector_code",
                "io_key",
                "sector_label",
                "output_usd_mn",
                "target_status",
            ]
        ].to_string(index=False)
    )

    save(targets, args.output_dir / "target_crosswalk_diagnostics.csv")

    # ------------------------------------------------------------------
    # 6. Show the highest-output sectors in each focus country.
    # ------------------------------------------------------------------
    top_sectors = (
        focus[focus["output_usd_mn"].notna()]
        .sort_values(
            ["country", "output_usd_mn"],
            ascending=[True, False],
        )
        .groupby("country", group_keys=False)
        .head(15)
    )

    print("\n--- TOP 15 OUTPUT SECTORS BY COUNTRY ---")
    print(
        top_sectors[
            ["country", "sector_code", "sector_label", "output_usd_mn"]
        ].to_string(index=False)
    )

    save(top_sectors, args.output_dir / "top_output_sectors.csv")

    # ------------------------------------------------------------------
    # 7. Optional comparison with the original ICIO source extract.
    # ------------------------------------------------------------------
    if args.raw_icio is not None:
        if not args.raw_output_column:
            raise ValueError(
                "--raw-output-column is required when --raw-icio is supplied."
            )

        raw = read_csv(args.raw_icio, "Raw ICIO source")
        require_columns(
            raw,
            [args.raw_output_column],
            "Raw ICIO source",
        )

        raw["raw_output_value"] = pd.to_numeric(
            raw[args.raw_output_column], errors="coerce"
        )

        if args.raw_key_column:
            require_columns(raw, [args.raw_key_column], "Raw ICIO source")
            raw["io_key"] = normalize_code(raw[args.raw_key_column])

        elif args.raw_country_column and args.raw_sector_column:
            require_columns(
                raw,
                [args.raw_country_column, args.raw_sector_column],
                "Raw ICIO source",
            )
            raw["io_key"] = (
                normalize_code(raw[args.raw_country_column])
                + "_"
                + normalize_code(raw[args.raw_sector_column])
            )
        else:
            raise ValueError(
                "For raw ICIO comparison, supply either:\n"
                "  --raw-key-column COLUMN\n"
                "or both:\n"
                "  --raw-country-column COLUMN --raw-sector-column COLUMN"
            )

        raw_duplicates = raw[
            raw.duplicated("io_key", keep=False)
        ].sort_values("io_key")

        save(
            raw_duplicates,
            args.output_dir / "raw_source_duplicate_keys.csv",
        )

        if not raw_duplicates.empty:
            raise ValueError(
                "Raw source has duplicate io_key values. This may indicate "
                "multiple years, measures, or other dimensions. Filter the "
                "source to the intended observation before comparing."
            )

        raw_comparison = merged.merge(
            raw[["io_key", "raw_output_value"]],
            on="io_key",
            how="left",
            validate="one_to_one",
        )

        raw_comparison["constructed_minus_raw"] = (
            raw_comparison["output_usd_mn"]
            - raw_comparison["raw_output_value"]
        )

        raw_comparison["raw_status"] = np.select(
            [
                raw_comparison["raw_output_value"].isna(),
                raw_comparison["raw_output_value"].eq(0),
                raw_comparison["raw_output_value"].gt(0),
                raw_comparison["raw_output_value"].lt(0),
            ],
            [
                "RAW_RECORD_MISSING",
                "RAW_ZERO",
                "RAW_POSITIVE",
                "RAW_NEGATIVE",
            ],
            default="OTHER",
        )

        print("\n--- RAW SOURCE COMPARISON: BRN, CHL, SGP ---")
        print(
            raw_comparison[
                raw_comparison["country"].isin(["BRN", "CHL", "SGP"])
            ][
                [
                    "country",
                    "sector_code",
                    "io_key",
                    "output_usd_mn",
                    "raw_output_value",
                    "constructed_minus_raw",
                    "raw_status",
                ]
            ].to_string(index=False)
        )

        save(
            raw_comparison,
            args.output_dir / "raw_source_output_comparison.csv",
        )

    print("\n" + "=" * 90)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 90)
    print(f"Results saved to: {args.output_dir.resolve()}")
    print(
        "\nInterpretation: positive country totals with zero target sectors "
        "indicate a sector-specific issue, not necessarily a country-wide "
        "output construction failure. A raw-source comparison is needed "
        "to establish whether the zeros originate in the OECD source or "
        "in the construction of icio_output.csv."
    )


if __name__ == "__main__":
    main()
