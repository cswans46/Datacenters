
#!/usr/bin/env python
"""
Compare GTAP 12 benchmark commodity output with OECD ICIO gross output.

GTAP source:
    GSDFDAT.GDX

GTAP benchmark output proxy:
    VDFB  Domestic sales to firms
    VDPB  Domestic sales to private households
    VDGB  Domestic sales to government
    VDIB  Domestic sales to investment
    VFOB  Exports, excluding domestic diagonal flows

All domestic sales and exports are aggregated by commodity and region.

IMPORTANT:
    This is a benchmark value-based output proxy, not the solved GTAP
    quantity variable qo. Do not assume that its units are directly
    comparable to OECD ICIO USD millions without checking metadata.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gams.transfer as gt
import pandas as pd


# ---------------------------------------------------------------------
# Paths and cases
# ---------------------------------------------------------------------

DEFAULT_GDX = Path(r"GDX12_23\GSDFDAT.GDX")

CASES = {
    ("BRN", "MVH"): {
        "aea_activity": "C29",
        "icio_sectors": ["29"],
    },
    ("CHL", "MVH"): {
        "aea_activity": "C29",
        "icio_sectors": ["29"],
    },
    ("SGP", "OXT"): {
        "aea_activity": "B",
        "icio_sectors": ["05", "06", "07", "08", "09"],
    },
    ("SGP", "OIL"): {
        "aea_activity": "B",
        "icio_sectors": ["05", "06", "07", "08", "09"],
    },
    ("SGP", "COA"): {
        "aea_activity": "B",
        "icio_sectors": ["05", "06", "07", "08", "09"],
    },
    ("SGP", "GAS"): {
        "aea_activity": "B",
        "icio_sectors": ["05", "06", "07", "08", "09"],
    },
    ("SGP", "FRS"): {
        "aea_activity": "A02",
        "icio_sectors": ["02"],
    },
}

DOMESTIC_SYMBOLS = {
    "VDFB": ["COMM", "ACTS", "REG"],
    "VDPB": ["COMM", "REG"],
    "VDGB": ["COMM", "REG"],
    "VDIB": ["COMM", "REG"],
}

EXPORT_SYMBOL = "VFOB"


# ---------------------------------------------------------------------
# GDX helpers
# ---------------------------------------------------------------------

def read_gdx(path: Path, system_directory: str):
    if not path.exists():
        raise FileNotFoundError(f"GDX file not found: {path}")

    container = gt.Container(system_directory=system_directory)
    container.read(str(path))
    return container


def read_parameter(container, name, expected_dimensions):
    if name not in container:
        raise KeyError(f"Required GTAP symbol not found: {name}")

    symbol = container[name]

    if symbol.records is None:
        raise ValueError(f"GTAP symbol {name} has no records.")

    records = symbol.records.copy()

    expected = set(expected_dimensions)
    actual = set(records.columns) - {"value"}

    if actual != expected:
        raise ValueError(
            f"Unexpected dimensions for {name}.\n"
            f"Expected: {expected_dimensions}\n"
            f"Actual: {records.columns.tolist()}"
        )

    records["value"] = pd.to_numeric(
        records["value"], errors="coerce"
    )

    for column in expected_dimensions:
        records[column] = (
            records[column].astype(str).str.strip().str.upper()
        )

    return records.dropna(subset=["value"])


# ---------------------------------------------------------------------
# Build GTAP benchmark output proxy
# ---------------------------------------------------------------------

def build_gtap_output(container):
    print("\n" + "=" * 85)
    print("GTAP BENCHMARK OUTPUT COMPONENTS")
    print("=" * 85)

    components = []

    # Domestic sales to firms, private households, government,
    # and investment.
    for symbol_name, dimensions in DOMESTIC_SYMBOLS.items():

        records = read_parameter(
            container,
            symbol_name,
            dimensions,
        )

        print(
            f"{symbol_name}: {len(records):,} records; "
            f"dimensions={dimensions}"
        )

        # Sum domestic sales by commodity and region.
        component = (
            records.groupby(["COMM", "REG"])["value"]
            .sum()
            .rename(symbol_name)
        )

        components.append(component)

    # Exports: aggregate across foreign destinations.
    exports = read_parameter(
        container,
        EXPORT_SYMBOL,
        ["COMM", "SRC", "DST"],
    )

    # Exclude the domestic diagonal. VFOB is used here as an
    # international export-flow component.
    exports = exports.loc[
        exports["SRC"].ne(exports["DST"])
    ]

    print(
        f"VFOB: {len(exports):,} international export records"
    )

    export_component = (
        exports.groupby(["COMM", "SRC"])["value"]
        .sum()
        .rename("VFOB")
    )

    export_component.index.names = ["COMM", "REG"]
    components.append(export_component)

    # Combine the five components.
    output = pd.concat(components, axis=1).fillna(0).reset_index()

    component_columns = [
        "VDFB",
        "VDPB",
        "VDGB",
        "VDIB",
        "VFOB",
    ]

    output["gtap_output_proxy"] = output[
        component_columns
    ].sum(axis=1)

    # In this GTAP database, COMM and ACTS are expected to use
    # corresponding sector codes. Validate this rather than
    # silently assuming it.
    comm_set = set(
        container["COMM"].records["uni"]
        .astype(str).str.strip().str.upper()
    )

    acts_set = set(
        container["ACTS"].records["uni"]
        .astype(str).str.strip().str.upper()
    )

    print("\nCOMM / ACTS correspondence:")
    print(f"COMM entries: {len(comm_set)}")
    print(f"ACTS entries: {len(acts_set)}")
    print(f"COMM not in ACTS: {sorted(comm_set - acts_set)}")
    print(f"ACTS not in COMM: {sorted(acts_set - comm_set)}")

    if comm_set != acts_set:
        raise ValueError(
            "COMM and ACTS are not identical. "
            "A commodity-to-activity mapping must be established "
            "before interpreting the output proxy as activity output."
        )

    output = output.rename(columns={"COMM": "activity"})
    output["activity"] = output["activity"].astype(str).str.upper()
    output["country"] = output["REG"].astype(str).str.upper()

    return output[
        [
            "country",
            "activity",
            *component_columns,
            "gtap_output_proxy",
        ]
    ]


# ---------------------------------------------------------------------
# Read ICIO output
# ---------------------------------------------------------------------

def read_icio_output(output_path, sector_map_path):

    output = pd.read_csv(output_path)
    sector_map = pd.read_csv(sector_map_path, dtype=str)

    output["output_usd_mn"] = pd.to_numeric(
        output["output_usd_mn"], errors="coerce"
    )

    sector_map["country"] = (
        sector_map["country"].astype(str).str.strip().str.upper()
    )

    sector_map["sector_code"] = (
        sector_map["sector_code"].astype(str).str.strip()
    )

    return sector_map.merge(
        output[["io_key", "output_usd_mn"]],
        on="io_key",
        how="left",
    )


# ---------------------------------------------------------------------
# Compare GTAP and ICIO
# ---------------------------------------------------------------------

def compare_outputs(gtap, icio):

    summary_rows = []
    detail_rows = []

    for (country, activity), case in CASES.items():

        aea_activity = case["aea_activity"]
        target_sectors = case["icio_sectors"]

        gtap_match = gtap.loc[
            gtap["country"].eq(country)
            & gtap["activity"].eq(activity)
        ]

        if gtap_match.empty:
            gtap_value = float("nan")
            gtap_status = "GTAP_RECORD_MISSING"
            components = {
                column: float("nan")
                for column in [
                    "VDFB", "VDPB", "VDGB", "VDIB", "VFOB"
                ]
            }
        else:
            gtap_value = gtap_match["gtap_output_proxy"].sum()

            components = {
                column: gtap_match[column].sum()
                for column in [
                    "VDFB", "VDPB", "VDGB", "VDIB", "VFOB"
                ]
            }

            if gtap_value > 0:
                gtap_status = "GTAP_POSITIVE_OUTPUT_PROXY"
            elif gtap_value == 0:
                gtap_status = "GTAP_ZERO_OUTPUT_PROXY"
            else:
                gtap_status = "GTAP_NEGATIVE_OUTPUT_PROXY"

        icio_match = icio.loc[
            icio["country"].eq(country)
            & icio["sector_code"].isin(target_sectors)
        ].copy()

        for _, row in icio_match.iterrows():
            detail_rows.append({
                "country": country,
                "gtap_activity": activity,
                "aea_activity": aea_activity,
                "icio_sector": row["sector_code"],
                "io_key": row["io_key"],
                "icio_output_usd_mn": row["output_usd_mn"],
            })

        icio_value = icio_match["output_usd_mn"].sum(
            min_count=1
        )

        if pd.isna(icio_value):
            icio_status = "ICIO_RECORD_MISSING"
        elif icio_value > 0:
            icio_status = "ICIO_POSITIVE_OUTPUT"
        elif icio_value == 0:
            icio_status = "ICIO_ZERO_OUTPUT"
        else:
            icio_status = "ICIO_NEGATIVE_OUTPUT"

        if gtap_status == "GTAP_RECORD_MISSING":
            conclusion = "INVESTIGATE_GTAP_MAPPING_OR_RECORD"
        elif (
            gtap_status == "GTAP_POSITIVE_OUTPUT_PROXY"
            and icio_status == "ICIO_ZERO_OUTPUT"
        ):
            conclusion = "GTAP_POSITIVE_BUT_ICIO_ZERO"
        elif (
            gtap_status == "GTAP_ZERO_OUTPUT_PROXY"
            and icio_status == "ICIO_ZERO_OUTPUT"
        ):
            conclusion = "ZERO_OUTPUT_IN_BOTH"
        elif (
            gtap_status == "GTAP_POSITIVE_OUTPUT_PROXY"
            and icio_status == "ICIO_POSITIVE_OUTPUT"
        ):
            conclusion = "POSITIVE_OUTPUT_IN_BOTH"
        elif (
            gtap_status == "GTAP_ZERO_OUTPUT_PROXY"
            and icio_status == "ICIO_POSITIVE_OUTPUT"
        ):
            conclusion = "GTAP_ZERO_BUT_ICIO_POSITIVE"
        else:
            conclusion = "REVIEW"

        summary_rows.append({
            "country": country,
            "gtap_activity": activity,
            "aea_activity": aea_activity,
            "icio_target_sectors": ";".join(target_sectors),
            **components,
            "gtap_output_proxy": gtap_value,
            "gtap_status": gtap_status,
            "icio_output_usd_mn": icio_value,
            "icio_status": icio_status,
            "conclusion": conclusion,
        })

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gtap-gdx",
        type=Path,
        default=DEFAULT_GDX,
    )

    parser.add_argument(
        "--system-directory",
        default=r"C:\GAMS\51",
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
        "--output-directory",
        type=Path,
        default=Path("gtap_benchmark_output_diagnostics"),
    )

    args = parser.parse_args()

    args.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Reading GTAP benchmark database...")
    container = read_gdx(
        args.gtap_gdx,
        args.system_directory,
    )

    gtap = build_gtap_output(container)

    print("\nReading OECD ICIO output...")
    icio = read_icio_output(
        args.icio_output,
        args.sector_map,
    )

    summary, details = compare_outputs(gtap, icio)

    # Save outputs.
    gtap_path = (
        args.output_directory / "gtap_benchmark_output_proxy.csv"
    )

    summary_path = (
        args.output_directory / "gtap_icio_output_comparison.csv"
    )

    details_path = (
        args.output_directory / "gtap_icio_sector_details.csv"
    )

    gtap.to_csv(gtap_path, index=False)
    summary.to_csv(summary_path, index=False)
    details.to_csv(details_path, index=False)

    print("\n" + "=" * 100)
    print("GTAP BENCHMARK OUTPUT / ICIO COMPARISON")
    print("=" * 100)

    print(summary.to_string(index=False))

    print("\n" + "=" * 100)
    print("DETAILED ICIO SECTOR OUTPUT")
    print("=" * 100)

    print(details.to_string(index=False))

    print("\nSaved diagnostic files:")
    print(gtap_path)
    print(summary_path)
    print(details_path)

    print(
        "\nNOTE: GTAP benchmark values and ICIO USD millions "
        "are not assumed to be directly comparable."
    )


if __name__ == "__main__":
    main()