
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gams import GamsWorkspace

def get_symbol_map(database) -> dict:
    """Build a reliable lookup of symbols actually loaded from a GDX."""
    return {
        str(symbol.name).upper(): symbol
        for symbol in database
    }

# ---------------------------------------------------------------------
# GTAP 12 emissions diagnostic
#
# Inspects the CO2 and NCO2 GDX files without combining emissions.
#
# Outputs:
#   1. gtap12_symbol_metadata.csv
#   2. gtap12_domain_values.csv
#   3. gtap12_raw_records.csv
#   4. gtap12_emissions_by_gas_country.csv
#   5. gtap12_emissions_by_activity.csv
#   6. gtap12_co2_account_comparison.csv
#   7. gtap12_gwp_ar6.csv
#
# Units are retained as stored in the GDX files.
# Do not interpret the resulting sums as final GHG totals.
# ---------------------------------------------------------------------


GAMS_SYSTEM_DIRECTORY = r"C:\GAMS\51"

CO2_SYMBOLS = [
    "MDF",
    "MMF",
]

NCO2_SYMBOLS = [
    "EMI_IO",
    "EMI_ENDW",
    "EMI_QO",
    "EMI_IOP",
    "EMI_HH",
    "EMI_LU",
    "GWP",
]

DOMAIN_SETS = [
    "REG",
    "ACTS",
    "COMM",
    "ENDW",
    "EM",
    "EMN",
    "GHG",
    "NGHG",
    "AR",
    "FGAS",
    "LU_CAT",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def get_record_keys(record) -> tuple[str, ...]:
    """
    Retrieve the index keys from a GAMS record.

    Supports both parameter and set records.
    """
    if hasattr(record, "keys"):
        keys = record.keys

        if callable(keys):
            keys = keys()

        if isinstance(keys, str):
            return (keys,)

        return tuple(str(x) for x in keys)

    if hasattr(record, "key"):
        return (str(record.key),)

    raise AttributeError(
        f"Cannot retrieve keys from record type {type(record).__name__}"
    )


def get_symbol_domains(symbol) -> list[str]:
    """
    Return the declared domain names for a GAMS symbol.
    """
    result = []

    try:
        domains = symbol.domains
    except Exception:
        return result

    for domain in domains:
        if isinstance(domain, str):
            name = domain
        else:
            name = getattr(domain, "name", str(domain))

        result.append(str(name))

    return result


def get_symbol_text(symbol) -> str:
    try:
        return str(symbol.text)
    except Exception:
        return ""


def get_symbol_records(symbol, symbol_name: str) -> list[dict]:
    """
    Convert a GAMS symbol into a list of dictionaries.
    """
    rows = []

    dimension = int(symbol.dimension)

    for record in symbol:
        keys = get_record_keys(record)

        if len(keys) != dimension:
            raise ValueError(
                f"{symbol_name}: expected {dimension} keys, "
                f"received {len(keys)}: {keys}"
            )

        row = {
            f"key_{i + 1}": keys[i]
            for i in range(dimension)
        }

        if hasattr(record, "value"):
            row["value"] = float(record.value)
        else:
            row["value"] = 1.0

        rows.append(row)

    return rows


def load_symbol_records(database, symbol_name: str) -> pd.DataFrame:
    """
    Load one symbol from a GAMS database.
    """
    if symbol_name not in database:
        print(f"WARNING: Symbol not found: {symbol_name}")
        return pd.DataFrame()

    symbol = database[symbol_name]

    rows = get_symbol_records(symbol, symbol_name)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.insert(0, "symbol", symbol_name)

    return df


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Wrote: {path.name} ({len(df):,} rows)")


# ---------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------

def inspect_metadata(
    symbol_map: dict,
    symbols: list[str],
    database_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    metadata_rows = []
    domain_rows = []

    for symbol_name in symbols:

        symbol = symbol_map.get(symbol_name.upper())

        if symbol is None:
            metadata_rows.append({
                "database": database_name,
                "symbol": symbol_name,
                "exists": False,
                "symbol_type": None,
                "dimension": None,
                "description": None,
                "domains": None,
            })
            continue

        domains = get_symbol_domains(symbol)

        metadata_rows.append({
            "database": database_name,
            "symbol": symbol_name,
            "exists": True,
            "symbol_type": type(symbol).__name__,
            "dimension": int(symbol.dimension),
            "description": get_symbol_text(symbol),
            "domains": "|".join(domains),
        })

        for position, domain_name in enumerate(domains, start=1):
            domain_rows.append({
                "database": database_name,
                "symbol": symbol_name,
                "dimension_position": position,
                "domain": domain_name,
            })

    return pd.DataFrame(metadata_rows), pd.DataFrame(domain_rows)


def inspect_domain_sets(database, database_name: str) -> pd.DataFrame:

    rows = []

    for set_name in DOMAIN_SETS:

        if set_name not in database:
            continue

        symbol = database[set_name]

        for record in symbol:

            keys = get_record_keys(record)

            rows.append({
                "database": database_name,
                "set_name": set_name,
                "key": "|".join(keys),
                "description": get_symbol_text(symbol),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Standardized raw record extraction
# ---------------------------------------------------------------------

def add_database_records(
    symbol_map: dict,
    database_name: str,
    symbol_names: list[str],
) -> pd.DataFrame:

    frames = []

    for symbol_name in symbol_names:

        symbol = symbol_map.get(symbol_name.upper())

        if symbol is None:
            print(f"WARNING: Symbol not found: {symbol_name}")
            continue

        rows = get_symbol_records(symbol, symbol_name)

        if not rows:
            continue

        df = pd.DataFrame(rows)
        df.insert(0, "symbol", symbol_name)
        df.insert(1, "database", database_name)

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True, sort=False)


# ---------------------------------------------------------------------
# Emissions summaries
# ---------------------------------------------------------------------

def summarize_emissions(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

    gas_country_rows = []
    activity_rows = []

    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    for _, row in raw.iterrows():

        symbol = row["symbol"]
        database = row["database"]
        value = row["value"]

        keys = [
            row.get(f"key_{i}")
            for i in range(1, 6)
            if pd.notna(row.get(f"key_{i}"))
        ]

        gas = None
        region = None
        activity = None
        commodity = None

        # MDF/MMF:
        # fuel | activity | region
        if symbol in {"MDF", "MMF"} and len(keys) == 3:
            activity = keys[1]
            region = keys[2]
            gas = "CO2"

        # EMI_IO:
        # gas | fuel | activity | region
        elif symbol == "EMI_IO" and len(keys) == 4:
            gas = keys[0]
            activity = keys[2]
            region = keys[3]

        # EMI_ENDW:
        # gas | endowment | activity | region
        elif symbol == "EMI_ENDW" and len(keys) == 4:
            gas = keys[0]
            activity = keys[2]
            region = keys[3]

        # EMI_QO:
        # gas | commodity | region
        elif symbol == "EMI_QO" and len(keys) == 3:
            gas = keys[0]
            commodity = keys[1]
            region = keys[2]

        # EMI_IOP:
        # gas | product | activity | region
        elif symbol == "EMI_IOP" and len(keys) == 4:
            gas = keys[0]
            activity = keys[2]
            region = keys[3]

        # EMI_HH:
        # Preserve generic keys; household account is excluded
        # from the production extension.
        elif symbol == "EMI_HH":
            continue

        # EMI_LU:
        # Preserve generic keys; land-use account is separate.
        elif symbol == "EMI_LU":
            continue

        else:
            continue

        gas_country_rows.append({
            "database": database,
            "symbol": symbol,
            "gas": gas,
            "region": region,
            "value": value,
        })

        activity_rows.append({
            "database": database,
            "symbol": symbol,
            "gas": gas,
            "region": region,
            "activity": activity,
            "commodity": commodity,
            "value": value,
        })

    gas_country = pd.DataFrame(gas_country_rows)
    activity = pd.DataFrame(activity_rows)

    if not gas_country.empty:
        gas_country = (
            gas_country
            .groupby(
                ["database", "symbol", "gas", "region"],
                dropna=False,
                as_index=False,
            )["value"]
            .sum()
        )

    if not activity.empty:
        activity = (
            activity
            .groupby(
                [
                    "database",
                    "symbol",
                    "gas",
                    "region",
                    "activity",
                    "commodity",
                ],
                dropna=False,
                as_index=False,
            )["value"]
            .sum()
        )

    return gas_country, activity


# ---------------------------------------------------------------------
# CO2 account comparison
# ---------------------------------------------------------------------

def build_co2_comparison(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Extract CO2 records from each relevant account.

    This is a diagnostic comparison only.
    It does not imply that the accounts should be summed.
    """

    rows = []

    if raw.empty:
        return pd.DataFrame()

    for _, row in raw.iterrows():

        symbol = row["symbol"]

        keys = [
            row.get(f"key_{i}")
            for i in range(1, 6)
            if pd.notna(row.get(f"key_{i}"))
        ]

        gas = None
        region = None
        activity = None
        commodity = None

        if symbol in {"MDF", "MMF"} and len(keys) == 3:
            gas = "CO2"
            activity = keys[1]
            region = keys[2]

        elif symbol in {"EMI_IO", "EMI_ENDW", "EMI_IOP"} and len(keys) == 4:
            gas = keys[0]
            activity = keys[2]
            region = keys[3]

        elif symbol == "EMI_QO" and len(keys) == 3:
            gas = keys[0]
            commodity = keys[1]
            region = keys[2]

        if gas != "CO2":
            continue

        rows.append({
            "symbol": symbol,
            "region": region,
            "activity": activity,
            "commodity": commodity,
            "value": row["value"],
        })

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    return (
        result
        .groupby(
            ["symbol", "region", "activity", "commodity"],
            dropna=False,
            as_index=False,
        )["value"]
        .sum()
    )


# ---------------------------------------------------------------------
# GWP AR6 extraction
# ---------------------------------------------------------------------

def extract_gwp_ar6(raw: pd.DataFrame) -> pd.DataFrame:

    if raw.empty:
        return pd.DataFrame()

    gwp = raw.loc[raw["symbol"] == "GWP"].copy()

    if gwp.empty:
        return gwp

    # Expected dimensions:
    # gas | region | assessment report

    gwp["gas"] = gwp["key_1"]
    gwp["region"] = gwp["key_2"]
    gwp["assessment_report"] = gwp["key_3"]

    gwp = gwp.loc[
        gwp["assessment_report"].astype(str).str.upper().eq("AR6")
    ].copy()

    return gwp[
        ["gas", "region", "assessment_report", "value"]
    ].rename(columns={"value": "gwp"})


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Diagnose GTAP 12 emissions GDX files."
    )

    parser.add_argument(
        "--co2",
        required=True,
        help="Path to GSDFEMISS.GDX",
    )

    parser.add_argument(
        "--nco2",
        required=True,
        help="Path to GSDFNCO2.GDX",
    )

    parser.add_argument(
        "--output-dir",
        default="gtap12_diagnostics",
        help="Output directory",
    )

    args = parser.parse_args()

    co2_path = Path(args.co2).resolve()
    nco2_path = Path(args.nco2).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not co2_path.exists():
        raise FileNotFoundError(co2_path)

    if not nco2_path.exists():
        raise FileNotFoundError(nco2_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("GTAP 12 EMISSIONS DIAGNOSTIC")
    print("=" * 75)

    print(f"CO2 GDX:  {co2_path}")
    print(f"NCO2 GDX: {nco2_path}")
    print(f"Output:   {output_dir}")

    workspace = GamsWorkspace(
        system_directory=GAMS_SYSTEM_DIRECTORY
    )

    co2_db = workspace.add_database_from_gdx(str(co2_path))
    nco2_db = workspace.add_database_from_gdx(str(nco2_path))

    # Explicitly inventory the loaded symbols.
    co2_symbols = get_symbol_map(co2_db)
    nco2_symbols = get_symbol_map(nco2_db)

    print("\nCO2 symbols loaded:")
    print(", ".join(sorted(co2_symbols.keys())))

    print("\nNCO2 symbols loaded:")
    print(", ".join(sorted(nco2_symbols.keys())))

    # -------------------------------------------------------------
    # 1. Metadata
    # -------------------------------------------------------------

    co2_metadata, co2_domains = inspect_metadata(
        co2_symbols,
        CO2_SYMBOLS,
        "CO2",
    )

    nco2_metadata, nco2_domains = inspect_metadata(
        nco2_symbols,
        NCO2_SYMBOLS,
        "NCO2",
    )

    metadata = pd.concat(
        [co2_metadata, nco2_metadata],
        ignore_index=True,
    )

    domains = pd.concat(
        [co2_domains, nco2_domains],
        ignore_index=True,
    )

    write_csv(
        metadata,
        output_dir / "gtap12_symbol_metadata.csv",
    )

    write_csv(
        domains,
        output_dir / "gtap12_symbol_domains.csv",
    )

    # -------------------------------------------------------------
    # 2. Domain set values
    # -------------------------------------------------------------

    co2_sets = inspect_domain_sets(co2_symbols, "CO2")
    nco2_sets = inspect_domain_sets(nco2_symbols, "NCO2")

    domain_values = pd.concat(
        [co2_sets, nco2_sets],
        ignore_index=True,
    )

    write_csv(
        domain_values,
        output_dir / "gtap12_domain_values.csv",
    )

    # -------------------------------------------------------------
    # 3. Raw emissions records
    # -------------------------------------------------------------

    co2_raw = add_database_records(
        co2_symbols,
        "CO2",
        CO2_SYMBOLS,
    )

    nco2_raw = add_database_records(
        nco2_symbols,
        "NCO2",
        NCO2_SYMBOLS,
    )

    raw = pd.concat(
        [co2_raw, nco2_raw],
        ignore_index=True,
        sort=False,
    )

    write_csv(
        raw,
        output_dir / "gtap12_raw_records.csv",
    )

    # -------------------------------------------------------------
    # 4. Emissions by gas and country
    # -------------------------------------------------------------

    gas_country, activity = summarize_emissions(raw)

    write_csv(
        gas_country,
        output_dir / "gtap12_emissions_by_gas_country.csv",
    )

    write_csv(
        activity,
        output_dir / "gtap12_emissions_by_activity.csv",
    )

    # -------------------------------------------------------------
    # 5. CO2 comparison
    # -------------------------------------------------------------

    co2_comparison = build_co2_comparison(raw)

    write_csv(
        co2_comparison,
        output_dir / "gtap12_co2_account_comparison.csv",
    )

    # -------------------------------------------------------------
    # 6. GWP AR6
    # -------------------------------------------------------------

    gwp_ar6 = extract_gwp_ar6(raw)

    write_csv(
        gwp_ar6,
        output_dir / "gtap12_gwp_ar6.csv",
    )

    # -------------------------------------------------------------
    # 7. Console summary
    # -------------------------------------------------------------

    print("\n" + "=" * 75)
    print("SYMBOL METADATA")
    print("=" * 75)

    print(
        metadata[
            [
                "database",
                "symbol",
                "exists",
                "dimension",
                "domains",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 75)
    print("RAW RECORD COUNTS")
    print("=" * 75)

    if not raw.empty:
        print(
            raw.groupby(
                ["database", "symbol"],
                dropna=False,
            )
            .size()
            .rename("records")
            .to_string()
        )

    print("\n" + "=" * 75)
    print("EMISSIONS BY SYMBOL AND GAS")
    print("=" * 75)

    if not gas_country.empty:
        print(
            gas_country.groupby(
                ["symbol", "gas"],
                dropna=False,
            )["value"]
            .sum()
            .to_string()
        )

    print("\n" + "=" * 75)
    print("CANADA: CO2 ACCOUNT COMPARISON")
    print("=" * 75)

    if not co2_comparison.empty:
        canada = co2_comparison.loc[
            co2_comparison["region"].astype(str).eq("CAN")
        ]

        print(
            canada.groupby("symbol")["value"]
            .sum()
            .to_string()
        )

    print("\n" + "=" * 75)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 75)

    print(f"\nOutput files are in:\n{output_dir}")


if __name__ == "__main__":
    main()
