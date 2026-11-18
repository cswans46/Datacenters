
#!/usr/bin/env python3
"""
Global data-centre expansion footprint model using OECD ICIO and AEA data.

Primary scenario
----------------
A data-centre gross-output path is supplied in dc_output_scenarios.csv.
The model calculates annual and cumulative changes relative to the first
scenario year and uses these as target gross-output shocks in the relevant
ICIO data-centre proxy sectors.

Electricity cross-check
-----------------------
An independent estimate of data-centre output changes is calculated from
facility electricity consumption, electricity prices, and the ICIO
electricity-input coefficient:

    delta_output_electricity =
        (delta_facility_GWh * price_2022_USD_per_MWh / 1000)
        / electricity_input_coefficient

This is a comparison estimate. It is NOT added to the direct output shock.

Emissions
---------
Direct emissions intensities are calculated from the OECD Air Emissions
Accounts (AEA), mapped to the ICIO classification, and divided by 2022
ICIO gross output.

Leontief calculation
--------------------
The supplied gross-output shock is converted to a final-demand shock so
that the resulting change in the targeted proxy-sector gross output equals
the specified target.

Attribution
-----------
All-expansion, US-customer, and US-parent scenarios are alternative
attribution lenses. They must not be added together.

The model does not identify individual companies in ICIO.

Important assumptions
---------------------
- ICIO A, X, FD and AEA emissions refer to the same reference year.
- Monetary values used with ICIO coefficients are in constant 2022 USD.
- The ICIO production structure and emissions intensities are held fixed
  over the scenario horizon.
- J62_63 (or another configured sector) is an aggregate proxy, not a
  separately identified data-centre industry.
- The AEA crosswalk must not double-count overlapping aggregate and
  detailed activity observations.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import pyreadr
from scipy.linalg import lu_factor, lu_solve


LOG = logging.getLogger("datacentre_icio")

REQUIRED_COUNTRIES = {"CAN", "IND"}
ATTRIBUTIONS = (
    "all_expansion",
    "us_customer_demand",
    "us_parent_owned_operator",
)

SCRIPT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------

def require_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    name: str,
) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def check_unique(
    df: pd.DataFrame,
    keys: list[str],
    name: str,
) -> None:
    duplicate = df.duplicated(keys, keep=False)
    if duplicate.any():
        examples = df.loc[duplicate, keys].head(10).to_dict("records")
        raise ValueError(
            f"{name} contains duplicate observations for {keys}. "
            f"Examples: {examples}"
        )


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    raise ValueError(
        f"Unsupported file type: {path}. Use CSV or Parquet."
    )


def numeric_column(
    df: pd.DataFrame,
    column: str,
    name: str,
) -> pd.Series:
    result = pd.to_numeric(df[column], errors="raise")

    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError(f"{name}.{column} contains non-finite values.")

    return result


def validate_shares(
    df: Optional[pd.DataFrame],
    column: str,
    name: str,
) -> None:
    if df is None:
        return

    values = numeric_column(df, column, name)

    if ((values < 0) | (values > 1)).any():
        raise ValueError(f"{name}.{column} must lie between zero and one.")


# ---------------------------------------------------------------------
# OECD RData loading
# ---------------------------------------------------------------------


def load_r_object(path, object_name=None, description="R object", icio_year=2022):
    """Load an OECD ICIO RData object and select the requested year."""

    import xarray as xr

    result = pyreadr.read_r(path)

    # Identify the object
    if object_name is None:
        candidates = [
            (name, obj)
            for name, obj in result.items()
            if isinstance(obj, (pd.DataFrame, xr.DataArray))
        ]

        if len(candidates) != 1:
            raise ValueError(
                f"{description}: expected one DataFrame or DataArray in "
                f"'{path}', found {[name for name, _ in candidates]}."
            )

        object_name, obj = candidates[0]

    else:
        if object_name not in result:
            raise KeyError(
                f"{description}: object '{object_name}' not found. "
                f"Available objects: {list(result.keys())}"
            )

        obj = result[object_name]

    # Process xarray objects
    if isinstance(obj, xr.DataArray):

        # Identify and select the requested year
        year_dims = []

        for dim in obj.dims:
            if dim in obj.coords:
                values = [str(v) for v in obj.coords[dim].values]

                if str(icio_year) in values:
                    year_dims.append(dim)

        if year_dims:
            if len(year_dims) > 1:
                raise ValueError(
                    f"{description}: multiple possible year dimensions: "
                    f"{year_dims}"
                )

            year_dim = year_dims[0]
            obj = obj.sel({year_dim: str(icio_year)})

        elif obj.ndim == 3:
            raise ValueError(
                f"{description}: cannot identify year dimension for "
                f"{icio_year}. Dimensions: {obj.dims}"
            )

        # Convert to pandas
        if obj.ndim == 2:
            obj = obj.to_pandas()

        elif obj.ndim == 1:
            obj = obj.to_series().to_frame()

        else:
            raise ValueError(
                f"{description}: unsupported dimensions after year "
                f"selection: {obj.dims}, shape={obj.shape}"
            )

    elif not isinstance(obj, pd.DataFrame):
        raise TypeError(
            f"{description}: unsupported object type {type(obj)}."
        )

    return obj



def normalize_io_matrix(
    raw: pd.DataFrame,
    name: str,
) -> pd.DataFrame:
    """
    Normalize a square ICIO matrix.

    Assumes the RData object contains industry labels as its pandas index
    and column names. If the index was exported as a regular column named
    io_key, that column is restored as the index.
    """
    df = raw.copy()

    if "io_key" in df.columns:
        df = df.set_index("io_key")

    df.index = df.index.astype(str).str.strip()
    df.columns = df.columns.astype(str).str.strip()

    if df.index.has_duplicates or df.columns.has_duplicates:
        raise ValueError(f"{name} contains duplicate industry labels.")

    if set(df.index) != set(df.columns):
        raise ValueError(
            f"{name} is not square with matching row and column labels. "
            f"Rows={len(df.index)}, columns={len(df.columns)}."
        )

    df = df.loc[df.index, df.index]
    df = df.apply(pd.to_numeric, errors="raise").astype(float)

    if not np.isfinite(df.to_numpy()).all():
        raise ValueError(f"{name} contains NaN or infinite values.")

    return df



def normalize_io_vector(
    df: pd.DataFrame,
    *,
    name: str,
    keys: list[str] | pd.Index | None = None,
    value_column: str | None = None,
    key_column: str | None = None,
    reference_year: int | str | None = None,
) -> pd.Series:
    """
    Normalize an OECD ICIO output/final-demand vector to a Series indexed
    by ICIO industry keys.

    Supported formats:
      1. Wide: one row, with IO industry codes as columns.
      2. Wide: one column, with IO industry codes as the index.
      3. Long: key_column and value_column.
      4. Long: two columns, one identifiable key column and one value column.
    """
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    df.index = df.index.astype(str).str.strip()

    expected_keys = set(map(str, keys)) if keys is not None else None
    
    
    # ---------------------------------------------------------
    # Case 0: OECD ICIO annual wide format.
    # Rows = years; columns = ICIO industry codes.
    # ---------------------------------------------------------
    if expected_keys is not None:
        matching_columns = [
            col for col in df.columns
            if col in expected_keys
        ]

        if len(matching_columns) == len(expected_keys):
            if reference_year is None:
                raise ValueError(
                    f"{name} contains annual observations, but no "
                    "reference_year was supplied."
                )

            year = str(reference_year)

            if year not in df.index:
                raise ValueError(
                    f"{name}: reference year {year} not found. "
                    f"Available years: {df.index.tolist()[:30]}"
                )

            result = df.loc[year, matching_columns].copy()
            result.index = result.index.astype(str)

            result = pd.to_numeric(result, errors="coerce")

            if result.isna().any():
                bad = result[result.isna()].index.tolist()[:10]
                raise ValueError(
                    f"{name} contains missing or non-numeric values "
                    f"for {year}: {bad}"
                )

            return result.reindex(keys).astype(float)


    # ---------------------------------------------------------
    # Case 1: Wide format, industry keys in columns.
    # ---------------------------------------------------------
    if expected_keys is not None:
        matching_columns = [
            col for col in df.columns
            if col in expected_keys
        ]

        if len(matching_columns) > 0:
            wide = df[matching_columns]

            if len(wide) != 1:
                raise ValueError(
                    f"{name} has {len(wide)} rows and ICIO keys in columns. "
                    "Expected a single-row vector."
                )

            result = wide.iloc[0]
            result.index = result.index.astype(str)
            result = pd.to_numeric(result, errors="coerce")

            if result.isna().any():
                bad = result[result.isna()].index.tolist()[:10]
                raise ValueError(
                    f"{name} contains non-numeric or missing values for: {bad}"
                )

            return result.reindex(keys).astype(float)

    # ---------------------------------------------------------
    # Case 2: Wide format, industry keys in the index.
    # ---------------------------------------------------------
    if expected_keys is not None:
        matching_index = [
            idx for idx in df.index
            if idx in expected_keys
        ]

        if len(matching_index) > 0 and df.shape[1] == 1:
            result = df.iloc[:, 0].copy()
            result.index = result.index.astype(str)
            result = pd.to_numeric(result, errors="coerce")

            if result.isna().any():
                raise ValueError(f"{name} contains missing or non-numeric values.")

            return result.reindex(keys).astype(float)

    # ---------------------------------------------------------
    # Case 3: Explicit long-format key/value columns.
    # ---------------------------------------------------------
    if key_column is not None and value_column is not None:
        if key_column not in df.columns or value_column not in df.columns:
            raise ValueError(
                f"{name} does not contain requested columns "
                f"{key_column!r} and {value_column!r}."
            )

        result = df.set_index(key_column)[value_column]
        result.index = result.index.astype(str)
        result = pd.to_numeric(result, errors="coerce")

        if result.index.has_duplicates:
            raise ValueError(f"{name} has duplicate IO keys.")

        if result.isna().any():
            raise ValueError(f"{name} contains missing or non-numeric values.")

        return result.reindex(keys).astype(float) if keys is not None else result

    # ---------------------------------------------------------
    # Case 4: One-column vector with industry keys in the index.
    # ---------------------------------------------------------
    if df.shape[1] == 1:
        result = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        result.index = df.index.astype(str)

        if result.isna().any():
            raise ValueError(f"{name} contains missing or non-numeric values.")

        return result.reindex(keys).astype(float) if keys is not None else result

    # ---------------------------------------------------------
    # Case 5: One-row vector with industry keys in columns.
    # ---------------------------------------------------------
    if df.shape[0] == 1:
        result = pd.to_numeric(df.iloc[0], errors="coerce")
        result.index = df.columns.astype(str)

        if result.isna().any():
            raise ValueError(f"{name} contains missing or non-numeric values.")

        return result.reindex(keys).astype(float) if keys is not None else result

    raise ValueError(
        f"Cannot normalize {name}. Shape={df.shape}; "
        f"columns={df.columns.tolist()[:20]}; "
        f"index={df.index.tolist()[:20]}"
    )



def normalize_final_demand(
    raw: pd.DataFrame,
    industry_keys: pd.Index,
) -> pd.DataFrame:
    """
    Normalize FD to industries in rows and final-demand categories in columns.

    The function accepts either orientation, provided one dimension matches
    the ICIO industry labels.
    """
    fd = raw.copy()

    if "io_key" in fd.columns:
        fd = fd.set_index("io_key")

    fd.index = fd.index.astype(str).str.strip()
    fd.columns = fd.columns.astype(str).str.strip()

    row_match = set(industry_keys).issubset(set(fd.index))
    col_match = set(industry_keys).issubset(set(fd.columns))

    if row_match and not col_match:
        result = fd.loc[industry_keys, :]

    elif col_match and not row_match:
        result = fd.loc[:, industry_keys].T

    elif row_match and col_match:
        raise ValueError(
            "FD has industry labels on both axes. Its orientation is "
            "ambiguous; inspect the RData object."
        )

    else:
        raise ValueError(
            "Could not identify the industry dimension of FD. "
            "Its labels do not match the A matrix."
        )

    result = result.apply(pd.to_numeric, errors="raise").astype(float)

    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("FD contains non-finite values.")

    return result



def load_icio(args):
    """Load A, X and FD and align them to a common industry ordering."""

    # ------------------------------------------------------------------
    # 1. Load raw ICIO objects
    # ------------------------------------------------------------------
    raw_A = load_r_object(
        args.A_file, args.A_object, "ICIO A", args.icio_year
    )

    raw_X = load_r_object(
        args.X_file, args.X_object, "ICIO X", args.icio_year
    )

    raw_FD = load_r_object(
        args.FD_file, args.FD_object, "ICIO FD", args.icio_year
    )

    # ------------------------------------------------------------------
    # 2. Inspect raw object structures
    # ------------------------------------------------------------------
    print(f"ICIO A dimensions: {raw_A.shape}")
    print(f"ICIO A index sample: {raw_A.index[:5].tolist()}")
    print(f"ICIO A columns sample: {raw_A.columns[:5].tolist()}")

    print(f"ICIO X dimensions: {raw_X.shape}")
    print(f"ICIO X index sample: {raw_X.index[:5].tolist()}")
    print(f"ICIO X columns sample: {raw_X.columns[:5].tolist()}")

    print(f"ICIO FD dimensions: {raw_FD.shape}")
    print(f"ICIO FD index sample: {raw_FD.index[:5].tolist()}")
    print(f"ICIO FD columns sample: {raw_FD.columns[:5].tolist()}")

    # ------------------------------------------------------------------
    # 3. Normalize A first
    # ------------------------------------------------------------------
    A = normalize_io_matrix(raw_A, "A")

    if A.shape[0] != A.shape[1]:
        raise ValueError(
            f"ICIO A must be square. Found dimensions {A.shape}."
        )

    if set(A.index) != set(A.columns):
        raise ValueError(
            "ICIO A row and column industry labels do not match."
        )

    # ------------------------------------------------------------------
    # 4. Normalize X
    # ------------------------------------------------------------------
    X = normalize_io_vector(
        raw_X,
        name="X",
        keys=A.index,
        value_column=args.X_value_column,
        reference_year=args.icio_year,
    )

    print("\nNormalized ICIO X:")
    print(f"X dimensions: {X.shape}")
    print(f"X index sample: {X.index[:5].tolist()}")

    # Check industry coverage.
    missing_X = sorted(set(A.index) - set(X.index))
    extra_X = sorted(set(X.index) - set(A.index))

    if missing_X or extra_X:
        raise ValueError(
            "X industry labels do not match A. "
            f"Missing in X={missing_X[:10]}; "
            f"extra in X={extra_X[:10]}"
        )

    X = X.loc[A.index]

    # ------------------------------------------------------------------
    # 5. Normalize final demand
    # ------------------------------------------------------------------
    FD = normalize_final_demand(raw_FD, A.index)

    missing_FD = sorted(set(A.index) - set(FD.index))
    extra_FD = sorted(set(FD.index) - set(A.index))

    if missing_FD or extra_FD:
        raise ValueError(
            "FD industry labels do not match A. "
            f"Missing in FD={missing_FD[:10]}; "
            f"extra in FD={extra_FD[:10]}"
        )

    FD = FD.loc[A.index]

    # ------------------------------------------------------------------
    # 6. Validate numerical values
    # ------------------------------------------------------------------
    if not np.isfinite(A.to_numpy()).all():
        raise ValueError("ICIO A contains non-finite values.")

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("ICIO X contains non-finite values.")

    if not np.isfinite(FD.to_numpy()).all():
        raise ValueError("ICIO FD contains non-finite values.")
    
    
    # Diagnose zero and negative gross-output observations.
    negative_X = X[X < 0]
    zero_X = X[X == 0]

    print("\nICIO gross-output diagnostics:")
    print(f"Minimum X:             {X.min():,.10f}")
    print(f"Maximum X:             {X.max():,.4f}")
    print(f"Negative observations: {len(negative_X)}")
    print(f"Zero observations:     {len(zero_X)}")

    if len(negative_X):
        print("\nMost negative gross-output observations:")
        print(negative_X.sort_values().head(20).to_string())

    if len(zero_X):
        print("\nZero gross-output observations:")
        print(zero_X.head(20).to_string())

    if len(negative_X):
        raise ValueError(
            "ICIO gross output contains negative values. "
            "Inspect the observations printed above."
        )
        
    if (A.to_numpy() < -1e-12).any():
        raise ValueError(
            "A contains negative coefficients. Inspect the ICIO matrix "
            "and confirm that it is the intended technical-coefficient matrix."
        )

    # ------------------------------------------------------------------
    # 7. Check base-year accounting identity
    # ------------------------------------------------------------------
    # X = A @ X + total final demand.
    intermediate_demand = A.to_numpy() @ X.to_numpy()
    final_demand = FD.sum(axis=1).to_numpy()

    reconstructed_X = intermediate_demand + final_demand

    relative_error = (
        np.abs(reconstructed_X - X.to_numpy())
        / np.maximum(np.abs(X.to_numpy()), 1.0)
    )

    max_error = float(np.nanmax(relative_error))

    LOG.info(
        "Maximum ICIO accounting identity relative error: %.6g",
        max_error,
    )

    if max_error > args.icio_identity_tolerance:
        LOG.warning(
            "ICIO accounting identity does not hold within tolerance %.3g. "
            "Check FD aggregation, valuation adjustments, taxes, margins, "
            "and the selected RData objects.",
            args.icio_identity_tolerance,
        )

    # ------------------------------------------------------------------
    # 8. Return aligned ICIO objects
    # ------------------------------------------------------------------
    LOG.info(
        "ICIO matrices successfully loaded for %s: A=%s, X=%s, FD=%s",
        args.icio_year,
        A.shape,
        X.shape,
        FD.shape,
    )

    return A, X, FD



# ---------------------------------------------------------------------
# Sector maps and input validation
# ---------------------------------------------------------------------


def load_sector_maps(A: pd.DataFrame):
    """
    Construct the ICIO sector map directly from the A matrix labels.

    OECD ICIO industry labels follow the convention:
        COUNTRY_SECTOR

    Examples:
        CAN_01
        CAN_D35
        CAN_J62_63

    The data-centre proxy and electricity supplier mappings remain
    user-specified in dc_sector_map.csv.
    """

    # ---------------------------------------------------------
    # 1. Automatically construct the country-sector mapping
    # ---------------------------------------------------------
    records = []

    for io_key in A.index.astype(str):
        if "_" not in io_key:
            raise ValueError(
                f"Cannot parse ICIO industry label: {io_key}"
            )

        country, sector_code = io_key.split("_", 1)

        records.append({
            "io_key": io_key,
            "country": country,
            "sector_code": sector_code,
            "sector_label": sector_code,
            # OECD ICIO D35 corresponds to electricity, gas,
            # steam and air conditioning supply.
            "is_electricity_sector": sector_code == "D35",
        })

    sector_map = pd.DataFrame(records)

    check_unique(sector_map, ["io_key"], "generated sector map")

    # Confirm that the generated mapping covers the entire A matrix.
    if set(sector_map["io_key"]) != set(A.index):
        raise ValueError(
            "Generated sector map does not match the ICIO A matrix."
        )

    LOG.info(
        "Generated sector map from ICIO labels: %s industries, %s countries, %s sectors.",
        len(sector_map),
        sector_map["country"].nunique(),
        sector_map["sector_code"].nunique(),
    )

    # ---------------------------------------------------------
    # 2. Load the data-centre-specific mapping
    # ---------------------------------------------------------
    dc_map = read_table(SCRIPT_DIR / "dc_sector_map.csv")

    require_columns(
        dc_map,
        ["country", "dc_proxy_io_key", "electricity_io_key"],
        "dc_sector_map.csv",
    )

    dc_map["country"] = (
        dc_map["country"].astype(str).str.strip()
    )

    dc_map["dc_proxy_io_key"] = (
        dc_map["dc_proxy_io_key"].astype(str).str.strip()
    )

    dc_map["electricity_io_key"] = (
        dc_map["electricity_io_key"].astype(str).str.strip()
    )

    check_unique(dc_map, ["country"], "dc_sector_map.csv")

    if not REQUIRED_COUNTRIES.issubset(set(dc_map["country"])):
        raise ValueError(
            f"dc_sector_map.csv must include "
            f"{sorted(REQUIRED_COUNTRIES)}."
        )

    for row in dc_map.itertuples(index=False):

        if row.dc_proxy_io_key not in A.columns:
            raise ValueError(
                f"Unknown data-centre proxy for {row.country}: "
                f"{row.dc_proxy_io_key}"
            )

        if row.electricity_io_key not in A.index:
            raise ValueError(
                f"Unknown electricity industry for {row.country}: "
                f"{row.electricity_io_key}"
            )

    LOG.info(
        "Loaded data-centre mappings for: %s",
        ", ".join(dc_map["country"].tolist()),
    )

    return sector_map, dc_map

def load_scenario_inputs():
    """Read and validate data-centre output, electricity, price, grid,
    water and attribution data.

    Rows with missing required observations are dropped and logged.
    Invalid non-missing values and duplicate country-year observations
    raise errors.
    """

    def clean_keys(df, name):
        """Normalize country/year keys and remove rows with missing keys."""
        df = df.copy()

        df["country"] = df["country"].astype("string").str.strip().str.upper()
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

        missing = (
            df["country"].isna()
            | df["country"].eq("")
            | df["year"].isna()
        )

        if missing.any():
            LOG.warning(
                "Dropping %s rows with missing country or year from %s: %s",
                missing.sum(),
                name,
                df.loc[missing, ["country", "year"]].to_dict("records"),
            )
            df = df.loc[~missing].copy()

        df["year"] = df["year"].astype(int)

        return df

    def clean_numeric_observations(df, columns, name):
        """Convert required numeric columns and drop incomplete observations."""
        df = df.copy()

        for column in columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        missing = df[columns].isna().any(axis=1)

        if missing.any():
            LOG.warning(
                "Dropping %s rows with missing or non-numeric values in "
                "%s: %s",
                missing.sum(),
                name,
                df.loc[missing, ["country", "year"]].to_dict("records"),
            )
            df = df.loc[~missing].copy()

        return df

    def validate_country_year(df, name):
        """Check that each country-year combination is unique."""
        check_unique(df, ["country", "year"], name)

    # ------------------------------------------------------------------
    # 1. Data-centre output scenarios
    # ------------------------------------------------------------------

    output = read_table(SCRIPT_DIR / "dc_output_scenarios.csv")

    require_columns(
        output,
        [
            "country",
            "year",
            "dc_capacity_mw",
            "output_mn_usd_per_mw_2022",
        ],
        "dc_output_scenarios.csv",
    )

    output = clean_keys(output, "dc_output_scenarios.csv")

    output = clean_numeric_observations(
        output,
        ["dc_capacity_mw", "output_mn_usd_per_mw_2022"],
        "dc_output_scenarios.csv",
    )

    validate_country_year(output, "dc_output_scenarios.csv")

    output["dc_capacity_mw"] = numeric_column(
        output,
        "dc_capacity_mw",
        "dc_output_scenarios",
    )

    output["output_mn_usd_per_mw_2022"] = numeric_column(
        output,
        "output_mn_usd_per_mw_2022",
        "dc_output_scenarios",
    )

    if (output["dc_capacity_mw"] < 0).any():
        raise ValueError("Data-centre capacity cannot be negative.")

    if (output["output_mn_usd_per_mw_2022"] <= 0).any():
        raise ValueError("Data-centre output coefficients must be positive.")

    if not REQUIRED_COUNTRIES.issubset(set(output["country"])):
        raise ValueError("dc_output_scenarios.csv must include CAN and IND.")

    output["dc_output_usd_mn"] = (
        output["dc_capacity_mw"]
        * output["output_mn_usd_per_mw_2022"]
    )

    # ------------------------------------------------------------------
    # 2. Electricity demand
    # ------------------------------------------------------------------

    demand = read_table(SCRIPT_DIR / "dc_electricity_demand.csv")

    require_columns(
        demand,
        ["country", "year", "demand_basis"],
        "dc_electricity_demand.csv",
    )

    if "electricity_gwh" in demand.columns:
        demand_column = "electricity_gwh"
    elif "electricity_mw" in demand.columns:
        demand_column = "electricity_mw"
    else:
        raise ValueError(
            "dc_electricity_demand.csv must contain either "
            "'electricity_gwh' or 'electricity_mw'."
        )

    demand = clean_keys(demand, "dc_electricity_demand.csv")

    # Normalize demand basis while preserving missing values.
    demand["demand_basis"] = (
        demand["demand_basis"]
        .astype("string")
        .str.lower()
        .str.strip()
    )

    demand["demand_basis"] = demand["demand_basis"].replace({
        "facility load": "facility",
        "facility-level": "facility",
        "facility level": "facility",
        "it load": "it",
        "it-level": "it",
        "it level": "it",
    })

    # Convert demand to numeric and drop incomplete observations.
    demand = clean_numeric_observations(
        demand,
        [demand_column],
        "dc_electricity_demand.csv",
    )

    missing_basis = (
        demand["demand_basis"].isna()
        | demand["demand_basis"].eq("")
    )

    if missing_basis.any():
        LOG.warning(
            "Dropping %s rows with missing demand_basis: %s",
            missing_basis.sum(),
            demand.loc[missing_basis, ["country", "year"]].to_dict("records"),
        )
        demand = demand.loc[~missing_basis].copy()

    invalid_basis = ~demand["demand_basis"].isin(["facility", "it"])

    if invalid_basis.any():
        invalid_values = (
            demand.loc[invalid_basis, "demand_basis"].unique().tolist()
        )
        raise ValueError(
            "Invalid demand_basis values. Expected 'facility' or 'it'. "
            f"Found: {invalid_values}"
        )

    # Convert average MW to annual GWh where necessary.
    if demand_column == "electricity_mw":
        demand["electricity_gwh"] = (
            demand["electricity_mw"] * 8760 / 1000
        )

    validate_country_year(demand, "dc_electricity_demand.csv")

    demand["electricity_gwh"] = numeric_column(
        demand,
        "electricity_gwh",
        "dc_electricity_demand",
    )

    if (demand["electricity_gwh"] < 0).any():
        raise ValueError("Electricity demand cannot be negative.")

    # ------------------------------------------------------------------
    # 3. Electricity prices (already in constant 2022 USD/MWh)
    # ------------------------------------------------------------------

    prices = read_table(SCRIPT_DIR / "electricity_prices.csv")

    require_columns(
        prices,
        ["country", "year", "price_2022_usd_mwh"],
        "electricity_prices.csv",
    )

    prices = clean_keys(prices, "electricity_prices.csv")

    prices = clean_numeric_observations(
        prices,
        ["price_2022_usd_mwh"],
        "electricity_prices.csv",
    )

    validate_country_year(prices, "electricity_prices.csv")

    prices["price_2022_usd_mwh"] = numeric_column(
        prices,
        "price_2022_usd_mwh",
        "electricity_prices",
    )

    if (prices["price_2022_usd_mwh"] <= 0).any():
        raise ValueError("Electricity prices must be positive.")

    # ------------------------------------------------------------------
    # 4. Grid emissions factors
    # ------------------------------------------------------------------

    grid = read_table(SCRIPT_DIR / "grid_emissions_factors.csv")

    require_columns(
        grid,
        ["country", "year", "kgco2_per_kwh"],
        "grid_emissions_factors.csv",
    )

    grid = clean_keys(grid, "grid_emissions_factors.csv")

    grid = clean_numeric_observations(
        grid,
        ["kgco2_per_kwh"],
        "grid_emissions_factors.csv",
    )

    validate_country_year(grid, "grid_emissions_factors.csv")

    grid["kgco2_per_kwh"] = numeric_column(
        grid,
        "kgco2_per_kwh",
        "grid_emissions_factors",
    )

    if (grid["kgco2_per_kwh"] < 0).any():
        raise ValueError("Grid emissions factors cannot be negative.")

    # ------------------------------------------------------------------
    # 5. Water coefficients
    # ------------------------------------------------------------------

    water = read_table(SCRIPT_DIR / "water_coefficients.csv")

    require_columns(
        water,
        ["country", "year", "pue", "wue_l_per_kwh_it"],
        "water_coefficients.csv",
    )

    water = clean_keys(water, "water_coefficients.csv")

    water = clean_numeric_observations(
        water,
        ["pue", "wue_l_per_kwh_it"],
        "water_coefficients.csv",
    )

    validate_country_year(water, "water_coefficients.csv")

    water["pue"] = numeric_column(
        water,
        "pue",
        "water_coefficients",
    )

    water["wue_l_per_kwh_it"] = numeric_column(
        water,
        "wue_l_per_kwh_it",
        "water_coefficients",
    )

    if (water["pue"] < 1).any():
        raise ValueError("PUE must be at least one.")

    if (water["wue_l_per_kwh_it"] < 0).any():
        raise ValueError("WUE cannot be negative.")

    # ------------------------------------------------------------------
    # 6. US customer shares
    # ------------------------------------------------------------------

    customer = read_table(SCRIPT_DIR / "us_customer_shares.csv")

    require_columns(
        customer,
        ["country", "year", "us_customer_share"],
        "us_customer_shares.csv",
    )

    customer = clean_keys(customer, "us_customer_shares.csv")

    customer = clean_numeric_observations(
        customer,
        ["us_customer_share"],
        "us_customer_shares.csv",
    )

    validate_country_year(customer, "us_customer_shares.csv")

    validate_shares(
        customer,
        "us_customer_share",
        "us_customer_shares",
    )

    # ------------------------------------------------------------------
    # 7. Optional US parent-operator shares
    # ------------------------------------------------------------------

    owner_path = SCRIPT_DIR / "us_parent_operator_shares.csv"

    if owner_path.exists():
        owner = read_table(owner_path)

        require_columns(
            owner,
            ["country", "year", "us_parent_operator_share"],
            "us_parent_operator_shares.csv",
        )

        owner = clean_keys(owner, "us_parent_operator_shares.csv")

        owner = clean_numeric_observations(
            owner,
            ["us_parent_operator_share"],
            "us_parent_operator_shares.csv",
        )

        validate_country_year(owner, "us_parent_operator_shares.csv")

        validate_shares(
            owner,
            "us_parent_operator_share",
            "us_parent_operator_shares",
        )

    else:
        owner = None

        LOG.warning(
            "No us_parent_operator_shares.csv found. "
            "US-parent attribution will not be calculated."
        )

    # ------------------------------------------------------------------
    # Return cleaned input data
    # ------------------------------------------------------------------

    return output, demand, prices, grid, water, customer, owner


# =====================================================================
# LOAD AEA EMISSIONS ALLOCATION WEIGHTS
# =====================================================================

def load_aea_emission_weights(
    weights_file: Path | None = None,
    reference_year: int | None = None,
) -> pd.DataFrame:
    """
    Load and validate independent AEA emissions allocation weights.

    Required columns:
        REF_AREA
        TIME_PERIOD
        aea_activity_code
        icio_sector_code
        emission_weight

    Weights are normalized later within each country-year-activity.
    """

    if weights_file is None:
        weights_file = SCRIPT_DIR / "aea_weights.csv"

    weights = pd.read_csv(
        weights_file,
        low_memory=False,
    )

    required = [
        "REF_AREA",
        "TIME_PERIOD",
        "aea_activity_code",
        "icio_sector_code",
        "emission_weight",
    ]

    require_columns(weights, required, "AEA emissions weights")

    weights = weights[required].copy()

    # ---------------------------------------------------------
    # Standardize identifiers
    # ---------------------------------------------------------

    for col in [
        "REF_AREA",
        "aea_activity_code",
        "icio_sector_code",
    ]:
        weights[col] = (
            weights[col].astype("string").str.strip()
        )

    weights["TIME_PERIOD"] = pd.to_numeric(
        weights["TIME_PERIOD"],
        errors="raise",
    )

    weights["emission_weight"] = pd.to_numeric(
        weights["emission_weight"],
        errors="raise",
    )

    if weights[
        ["REF_AREA", "aea_activity_code", "icio_sector_code"]
    ].isna().any().any():
        raise ValueError(
            "AEA emissions weights contain missing identifiers."
        )

    if weights["TIME_PERIOD"].isna().any():
        raise ValueError(
            "AEA emissions weights contain missing years."
        )

    if not np.isfinite(weights["emission_weight"]).all():
        raise ValueError(
            "AEA emissions weights contain non-finite values."
        )

    if (weights["emission_weight"] < 0).any():
        raise ValueError(
            "AEA emissions weights cannot be negative."
        )

    # ---------------------------------------------------------
    # Select reference year
    # ---------------------------------------------------------

    if reference_year is not None:

        weights = weights.loc[
            weights["TIME_PERIOD"].eq(reference_year)
        ].copy()

        if weights.empty:
            raise ValueError(
                f"No AEA emissions weights found for {reference_year}."
            )

    # ---------------------------------------------------------
    # Check uniqueness
    # ---------------------------------------------------------

    weight_keys = [
        "REF_AREA",
        "TIME_PERIOD",
        "aea_activity_code",
        "icio_sector_code",
    ]

    check_unique(
        weights,
        weight_keys,
        "AEA emissions weights",
    )

    # ---------------------------------------------------------
    # Check positive total weights
    # ---------------------------------------------------------

    group_keys = [
        "REF_AREA",
        "TIME_PERIOD",
        "aea_activity_code",
    ]

    weight_totals = (
        weights.groupby(group_keys)["emission_weight"].sum()
    )

    if (weight_totals <= 0).any():
        raise ValueError(
            "Some country-year-activity groups have zero total "
            "emissions allocation weights:\n"
            f"{weight_totals[weight_totals <= 0]}"
        )

    # ---------------------------------------------------------
    # Normalize weights
    # ---------------------------------------------------------

    weights["emission_weight"] = (
        weights["emission_weight"]
        / weights.groupby(group_keys)["emission_weight"]
            .transform("sum")
    )

    LOG.info(
        "Loaded %s AEA emissions allocation weights from %s.",
        len(weights),
        weights_file,
    )

    return weights



# =====================================================================
# BUILD AEA EMISSIONS EXTENSION
# =====================================================================

def build_aea_extension(
    args,
    X: pd.Series,
    sector_map: pd.DataFrame,
    weights: pd.DataFrame,
):
    """
    Build country-industry emissions and emissions intensities.

    Mapping types
    -------------
    direct:
        Assign AEA emissions directly to one ICIO industry.

    emission_weight:
        Allocate aggregate AEA emissions using independent,
        country-year-activity-specific emissions weights.

    aggregate:
        Not allocated by this function. The AEA hierarchy must first
        be resolved to avoid double counting aggregate and detailed
        observations.

    Returns
    -------
    extension, coverage, coverage_summary, allocation
    """

    # ---------------------------------------------------------
    # 1. Load AEA observations and crosswalk
    # ---------------------------------------------------------

    aea = pd.read_csv(
        args.AEA_file,
        low_memory=False,
    )

    required_aea_columns = [
        args.aea_country_column,
        args.aea_activity_column,
        args.aea_year_column,
        args.aea_value_column,
    ]

    if args.aea_pollutant_column:
        required_aea_columns.append(args.aea_pollutant_column)

    if args.aea_measure_column:
        required_aea_columns.append(args.aea_measure_column)

    if args.aea_unit_column:
        required_aea_columns.append(args.aea_unit_column)

    require_columns(aea, required_aea_columns, "AEA file")

    crosswalk = read_table(
        SCRIPT_DIR / "aea_crosswalk.csv"
    )

    require_columns(
        crosswalk,
        [
            "aea_activity_code",
            "icio_sector_code",
            "mapping_type",
        ],
        "aea_crosswalk.csv",
    )

    crosswalk = crosswalk.copy()

    for col in [
        "aea_activity_code",
        "icio_sector_code",
        "mapping_type",
    ]:
        crosswalk[col] = (
            crosswalk[col].astype("string").str.strip()
        )

    check_unique(
        crosswalk,
        ["aea_activity_code", "icio_sector_code"],
        "aea_crosswalk.csv",
    )

    # ---------------------------------------------------------
    # 2. Filter and prepare AEA observations
    # ---------------------------------------------------------

    aea = aea.copy()

    aea["country"] = (
        aea[args.aea_country_column].astype(str).str.strip()
    )

    aea["aea_activity_code"] = (
        aea[args.aea_activity_column].astype(str).str.strip()
    )

    aea["year"] = pd.to_numeric(
        aea[args.aea_year_column],
        errors="coerce",
    )

    aea["ghg_tonnes_co2e"] = pd.to_numeric(
        aea[args.aea_value_column],
        errors="coerce",
    )

    aea = aea.loc[
        aea["year"].eq(args.icio_year)
    ].copy()

    if args.aea_pollutant_column:
        aea = aea.loc[
            aea[args.aea_pollutant_column]
            .astype(str)
            .eq(args.aea_pollutant_value)
        ].copy()

    if args.aea_measure_column:
        aea = aea.loc[
            aea[args.aea_measure_column]
            .astype(str)
            .eq(args.aea_measure_value)
        ].copy()

    if args.aea_unit_column:
        aea = aea.loc[
            aea[args.aea_unit_column]
            .astype(str)
            .eq(args.aea_unit_value)
        ].copy()

    if aea.empty:
        raise ValueError(
            "No AEA observations remain after applying the "
            "year and dimension filters."
        )

    if aea["ghg_tonnes_co2e"].isna().any():
        raise ValueError(
            "AEA contains missing or non-numeric emissions values."
        )

    aea["ghg_tonnes_co2e"] *= args.aea_unit_to_tonnes

    if not np.isfinite(aea["ghg_tonnes_co2e"]).all():
        raise ValueError(
            "AEA contains non-finite emissions values."
        )

    if (aea["ghg_tonnes_co2e"] < 0).any():
        raise ValueError(
            "Negative AEA emissions were found."
        )

    # Sum any remaining observations only after the relevant AEA
    # dimensions have been filtered.
    aea = (
        aea.groupby(
            ["country", "year", "aea_activity_code"],
            as_index=False,
        )["ghg_tonnes_co2e"].sum()
    )

    # ---------------------------------------------------------
    # 3. Separate crosswalk mapping types
    # ---------------------------------------------------------

    direct_map = crosswalk.loc[
        crosswalk["mapping_type"].eq("direct")
    ].copy()

    weighted_map = crosswalk.loc[
        crosswalk["mapping_type"].eq("emission_weight")
    ].copy()

    aggregate_map = crosswalk.loc[
        crosswalk["mapping_type"].eq("aggregate")
    ].copy()

    unsupported = (
        set(crosswalk["mapping_type"].dropna().unique())
        - {"direct", "emission_weight", "aggregate"}
    )

    if unsupported:
        raise ValueError(
            f"Unsupported crosswalk mapping types: {sorted(unsupported)}"
        )

    # Direct mappings must have exactly one destination.
    direct_counts = (
        direct_map.groupby("aea_activity_code")["icio_sector_code"]
        .nunique()
    )

    if (direct_counts > 1).any():
        raise ValueError(
            "Direct AEA mappings must have exactly one ICIO destination."
        )

    # ---------------------------------------------------------
    # 4. Identify unmatched and aggregate-only observations
    # ---------------------------------------------------------

    supported_activities = (
        set(direct_map["aea_activity_code"])
        | set(weighted_map["aea_activity_code"])
    )

    aggregate_activities = set(
        aggregate_map["aea_activity_code"]
    )

    unmatched = sorted(
        set(aea["aea_activity_code"])
        - set(crosswalk["aea_activity_code"])
    )

    if unmatched:
        LOG.warning(
            "%s AEA activities are not in the crosswalk. Examples: %s",
            len(unmatched),
            unmatched[:20],
        )

    aggregate_only = sorted(
        set(aea["aea_activity_code"])
        & aggregate_activities
        - supported_activities
    )

    if aggregate_only:
        LOG.warning(
            "AEA aggregate activities are not allocated by this function. "
            "Resolve their hierarchy or provide explicit allocation weights: %s",
            aggregate_only,
        )

    # Only direct and explicitly weighted activities are allocated.
    aea_eligible = aea.loc[
        aea["aea_activity_code"].isin(supported_activities)
    ].copy()

    # ---------------------------------------------------------
    # 5. Build ICIO output lookup
    # ---------------------------------------------------------

    output = pd.DataFrame({
        "io_key": X.index.astype(str),
        "output_usd_mn": X.to_numpy(dtype=float),
    })

    require_columns(
        sector_map,
        ["io_key", "country", "sector_code"],
        "ICIO sector map",
    )

    output = output.merge(
        sector_map[["io_key", "country", "sector_code"]],
        on="io_key",
        how="left",
        validate="one_to_one",
    )

    if output[["country", "sector_code"]].isna().any().any():
        raise ValueError(
            "Some ICIO output industries could not be matched to "
            "the country-sector map."
        )

    if not np.isfinite(output["output_usd_mn"]).all():
        raise ValueError("ICIO gross output contains non-finite values.")

    if (output["output_usd_mn"] < 0).any():
        raise ValueError("ICIO gross output contains negative values.")

    # ---------------------------------------------------------
    # 6. Direct mappings
    # ---------------------------------------------------------

    direct = aea_eligible.loc[
        aea_eligible["aea_activity_code"].isin(
            direct_map["aea_activity_code"]
        )
    ].merge(
        direct_map,
        on="aea_activity_code",
        how="inner",
        validate="many_to_one",
    )

    direct = direct.merge(
        output,
        left_on=["country", "icio_sector_code"],
        right_on=["country", "sector_code"],
        how="left",
        validate="many_to_one",
    )

    if direct["io_key"].isna().any():
        missing = direct.loc[
            direct["io_key"].isna(),
            ["country", "aea_activity_code", "icio_sector_code"],
        ].drop_duplicates()

        raise ValueError(
            "Direct AEA mappings refer to ICIO sectors that do not exist:\n"
            f"{missing.head(20).to_string(index=False)}"
        )

    direct["allocation_weight"] = 1.0

    direct["allocated_ghg_tonnes_co2e"] = (
        direct["ghg_tonnes_co2e"]
    )

    direct["allocation_method"] = "direct"

    # ---------------------------------------------------------
    # 7. Emissions-weighted mappings
    # ---------------------------------------------------------

    weighted_aea = aea_eligible.loc[
        aea_eligible["aea_activity_code"].isin(
            weighted_map["aea_activity_code"]
        )
    ].copy()

    if not weighted_aea.empty:

        # Attach the destination ICIO sectors.
        weighted = weighted_aea.merge(
            weighted_map,
            on="aea_activity_code",
            how="inner",
            validate="many_to_many",
        )

        # Match the independent emissions weights.
        weighted = weighted.merge(
            weights,
            left_on=[
                "country",
                "year",
                "aea_activity_code",
                "icio_sector_code",
            ],
            right_on=[
                "REF_AREA",
                "TIME_PERIOD",
                "aea_activity_code",
                "icio_sector_code",
            ],
            how="left",
            validate="many_to_one",
        )

        if weighted["emission_weight"].isna().any():
            missing = weighted.loc[
                weighted["emission_weight"].isna(),
                [
                    "country",
                    "year",
                    "aea_activity_code",
                    "icio_sector_code",
                ],
            ].drop_duplicates()

            raise ValueError(
                "Missing independent emissions weights for:\n"
                f"{missing.head(20).to_string(index=False)}"
            )

        # Verify that every allocation group has positive weights.
        weight_keys = [
            "country",
            "year",
            "aea_activity_code",
        ]

        weight_totals = (
            weighted.groupby(weight_keys)["emission_weight"].sum()
        )

        if (weight_totals <= 0).any():
            raise ValueError(
                "Some emissions allocation groups have zero total weight."
            )

        # Normalize weights within each country-year-activity.
        weighted["allocation_weight"] = (
            weighted["emission_weight"]
            / weighted.groupby(weight_keys)["emission_weight"]
                .transform("sum")
        )

        weighted["allocated_ghg_tonnes_co2e"] = (
            weighted["ghg_tonnes_co2e"]
            * weighted["allocation_weight"]
        )

        weighted["allocation_method"] = "aea_emissions_weight"

        # Match destination industries to ICIO gross output.
        weighted = weighted.merge(
            output,
            left_on=["country", "icio_sector_code"],
            right_on=["country", "sector_code"],
            how="left",
            validate="many_to_one",
        )

        if weighted["io_key"].isna().any():
            missing = weighted.loc[
                weighted["io_key"].isna(),
                ["country", "aea_activity_code", "icio_sector_code"],
            ].drop_duplicates()

            raise ValueError(
                "Emissions-weighted mappings refer to ICIO sectors "
                "that do not exist:\n"
                f"{missing.head(20).to_string(index=False)}"
            )

    else:
        weighted = direct.iloc[0:0].copy()

    # ---------------------------------------------------------
    # 8. Combine allocation results
    # ---------------------------------------------------------

    allocation_columns = [
        "country",
        "year",
        "aea_activity_code",
        "icio_sector_code",
        "io_key",
        "output_usd_mn",
        "ghg_tonnes_co2e",
        "allocation_weight",
        "allocated_ghg_tonnes_co2e",
        "allocation_method",
    ]

    allocation = pd.concat(
        [
            direct[allocation_columns],
            weighted[allocation_columns],
        ],
        ignore_index=True,
    )

    # ---------------------------------------------------------
    # 9. Verify emissions conservation
    # ---------------------------------------------------------

    original_totals = (
        aea_eligible.groupby(
            ["country", "year", "aea_activity_code"]
        )["ghg_tonnes_co2e"].sum()
    )

    allocated_totals = (
        allocation.groupby(
            ["country", "year", "aea_activity_code"]
        )["allocated_ghg_tonnes_co2e"].sum()
    )

    conservation = pd.concat(
        [
            original_totals.rename("original"),
            allocated_totals.rename("allocated"),
        ],
        axis=1,
    )

    if not np.allclose(
        conservation["original"],
        conservation["allocated"],
        rtol=1e-8,
        atol=1e-8,
    ):
        raise ValueError(
            "AEA emissions allocation does not conserve the original "
            "emissions totals:\n"
            f"{conservation.loc[~np.isclose(conservation['original'], conservation['allocated'])]}"
        )

    # ---------------------------------------------------------
    # 10. Aggregate emissions by ICIO industry
    # ---------------------------------------------------------

    emissions = (
        allocation.groupby(
            ["io_key", "country", "icio_sector_code"],
            as_index=False,
        )["allocated_ghg_tonnes_co2e"].sum()
        .rename(columns={
            "icio_sector_code": "sector_code",
            "allocated_ghg_tonnes_co2e": "ghg_tonnes_co2e",
        })
    )

    emissions = emissions.merge(
        output[["io_key", "output_usd_mn"]],
        on="io_key",
        how="left",
        validate="one_to_one",
    )

    if (emissions["output_usd_mn"] <= 0).any():
        raise ValueError(
            "Cannot calculate emissions intensity for an industry "
            "with zero or negative ICIO gross output."
        )

    emissions["ghg_tonnes_per_usd_mn"] = (
        emissions["ghg_tonnes_co2e"]
        / emissions["output_usd_mn"]
    )

    # ---------------------------------------------------------
    # 11. AEA coverage report
    # ---------------------------------------------------------

    coverage = output.merge(
        emissions[["io_key", "ghg_tonnes_co2e"]],
        on="io_key",
        how="left",
        validate="one_to_one",
    )

    coverage["has_aea_emissions"] = (
        coverage["ghg_tonnes_co2e"].notna()
    )

    coverage_summary = (
        coverage.groupby("country")
        .agg(
            icio_industries=("io_key", "size"),
            industries_with_aea=("has_aea_emissions", "sum"),
            covered_output=("output_usd_mn", lambda s: s[
                coverage.loc[s.index, "has_aea_emissions"]
            ].sum()),
            total_output=("output_usd_mn", "sum"),
        )
        .reset_index()
    )

    coverage_summary["output_coverage_share"] = (
        coverage_summary["covered_output"]
        / coverage_summary["total_output"]
    )

    coverage_summary = coverage_summary.drop(
        columns=["covered_output", "total_output"]
    )

    missing = coverage.loc[
        ~coverage["has_aea_emissions"],
        ["io_key", "country", "sector_code", "output_usd_mn"],
    ]

    if not missing.empty and not args.allow_missing_aea:
        raise ValueError(
            f"AEA emissions are missing for {len(missing)} ICIO industries. "
            "Inspect aea_emissions_coverage.csv."
        )

    if not missing.empty:
        LOG.warning(
            "%s ICIO industries have no AEA emissions observation. "
            "Their emissions are excluded from the footprint.",
            len(missing),
        )

    # ---------------------------------------------------------
    # 12. Build final extension
    # ---------------------------------------------------------

    extension = output.merge(
        emissions[
            ["io_key", "ghg_tonnes_co2e", "ghg_tonnes_per_usd_mn"]
        ],
        on="io_key",
        how="left",
        validate="one_to_one",
    )

    return extension, coverage, coverage_summary, allocation

"""
Allocate aggregate AEA emissions to multiple ICIO industries
using independent emissions weights.

Inputs
------
OECD_AEA.csv
aea_crosswalk.csv
aea_weights.csv

Output
------
aea_icio_emissions.csv
aea_allocation_diagnostics.csv

aea_crosswalk.csv:
    aea_activity_code,icio_sector_code,mapping_type,hierarchy_group,hierarchy_level,mapping_note

aea_weights.csv:
    REF_AREA,TIME_PERIOD,icio_sector_code,emission_weight

The emission weights should be independently estimated sector emissions,
or another documented emissions-based allocation proxy.

Weights are normalized within each country, year and AEA activity.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

AEA_FILE = Path(
    r"D:\Research\Datacenters\OECD_AEA.csv"
)

CROSSWALK_FILE = SCRIPT_DIR / "aea_crosswalk.csv"

WEIGHTS_FILE = SCRIPT_DIR / "aea_weights.csv"

OUTPUT_FILE = SCRIPT_DIR / "aea_icio_emissions.csv"

DIAGNOSTICS_FILE = SCRIPT_DIR / "aea_allocation_diagnostics.csv"


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

REFERENCE_YEAR = 2022

# The emissions measure used in the extension.
POLLUTANT = "GHG"

UNIT_MEASURE = "T_CO2E"

# Only resident production activities are allocated to domestic
# producing industries.
ACTIVITY_SCOPE = "RES"

# AEA activity codes requiring allocation across multiple ICIO sectors.
#
# These codes must also appear in aea_crosswalk.csv with mapping_type
# equal to "emission_weight".

ALLOCATED_ACTIVITIES = {
    "B",
    "C24",
    "C30",
}

# Directly mapped activities are retained without allocation.
DIRECT_MAPPING_TYPE = "direct"

ALLOCATED_MAPPING_TYPE = "emission_weight"


# ---------------------------------------------------------------------
# LOAD AND VALIDATE INPUTS
# ---------------------------------------------------------------------

def load_inputs():

    aea = pd.read_csv(AEA_FILE, low_memory=False)

    crosswalk = pd.read_csv(CROSSWALK_FILE)

    weights = pd.read_csv(WEIGHTS_FILE)

    required_aea = {
        "REF_AREA",
        "TIME_PERIOD",
        "ACTIVITY",
        "ACTIVITY_SCOPE",
        "POLLUTANT",
        "UNIT_MEASURE",
        "OBS_VALUE",
    }

    required_crosswalk = {
        "aea_activity_code",
        "icio_sector_code",
        "mapping_type",
    }

    required_weights = {
        "REF_AREA",
        "TIME_PERIOD",
        "aea_activity_code",
        "icio_sector_code",
        "emission_weight",
    }

    for name, df, required in [
        ("AEA", aea, required_aea),
        ("Crosswalk", crosswalk, required_crosswalk),
        ("Weights", weights, required_weights),
    ]:

        missing = required.difference(df.columns)

        if missing:
            raise ValueError(
                f"{name} is missing required columns: {sorted(missing)}"
            )

    return aea, crosswalk, weights


# ---------------------------------------------------------------------
# PREPARE AEA OBSERVATIONS
# ---------------------------------------------------------------------

def prepare_aea(aea):

    aea = aea.copy()

    aea["REF_AREA"] = aea["REF_AREA"].astype(str).str.strip()

    aea["ACTIVITY"] = aea["ACTIVITY"].astype(str).str.strip()

    aea["TIME_PERIOD"] = pd.to_numeric(
        aea["TIME_PERIOD"],
        errors="coerce",
    )

    aea["OBS_VALUE"] = pd.to_numeric(
        aea["OBS_VALUE"],
        errors="coerce",
    )

    aea = aea.loc[
        (aea["TIME_PERIOD"] == REFERENCE_YEAR)
        & (aea["POLLUTANT"] == POLLUTANT)
        & (aea["UNIT_MEASURE"] == UNIT_MEASURE)
        & (aea["ACTIVITY_SCOPE"] == ACTIVITY_SCOPE)
    ].copy()

    aea = aea.dropna(
        subset=["REF_AREA", "ACTIVITY", "OBS_VALUE"]
    )

    aea["TIME_PERIOD"] = aea["TIME_PERIOD"].astype(int)

    # There should be one observation per country/activity/year
    # after the selected filters.
    #
    # If this fails, inspect other AEA dimensions such as MEASURE,
    # ADJUSTMENT, METHODOLOGY and UNIT_MULT before aggregating.

    duplicate_keys = [
        "REF_AREA",
        "TIME_PERIOD",
        "ACTIVITY",
    ]

    duplicates = aea.duplicated(
        subset=duplicate_keys,
        keep=False,
    )

    if duplicates.any():

        print(
            "\nWARNING: Duplicate AEA observations after filtering:"
        )

        print(
            aea.loc[duplicates, duplicate_keys + ["OBS_VALUE"]]
            .sort_values(duplicate_keys)
            .to_string(index=False)
        )

        raise ValueError(
            "AEA observations are not unique. Inspect the remaining "
            "dimensions before proceeding."
        )

    return aea


# ---------------------------------------------------------------------
# PREPARE CROSSWALK
# ---------------------------------------------------------------------

def prepare_crosswalk(crosswalk):

    crosswalk = crosswalk.copy()

    for column in [
        "aea_activity_code",
        "icio_sector_code",
        "mapping_type",
    ]:

        crosswalk[column] = (
            crosswalk[column]
            .astype(str)
            .str.strip()
        )

    if crosswalk.duplicated(
        ["aea_activity_code", "icio_sector_code"]
    ).any():

        raise ValueError(
            "Duplicate AEA-to-ICIO mappings found in crosswalk."
        )

    return crosswalk


# ---------------------------------------------------------------------
# PREPARE EMISSION WEIGHTS
# ---------------------------------------------------------------------

def prepare_weights(weights):

    weights = weights.copy()

    weights["REF_AREA"] = (
        weights["REF_AREA"].astype(str).str.strip()
    )

    weights["icio_sector_code"] = (
        weights["icio_sector_code"].astype(str).str.strip()
    )

    weights["TIME_PERIOD"] = pd.to_numeric(
        weights["TIME_PERIOD"],
        errors="coerce",
    )

    weights["emission_weight"] = pd.to_numeric(
        weights["emission_weight"],
        errors="coerce",
    )
    
    weights["aea_activity_code"] = (
        weights["aea_activity_code"].astype(str).str.strip()
    )

    weights = weights.dropna(
        subset=[
            "REF_AREA",
            "TIME_PERIOD",
            "icio_sector_code",
            "emission_weight",
        ]
    )

    weights["TIME_PERIOD"] = weights["TIME_PERIOD"].astype(int)

    if (weights["emission_weight"] < 0).any():

        raise ValueError(
            "Emission weights cannot be negative."
        )

    keys = [
        "REF_AREA",
        "TIME_PERIOD",
        "aea_activity_code",
        "icio_sector_code",
    ]

    if weights.duplicated(keys).any():

        raise ValueError(
            "Duplicate country-year-ICIO-sector emission weights."
        )

    return weights


# ---------------------------------------------------------------------
# ALLOCATE AEA EMISSIONS
# ---------------------------------------------------------------------


def allocate_emissions(
    aea: pd.DataFrame,
    crosswalk: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """
    Map AEA emissions to ICIO industries.

    Direct AEA-to-ICIO mappings are assigned in full.
    B, C24, and C30 are allocated using independent, country-year-specific
    emissions weights.

    Aggregate crosswalk mappings are not allocated here. The AEA input must
    first be filtered to a non-overlapping hierarchy of activities so that
    aggregate and detailed observations are not both counted.

    Parameters
    ----------
    aea : pd.DataFrame
        AEA observations, already filtered to the desired year, pollutant,
        unit, and activity scope. Required columns:
        REF_AREA, TIME_PERIOD, ACTIVITY, OBS_VALUE.

    crosswalk : pd.DataFrame
        AEA-to-ICIO concordance. Required columns:
        aea_activity_code, icio_sector_code, mapping_type.

    weights : pd.DataFrame
        Independent emissions allocation weights. Required columns:
        REF_AREA, TIME_PERIOD, aea_activity_code, icio_sector_code,
        emission_weight.

    Returns
    -------
    pd.DataFrame
        Allocated emissions by country, year, AEA activity, and ICIO sector.
        The original AEA emissions value is retained for auditing.
    """

    ALLOCATED_ACTIVITIES = {"B", "C24", "C30"}

    # ---------------------------------------------------------
    # 1. Validate required columns
    # ---------------------------------------------------------

    required_aea = {
        "REF_AREA",
        "TIME_PERIOD",
        "ACTIVITY",
        "OBS_VALUE",
    }

    required_crosswalk = {
        "aea_activity_code",
        "icio_sector_code",
        "mapping_type",
    }

    required_weights = {
        "REF_AREA",
        "TIME_PERIOD",
        "aea_activity_code",
        "icio_sector_code",
        "emission_weight",
    }

    for name, frame, required in [
        ("AEA", aea, required_aea),
        ("Crosswalk", crosswalk, required_crosswalk),
        ("Weights", weights, required_weights),
    ]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"{name} is missing required columns: {sorted(missing)}"
            )

    # ---------------------------------------------------------
    # 2. Standardize keys and numeric fields
    # ---------------------------------------------------------

    aea = aea.copy()
    crosswalk = crosswalk.copy()
    weights = weights.copy()

    for col in ["REF_AREA", "ACTIVITY"]:
        aea[col] = aea[col].astype("string").str.strip()

    aea["TIME_PERIOD"] = pd.to_numeric(
        aea["TIME_PERIOD"], errors="raise"
    ).astype(int)

    aea["OBS_VALUE"] = pd.to_numeric(
        aea["OBS_VALUE"], errors="raise"
    )

    for col in ["aea_activity_code", "icio_sector_code", "mapping_type"]:
        crosswalk[col] = crosswalk[col].astype("string").str.strip()

    for col in ["REF_AREA", "aea_activity_code", "icio_sector_code"]:
        weights[col] = weights[col].astype("string").str.strip()

    weights["TIME_PERIOD"] = pd.to_numeric(
        weights["TIME_PERIOD"], errors="raise"
    ).astype(int)

    weights["emission_weight"] = pd.to_numeric(
        weights["emission_weight"], errors="raise"
    )

    if not np.isfinite(aea["OBS_VALUE"]).all():
        raise ValueError("AEA OBS_VALUE contains non-finite values.")

    if not np.isfinite(weights["emission_weight"]).all():
        raise ValueError("Allocation weights contain non-finite values.")

    if (aea["OBS_VALUE"] < 0).any():
        raise ValueError("AEA emissions must be non-negative.")

    if (weights["emission_weight"] < 0).any():
        raise ValueError("Emission weights must be non-negative.")

    # ---------------------------------------------------------
    # 3. Check AEA observation uniqueness
    # ---------------------------------------------------------

    aea_keys = ["REF_AREA", "TIME_PERIOD", "ACTIVITY"]

    duplicates = aea.duplicated(aea_keys, keep=False)

    if duplicates.any():
        examples = aea.loc[duplicates, aea_keys].head(10)

        raise ValueError(
            "AEA contains duplicate country-year-activity observations "
            "after filtering. Check ACTIVITY_SCOPE, POLLUTANT, UNIT_MEASURE, "
            "and any other dimensions before allocation.\n"
            f"{examples.to_string(index=False)}"
        )

    # ---------------------------------------------------------
    # 4. Validate crosswalk
    # ---------------------------------------------------------

    crosswalk = crosswalk.drop_duplicates()

    cw_keys = ["aea_activity_code", "icio_sector_code"]

    if crosswalk.duplicated(cw_keys).any():
        raise ValueError(
            "Crosswalk contains duplicate AEA-to-ICIO mappings."
        )

    # Only direct mappings and explicit emissions allocations are handled.
    supported_types = {"direct", "emission_weight"}

    unsupported = (
        set(crosswalk["mapping_type"].dropna().unique())
        - supported_types
    )

    # Aggregate mappings are intentionally excluded. They must be resolved
    # by the hierarchy-selection procedure before allocation.
    if unsupported - {"aggregate"}:
        raise ValueError(
            f"Unsupported crosswalk mapping types: {sorted(unsupported)}"
        )

    # Verify that the three allocation activities have the expected
    # destinations and mapping type.
    expected_targets = {
        "B": {"05", "06", "07", "08", "09"},
        "C24": {"24A", "24B"},
        "C30": {"301", "302T309"},
    }

    for activity, expected in expected_targets.items():
        rows = crosswalk.loc[
            crosswalk["aea_activity_code"].eq(activity)
        ]

        actual = set(rows["icio_sector_code"])

        if actual != expected:
            raise ValueError(
                f"Crosswalk destinations for {activity} are incorrect. "
                f"Expected {sorted(expected)}, found {sorted(actual)}."
            )

        if not rows["mapping_type"].eq("emission_weight").all():
            raise ValueError(
                f"All crosswalk rows for {activity} must have "
                "mapping_type='emission_weight'."
            )

    # ---------------------------------------------------------
    # 5. Separate direct mappings from allocated mappings
    # ---------------------------------------------------------

    direct_map = crosswalk.loc[
        crosswalk["mapping_type"].eq("direct")
    ].copy()

    allocation_map = crosswalk.loc[
        crosswalk["mapping_type"].eq("emission_weight")
    ].copy()

    # A direct mapping must have exactly one destination.
    direct_counts = (
        direct_map.groupby("aea_activity_code")["icio_sector_code"]
        .nunique()
    )

    if (direct_counts > 1).any():
        bad = direct_counts[direct_counts > 1].index.tolist()

        raise ValueError(
            "Activities classified as direct have multiple ICIO "
            f"destinations: {bad}"
        )

    # This function only supports the three explicitly defined
    # aggregate allocations. Other emission_weight activities need
    # their own allocation methodology and weights.
    unsupported_allocations = (
        set(allocation_map["aea_activity_code"])
        - ALLOCATED_ACTIVITIES
    )

    if unsupported_allocations:
        raise ValueError(
            "The crosswalk contains emission_weight activities without "
            "an allocation rule in this function: "
            f"{sorted(unsupported_allocations)}. "
            "Either add their weights and allocation rules or leave "
            "them out of this allocation workflow."
        )

    # ---------------------------------------------------------
    # 6. Map direct activities
    # ---------------------------------------------------------

    direct = aea.merge(
        direct_map,
        left_on="ACTIVITY",
        right_on="aea_activity_code",
        how="inner",
        validate="many_to_one",
    )

    direct["allocation_weight"] = 1.0
    direct["allocated_emissions"] = direct["OBS_VALUE"]

    direct["allocation_method"] = "direct"

    # ---------------------------------------------------------
    # 7. Allocate B, C24, and C30 using emissions weights
    # ---------------------------------------------------------

    aggregate = aea.loc[
        aea["ACTIVITY"].isin(ALLOCATED_ACTIVITIES)
    ].copy()

    if not aggregate.empty:

        # Each selected aggregate activity must have a corresponding
        # allocation mapping.
        mapped_activities = set(allocation_map["aea_activity_code"])
        missing_mappings = (
            set(aggregate["ACTIVITY"]) - mapped_activities
        )

        if missing_mappings:
            raise ValueError(
                "No allocation crosswalk found for activities: "
                f"{sorted(missing_mappings)}"
            )

        # Attach the destination sectors.
        aggregate = aggregate.merge(
            allocation_map,
            left_on="ACTIVITY",
            right_on="aea_activity_code",
            how="left",
            validate="many_to_many",
        )

        # Restrict weights to the same country, year, activity,
        # and destination sector.
        aggregate = aggregate.merge(
            weights,
            left_on=[
                "REF_AREA",
                "TIME_PERIOD",
                "ACTIVITY",
                "icio_sector_code",
            ],
            right_on=[
                "REF_AREA",
                "TIME_PERIOD",
                "aea_activity_code",
                "icio_sector_code",
            ],
            how="left",
            validate="many_to_one",
            suffixes=("", "_weight"),
        )

        if aggregate["emission_weight"].isna().any():
            missing = aggregate.loc[
                aggregate["emission_weight"].isna(),
                [
                    "REF_AREA",
                    "TIME_PERIOD",
                    "ACTIVITY",
                    "icio_sector_code",
                ],
            ].drop_duplicates()

            raise ValueError(
                "Missing emissions allocation weights for:\n"
                f"{missing.to_string(index=False)}"
            )

        # Check that every country-year-activity has positive total weight.
        weight_keys = ["REF_AREA", "TIME_PERIOD", "ACTIVITY"]

        weight_sums = (
            aggregate.groupby(weight_keys)["emission_weight"]
            .sum()
        )

        if (weight_sums <= 0).any():
            bad = weight_sums[weight_sums <= 0]

            raise ValueError(
                "Allocation weights must have a positive sum for every "
                "country-year-activity:\n"
                f"{bad.to_string()}"
            )

        # Normalize within country-year-activity.
        aggregate["allocation_weight"] = (
            aggregate["emission_weight"]
            / aggregate.groupby(weight_keys)["emission_weight"]
                .transform("sum")
        )

        aggregate["allocated_emissions"] = (
            aggregate["OBS_VALUE"]
            * aggregate["allocation_weight"]
        )

        aggregate["allocation_method"] = "aea_emissions_weight"

    # ---------------------------------------------------------
    # 8. Combine allocated and directly mapped observations
    # ---------------------------------------------------------

    result = pd.concat(
        [direct, aggregate],
        ignore_index=True,
        sort=False,
    )

    # ---------------------------------------------------------
    # 9. Check that no AEA emissions observations were silently lost
    # ---------------------------------------------------------

    eligible_activities = (
        set(direct_map["aea_activity_code"])
        | ALLOCATED_ACTIVITIES
    )

    eligible_aea = aea.loc[
        aea["ACTIVITY"].isin(eligible_activities)
    ]

    mapped_keys = set(
        zip(
            result["REF_AREA"],
            result["TIME_PERIOD"],
            result["ACTIVITY"],
        )
    )

    eligible_keys = set(
        zip(
            eligible_aea["REF_AREA"],
            eligible_aea["TIME_PERIOD"],
            eligible_aea["ACTIVITY"],
        )
    )

    missing_keys = eligible_keys - mapped_keys

    if missing_keys:
        raise ValueError(
            "Some eligible AEA observations were not allocated: "
            f"{sorted(missing_keys)[:20]}"
        )

    # ---------------------------------------------------------
    # 10. Verify emissions conservation
    # ---------------------------------------------------------

    allocated_totals = (
        result.groupby(aea_keys)["allocated_emissions"]
        .sum()
        .rename("allocated_total")
    )

    original_totals = (
        eligible_aea.set_index(aea_keys)["OBS_VALUE"]
        .rename("original_total")
    )

    check = pd.concat(
        [original_totals, allocated_totals],
        axis=1,
    ).fillna(0)

    tolerance = 1e-8

    if not np.allclose(
        check["original_total"],
        check["allocated_total"],
        rtol=tolerance,
        atol=tolerance,
    ):
        raise ValueError(
            "Emissions conservation check failed:\n"
            f"{check.loc[~np.isclose(check['original_total'], check['allocated_total'])].head(10)}"
        )

    # ---------------------------------------------------------
    # 11. Create the ICIO industry key
    # ---------------------------------------------------------

    result["io_key"] = (
        result["REF_AREA"].astype(str)
        + "_"
        + result["icio_sector_code"].astype(str)
    )

    result, allocated = allocate_emissions(
        aea,
        crosswalk,
        weights,
    )


# ---------------------------------------------------------------------
# VALIDATE ALLOCATION
# ---------------------------------------------------------------------

def validate_allocation(allocated):

    if allocated.empty:

        print(
            "\nNo aggregate AEA observations required allocation."
        )

        return

    group_keys = [
        "REF_AREA",
        "TIME_PERIOD",
        "ACTIVITY",
    ]

    check = allocated.groupby(group_keys).agg(
        original_emissions=("OBS_VALUE", "first"),
        allocated_emissions=("allocated_emissions", "sum"),
        weight_sum=("allocation_weight", "sum"),
    )

    check["emissions_difference"] = (
        check["allocated_emissions"]
        - check["original_emissions"]
    )

    if not np.allclose(
        check["weight_sum"],
        1.0,
        atol=1e-10,
    ):

        raise ValueError(
            "Allocation weights do not sum to one."
        )

    if not np.allclose(
        check["allocated_emissions"],
        check["original_emissions"],
        rtol=1e-10,
        atol=1e-8,
    ):

        raise ValueError(
            "Allocated emissions do not reconcile to original AEA totals."
        )

    print("\nAllocation reconciliation:")

    print(check.to_string())


# ---------------------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------------------

if __name__ == "__main__":

    aea, crosswalk, weights = load_inputs()

    aea = prepare_aea(aea)

    crosswalk = prepare_crosswalk(crosswalk)

    weights = prepare_weights(weights)

    result, allocated = allocate_emissions(
        aea,
        crosswalk,
        weights,
    )

    validate_allocation(allocated)

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    if not allocated.empty:

        allocated.to_csv(
            DIAGNOSTICS_FILE,
            index=False,
        )

    print("\nAEA-to-ICIO emissions extension saved:")
    print(OUTPUT_FILE)

    if not allocated.empty:

        print("\nAllocation diagnostics saved:")
        print(DIAGNOSTICS_FILE)


# ---------------------------------------------------------------------
# Physical metrics and output-shock calibration
# ---------------------------------------------------------------------

def build_scenarios(
    output,
    demand,
    prices,
    grid,
    water,
    customer,
    owner,
    dc_map,
    A,
):
    """Create physical metrics and direct/electricity-implied output shocks."""
    physical = (
        demand.merge(prices, on=["country", "year"], how="left", validate="one_to_one")
        .merge(grid, on=["country", "year"], how="left", validate="one_to_one")
        .merge(water, on=["country", "year"], how="left", validate="one_to_one")
    )

    if physical[
        ["price_2022_usd_mwh", "kgco2_per_kwh", "pue", "wue_l_per_kwh_it"]
    ].isna().any().any():
        raise ValueError(
            "Electricity demand is missing price, grid, PUE or WUE observations."
        )

    physical["facility_electricity_gwh"] = np.where(
        physical["demand_basis"].eq("facility"),
        physical["electricity_gwh"],
        physical["electricity_gwh"] * physical["pue"],
    )

    physical["it_electricity_gwh"] = np.where(
        physical["demand_basis"].eq("it"),
        physical["electricity_gwh"],
        physical["electricity_gwh"] / physical["pue"],
    )

    # GWh x kg/kWh = thousand tonnes.
    physical["direct_grid_emissions_ktco2"] = (
        physical["facility_electricity_gwh"]
        * physical["kgco2_per_kwh"]
    )

    # GWh x L/kWh = million litres.
    physical["operational_water_ml"] = (
        physical["it_electricity_gwh"]
        * physical["wue_l_per_kwh_it"]
    )

    physical = physical.sort_values(["country", "year"]).reset_index(drop=True)

    physical["facility_gwh_yoy"] = (
        physical.groupby("country")["facility_electricity_gwh"].diff()
    )

    physical["it_gwh_yoy"] = (
        physical.groupby("country")["it_electricity_gwh"].diff()
    )

    physical["facility_gwh_yoy"] = physical["facility_gwh_yoy"].fillna(0.0)
    physical["it_gwh_yoy"] = physical["it_gwh_yoy"].fillna(0.0)

    physical["facility_gwh_cumulative"] = (
        physical["facility_electricity_gwh"]
        - physical.groupby("country")["facility_electricity_gwh"].transform("first")
    )

    physical["it_gwh_cumulative"] = (
        physical["it_electricity_gwh"]
        - physical.groupby("country")["it_electricity_gwh"].transform("first")
    )

    # Merge the direct data-centre output path with ancillary shares.
    scenarios = (
        output.merge(customer, on=["country", "year"], how="left", validate="one_to_one")
        .merge(
            physical[
                [
                    "country",
                    "year",
                    "facility_electricity_gwh",
                    "it_electricity_gwh",
                    "facility_gwh_yoy",
                    "it_gwh_yoy",
                    "facility_gwh_cumulative",
                    "it_gwh_cumulative",
                    "price_2022_usd_mwh",
                    "kgco2_per_kwh",
                    "pue",
                    "wue_l_per_kwh_it",
                    "direct_grid_emissions_ktco2",
                    "operational_water_ml",
                ]
            ],
            on=["country", "year"],
            how="left",
            validate="one_to_one",
        )
    )

    if scenarios["us_customer_share"].isna().any():
        raise ValueError(
            "Missing US customer shares for one or more output scenario years."
        )

    if owner is not None:
        scenarios = scenarios.merge(
            owner,
            on=["country", "year"],
            how="left",
            validate="one_to_one",
        )
    else:
        scenarios["us_parent_operator_share"] = np.nan

    scenarios = scenarios.sort_values(["country", "year"]).reset_index(drop=True)

    # Direct output scenario: output levels are in constant 2022 USD mn.
    scenarios["dc_output_yoy_usd_mn"] = (
        scenarios.groupby("country")["dc_output_usd_mn"].diff().fillna(0.0)
    )

    scenarios["dc_output_cumulative_usd_mn"] = (
        scenarios["dc_output_usd_mn"]
        - scenarios.groupby("country")["dc_output_usd_mn"].transform("first")
    )

    dc_lookup = dc_map.set_index("country")

    electricity_coefficients = []

    for row in scenarios.itertuples(index=False):
        cfg = dc_lookup.loc[row.country]

        coefficient = float(
            A.loc[cfg.electricity_io_key, cfg.dc_proxy_io_key]
        )

        if not np.isfinite(coefficient) or coefficient <= 0:
            raise ValueError(
                f"Invalid electricity coefficient for {row.country}: "
                f"A[{cfg.electricity_io_key}, {cfg.dc_proxy_io_key}]={coefficient}"
            )

        electricity_coefficients.append(coefficient)

    scenarios["electricity_input_coefficient"] = electricity_coefficients

    # Electricity-implied output estimate.
    # GWh x USD/MWh / 1000 = USD mn.
    scenarios["electricity_expenditure_yoy_usd_mn"] = (
        scenarios["facility_gwh_yoy"]
        * scenarios["price_2022_usd_mwh"]
        / 1000.0
    )

    scenarios["electricity_expenditure_cumulative_usd_mn"] = (
        scenarios["facility_gwh_cumulative"]
        * scenarios["price_2022_usd_mwh"]
        / 1000.0
    )

    scenarios["electricity_implied_output_yoy_usd_mn"] = (
        scenarios["electricity_expenditure_yoy_usd_mn"]
        / scenarios["electricity_input_coefficient"]
    )

    scenarios["electricity_implied_output_cumulative_usd_mn"] = (
        scenarios["electricity_expenditure_cumulative_usd_mn"]
        / scenarios["electricity_input_coefficient"]
    )

    # Attribution shares apply to the direct output scenario.
    for basis in ["yoy", "cumulative"]:
        output_col = f"dc_output_{basis}_usd_mn"

        scenarios[f"us_customer_output_{basis}_usd_mn"] = (
            scenarios[output_col] * scenarios["us_customer_share"]
        )

        scenarios[f"us_parent_output_{basis}_usd_mn"] = (
            scenarios[output_col]
            * scenarios["us_parent_operator_share"]
        )

    return physical, scenarios


# ---------------------------------------------------------------------
# Leontief calculation
# ---------------------------------------------------------------------

def gross_output_to_final_demand(
    lu,
    piv,
    n: int,
    target_indices: list[int],
    delta_x_target: np.ndarray,
):
    """
    Convert target gross-output changes into final-demand changes.

    If delta_x_target is the desired change in gross output for the target
    industries, calculate:

        delta_y_target = inverse(L_SS) @ delta_x_target

        delta_x = L[:, S] @ delta_y_target

    This ensures that the resulting gross-output changes in the target
    industries equal the specified targets.
    """
    L_columns = lu_solve(
        (lu, piv),
        np.eye(n, dtype=float)[:, target_indices],
    )

    L_ss = L_columns[target_indices, :]

    delta_y_target = np.linalg.solve(L_ss, delta_x_target)

    delta_y = np.zeros(n, dtype=float)
    delta_y[target_indices] = delta_y_target

    delta_x = lu_solve((lu, piv), delta_y)

    if not np.allclose(
        delta_x[target_indices],
        delta_x_target,
        rtol=1e-7,
        atol=1e-6,
    ):
        raise RuntimeError(
            "Leontief calculation did not reproduce the specified "
            "target-sector gross-output changes."
        )

    return delta_y, delta_x


def run_leontief(
    A: pd.DataFrame,
    X: pd.Series,
    sector_map: pd.DataFrame,
    extension: pd.DataFrame,
    dc_map: pd.DataFrame,
    scenarios: pd.DataFrame,
    allow_missing_aea: bool,
):
    """Calculate upstream gross output and direct/embodied GHG emissions."""
    keys = A.index.tolist()
    n = len(keys)

    LOG.info("Factorizing Leontief system with %s industries.", n)

    M = np.eye(n, dtype=float) - A.to_numpy(dtype=float)
    lu, piv = lu_factor(M, check_finite=True)

    extension = extension.set_index("io_key").loc[keys]

    intensity = extension["ghg_tonnes_per_usd_mn"].to_numpy(dtype=float)

    if np.isnan(intensity).any() and not allow_missing_aea:
        raise ValueError("Missing emissions intensities remain in the extension.")

    # Under explicit allow_missing_aea, missing values are set to zero only
    # for computation. Results are incomplete and must be reported as such.
    intensity_for_calculation = np.nan_to_num(
        intensity,
        nan=0.0,
    )

    sector_lookup = sector_map.set_index("io_key").loc[keys]
    country_by_key = sector_lookup["country"].to_numpy()

    electricity_codes = set()
    if "is_electricity_sector" in sector_map.columns:
        flag = sector_map["is_electricity_sector"].astype(str).str.lower()
        electricity_codes = set(
            sector_map.loc[flag.isin(["true", "1", "yes"]), "io_key"]
        )

    target_key_by_country = (
        dc_map.set_index("country")["dc_proxy_io_key"].to_dict()
    )

    target_index_by_country = {
        country: keys.index(key)
        for country, key in target_key_by_country.items()
    }

    detail_rows = []
    summary_rows = []
    final_demand_rows = []

    for row in scenarios.itertuples(index=False):
        country = row.country
        year = int(row.year)

        target_index = target_index_by_country[country]

        shock_definitions = {
            "all_expansion": row.dc_output_yoy_usd_mn,
            "us_customer_demand": row.us_customer_output_yoy_usd_mn,
            "us_parent_owned_operator": row.us_parent_output_yoy_usd_mn,
        }

        cumulative_definitions = {
            "all_expansion": row.dc_output_cumulative_usd_mn,
            "us_customer_demand": row.us_customer_output_cumulative_usd_mn,
            "us_parent_owned_operator": row.us_parent_output_cumulative_usd_mn,
        }

        for shock_basis, definitions in [
            ("annual_increment", shock_definitions),
            ("cumulative_vs_first_year", cumulative_definitions),
        ]:
            for attribution, shock in definitions.items():
                if pd.isna(shock):
                    continue

                shock = float(shock)

                # Skip exact zero shocks to avoid unnecessary solves.
                if abs(shock) < 1e-12:
                    continue

                delta_y, delta_x = gross_output_to_final_demand(
                    lu=lu,
                    piv=piv,
                    n=n,
                    target_indices=[target_index],
                    delta_x_target=np.array([shock], dtype=float),
                )

                emissions = delta_x * intensity_for_calculation

                origin_df = pd.DataFrame({
                    "year": year,
                    "shock_basis": shock_basis,
                    "shock_destination": country,
                    "attribution": attribution,
                    "io_key": keys,
                    "output_change_usd_mn": delta_x,
                    "ghg_tonnes_co2e": emissions,
                    "emissions_intensity_tonnes_per_usd_mn": intensity_for_calculation,
                }).merge(
                    sector_map,
                    on="io_key",
                    how="left",
                    validate="many_to_one",
                )

                detail_rows.append(origin_df)

                total_emissions = float(np.sum(emissions))
                us_emissions = float(
                    emissions[country_by_key == "USA"].sum()
                )
                domestic_emissions = float(
                    emissions[country_by_key == country].sum()
                )

                electricity_emissions = (
                    float(emissions[keys.index(k)] for k in [])
                    if False else 0.0
                )

                if electricity_codes:
                    electricity_indices = [
                        i for i, key in enumerate(keys)
                        if key in electricity_codes
                    ]
                    electricity_emissions = float(
                        emissions[electricity_indices].sum()
                    )

                summary_rows.append({
                    "year": year,
                    "shock_basis": shock_basis,
                    "shock_destination": country,
                    "attribution": attribution,
                    "target_gross_output_change_usd_mn": shock,
                    "global_upstream_ghg_tonnes_co2e": total_emissions,
                    "us_origin_ghg_tonnes_co2e": us_emissions,
                    "destination_country_origin_ghg_tonnes_co2e": domestic_emissions,
                    "foreign_origin_ghg_tonnes_co2e": total_emissions - domestic_emissions,
                    "electricity_sector_origin_ghg_tonnes_co2e": electricity_emissions,
                    "us_origin_share": (
                        us_emissions / total_emissions
                        if abs(total_emissions) > 1e-12 else np.nan
                    ),
                    "destination_origin_share": (
                        domestic_emissions / total_emissions
                        if abs(total_emissions) > 1e-12 else np.nan
                    ),
                })

                final_demand_rows.append(pd.DataFrame({
                    "year": year,
                    "shock_basis": shock_basis,
                    "shock_destination": country,
                    "attribution": attribution,
                    "io_key": keys,
                    "final_demand_change_usd_mn": delta_y,
                }))

    if not detail_rows:
        raise ValueError(
            "No Leontief results were generated. Check whether the output "
            "scenario contains non-zero changes after its baseline year."
        )

    detail = pd.concat(detail_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    final_demand = pd.concat(final_demand_rows, ignore_index=True)

    return detail, summary, final_demand


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))

    parser.add_argument("--icio-year", type=int, default=2022)

    parser.add_argument("--A-file", type=Path, required=True)
    parser.add_argument("--X-file", type=Path, required=True)
    parser.add_argument("--FD-file", type=Path, required=True)
    parser.add_argument("--AEA-file", type=Path, required=True)

    parser.add_argument("--A-object", type=str, default=None)
    parser.add_argument("--X-object", type=str, default=None)
    parser.add_argument("--FD-object", type=str, default=None)
    parser.add_argument("--X-value-column", type=str, default=None)
    
    parser.add_argument(
        "--icio-identity-tolerance",
        type=float,
        default=0.02,
        help="Maximum relative discrepancy for X = A@X + FD.",
    )

    # AEA column names and dimension filters.
    parser.add_argument("--aea-country-column", default="REF_AREA")
    parser.add_argument("--aea-activity-column", default="ACTIVITY")
    parser.add_argument("--aea-year-column", default="TIME_PERIOD")
    parser.add_argument("--aea-value-column", default="OBS_VALUE")

    parser.add_argument("--aea-pollutant-column", default=None)
    parser.add_argument("--aea-pollutant-value", default=None)

    parser.add_argument("--aea-measure-column", default=None)
    parser.add_argument("--aea-measure-value", default=None)

    parser.add_argument("--aea-unit-column", default=None)
    parser.add_argument("--aea-unit-value", default=None)

    parser.add_argument(
        "--aea-unit-to-tonnes",
        type=float,
        default=1.0,
        help="Conversion multiplier from the selected AEA unit to tonnes.",
    )

    parser.add_argument(
        "--allow-missing-aea",
        action="store_true",
        help=(
            "Allow missing AEA industry observations. Missing intensities "
            "are excluded from the emissions calculation, so totals are incomplete."
        ),
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Load and validate data without calculating Leontief footprints.",
    )

    parser.add_argument("--log-level", default="INFO")

    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Starting data-centre footprint model.")
    LOG.info("ICIO reference year: %s", args.icio_year)

    # 1. Load ICIO.
    A, X, FD = load_icio(args)

    LOG.info("ICIO A dimensions: %s x %s", *A.shape)
    LOG.info("ICIO X industries: %s", len(X))
    LOG.info("ICIO FD dimensions: %s", FD.shape)

    # 2. Load sector maps.
    sector_map, dc_map = load_sector_maps(A)

    # 3. Load scenario inputs.
    (
        output,
        demand,
        prices,
        grid,
        water,
        customer,
        owner,
    ) = load_scenario_inputs()

    # Weights for allocating AEA emissions across ICIO industries.
    aea_weights = load_aea_emission_weights(
        reference_year=args.icio_year,
    )
    # Build AEA emissions extension.
    extension, coverage, coverage_summary, allocation = (
        build_aea_extension(
            args,
            X=X,
            sector_map=sector_map,
            weights=aea_weights,
        )
    )

    # Save input and mapping diagnostics.
    coverage.to_csv(
        args.output_dir / "aea_emissions_coverage.csv",
        index=False,
    )

    coverage_summary.to_csv(
        args.output_dir / "aea_emissions_coverage_summary.csv",
        index=False,
    )

    allocation.to_csv(
        args.output_dir / "aea_crosswalk_allocation.csv",
        index=False,
    )

    extension.to_csv(
        args.output_dir / "icio_emissions_extension.csv",
        index=False,
    )

    # 5. Build physical metrics and output scenarios.
    physical, scenarios = build_scenarios(
        output=output,
        demand=demand,
        prices=prices,
        grid=grid,
        water=water,
        customer=customer,
        owner=owner,
        dc_map=dc_map,
        A=A,
    )

    physical.to_csv(
        args.output_dir / "physical_operational_metrics.csv",
        index=False,
    )

    scenarios.to_csv(
        args.output_dir / "dc_output_and_electricity_calibration.csv",
        index=False,
    )

    # Separate output-shock comparison table.
    comparison = scenarios[
        [
            "country",
            "year",
            "dc_output_yoy_usd_mn",
            "electricity_implied_output_yoy_usd_mn",
            "dc_output_cumulative_usd_mn",
            "electricity_implied_output_cumulative_usd_mn",
        ]
    ].copy()

    comparison["annual_difference_usd_mn"] = (
        comparison["dc_output_yoy_usd_mn"]
        - comparison["electricity_implied_output_yoy_usd_mn"]
    )

    comparison["annual_difference_pct"] = np.where(
        comparison["electricity_implied_output_yoy_usd_mn"].abs() > 1e-12,
        100.0 * comparison["annual_difference_usd_mn"]
        / comparison["electricity_implied_output_yoy_usd_mn"],
        np.nan,
    )

    comparison.to_csv(
        args.output_dir / "output_shock_comparison.csv",
        index=False,
    )

    LOG.info("Input validation and scenario construction completed.")

    if args.check_only:
        LOG.info("Check-only selected; stopping before Leontief calculation.")
        return

    # 6. Calculate global embodied-emissions footprints.
    detail, summary, final_demand = run_leontief(
        A=A,
        X=X,
        sector_map=sector_map,
        extension=extension,
        dc_map=dc_map,
        scenarios=scenarios,
        allow_missing_aea=args.allow_missing_aea,
    )

    detail.to_parquet(
        args.output_dir / "upstream_emissions_by_origin_industry.parquet",
        index=False,
    )

    summary.to_csv(
        args.output_dir / "upstream_emissions_summary.csv",
        index=False,
    )

    final_demand.to_parquet(
        args.output_dir / "implied_final_demand_shocks.parquet",
        index=False,
    )

    by_origin = (
        detail.groupby(
            [
                "year",
                "shock_basis",
                "shock_destination",
                "attribution",
                "country",
                "sector_code",
                "sector_label",
            ],
            as_index=False,
        )[
            ["output_change_usd_mn", "ghg_tonnes_co2e"]
        ].sum()
    )

    by_origin.to_csv(
        args.output_dir / "upstream_emissions_by_origin_country_sector.csv",
        index=False,
    )

    LOG.info("Model completed successfully.")
    LOG.info("Results written to %s", args.output_dir.resolve())


if __name__ == "__main__":
    main()
