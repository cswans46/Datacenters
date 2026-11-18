
#!/usr/bin/env python3
"""
Global data-centre expansion footprint model using OECD ICIO
and a hybrid emissions extension.

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
Direct industry emissions are loaded from the completed hybrid emissions
extension produced by build_hybrid_emissions_extension.py.

The extension combines the appropriate country-level emissions sources
and mappings upstream. This model does not independently process AEA data,
allocate aggregate AEA activities, or reconstruct emissions by ICIO sector.

Industry emissions intensities are calculated as:

    emissions intensity = direct industry GHG emissions / ICIO gross output

The source field from the hybrid extension is retained for auditing.

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
- ICIO A, X, FD and the hybrid emissions extension refer to the same
  reference year.
- Monetary values used with ICIO coefficients are in constant 2022 USD.
- The ICIO production structure and emissions intensities are held fixed
  over the scenario horizon.
- J62_63 (or another configured sector) is an aggregate proxy, not a
  separately identified data-centre industry.
- The hybrid emissions extension is authoritative for direct industry
  emissions and their source attribution.
- The hybrid extension must cover every ICIO industry exactly once.
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
            year_value = next(
                value
                for value in obj.coords[year_dim].values
                if str(value) == str(icio_year)
            )

            obj = obj.sel({year_dim: year_value})

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
    
    def validate_vector_keys(result: pd.Series) -> pd.Series:
        result = result.copy()
        result.index = result.index.astype(str).str.strip()

        if result.index.has_duplicates:
            raise ValueError(f"{name} contains duplicate industry keys.")

        if expected_keys is not None:
            actual = set(result.index)
            missing = sorted(expected_keys - actual)
            extra = sorted(actual - expected_keys)

            if missing or extra:
                raise ValueError(
                    f"{name} industry keys do not match ICIO. "
                    f"Missing={missing[:10]}; extra={extra[:10]}"
                )

            result = result.reindex(keys)

        return result.astype(float)

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

            return validate_vector_keys(result)


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

            return validate_vector_keys(result)

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

            return validate_vector_keys(result)

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

        return validate_vector_keys(result)

    # ---------------------------------------------------------
    # Case 4: One-column vector with industry keys in the index.
    # ---------------------------------------------------------
    if df.shape[1] == 1:
        result = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        result.index = df.index.astype(str)

        if result.isna().any():
            raise ValueError(f"{name} contains missing or non-numeric values.")

        return validate_vector_keys(result)

    # ---------------------------------------------------------
    # Case 5: One-row vector with industry keys in columns.
    # ---------------------------------------------------------
    if df.shape[0] == 1:
        result = pd.to_numeric(df.iloc[0], errors="coerce")
        result.index = df.columns.astype(str)

        if result.isna().any():
            raise ValueError(f"{name} contains missing or non-numeric values.")

        return validate_vector_keys(result)

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

    expected = set(map(str, industry_keys))

    row_match = set(fd.index) == expected
    col_match = set(fd.columns) == expected

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
        key_column=args.X_key_column,
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


def load_sector_maps(A: pd.DataFrame, data_dir: Path):
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
            "is_electricity_sector": sector_code in {"35", "D35"},
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
    dc_map = read_table(data_dir / "dc_sector_map.csv")

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
    
    check_unique(
        dc_map,
        ["country"],
        "data-centre sector map",
    )

    if dc_map["dc_proxy_io_key"].duplicated().any():
        raise ValueError(
            "Multiple countries are mapped to the same data-centre proxy sector."
        )

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
            
        if not row.dc_proxy_io_key.startswith(f"{row.country}_"):
            raise ValueError(
                f"Data-centre proxy for {row.country} does not belong to "
                f"that country: {row.dc_proxy_io_key}"
            )

        if not row.electricity_io_key.startswith(f"{row.country}_"):
            raise ValueError(
                f"Electricity industry for {row.country} does not belong to "
                f"that country: {row.electricity_io_key}"
            )
            
        electricity_sector_code = row.electricity_io_key.split("_", 1)[1]

        if electricity_sector_code not in {"35", "D35"}:
            raise ValueError(
                f"Expected a sector 35 electricity supplier for {row.country}, "
                f"found {row.electricity_io_key}"
            )

    LOG.info(
        "Loaded data-centre mappings for: %s",
        ", ".join(dc_map["country"].tolist()),
    )

    return sector_map, dc_map

def load_scenario_inputs(data_dir: Path):
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

        non_integer_years = df["year"].notna() & (
            df["year"] != np.floor(df["year"])
        )

        if non_integer_years.any():
            raise ValueError(
                f"{name} contains non-integer years: "
                f"{df.loc[non_integer_years, 'year'].tolist()[:10]}"
            )

        df["year"] = df["year"].astype(int)

        return df

    def clean_numeric_observations(df, columns, name):
        """Convert required numeric columns; drop genuinely missing observations."""
        df = df.copy()

        for column in columns:
            original = df[column]
            converted = pd.to_numeric(original, errors="coerce")

            invalid = original.notna() & converted.isna()

            if invalid.any():
                examples = original.loc[invalid].head(10).tolist()
                raise ValueError(
                    f"{name}.{column} contains non-numeric, non-missing "
                    f"values: {examples}"
                )

            df[column] = converted

        missing = df[columns].isna().any(axis=1)

        if missing.any():
            LOG.warning(
                "Dropping %s rows with missing values in %s: %s",
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

    output = read_table(data_dir / "dc_output_scenarios.csv")

    require_columns(
        output,
        [
            "country",
            "year",
            "dc_capacity_mw",
            "price_mn_MW_2022",
            "output_mn_usd_per_mw_2022",
        ],
        "dc_output_scenarios.csv",
    )

    output = clean_keys(output, "dc_output_scenarios.csv")

    output = clean_numeric_observations(
        output,
        [
            "dc_capacity_mw",
            "price_mn_MW_2022",
            "output_mn_usd_per_mw_2022",
        ],
        "dc_output_scenarios.csv",
    )

    validate_country_year(output, "dc_output_scenarios.csv")
    output["dc_capacity_mw"] = numeric_column(
        output,
        "dc_capacity_mw",
        "dc_output_scenarios",
    )

    output["price_mn_MW_2022"] = numeric_column(
        output,
        "price_mn_MW_2022",
        "dc_output_scenarios",
    )

    output["output_mn_usd_per_mw_2022"] = numeric_column(
        output,
        "output_mn_usd_per_mw_2022",
        "dc_output_scenarios",
    )

    if (output["dc_capacity_mw"] < 0).any():
        raise ValueError("Data-centre capacity cannot be negative.")

    if (output["price_mn_MW_2022"] <= 0).any():
        raise ValueError("Output value per MW must be positive.")

    if not REQUIRED_COUNTRIES.issubset(set(output["country"])):
        raise ValueError("dc_output_scenarios.csv must include CAN and IND.")

    # Calculate total data-centre output in constant 2022 USD millions.
    output["dc_output_usd_mn"] = (
        output["dc_capacity_mw"]
        * output["price_mn_MW_2022"]
    )

    # Check the supplied output column against the independently
    # reconstructed total output. This also documents its meaning.
    if not np.allclose(
        output["dc_output_usd_mn"],
        output["output_mn_usd_per_mw_2022"],
        rtol=1e-6,
        atol=1e-6,
    ):
        raise ValueError(
            "The supplied output_mn_usd_per_mw_2022 column does not match "
            "capacity multiplied by price_mn_MW_2022. Check the input units "
            "and output calculation."
        )
    
    for country, group in output.groupby("country"):
        years = np.sort(group["year"].unique())

        if len(years) > 1 and not np.all(np.diff(years) == 1):
            raise ValueError(
                f"Output scenario years for {country} are not consecutive: "
                f"{years.tolist()}"
            )

    # ------------------------------------------------------------------
    # 2. Electricity demand
    # ------------------------------------------------------------------

    demand = read_table(data_dir / "dc_electricity_demand.csv")

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

    prices = read_table(data_dir / "electricity_prices.csv")

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

    
    # ------------------------------------------------------------------
    # 4. Grid and natural-gas emissions factors
    # ------------------------------------------------------------------

    grid = read_table(data_dir / "grid_emissions_factors.csv")

    require_columns(
        grid,
        [
            "country",
            "year",
            "kgco2_per_kwh",
            "kgco2_per_kwh_natgas",
        ],
        "grid_emissions_factors.csv",
    )

    grid = clean_keys(grid, "grid_emissions_factors.csv")

    # The average-grid factor is required for all observations.
    grid = clean_numeric_observations(
        grid,
        ["kgco2_per_kwh"],
        "grid_emissions_factors.csv",
    )

    # The natural-gas factor is only required for Canadian observations.
    original_natgas = grid["kgco2_per_kwh_natgas"]
    grid["kgco2_per_kwh_natgas"] = pd.to_numeric(
        original_natgas,
        errors="coerce",
    )

    invalid_natgas = (
        original_natgas.notna()
        & grid["kgco2_per_kwh_natgas"].isna()
    )

    if invalid_natgas.any():
        raise ValueError(
            "grid_emissions_factors.csv contains non-numeric "
            "natural-gas emissions factors."
        )

    validate_country_year(grid, "grid_emissions_factors.csv")

    grid["kgco2_per_kwh"] = numeric_column(
        grid,
        "kgco2_per_kwh",
        "grid_emissions_factors",
    )

    if (grid["kgco2_per_kwh"] < 0).any():
        raise ValueError("Average-grid emissions factors cannot be negative.")

    available_natgas = grid["kgco2_per_kwh_natgas"].notna()

    if (
        grid.loc[available_natgas, "kgco2_per_kwh_natgas"] < 0
    ).any():
        raise ValueError("Natural-gas emissions factors cannot be negative.")

    # Every Canadian observation must have a natural-gas factor.
    missing_canadian_natgas = (
        grid["country"].eq("CAN")
        & grid["kgco2_per_kwh_natgas"].isna()
    )

    if missing_canadian_natgas.any():
        raise ValueError(
            "Missing Canadian natural-gas emissions factors for: "
            f"{grid.loc[missing_canadian_natgas, ['country', 'year']].to_dict('records')}"
        )


    # ------------------------------------------------------------------
    # 5. Water coefficients
    # ------------------------------------------------------------------

    water = read_table(data_dir / "water_coefficients.csv")

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

    customer = read_table(data_dir / "us_customer_shares.csv")

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

    owner_path = data_dir / "us_parent_operator_shares.csv"

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

def load_emissions_extension(
    emissions_file: Path,
    X: pd.Series,
    sector_map: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the completed hybrid emissions extension and calculate ICIO
    industry emissions intensities.

    The hybrid extension is the authoritative source of direct industry
    emissions. No AEA mapping or emissions allocation is performed here.

    Expected input columns:
        io_key
        ghg_tonnes_co2e
        source

    Returns
    -------
    extension:
        One row per ICIO industry, including gross output, direct emissions,
        emissions intensity, and emissions source.

    coverage_summary:
        Country-level coverage diagnostics.
    """
    required = {
        "io_key",
        "ghg_tonnes_co2e",
        "source",
    }
    
    if not emissions_file.is_file():
        raise FileNotFoundError(
            f"Hybrid emissions extension not found: {emissions_file}"
        )

    emissions = pd.read_csv(emissions_file, low_memory=False)

    missing_columns = required.difference(emissions.columns)
    if missing_columns:
        raise ValueError(
            "Hybrid emissions extension is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    emissions = emissions[
        ["io_key", "ghg_tonnes_co2e", "source"]
    ].copy()

    emissions["io_key"] = (
        emissions["io_key"].astype("string").str.strip()
    )
    emissions["source"] = (
        emissions["source"].astype("string").str.strip()
    )
    emissions["ghg_tonnes_co2e"] = pd.to_numeric(
        emissions["ghg_tonnes_co2e"],
        errors="raise",
    )

    if emissions["io_key"].isna().any() or emissions["io_key"].eq("").any():
        raise ValueError(
            "Hybrid emissions extension contains missing or empty io_key values."
        )

    if emissions["source"].isna().any() or emissions["source"].eq("").any():
        raise ValueError(
            "Hybrid emissions extension contains missing or empty source labels."
        )

    if not np.isfinite(emissions["ghg_tonnes_co2e"]).all():
        raise ValueError(
            "Hybrid emissions extension contains non-finite emissions values."
        )

    # Negative emissions are retained: the hybrid extension is responsible
    # for determining and documenting their treatment.
    check_unique(
        emissions,
        ["io_key"],
        "Hybrid emissions extension",
    )

    icio_keys = pd.Index(X.index.astype(str), name="io_key")
    emissions_keys = pd.Index(emissions["io_key"], name="io_key")

    missing_keys = icio_keys.difference(emissions_keys)
    extra_keys = emissions_keys.difference(icio_keys)

    if len(missing_keys) or len(extra_keys):
        messages = [
            "Hybrid emissions extension does not match the ICIO industry set."
        ]

        if len(missing_keys):
            messages.append(
                f"Missing ICIO industries ({len(missing_keys)}): "
                f"{missing_keys[:20].tolist()}"
            )

        if len(extra_keys):
            messages.append(
                f"Unknown extension industries ({len(extra_keys)}): "
                f"{extra_keys[:20].tolist()}"
            )

        raise ValueError("\n".join(messages))

    output = (
        X.rename("output_usd_mn")
        .rename_axis("io_key")
        .reset_index()
    )
    output["io_key"] = output["io_key"].astype(str)

    output["output_usd_mn"] = pd.to_numeric(
        output["output_usd_mn"],
        errors="raise",
    )

    if (
        ~np.isfinite(output["output_usd_mn"])
    ).any() or (output["output_usd_mn"] < 0).any():
        raise ValueError(
            "ICIO gross output must be finite and non-negative "
            "for every industry."
        )

    extension = (
        output.merge(
            emissions,
            on="io_key",
            how="left",
            validate="one_to_one",
        )
        .merge(
            sector_map[
                ["io_key", "country", "sector_code", "sector_label"]
            ],
            on="io_key",
            how="left",
            validate="one_to_one",
        )
    )

    if extension["ghg_tonnes_co2e"].isna().any():
        raise ValueError(
            "Missing emissions remain after matching the hybrid extension "
            "to ICIO industries."
        )

    if extension["country"].isna().any():
        raise ValueError(
            "Some ICIO industries could not be matched to the sector map."
        )

    # Validate direct emissions before calculating emissions intensities.
    if not np.isfinite(extension["ghg_tonnes_co2e"]).all():
        raise ValueError(
            "Direct emissions must be finite for every ICIO industry."
        )

    # Identify industries with zero gross output.
    zero_output = extension["output_usd_mn"].eq(0)

    # Zero output is only consistent with zero direct emissions.
    zero_output_nonzero_emissions = zero_output & (
        extension["ghg_tonnes_co2e"] != 0
    )

    if zero_output_nonzero_emissions.any():
        problem_industries = extension.loc[
            zero_output_nonzero_emissions,
            ["io_key", "output_usd_mn", "ghg_tonnes_co2e"],
        ]

        raise ValueError(
            "Industries have zero ICIO gross output but nonzero direct "
            "emissions. Cannot calculate emissions intensity:\n"
            f"{problem_industries.head(20).to_string(index=False)}"
        )

    # Calculate emissions intensity.
    # Zero-output industries with zero emissions receive zero intensity.
    extension["ghg_tonnes_per_usd_mn"] = 0.0

    positive_output = extension["output_usd_mn"] > 0

    extension.loc[
        positive_output,
        "ghg_tonnes_per_usd_mn",
    ] = (
        extension.loc[positive_output, "ghg_tonnes_co2e"]
        / extension.loc[positive_output, "output_usd_mn"]
    )
    
    LOG.info(
    "ICIO industries with zero gross output: %s",
    int(zero_output.sum()),
    )

    LOG.info(
        "Zero-output industries with zero direct emissions: %s",
        int((zero_output & extension["ghg_tonnes_co2e"].eq(0)).sum()),
    )

    coverage_summary = (
        extension.groupby("country", as_index=False)
        .agg(
            icio_industries=("io_key", "size"),
            industries_with_emissions=("ghg_tonnes_co2e", "count"),
            total_output_usd_mn=("output_usd_mn", "sum"),
            emissions_tonnes_co2e=("ghg_tonnes_co2e", "sum"),
        )
    )

    coverage_summary["industry_coverage_share"] = (
        coverage_summary["industries_with_emissions"]
        / coverage_summary["icio_industries"]
    )

    coverage_summary["output_coverage_share"] = 1.0

    LOG.info(
        "Loaded hybrid emissions extension for %s ICIO industries.",
        len(extension),
    )
    LOG.info(
        "Emissions sources: %s",
        extension["source"].value_counts().to_dict(),
    )

    return extension, coverage_summary

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

    # Require a 2030 output scenario for every required country.
    countries_with_2030 = set(
        output.loc[output["year"].eq(2030), "country"]
    )

    missing_2030 = REQUIRED_COUNTRIES - countries_with_2030

    if missing_2030:
        raise ValueError(
            f"Missing 2030 output scenarios for: {sorted(missing_2030)}"
        )

    missing_mappings = (
        set(output["country"]) - set(dc_map["country"])
    )

    if missing_mappings:
        raise ValueError(
            "Missing data-centre sector mappings for scenario countries: "
            f"{sorted(missing_mappings)}"
        )
    
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
                    "kgco2_per_kwh_natgas",
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
    
    # Validate electricity demand basis.
    valid_demand_basis = physical["demand_basis"].isin(["facility", "it"])

    if not valid_demand_basis.all():
        invalid = physical.loc[
            ~valid_demand_basis,
            ["country", "year", "demand_basis"],
        ]

        raise ValueError(
            "Invalid electricity demand basis. Expected 'facility' or 'it':\n"
            f"{invalid.to_string(index=False)}"
        )

    # Validate physical input values.
    positive_columns = [
        "price_2022_usd_mwh",
        "pue",
    ]

    for column in positive_columns:
        values = pd.to_numeric(physical[column], errors="coerce")

        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(
                f"{column} must contain finite, strictly positive values."
            )

    nonnegative_columns = [
        "electricity_gwh",
        "kgco2_per_kwh",
        "wue_l_per_kwh_it",
    ]

    for column in nonnegative_columns:
        values = pd.to_numeric(physical[column], errors="coerce")

        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(
                f"{column} must contain finite, non-negative values."
            )
    
    physical_columns = [
        "facility_electricity_gwh",
        "it_electricity_gwh",
        "price_2022_usd_mwh",
        "kgco2_per_kwh",
        "pue",
        "wue_l_per_kwh_it",
    ]

    missing_physical = scenarios[physical_columns].isna().any(axis=1)

    if missing_physical.any():
        raise ValueError(
            "Missing physical observations for data-centre output "
            "scenario years: "
            f"{scenarios.loc[missing_physical, ['country', 'year']].to_dict('records')}"
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

        if scenarios["us_parent_operator_share"].isna().any():
            missing = scenarios.loc[
                scenarios["us_parent_operator_share"].isna(),
                ["country", "year"],
            ].to_dict("records")

            raise ValueError(
                "Missing US-parent operator shares for output scenario years: "
                f"{missing}"
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
    
    # Full 2030 data-centre output level is treated as additional gross output.
    scenarios["dc_output_shock_usd_mn"] = np.where(
        scenarios["year"].eq(2030),
        scenarios["dc_output_usd_mn"],
        0.0,
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

    scenarios["electricity_expenditure_usd_mn"] = (
        scenarios["facility_electricity_gwh"]
        * scenarios["price_2022_usd_mwh"]
        / 1000.0
    )

    scenarios["electricity_expenditure_yoy_usd_mn"] = (
        scenarios.groupby("country")["electricity_expenditure_usd_mn"]
        .diff()
        .fillna(0.0)
    )

    scenarios["electricity_expenditure_cumulative_usd_mn"] = (
        scenarios["electricity_expenditure_usd_mn"]
        - scenarios.groupby("country")["electricity_expenditure_usd_mn"]
        .transform("first")
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
    for basis in ["yoy", "cumulative", "shock"]:
        output_col = f"dc_output_{basis}_usd_mn"

        scenarios[f"us_customer_output_{basis}_usd_mn"] = (
            scenarios[output_col] * scenarios["us_customer_share"]
        )

        scenarios[f"us_parent_output_{basis}_usd_mn"] = (
            scenarios[output_col]
            * scenarios["us_parent_operator_share"]
        )

    return physical, scenarios


def build_ghosh_system(A, x_ref):
    """
    Build and factorize the Ghosh supply-driven IO system.

    A[i, j] = input from supplier i per unit of purchaser j output.
    Z[i, j] = intermediate sales from supplier i to purchaser j.
    C[i, j] = share of supplier i output sold to purchaser j.

    The Ghosh model is:
        delta_x = (I - C.T)^(-1) delta_v

    where delta_v is an exogenous primary-input shock.
    """
    a = A.to_numpy(dtype=float) if hasattr(A, "to_numpy") else np.asarray(A, dtype=float)
    x_ref = np.asarray(x_ref, dtype=float).reshape(-1)

    n = len(x_ref)

    if a.shape != (n, n):
        raise ValueError("A and x_ref dimensions do not match.")

    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(x_ref)):
        raise ValueError("A and x_ref must contain finite values.")

    if np.any(x_ref < 0):
        raise ValueError("Reference gross output cannot be negative.")

    # Reconstruct intermediate transactions from technical coefficients.
    # A[i, j] * X[j] = Z[i, j].
    Z = a * x_ref[np.newaxis, :]

    # Sales-allocation coefficients:
    # C[i, j] = Z[i, j] / X[i].
    C = np.divide(
        Z,
        x_ref[:, np.newaxis],
        out=np.zeros_like(Z),
        where=x_ref[:, np.newaxis] > 0,
    )

    # Ghosh system: delta_x = (I - C.T)^(-1) delta_v.
    H = np.eye(n) - C.T

    return lu_factor(H)

def gross_output_to_primary_inputs(
    ghosh_lu,
    n: int,
    target_indices,
    target_gross_output_changes,
):
    """
    Calibrate Ghosh primary-input shocks to reproduce specified
    gross-output changes at multiple target industries.

    Primary-input shocks are applied only to the target industries.
    Their effects propagate through the complete Ghosh system.
    """

    target_indices = np.asarray(target_indices, dtype=int)
    target_gross_output_changes = np.asarray(
        target_gross_output_changes,
        dtype=float,
    )

    if len(target_indices) != len(target_gross_output_changes):
        raise ValueError(
            "Ghosh target indices and gross-output changes must "
            "have the same length."
        )

    if len(np.unique(target_indices)) != len(target_indices):
        raise ValueError(
            "Ghosh target indices must be unique."
        )

    if (
        (target_indices < 0).any()
        or (target_indices >= n).any()
    ):
        raise ValueError(
            "Ghosh target indices are outside the industry range."
        )

    if not np.isfinite(target_gross_output_changes).all():
        raise ValueError(
            "Ghosh target gross-output changes contain non-finite values."
        )

    # Construct one unit primary-input shock for each target industry.
    selector = np.zeros(
        (n, len(target_indices)),
        dtype=float,
    )

    selector[
        target_indices,
        np.arange(len(target_indices)),
    ] = 1.0

    # Calculate the output response to each unit primary-input shock.
    response_matrix = lu_solve(
        ghosh_lu,
        selector,
    )

    # Extract the responses at the target industries.
    calibration_matrix = response_matrix[
        target_indices,
        :,
    ]

    # Solve for the primary-input shocks that reproduce all target
    # gross-output changes simultaneously.
    
    if not np.isfinite(calibration_matrix).all():
        raise ValueError(
            "Ghosh calibration matrix contains non-finite values."
        )

    if np.linalg.cond(calibration_matrix) > 1e12:
        raise ValueError(
            "Ghosh calibration matrix is numerically ill-conditioned."
        )
    
    try:
        target_primary_input_shocks = np.linalg.solve(
            calibration_matrix,
            target_gross_output_changes,
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "The Ghosh multi-target calibration system is singular."
        ) from exc

    primary_input_shock = np.zeros(n, dtype=float)
    primary_input_shock[target_indices] = target_primary_input_shocks

    # Calculate the complete Ghosh output response.
    output_change = lu_solve(
        ghosh_lu,
        primary_input_shock,
    )

    if (
        not np.isfinite(primary_input_shock).all()
        or not np.isfinite(output_change).all()
    ):
        raise ValueError(
            "Ghosh calibration produced non-finite results."
        )

    # Verify that all target gross-output changes are reproduced.
    if not np.allclose(
        output_change[target_indices],
        target_gross_output_changes,
        rtol=1e-8,
        atol=1e-8,
    ):
        raise RuntimeError(
            "Ghosh calibration did not reproduce the specified "
            "target gross-output changes."
        )

    return primary_input_shock, output_change

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
    rhs = np.zeros((n, len(target_indices)), dtype=float)
    rhs[target_indices, np.arange(len(target_indices))] = 1.0

    L_columns = lu_solve((lu, piv), rhs)

    L_ss = L_columns[target_indices, :]
    
    if not np.isfinite(L_ss).all():
        raise ValueError("Target-sector Leontief submatrix contains non-finite values.")

    if np.linalg.cond(L_ss) > 1e12:
        raise ValueError(
            "Target-sector Leontief submatrix is numerically ill-conditioned."
        )

    delta_y_target = np.linalg.solve(L_ss, delta_x_target)

    delta_y = np.zeros(n, dtype=float)
    delta_y[target_indices] = delta_y_target

    delta_x = lu_solve((lu, piv), delta_y)
    
    if not np.isfinite(delta_y).all() or not np.isfinite(delta_x).all():
        raise ValueError(
            "Leontief solution contains non-finite values."
        )

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
):
    """Calculate upstream and downstream output changes and GHG emissions."""

    keys = A.index.tolist()
    n = len(keys)

    # ------------------------------------------------------------
    # Align and validate reference gross output.
    # ------------------------------------------------------------

    x_ref = X.reindex(keys).to_numpy(dtype=float)

    if not np.isfinite(x_ref).all():
        raise ValueError("Reference gross output contains non-finite values.")

    if (x_ref < 0).any():
        raise ValueError("Reference gross output contains negative values.")

    if A.index.has_duplicates or A.columns.has_duplicates:
        raise ValueError("ICIO matrix contains duplicate industry keys.")

    if A.columns.tolist() != keys:
        raise ValueError(
            "ICIO matrix rows and columns must have identical industry ordering."
        )

    # ------------------------------------------------------------
    # Factorize Leontief and Ghosh systems.
    # ------------------------------------------------------------

    LOG.info(
        "Factorizing Leontief and Ghosh systems with %s industries.",
        n,
    )

    # Leontief: x = (I - A)^(-1) y
    M = np.eye(n, dtype=float) - A.to_numpy(dtype=float)

    lu, piv = lu_factor(M, check_finite=True)

    if (
        not np.isfinite(lu).all()
        or np.any(np.abs(np.diag(lu)) < 1e-12)
    ):
        raise ValueError(
            "The Leontief system is singular or numerically unstable."
        )

    # Ghosh: x = (I - C.T)^(-1) v
    ghosh_lu = build_ghosh_system(A, x_ref)

    if (
        not np.isfinite(ghosh_lu[0]).all()
        or np.any(np.abs(np.diag(ghosh_lu[0])) < 1e-12)
    ):
        raise ValueError(
            "The Ghosh system is singular or numerically unstable."
        )

    # ------------------------------------------------------------
    # Prepare emissions extension and industry lookups.
    # ------------------------------------------------------------

    extension = (
        extension
        .set_index("io_key")
        .loc[keys]
    )

    average_grid_intensity = (
        extension["ghg_tonnes_per_usd_mn"].to_numpy(dtype=float)
    )

    if not np.isfinite(average_grid_intensity).all():
        raise ValueError(
            "The hybrid emissions extension contains non-finite "
            "emissions intensities."
        )

    emissions_source = extension["source"].to_numpy()

    # Share of Canadian data-centre electricity demand supplied by BTM gas.
    BTM_NATGAS_SHARE = 0.92

    LOG.info(
        "BTM natural-gas share of Canadian data-centre electricity demand: %.1f%%",
        100.0 * BTM_NATGAS_SHARE,
    )

    sector_lookup = sector_map.set_index("io_key").loc[keys]
    country_by_key = sector_lookup["country"].to_numpy()

    electricity_codes = set()

    if "is_electricity_sector" in sector_map.columns:
        flag = (
            sector_map["is_electricity_sector"]
            .astype(str)
            .str.lower()
        )

        electricity_codes = set(
            sector_map.loc[
                flag.isin(["true", "1", "yes"]),
                "io_key",
            ]
        )

    electricity_mask = np.isin(
        keys,
        list(electricity_codes),
    )

    # ------------------------------------------------------------
    # Calculate 2022 baseline emissions.
    # ------------------------------------------------------------

    baseline_emissions = x_ref * average_grid_intensity

    if not np.isfinite(baseline_emissions).all():
        raise ValueError(
            "The 2022 baseline emissions calculation produced "
            "non-finite values."
        )

    baseline_total_emissions = float(baseline_emissions.sum())

    baseline_us_emissions = float(
        baseline_emissions[country_by_key == "USA"].sum()
    )

    baseline_electricity_emissions = float(
        baseline_emissions[electricity_mask].sum()
    )

    # ------------------------------------------------------------
    # Construct target industry lookups.
    # ------------------------------------------------------------

    target_key_by_country = (
        dc_map
        .set_index("country")["dc_proxy_io_key"]
        .to_dict()
    )

    key_to_index = {
        key: i
        for i, key in enumerate(keys)
    }

    missing_target_keys = [
        key
        for key in target_key_by_country.values()
        if key not in key_to_index
    ]

    if missing_target_keys:
        raise ValueError(
            f"Data-centre proxy industries are missing from ICIO: "
            f"{missing_target_keys}"
        )

    target_index_by_country = {
        country: key_to_index[key]
        for country, key in target_key_by_country.items()
    }

    electricity_index_by_country = {}

    for row in dc_map.itertuples(index=False):
        if row.electricity_io_key not in key_to_index:
            raise ValueError(
                f"Electricity industry {row.electricity_io_key} "
                f"for {row.country} is missing from ICIO."
            )

        electricity_index_by_country[row.country] = (
            key_to_index[row.electricity_io_key]
        )

    # ------------------------------------------------------------
    # Prepare result containers.
    # ------------------------------------------------------------

    detail_rows = []
    summary_rows = []
    final_demand_rows = []

    downstream_detail_rows = []
    downstream_summary_rows = []

    required_shock_columns = [
        "dc_output_shock_usd_mn",
        "us_customer_output_shock_usd_mn",
        "us_parent_output_shock_usd_mn",
    ]

    missing_columns = [
        col
        for col in required_shock_columns
        if col not in scenarios.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing 2030 output shock columns: {missing_columns}"
        )

    if not scenarios["year"].eq(2030).any():
        raise ValueError(
            "The scenario data contains no 2030 observations."
        )

    # ------------------------------------------------------------
    # Main scenario loop.
    # ------------------------------------------------------------

    for row in scenarios.itertuples(index=False):

        country = row.country
        year = int(row.year)

        if country not in target_index_by_country:
            raise ValueError(
                f"No data-centre proxy industry is mapped for {country}."
            )

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

        if year == 2030:
            additional_output_definitions = {
                "all_expansion": row.dc_output_shock_usd_mn,
                "us_customer_demand": row.us_customer_output_shock_usd_mn,
                "us_parent_owned_operator": row.us_parent_output_shock_usd_mn,
            }
        else:
            additional_output_definitions = {}

        shock_basis_definitions = [
            ("annual_increment", shock_definitions),
            ("cumulative_vs_first_year", cumulative_definitions),
            ("2030_additional_output", additional_output_definitions),
        ]

        for shock_basis, definitions in shock_basis_definitions:

            for attribution, shock in definitions.items():

                if pd.isna(shock):
                    continue

                shock = float(shock)

                if not np.isfinite(shock):
                    raise ValueError(
                        "ITC gross-output shock is non-finite."
                    )

                # --------------------------------------------------------
                # Determine physical electricity demand and expenditure.
                # --------------------------------------------------------

                if shock_basis == "annual_increment":
                    electricity_change_gwh = float(row.facility_gwh_yoy)

                elif shock_basis == "cumulative_vs_first_year":
                    electricity_change_gwh = float(
                        row.facility_gwh_cumulative
                    )

                elif shock_basis == "2030_additional_output":
                    electricity_change_gwh = float(
                        row.facility_electricity_gwh
                    )

                else:
                    raise ValueError(
                        f"Unrecognized shock basis: {shock_basis}"
                    )

                if not np.isfinite(electricity_change_gwh):
                    raise ValueError(
                        "Physical electricity-demand change is non-finite."
                    )

                electricity_price = float(row.price_2022_usd_mwh)

                if (
                    not np.isfinite(electricity_price)
                    or electricity_price <= 0
                ):
                    raise ValueError(
                        "Electricity price must be positive and finite."
                    )

                # GWh * USD/MWh / 1,000 = USD million.
                electricity_expenditure_usd_mn = (
                    electricity_change_gwh
                    * electricity_price
                    / 1_000.0
                )

                if not np.isfinite(electricity_expenditure_usd_mn):
                    raise ValueError(
                        "Electricity expenditure is non-finite."
                    )

                # Skip only if neither the ITC shock nor the physical
                # electricity-demand shock is non-zero.
                if (
                    abs(shock) < 1e-12
                    and abs(electricity_expenditure_usd_mn) < 1e-12
                ):
                    continue

                # --------------------------------------------------------
                # Define electricity-demand sensitivities.
                # --------------------------------------------------------

                demand_sensitivities = {
                    "icio_endogenous": None,
                }

                if country in electricity_index_by_country:
                    demand_sensitivities[
                        "physical_facility_demand"
                    ] = electricity_expenditure_usd_mn

                electricity_index = electricity_index_by_country.get(
                    country
                )

                # --------------------------------------------------------
                # Run each electricity-demand sensitivity.
                # --------------------------------------------------------

                for (
                    electricity_demand_sensitivity,
                    physical_electricity_shock,
                ) in demand_sensitivities.items():

                    # ----------------------------------------------------
                    # Construct the common target output shocks.
                    #
                    # These exact target indices and target output
                    # changes are used in BOTH Leontief and Ghosh.
                    # ----------------------------------------------------

                    if physical_electricity_shock is None:

                        target_indices = [target_index]

                        target_shocks = np.array(
                            [shock],
                            dtype=float,
                        )

                    else:

                        target_indices = [
                            target_index,
                            electricity_index,
                        ]

                        target_shocks = np.array(
                            [
                                shock,
                                physical_electricity_shock,
                            ],
                            dtype=float,
                        )

                    if len(set(target_indices)) != len(target_indices):
                        raise ValueError(
                            "The ITC and electricity target industries "
                            "must be distinct."
                        )

                    # ----------------------------------------------------
                    # Leontief: calibrate final demand to reproduce the
                    # common target gross-output shocks.
                    # ----------------------------------------------------

                    delta_y, delta_x = gross_output_to_final_demand(
                        lu=lu,
                        piv=piv,
                        n=n,
                        target_indices=target_indices,
                        delta_x_target=target_shocks,
                    )

                    if (
                        not np.isfinite(delta_y).all()
                        or not np.isfinite(delta_x).all()
                    ):
                        raise ValueError(
                            "Leontief final-demand or output changes "
                            "contain non-finite values."
                        )

                    # ----------------------------------------------------
                    # Ghosh: calibrate primary-input shocks to reproduce
                    # the EXACT SAME target gross-output shocks.
                    # ----------------------------------------------------

                    delta_v_downstream, delta_x_downstream = (
                        gross_output_to_primary_inputs(
                            ghosh_lu=ghosh_lu,
                            n=n,
                            target_indices=target_indices,
                            target_gross_output_changes=target_shocks,
                        )
                    )

                    if not np.allclose(
                        delta_x_downstream[target_indices],
                        delta_x[target_indices],
                        rtol=1e-8,
                        atol=1e-8,
                    ):
                        raise RuntimeError(
                            "Leontief and Ghosh do not reproduce the "
                            "same target gross-output shocks."
                        )

                    # ----------------------------------------------------
                    # Calculate incremental BTM natural-gas emissions.
                    #
                    # This adjustment is calculated once and applied
                    # only in the btm_natgas emissions sensitivity.
                    # ----------------------------------------------------

                    additional_btm_emissions_tonnes = 0.0

                    if (
                        country == "CAN"
                        and electricity_demand_sensitivity
                        == "physical_facility_demand"
                    ):

                        natgas_factor = float(
                            row.kgco2_per_kwh_natgas
                        )

                        if (
                            not np.isfinite(natgas_factor)
                            or natgas_factor < 0
                        ):
                            raise ValueError(
                                "Canadian natural-gas emissions factor "
                                "must be finite and non-negative."
                            )

                        # Convert the extension-derived electricity-sector
                        # emissions intensity into kg CO2e/kWh.
                        #
                        # The numerical value is also tonnes CO2e/MWh.
                        extension_grid_factor = (
                            average_grid_intensity[electricity_index]
                            * electricity_price
                            / 1_000_000.0
                        )

                        # GWh * 1,000 = MWh.
                        additional_btm_emissions_tonnes = (
                            electricity_change_gwh
                            * 1_000.0
                            * BTM_NATGAS_SHARE
                            * (
                                natgas_factor
                                - extension_grid_factor
                            )
                        )

                        if not np.isfinite(
                            additional_btm_emissions_tonnes
                        ):
                            raise ValueError(
                                "Additional BTM emissions are non-finite."
                            )

                    # ----------------------------------------------------
                    # Save the Leontief final-demand shock.
                    # ----------------------------------------------------

                    final_demand_rows.append(
                        pd.DataFrame({
                            "year": year,
                            "shock_basis": shock_basis,
                            "shock_destination": country,
                            "attribution": attribution,
                            "electricity_demand_sensitivity": (
                                electricity_demand_sensitivity
                            ),
                            "io_key": keys,
                            "final_demand_change_usd_mn": delta_y,
                        })
                    )

                    # ----------------------------------------------------
                    # Calculate emissions under both grid sensitivities.
                    # ----------------------------------------------------

                    emissions_sensitivities = [
                        "average_grid",
                        "btm_natgas",
                    ]

                    for grid_sensitivity in emissions_sensitivities:

                        # Both cases use the hybrid average-grid
                        # emissions extension as their starting point.
                        # The BTM case adds the incremental natural-gas
                        # emissions adjustment below.
                        intensity_vector = average_grid_intensity

                        # =================================================
                        # UPSTREAM: Leontief emissions.
                        # =================================================

                        emissions = delta_x * intensity_vector

                        direct_btm_adjustment = np.zeros(n)

                        if (
                            grid_sensitivity == "btm_natgas"
                            and country == "CAN"
                            and electricity_demand_sensitivity
                            == "physical_facility_demand"
                        ):

                            direct_btm_adjustment[electricity_index] = (
                                additional_btm_emissions_tonnes
                            )

                            emissions = (
                                emissions + direct_btm_adjustment
                            )

                        if not np.isfinite(emissions).all():
                            raise ValueError(
                                "Calculated upstream emissions contain non-finite values."
                            )

                        origin_df = pd.DataFrame({
                            "year": year,
                            "shock_basis": shock_basis,
                            "shock_destination": country,
                            "attribution": attribution,
                            "electricity_demand_sensitivity": (
                                electricity_demand_sensitivity
                            ),
                            "grid_emissions_sensitivity": grid_sensitivity,
                            "io_key": keys,
                            "output_change_usd_mn": delta_x,
                            "ghg_tonnes_co2e": emissions,
                            "emissions_intensity_tonnes_per_usd_mn": (
                                intensity_vector
                            ),
                            "emissions_source": emissions_source,
                            "direct_btm_adjustment_tonnes_co2e": (
                                direct_btm_adjustment
                            ),
                        }).merge(
                            sector_map,
                            on="io_key",
                            how="left",
                            validate="one_to_one",
                        )

                        detail_rows.append(origin_df)

                        total_emissions = float(emissions.sum())

                        us_emissions = float(
                            emissions[country_by_key == "USA"].sum()
                        )

                        domestic_emissions = float(
                            emissions[country_by_key == country].sum()
                        )

                        electricity_emissions = float(
                            emissions[electricity_mask].sum()
                        )

                        destination_baseline = float(
                            baseline_emissions[
                                country_by_key == country
                            ].sum()
                        )

                        summary_rows.append({
                            "year": year,
                            "shock_basis": shock_basis,
                            "shock_destination": country,
                            "attribution": attribution,
                            "electricity_demand_sensitivity": (
                                electricity_demand_sensitivity
                            ),
                            "grid_emissions_sensitivity": grid_sensitivity,
                            "target_gross_output_change_usd_mn": shock,
                            "physical_electricity_expenditure_usd_mn": (
                                physical_electricity_shock
                                if physical_electricity_shock is not None
                                else np.nan
                            ),
                            "global_upstream_ghg_tonnes_co2e": (
                                total_emissions
                            ),
                            "us_origin_ghg_tonnes_co2e": us_emissions,
                            "destination_country_origin_ghg_tonnes_co2e": (
                                domestic_emissions
                            ),
                            "foreign_origin_ghg_tonnes_co2e": (
                                total_emissions - domestic_emissions
                            ),
                            "electricity_sector_origin_ghg_tonnes_co2e": (
                                electricity_emissions
                            ),
                            "us_origin_share": (
                                us_emissions / total_emissions
                                if abs(total_emissions) > 1e-12
                                else np.nan
                            ),
                            "destination_origin_share": (
                                domestic_emissions / total_emissions
                                if abs(total_emissions) > 1e-12
                                else np.nan
                            ),
                            "baseline_global_ghg_tonnes_co2e": (
                                baseline_total_emissions
                            ),
                            "baseline_us_origin_ghg_tonnes_co2e": (
                                baseline_us_emissions
                            ),
                            "baseline_destination_country_origin_ghg_tonnes_co2e": (
                                destination_baseline
                            ),
                            "baseline_foreign_origin_ghg_tonnes_co2e": (
                                baseline_total_emissions
                                - destination_baseline
                            ),
                            "baseline_electricity_sector_origin_ghg_tonnes_co2e": (
                                baseline_electricity_emissions
                            ),
                        })

                        # =================================================
                        # DOWNSTREAM: Ghosh emissions.
                        # =================================================

                        downstream_emissions = (
                            delta_x_downstream * intensity_vector
                        )

                        downstream_direct_btm_adjustment = np.zeros(n)

                        if (
                            grid_sensitivity == "btm_natgas"
                            and country == "CAN"
                            and electricity_demand_sensitivity
                            == "physical_facility_demand"
                        ):

                            downstream_direct_btm_adjustment[
                                electricity_index
                            ] = additional_btm_emissions_tonnes

                            downstream_emissions = (
                                downstream_emissions
                                + downstream_direct_btm_adjustment
                            )

                        if not np.isfinite(downstream_emissions).all():
                            raise ValueError(
                                "Calculated downstream emissions contain non-finite values."
                            )

                        downstream_total_emissions = float(
                            downstream_emissions.sum()
                        )

                        downstream_us_emissions = float(
                            downstream_emissions[
                                country_by_key == "USA"
                            ].sum()
                        )

                        downstream_domestic_emissions = float(
                            downstream_emissions[
                                country_by_key == country
                            ].sum()
                        )

                        downstream_electricity_emissions = float(
                            downstream_emissions[electricity_mask].sum()
                        )

                        downstream_destination_baseline = (
                            destination_baseline
                        )

                        downstream_foreign_baseline = (
                            baseline_total_emissions
                            - downstream_destination_baseline
                        )

                        # ------------------------------------------------
                        # Downstream detail.
                        # ------------------------------------------------

                        downstream_detail = pd.DataFrame({
                            "io_key": keys,
                            "output_change_usd_mn": (
                                delta_x_downstream
                            ),
                            "primary_input_change_usd_mn": (
                                delta_v_downstream
                            ),
                            "ghg_tonnes_co2e": downstream_emissions,
                            "emissions_intensity_tonnes_per_usd_mn": (
                                intensity_vector
                            ),
                            "emissions_source": emissions_source,
                            "direct_btm_adjustment_tonnes_co2e": (
                                downstream_direct_btm_adjustment
                            ),
                        }).merge(
                            sector_map,
                            on="io_key",
                            how="left",
                            validate="one_to_one",
                        )

                        downstream_detail["year"] = year
                        downstream_detail["shock_basis"] = shock_basis
                        downstream_detail["shock_destination"] = country
                        downstream_detail["attribution"] = attribution
                        downstream_detail[
                            "electricity_demand_sensitivity"
                        ] = electricity_demand_sensitivity
                        downstream_detail[
                            "grid_emissions_sensitivity"
                        ] = grid_sensitivity

                        downstream_detail_rows.append(
                            downstream_detail
                        )

                        # ------------------------------------------------
                        # Downstream summary.
                        # ------------------------------------------------

                        downstream_summary_rows.append({
                            "year": year,
                            "shock_basis": shock_basis,
                            "shock_destination": country,
                            "attribution": attribution,
                            "electricity_demand_sensitivity": (
                                electricity_demand_sensitivity
                            ),
                            "grid_emissions_sensitivity": grid_sensitivity,
                            "itc_output_change_usd_mn": shock,
                            "physical_electricity_expenditure_usd_mn": (
                                physical_electricity_shock
                                if physical_electricity_shock is not None
                                else np.nan
                            ),
                            "global_downstream_ghg_tonnes_co2e": (
                                downstream_total_emissions
                            ),
                            "destination_country_origin_ghg_tonnes_co2e": (
                                downstream_domestic_emissions
                            ),
                            "us_origin_ghg_tonnes_co2e": (
                                downstream_us_emissions
                            ),
                            "foreign_origin_ghg_tonnes_co2e": (
                                downstream_total_emissions
                                - downstream_domestic_emissions
                            ),
                            "electricity_sector_origin_ghg_tonnes_co2e": (
                                downstream_electricity_emissions
                            ),
                            "btm_natgas_adjustment_tonnes_co2e": (
                                float(
                                    downstream_direct_btm_adjustment.sum()
                                )
                            ),
                            "baseline_global_ghg_tonnes_co2e": (
                                baseline_total_emissions
                            ),
                            "baseline_us_origin_ghg_tonnes_co2e": (
                                baseline_us_emissions
                            ),
                            "baseline_destination_country_origin_ghg_tonnes_co2e": (
                                downstream_destination_baseline
                            ),
                            "baseline_foreign_origin_ghg_tonnes_co2e": (
                                downstream_foreign_baseline
                            ),
                            "baseline_electricity_sector_origin_ghg_tonnes_co2e": (
                                baseline_electricity_emissions
                            ),
                        })

    # ------------------------------------------------------------
    # Assemble upstream results.
    # ------------------------------------------------------------

    if not detail_rows:
        raise ValueError(
            "No Leontief results were generated. Check the output "
            "scenario and whether the 2030 additional-output shock "
            "is non-zero."
        )

    detail = pd.concat(
        detail_rows,
        ignore_index=True,
    )

    summary = pd.DataFrame(summary_rows)

    total = summary[
        "global_upstream_ghg_tonnes_co2e"
    ].replace(0, np.nan)

    summary["us_origin_share_pct"] = (
        100.0
        * summary["us_origin_ghg_tonnes_co2e"]
        / total
    )

    summary["destination_origin_share_pct"] = (
        100.0
        * summary["destination_country_origin_ghg_tonnes_co2e"]
        / total
    )

    summary["foreign_origin_share_pct"] = (
        100.0
        * summary["foreign_origin_ghg_tonnes_co2e"]
        / total
    )

    summary["electricity_sector_origin_share_pct"] = (
        100.0
        * summary["electricity_sector_origin_ghg_tonnes_co2e"]
        / total
    )

    # ------------------------------------------------------------
    # Upstream: percentage changes relative to average-grid results.
    # ------------------------------------------------------------

    emission_columns = [
        "global_upstream_ghg_tonnes_co2e",
        "us_origin_ghg_tonnes_co2e",
        "destination_country_origin_ghg_tonnes_co2e",
        "foreign_origin_ghg_tonnes_co2e",
        "electricity_sector_origin_ghg_tonnes_co2e",
    ]

    comparison_keys = [
        "year",
        "shock_basis",
        "shock_destination",
        "attribution",
        "electricity_demand_sensitivity",
    ]

    upstream_average_grid = (
        summary.loc[
            summary["grid_emissions_sensitivity"] == "average_grid",
            comparison_keys + emission_columns,
        ]
        .set_index(comparison_keys)
    )

    summary_indexed = summary.set_index(comparison_keys)

    for column in emission_columns:

        baseline_values = upstream_average_grid[column].reindex(
            summary_indexed.index
        )

        summary[
            f"{column}_pct_change_vs_average_grid"
        ] = (
            (
                summary_indexed[column] / baseline_values
                - 1.0
            )
            * 100.0
        ).to_numpy()

    # ------------------------------------------------------------
    # Upstream: emissions as a percentage of the 2022 IO baseline.
    # ------------------------------------------------------------

    baseline_comparisons = {
        "global_upstream_ghg_tonnes_co2e": (
            "baseline_global_ghg_tonnes_co2e"
        ),
        "us_origin_ghg_tonnes_co2e": (
            "baseline_us_origin_ghg_tonnes_co2e"
        ),
        "destination_country_origin_ghg_tonnes_co2e": (
            "baseline_destination_country_origin_ghg_tonnes_co2e"
        ),
        "foreign_origin_ghg_tonnes_co2e": (
            "baseline_foreign_origin_ghg_tonnes_co2e"
        ),
        "electricity_sector_origin_ghg_tonnes_co2e": (
            "baseline_electricity_sector_origin_ghg_tonnes_co2e"
        ),
    }

    for emission_column, baseline_column in baseline_comparisons.items():

        summary[
            f"{emission_column}_pct_change_vs_2022_baseline"
        ] = (
            100.0
            * summary[emission_column]
            / summary[baseline_column].replace(0, np.nan)
        )

    final_demand = pd.concat(
        final_demand_rows,
        ignore_index=True,
    )

    # ------------------------------------------------------------
    # Assemble downstream results.
    # ------------------------------------------------------------

    if downstream_detail_rows:
        downstream_detail = pd.concat(
            downstream_detail_rows,
            ignore_index=True,
        )
    else:
        downstream_detail = pd.DataFrame()

    if downstream_summary_rows:

        downstream_summary = pd.DataFrame(
            downstream_summary_rows
        )

        # --------------------------------------------------------
        # Downstream: emissions origin shares.
        # --------------------------------------------------------

        downstream_total = downstream_summary[
            "global_downstream_ghg_tonnes_co2e"
        ].replace(0, np.nan)

        downstream_summary["us_origin_share_pct"] = (
            100.0
            * downstream_summary["us_origin_ghg_tonnes_co2e"]
            / downstream_total
        )

        downstream_summary["destination_origin_share_pct"] = (
            100.0
            * downstream_summary[
                "destination_country_origin_ghg_tonnes_co2e"
            ]
            / downstream_total
        )

        downstream_summary["foreign_origin_share_pct"] = (
            100.0
            * downstream_summary[
                "foreign_origin_ghg_tonnes_co2e"
            ]
            / downstream_total
        )

        downstream_summary[
            "electricity_sector_origin_share_pct"
        ] = (
            100.0
            * downstream_summary[
                "electricity_sector_origin_ghg_tonnes_co2e"
            ]
            / downstream_total
        )

        # --------------------------------------------------------
        # Downstream: percentage changes relative to average-grid
        # results, matching the same scenario dimensions.
        # --------------------------------------------------------

        downstream_emission_columns = [
            "global_downstream_ghg_tonnes_co2e",
            "us_origin_ghg_tonnes_co2e",
            "destination_country_origin_ghg_tonnes_co2e",
            "foreign_origin_ghg_tonnes_co2e",
            "electricity_sector_origin_ghg_tonnes_co2e",
        ]

        downstream_comparison_keys = [
            "year",
            "shock_basis",
            "shock_destination",
            "attribution",
            "electricity_demand_sensitivity",
        ]

        downstream_average_grid = (
            downstream_summary.loc[
                downstream_summary["grid_emissions_sensitivity"]
                == "average_grid",
                downstream_comparison_keys
                + downstream_emission_columns,
            ]
            .set_index(downstream_comparison_keys)
        )

        downstream_summary_indexed = downstream_summary.set_index(
            downstream_comparison_keys
        )

        for column in downstream_emission_columns:

            baseline_values = downstream_average_grid[column].reindex(
                downstream_summary_indexed.index
            )

            downstream_summary[
                f"{column}_pct_change_vs_average_grid"
            ] = (
                (
                    downstream_summary_indexed[column]
                    / baseline_values
                    - 1.0
                )
                * 100.0
            ).to_numpy()

        # --------------------------------------------------------
        # Downstream: emissions as a percentage of the 2022
        # IO baseline.
        # --------------------------------------------------------

        downstream_baseline_comparisons = {
            "global_downstream_ghg_tonnes_co2e": (
                "baseline_global_ghg_tonnes_co2e"
            ),
            "us_origin_ghg_tonnes_co2e": (
                "baseline_us_origin_ghg_tonnes_co2e"
            ),
            "destination_country_origin_ghg_tonnes_co2e": (
                "baseline_destination_country_origin_ghg_tonnes_co2e"
            ),
            "foreign_origin_ghg_tonnes_co2e": (
                "baseline_foreign_origin_ghg_tonnes_co2e"
            ),
            "electricity_sector_origin_ghg_tonnes_co2e": (
                "baseline_electricity_sector_origin_ghg_tonnes_co2e"
            ),
        }

        for emission_column, baseline_column in (
            downstream_baseline_comparisons.items()
        ):

            downstream_summary[
                f"{emission_column}_pct_change_vs_2022_baseline"
            ] = (
                100.0
                * downstream_summary[emission_column]
                / downstream_summary[baseline_column].replace(
                    0,
                    np.nan,
                )
            )

    else:
        downstream_summary = pd.DataFrame()

    return (
        detail,
        summary,
        final_demand,
        downstream_detail,
        downstream_summary,
    )

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

    parser.add_argument("--A-object", type=str, default=None)
    parser.add_argument("--X-object", type=str, default=None)
    parser.add_argument("--FD-object", type=str, default=None)
    parser.add_argument("--X-key-column", type=str, default=None)
    
    parser.add_argument(
        "--icio-identity-tolerance",
        type=float,
        default=0.02,
        help="Maximum relative discrepancy for X = A@X + FD.",
    )
    
    parser.add_argument(
        "--emissions-extension",
        type=Path,
        default=SCRIPT_DIR / "hybrid_emissions" / "direct_emissions.csv",
        help=(
            "Completed hybrid emissions extension produced by "
            "build_hybrid_emissions_extension.py."
        ),
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Load and validate data without calculating Leontief footprints.",
    )
    
    parser.add_argument(
        "--X-value-column",
        type=str,
        default=None,
    )

    parser.add_argument("--log-level", default="INFO")

    return parser.parse_args()


def main():
    args = parse_args()
    
    if (
        not np.isfinite(args.icio_identity_tolerance)
        or args.icio_identity_tolerance < 0
    ):
        raise ValueError(
            "--icio-identity-tolerance must be finite and non-negative."
        )
    
    args.data_dir = args.data_dir.resolve()

    if not args.data_dir.is_dir():
        raise FileNotFoundError(
            f"Input data directory does not exist: {args.data_dir}"
        )

    args.emissions_extension = args.emissions_extension.resolve()

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
    sector_map, dc_map = load_sector_maps(A, args.data_dir)

    (
        output,
        demand,
        prices,
        grid,
        water,
        customer,
        owner,
    ) = load_scenario_inputs(args.data_dir)

    # 4. Load the completed hybrid emissions extension.
    LOG.info(
        "Loading hybrid emissions extension: %s",
        args.emissions_extension.resolve(),
    )

    extension, coverage_summary = load_emissions_extension(
        emissions_file=args.emissions_extension,
        X=X,
        sector_map=sector_map,
    )

    # Save the validated extension and its coverage diagnostics.
    extension.to_csv(
        args.output_dir / "icio_emissions_extension.csv",
        index=False,
    )

    coverage_summary.to_csv(
        args.output_dir / "emissions_extension_coverage_summary.csv",
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
    (
        detail,
        summary,
        final_demand,
        downstream_detail,
        downstream_summary,
    ) = run_leontief(
        A=A,
        X=X,
        sector_map=sector_map,
        extension=extension,
        dc_map=dc_map,
        scenarios=scenarios,
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
    
    # ------------------------------------------------------------
    # Write downstream emissions results.
    # ------------------------------------------------------------

    if not downstream_detail.empty:
        downstream_detail.to_parquet(
            args.output_dir / "downstream_emissions_by_origin_industry.parquet",
            index=False,
        )

    if not downstream_summary.empty:
        downstream_summary.to_csv(
            args.output_dir / "downstream_emissions_summary.csv",
            index=False,
        )

    by_origin = (
        detail.groupby(
            [
                "year",
                "shock_basis",
                "shock_destination",
                "grid_emissions_sensitivity",
                "electricity_demand_sensitivity",
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
