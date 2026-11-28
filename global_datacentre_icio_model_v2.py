
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

Leontief calculation (upstream)
------------------------------
The supplied gross-output shock is converted to a final-demand shock so
that the resulting change in the targeted proxy-sector gross output equals
the specified target (exogenous-output, or "mixed", model).

In the physical_facility_demand sensitivity the data-centre column is
hybridised: its domestic-electricity coefficient is replaced by the
measured electricity cost share (facility GWh x price / output) and the
other inputs are scaled down so the column still sums to one. Electricity
demand therefore scales with the output shock and the attribution lens,
and indirect electricity use by suppliers is still modelled.

Ghosh calculation (downstream)
------------------------------
Primary-input shocks at the data-centre industries reproduce the same
output change; the resulting supply push gives income-based ("enabled")
emissions. There is no electricity target in the Ghosh run.

Emissions components (never add upstream and downstream)
--------------------------------------------------------
direct      data-centre industries' own emissions + on-site BTM generation
upstream    supply chain excluding direct (incl. purchased electricity)
downstream  Ghosh emissions excluding direct

Grid sensitivities: average_grid (2022 extension), projected_grid
(scenario-year grid factor for the destination electricity sector) and
btm_natgas (projected grid with --btm-natgas-share generated on site).

See docs/methodology.md for the full mathematical description.

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

Canadian DSUT split (optional, --split-canada-dsut)
--------------------------------------------------
CAN_J62_63 is split into CAN_J62_63_CLD (cloud computing services),
CAN_J62_63_DPH (other NAICS 51821: infrastructure, data processing, web
hosting) and CAN_J62_63_OTH (rest of J62_63) using the 2020 Canadian
DSUT and the 2022 StatCan SUT; see canada_dsut_split.py. The Canadian
data-centre shock is then applied to CLD and DPH in proportion to their
output (--canada-dc-target cloud_and_hosting, default) or to CLD only
(--canada-dc-target cloud). Other countries, including India, are not
split.

TiVA and GHG footprint indicators (optional, --tiva-ghg-indicators)
are computed on the (split) table; see tiva_ghg_indicators.py.
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

import canada_dsut_split as cds
import ewaste
import tiva_ghg_indicators as tgi


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


def load_sector_maps(
    A: pd.DataFrame,
    data_dir: Path,
    dc_target_overrides: Optional[dict[str, tuple[str, list[tuple[str, float]]]]] = None,
):
    """
    dc_target_overrides maps a country to (original_proxy_key, targets),
    where targets is a list of (io_key, weight) pairs. It is used when an
    ICIO sector has been split (e.g. CAN_J62_63 by the Canadian DSUT): the
    country's proxy in dc_sector_map.csv must equal original_proxy_key and
    the shock is then distributed over the target subsectors.
    """
    sector_map, dc_map = _load_sector_maps(A, data_dir, dc_target_overrides or {})
    return sector_map, dc_map


def dc_targets_of(cfg) -> list[tuple[str, float]]:
    """Data-centre shock targets (io_key, weight) for one dc_map row."""
    targets = getattr(cfg, "dc_targets", None)
    if isinstance(targets, (list, tuple)) and targets:
        return [(str(k), float(w)) for k, w in targets]
    return [(str(cfg.dc_proxy_io_key), 1.0)]


def _load_sector_maps(A: pd.DataFrame, data_dir: Path, dc_target_overrides: dict):
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

    # Apply split-sector target overrides (e.g. CAN_J62_63 -> CLD + DPH).
    targets_column = []
    for row in dc_map.itertuples(index=False):
        override = dc_target_overrides.get(row.country)
        if override is None:
            targets_column.append([(row.dc_proxy_io_key, 1.0)])
            continue
        original_key, targets = override
        if row.dc_proxy_io_key != original_key:
            raise ValueError(
                f"dc_sector_map.csv maps {row.country} to {row.dc_proxy_io_key}, "
                f"but the split replaces {original_key}. Map {row.country} to "
                f"{original_key} or disable the split."
            )
        weights = np.array([w for _, w in targets], dtype=float)
        if (weights < 0).any() or not np.isclose(weights.sum(), 1.0):
            raise ValueError(f"Data-centre target weights for {row.country} must sum to one.")
        targets_column.append(list(targets))
        LOG.info(
            "Data-centre proxy for %s: %s replaced by %s",
            row.country, original_key,
            ", ".join(f"{k} ({w:.4f})" for k, w in targets),
        )

    dc_map["dc_targets"] = targets_column
    dc_map["dc_proxy_io_key"] = [t[0][0] for t in targets_column]
    dc_map["dc_proxy_label"] = [
        "+".join(k for k, _ in t) for t in targets_column
    ]

    for row in dc_map.itertuples(index=False):

        for target_key, _ in row.dc_targets:
            if target_key not in A.columns:
                raise ValueError(
                    f"Unknown data-centre proxy for {row.country}: "
                    f"{target_key}"
                )
            if not target_key.startswith(f"{row.country}_"):
                raise ValueError(
                    f"Data-centre proxy for {row.country} does not belong to "
                    f"that country: {target_key}"
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
    transform=None,
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

    # Optional transformation, e.g. splitting the CAN_J62_63 row to match a
    # split ICIO (see canada_dsut_split.split_emissions_frame).
    if transform is not None:
        emissions = transform(emissions)

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
    it_utilisation: float = 1.0,
    output_includes_power_passthrough: bool = False,
):
    """
    Create physical metrics and direct/electricity-implied output shocks.

    it_utilisation
        Average share of capacity drawn over the year. Multiplies the
        electricity demand in dc_electricity_demand.csv; use it when that file
        assumes full load (electricity_mw = capacity x PUE).
    output_includes_power_passthrough
        Treat output per MW (price_mn_MW_2022) as rent that excludes power,
        as in wholesale colocation leases, and add the facility electricity
        bill to data-centre output (power resold to tenants):
        output = capacity x rent per MW + facility GWh x price / 1000.
    """
    if not 0 < it_utilisation <= 1:
        raise ValueError("--it-utilisation must lie in (0, 1].")

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

    if it_utilisation != 1.0:
        LOG.info("Scaling electricity demand by IT utilisation %.2f.", it_utilisation)
        physical["electricity_gwh"] = physical["electricity_gwh"] * it_utilisation
    physical["it_utilisation"] = it_utilisation

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

    # Output with power pass-through (colocation rent excludes electricity).
    scenarios["dc_rent_usd_mn"] = scenarios["dc_output_usd_mn"]
    scenarios["power_passthrough_usd_mn"] = 0.0
    if output_includes_power_passthrough:
        scenarios["power_passthrough_usd_mn"] = (
            scenarios["facility_electricity_gwh"] * scenarios["price_2022_usd_mwh"] / 1000.0
        )
        scenarios["dc_output_usd_mn"] = (
            scenarios["dc_rent_usd_mn"] + scenarios["power_passthrough_usd_mn"]
        )
        LOG.info("Data-centre output includes the power pass-through (rent + electricity bill).")

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

        # Output-weighted electricity coefficient over the proxy targets
        # (a single sector unless the proxy has been split).
        coefficient = float(sum(
            weight * A.loc[cfg.electricity_io_key, target_key]
            for target_key, weight in dc_targets_of(cfg)
        ))

        if not np.isfinite(coefficient) or coefficient <= 0:
            raise ValueError(
                f"Invalid electricity coefficient for {row.country}: "
                f"A[{cfg.electricity_io_key}, {cfg.dc_proxy_label}]={coefficient}"
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

GRID_SENSITIVITIES = ("average_grid", "projected_grid", "btm_natgas")

SHOCK_BASIS_NOTES = {
    "annual_increment": (
        "Change in data-centre output from the previous scenario year."
    ),
    "cumulative_vs_first_year": (
        "Change in data-centre output relative to the first scenario year "
        "(preferred incremental measure)."
    ),
    "2030_additional_output": (
        "UPPER BOUND: the entire 2030 data-centre output is treated as new, "
        "although the 2022 ICIO already contains existing data centres."
    ),
}


def gross_output_to_final_demand(
    lu,
    piv,
    n: int,
    target_indices: list[int],
    delta_x_target: np.ndarray,
    extra_demand: Optional[np.ndarray] = None,
):
    """
    Convert target gross-output changes into final-demand changes.

    The target industries S are treated as exogenous (Miller and Blair's
    "mixed" model). Without ``extra_demand`` the result is

        delta_y_S = inverse(L_SS) @ delta_x_S
        delta_x   = L[:, S] @ delta_y_S

    which is identical to delta_x_o = (I - A_oo)^-1 A_oS delta_x_S.

    ``extra_demand`` is an additional intermediate-demand vector on the
    non-target industries. It is used when the purchases of the target
    industries differ from their ICIO column (hybrid electricity column):
    extra = (A'[:, S] - A[:, S]) @ delta_x_S. The result is then

        delta_x_o = (I - A_oo)^-1 A'_oS delta_x_S

    while delta_x_S still equals the specified targets.
    """
    target_indices = list(target_indices)
    delta_x_target = np.asarray(delta_x_target, dtype=float)

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

    if extra_demand is None:
        induced_by_extra = np.zeros(n, dtype=float)
    else:
        extra = np.asarray(extra_demand, dtype=float).copy()
        if extra.shape != (n,):
            raise ValueError("extra_demand must have one value per industry.")
        # Purchases from the target industries themselves do not induce
        # anything: target output is exogenous.
        extra[target_indices] = 0.0
        induced_by_extra = lu_solve((lu, piv), extra) if np.any(extra) else np.zeros(n)

    delta_y_target = np.linalg.solve(
        L_ss,
        delta_x_target - induced_by_extra[target_indices],
    )

    delta_y = np.zeros(n, dtype=float)
    delta_y[target_indices] = delta_y_target

    delta_x = L_columns @ delta_y_target + induced_by_extra

    if not np.isfinite(delta_y).all() or not np.isfinite(delta_x).all():
        raise ValueError("Leontief solution contains non-finite values.")

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


def hybrid_target_columns(
    A_np: np.ndarray,
    target_indices: list[int],
    electricity_index: int,
    electricity_cost_share: float,
    grid_share: float = 1.0,
    rescale_other_inputs: bool = True,
):
    """
    Replace the domestic-electricity coefficient of the data-centre columns
    with the physically measured electricity cost share.

    electricity_cost_share  e = facility GWh x price / data-centre output
    grid_share              share of that electricity bought from the grid
                            (1 - behind-the-meter share)

    With ``rescale_other_inputs`` every other coefficient of the column
    (including value added, implicitly) is multiplied by

        r_k = (1 - e) / (1 - a_Ek)

    so that the column still sums to at most one: the more of each dollar
    of data-centre output goes to electricity, the less goes to everything
    else. Behind-the-meter electricity is not bought from the grid; its
    cost stays in the data-centre column as own-account generation.

    Returns (new_columns, rescale_factors).
    """
    if not np.isfinite(electricity_cost_share):
        raise ValueError("Electricity cost share must be finite.")

    if electricity_cost_share < 0 or electricity_cost_share >= 1:
        raise ValueError(
            "Electricity expenditure must be between 0% and 100% of "
            f"data-centre output; found {electricity_cost_share:.3f}. "
            "Check output per MW (price_mn_MW_2022), the electricity price, "
            "PUE and the utilisation implied by electricity_mw."
        )

    if not 0 <= grid_share <= 1:
        raise ValueError("grid_share must lie between zero and one.")

    columns = A_np[:, target_indices].copy()
    old_electricity = columns[electricity_index, :].copy()

    if rescale_other_inputs:
        factors = (1.0 - electricity_cost_share) / (1.0 - old_electricity)
        if (factors < 0).any() or not np.isfinite(factors).all():
            raise ValueError("Invalid input rescaling factor for the hybrid column.")
        columns = columns * factors[np.newaxis, :]
    else:
        factors = np.ones(len(target_indices))

    columns[electricity_index, :] = electricity_cost_share * grid_share

    if (columns.sum(axis=0) > 1 + 1e-9).any():
        raise ValueError(
            "The hybrid data-centre column sums to more than one (negative "
            "value added). Enable rescaling of the other inputs."
        )

    return columns, factors


def _attribution_shares(row) -> dict[str, float]:
    return {
        "all_expansion": 1.0,
        "us_customer_demand": float(row.us_customer_share),
        "us_parent_owned_operator": (
            float(row.us_parent_operator_share)
            if pd.notna(row.us_parent_operator_share)
            else np.nan
        ),
    }


def run_leontief(
    A: pd.DataFrame,
    X: pd.Series,
    sector_map: pd.DataFrame,
    extension: pd.DataFrame,
    dc_map: pd.DataFrame,
    scenarios: pd.DataFrame,
    btm_natgas_share: float = 0.92,
    btm_countries: Iterable[str] = ("CAN",),
    natgas_upstream_kgco2e_per_kwh: float = 0.0,
    rescale_other_inputs: bool = True,
):
    """
    Upstream (Leontief), direct and downstream (Ghosh) output changes and
    GHG emissions for every scenario row, shock basis and attribution lens.

    Electricity-demand sensitivities
        icio_endogenous           data-centre purchases follow the ICIO column
        physical_facility_demand  hybrid column: the domestic-electricity
                                  coefficient equals the measured electricity
                                  cost share (see hybrid_target_columns)

    Grid-emissions sensitivities
        average_grid     2022 hybrid emissions extension (ICIO-consistent)
        projected_grid   destination electricity sector intensity replaced by
                         the scenario-year grid factor (kg CO2e/kWh), converted
                         to t/USD mn with the data-centre electricity price
        btm_natgas       projected_grid, with a share of facility electricity
                         generated behind the meter from natural gas (physical
                         sensitivity and btm_countries only)

    Emissions are reported as
        direct      data-centre industries' own emissions plus on-site
                    (behind-the-meter) generation
        upstream    supply-chain emissions excluding direct (includes
                    purchased grid electricity, reported separately)
        downstream  Ghosh (income-based) emissions excluding direct, from a
                    supply push at the data-centre industries only
    """
    keys = A.index.tolist()
    n = len(keys)

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

    if not 0 <= btm_natgas_share <= 1:
        raise ValueError("btm_natgas_share must lie between zero and one.")

    btm_countries = {str(c).upper() for c in btm_countries}

    LOG.info("Factorizing Leontief and Ghosh systems with %s industries.", n)

    A_np = A.to_numpy(dtype=float)
    lu, piv = lu_factor(np.eye(n) - A_np, check_finite=True)

    if not np.isfinite(lu).all() or np.any(np.abs(np.diag(lu)) < 1e-12):
        raise ValueError("The Leontief system is singular or numerically unstable.")

    ghosh_lu = build_ghosh_system(A, x_ref)

    if (
        not np.isfinite(ghosh_lu[0]).all()
        or np.any(np.abs(np.diag(ghosh_lu[0])) < 1e-12)
    ):
        raise ValueError("The Ghosh system is singular or numerically unstable.")

    # Value-added coefficients (value added plus net taxes on products).
    va_coefficients = 1.0 - A_np.sum(axis=0)

    extension = extension.set_index("io_key").loc[keys]
    base_intensity = extension["ghg_tonnes_per_usd_mn"].to_numpy(dtype=float)

    if not np.isfinite(base_intensity).all():
        raise ValueError(
            "The hybrid emissions extension contains non-finite emissions intensities."
        )

    emissions_source = extension["source"].to_numpy()

    sector_lookup = sector_map.set_index("io_key").loc[keys]
    country_by_key = sector_lookup["country"].to_numpy()
    sector_by_key = sector_lookup["sector_code"].astype(str).to_numpy()

    electricity_mask = np.isin(
        sector_by_key, ["35", "D35"]
    )

    country_blocks = {
        country: np.flatnonzero(country_by_key == country)
        for country in pd.unique(country_by_key)
    }

    baseline_emissions = x_ref * base_intensity
    baseline_total = float(baseline_emissions.sum())
    baseline_us = float(baseline_emissions[country_by_key == "USA"].sum())
    baseline_electricity = float(baseline_emissions[electricity_mask].sum())

    key_to_index = {key: i for i, key in enumerate(keys)}

    target_index_by_country = {}
    electricity_index_by_country = {}

    for row in dc_map.itertuples(index=False):
        targets = dc_targets_of(row)
        missing = [k for k, _ in targets if k not in key_to_index]
        if missing:
            raise ValueError(f"Data-centre proxy industries are missing from ICIO: {missing}")
        target_index_by_country[row.country] = (
            [key_to_index[k] for k, _ in targets],
            np.array([w for _, w in targets], dtype=float),
        )
        if row.electricity_io_key not in key_to_index:
            raise ValueError(
                f"Electricity industry {row.electricity_io_key} for "
                f"{row.country} is missing from ICIO."
            )
        electricity_index_by_country[row.country] = key_to_index[row.electricity_io_key]

    required = [
        "dc_output_shock_usd_mn",
        "us_customer_output_shock_usd_mn",
        "us_parent_output_shock_usd_mn",
    ]
    missing_columns = [c for c in required if c not in scenarios.columns]
    if missing_columns:
        raise ValueError(f"Missing 2030 output shock columns: {missing_columns}")

    if not scenarios["year"].eq(2030).any():
        raise ValueError("The scenario data contains no 2030 observations.")

    detail_rows = []
    summary_rows = []
    final_demand_rows = []
    downstream_detail_rows = []
    downstream_summary_rows = []

    def basis_values(row, basis):
        """(data-centre output, facility GWh) for the full expansion."""
        if basis == "annual_increment":
            return float(row.dc_output_yoy_usd_mn), float(row.facility_gwh_yoy)
        if basis == "cumulative_vs_first_year":
            return (
                float(row.dc_output_cumulative_usd_mn),
                float(row.facility_gwh_cumulative),
            )
        if basis == "2030_additional_output":
            return float(row.dc_output_shock_usd_mn), float(row.facility_electricity_gwh)
        raise ValueError(f"Unrecognized shock basis: {basis}")

    for row in scenarios.itertuples(index=False):

        country = row.country
        year = int(row.year)

        if country not in target_index_by_country:
            raise ValueError(f"No data-centre proxy industry is mapped for {country}.")

        target_dc_indices, target_dc_weights = target_index_by_country[country]
        electricity_index = electricity_index_by_country[country]
        destination_mask = country_by_key == country
        destination_block = country_blocks[country]

        price = float(row.price_2022_usd_mwh)
        if not np.isfinite(price) or price <= 0:
            raise ValueError("Electricity price must be positive and finite.")

        # Projected-grid intensity of the destination electricity sector:
        # kg/kWh = t/MWh; t/MWh / (USD/MWh) * 1e6 = t per USD mn.
        projected_intensity = base_intensity.copy()
        projected_intensity[electricity_index] = (
            float(row.kgco2_per_kwh) * 1e6 / price
        )

        natgas_factor = (
            float(row.kgco2_per_kwh_natgas)
            if pd.notna(row.kgco2_per_kwh_natgas)
            else np.nan
        )

        shares = _attribution_shares(row)

        bases = ["annual_increment", "cumulative_vs_first_year"]
        if year == 2030:
            bases.append("2030_additional_output")

        for shock_basis in bases:

            dc_output_full, facility_gwh_full = basis_values(row, shock_basis)

            # Electricity cost share of the data-centre output change; the
            # same for every attribution lens because both scale together.
            if abs(dc_output_full) > 1e-12:
                electricity_cost_share = (
                    facility_gwh_full * price / 1_000.0 / dc_output_full
                )
            else:
                electricity_cost_share = np.nan

            for attribution, share in shares.items():

                if pd.isna(share):
                    continue

                shock = dc_output_full * share
                facility_gwh = facility_gwh_full * share

                if abs(shock) < 1e-12:
                    # No output change: nothing to propagate. (Electricity
                    # demand is attached to output through the hybrid column,
                    # so it scales with the attribution share.)
                    continue

                dc_shocks = shock * target_dc_weights

                # -------------------- Downstream (Ghosh) --------------------
                # Supply push at the data-centre industries only.
                delta_v_down, delta_x_down = gross_output_to_primary_inputs(
                    ghosh_lu=ghosh_lu,
                    n=n,
                    target_indices=target_dc_indices,
                    target_gross_output_changes=dc_shocks,
                )

                # -------------------- Electricity sensitivities ------------
                electricity_sensitivities = ["icio_endogenous"]

                physical_ok = (
                    np.isfinite(electricity_cost_share)
                    and 0 <= electricity_cost_share < 1
                )

                if np.isfinite(electricity_cost_share) and not physical_ok:
                    if electricity_cost_share >= 1:
                        raise ValueError(
                            f"{country} {year} {shock_basis}: electricity "
                            f"expenditure is {100 * electricity_cost_share:.1f}% "
                            "of data-centre output. Output per MW is too low "
                            "for the electricity demand supplied."
                        )
                    LOG.warning(
                        "%s %s %s: negative electricity cost share (%.3f); "
                        "physical sensitivity skipped.",
                        country, year, shock_basis, electricity_cost_share,
                    )
                elif physical_ok:
                    electricity_sensitivities.append("physical_facility_demand")
                    if electricity_cost_share > 0.5:
                        LOG.warning(
                            "%s %s %s: electricity is %.0f%% of data-centre "
                            "output; other inputs are scaled to %.0f%% of the "
                            "ICIO column. Check output per MW.",
                            country, year, shock_basis,
                            100 * electricity_cost_share,
                            100 * (1 - electricity_cost_share),
                        )

                old_columns = A_np[:, target_dc_indices]

                for electricity_sensitivity in electricity_sensitivities:

                    grid_sensitivities = ["average_grid", "projected_grid"]

                    btm_available = (
                        electricity_sensitivity == "physical_facility_demand"
                        and country in btm_countries
                        and np.isfinite(natgas_factor)
                        and btm_natgas_share > 0
                    )

                    if btm_available:
                        grid_sensitivities.append("btm_natgas")

                    leontief_cache = {}

                    for grid_sensitivity in grid_sensitivities:

                        btm = grid_sensitivity == "btm_natgas"
                        grid_share = 1.0 - btm_natgas_share if btm else 1.0

                        if electricity_sensitivity == "physical_facility_demand":
                            new_columns, rescale = hybrid_target_columns(
                                A_np,
                                target_dc_indices,
                                electricity_index,
                                electricity_cost_share,
                                grid_share=grid_share,
                                rescale_other_inputs=rescale_other_inputs,
                            )
                        else:
                            new_columns = old_columns
                            rescale = np.ones(len(target_dc_indices))

                        cache_key = (electricity_sensitivity, grid_share)

                        if cache_key not in leontief_cache:
                            extra = (
                                (new_columns - old_columns) @ dc_shocks
                                if electricity_sensitivity == "physical_facility_demand"
                                else None
                            )
                            leontief_cache[cache_key] = gross_output_to_final_demand(
                                lu=lu,
                                piv=piv,
                                n=n,
                                target_indices=target_dc_indices,
                                delta_x_target=dc_shocks,
                                extra_demand=extra,
                            )

                        delta_y, delta_x = leontief_cache[cache_key]

                        if grid_sensitivity == "btm_natgas" or grid_sensitivity == "projected_grid":
                            intensity = projected_intensity
                        else:
                            intensity = base_intensity

                        # ---------------- Emissions ----------------------
                        emissions = delta_x * intensity

                        onsite_btm = 0.0
                        upstream_gas = 0.0

                        if btm:
                            # GWh * 1,000 = MWh; kg/kWh = t/MWh.
                            onsite_btm = (
                                btm_natgas_share * facility_gwh * 1_000.0 * natgas_factor
                            )
                            upstream_gas = (
                                btm_natgas_share
                                * facility_gwh
                                * 1_000.0
                                * natgas_upstream_kgco2e_per_kwh
                            )

                        btm_vector = np.zeros(n)
                        btm_vector[target_dc_indices[0]] += onsite_btm
                        gas_keys = np.flatnonzero(
                            destination_mask & np.isin(sector_by_key, ["06", "B06"])
                        )
                        if upstream_gas:
                            btm_vector[
                                gas_keys[0] if len(gas_keys) else target_dc_indices[0]
                            ] += upstream_gas

                        emissions_total = emissions + btm_vector

                        if not np.isfinite(emissions_total).all():
                            raise ValueError("Calculated upstream emissions contain non-finite values.")

                        direct_industry = float(emissions[target_dc_indices].sum())
                        direct_total = direct_industry + onsite_btm

                        purchased_electricity_output = float(
                            new_columns[electricity_index, :] @ dc_shocks
                        )
                        purchased_electricity_emissions = (
                            purchased_electricity_output * intensity[electricity_index]
                        )

                        upstream_excl_direct = (
                            float(emissions.sum()) - direct_industry + upstream_gas
                        )
                        total_footprint = direct_total + upstream_excl_direct

                        # ---------------- Output and value added ---------
                        va = va_coefficients.copy()
                        va[target_dc_indices] = 1.0 - new_columns.sum(axis=0)
                        value_added_change = va * delta_x

                        # ---------------- Cross-border intermediates -----
                        # Sales of each industry to all purchasers, with the
                        # data-centre columns replaced by the hybrid columns.
                        sales = A_np @ delta_x + (new_columns - old_columns) @ dc_shocks
                        domestic_sales = np.zeros(n)
                        for block in country_blocks.values():
                            domestic_sales[block] = A_np[np.ix_(block, block)] @ delta_x[block]
                        in_block = np.isin(target_dc_indices, destination_block)
                        domestic_sales[destination_block] += (
                            (new_columns - old_columns)[destination_block][:, in_block]
                            @ dc_shocks[in_block]
                        )
                        exports = sales - domestic_sales

                        exports_to_destination = (
                            A_np[:, destination_block] @ delta_x[destination_block]
                            + (new_columns - old_columns) @ dc_shocks
                        )
                        exports_to_destination[destination_block] = 0.0

                        common = {
                            "year": year,
                            "shock_basis": shock_basis,
                            "shock_destination": country,
                            "attribution": attribution,
                            "electricity_demand_sensitivity": electricity_sensitivity,
                            "grid_emissions_sensitivity": grid_sensitivity,
                        }

                        # ---------------- Upstream detail ----------------
                        origin_df = pd.DataFrame({
                            **common,
                            "io_key": keys,
                            "output_change_usd_mn": delta_x,
                            "value_added_change_usd_mn": value_added_change,
                            "intermediate_exports_usd_mn": exports,
                            "intermediate_exports_to_destination_usd_mn": exports_to_destination,
                            "ghg_tonnes_co2e": emissions_total,
                            "emissions_intensity_tonnes_per_usd_mn": intensity,
                            "emissions_source": emissions_source,
                            "direct_btm_adjustment_tonnes_co2e": btm_vector,
                            "is_datacentre_industry": np.isin(np.arange(n), target_dc_indices),
                        }).merge(sector_map, on="io_key", how="left", validate="one_to_one")

                        detail_rows.append(origin_df)

                        # One final-demand vector per Leontief solution
                        # (average and projected grid share the solution).
                        if grid_sensitivity != "projected_grid":
                            final_demand_rows.append(pd.DataFrame({
                                **{
                                    k: v for k, v in common.items()
                                    if k != "grid_emissions_sensitivity"
                                },
                                "grid_share": grid_share,
                                "io_key": keys,
                                "final_demand_change_usd_mn": delta_y,
                            }))

                        total = float(emissions_total.sum())
                        domestic = float(emissions_total[destination_mask].sum())
                        destination_baseline = float(baseline_emissions[destination_mask].sum())

                        summary_rows.append({
                            **common,
                            "shock_basis_note": SHOCK_BASIS_NOTES[shock_basis],
                            "target_gross_output_change_usd_mn": shock,
                            "attribution_share": share,
                            "facility_electricity_gwh": facility_gwh,
                            "physical_electricity_expenditure_usd_mn": (
                                facility_gwh * price / 1_000.0
                                if electricity_sensitivity == "physical_facility_demand"
                                else np.nan
                            ),
                            "electricity_cost_share_of_output": (
                                electricity_cost_share
                                if electricity_sensitivity == "physical_facility_demand"
                                else float(old_columns[electricity_index, :] @ target_dc_weights)
                            ),
                            "other_inputs_rescale_factor": float(rescale @ target_dc_weights),
                            "btm_natgas_share": btm_natgas_share if btm else 0.0,
                            # Footprint components (direct is NOT in upstream).
                            "direct_ghg_tonnes_co2e": direct_total,
                            "direct_datacentre_industry_ghg_tonnes_co2e": direct_industry,
                            "onsite_btm_generation_ghg_tonnes_co2e": onsite_btm,
                            "upstream_excl_direct_ghg_tonnes_co2e": upstream_excl_direct,
                            "purchased_electricity_ghg_tonnes_co2e": purchased_electricity_emissions,
                            "btm_natgas_upstream_ghg_tonnes_co2e": upstream_gas,
                            # Total footprint = direct + upstream (kept under
                            # its historical name).
                            "global_upstream_ghg_tonnes_co2e": total_footprint,
                            "us_origin_ghg_tonnes_co2e": float(emissions_total[country_by_key == "USA"].sum()),
                            "destination_country_origin_ghg_tonnes_co2e": domestic,
                            "foreign_origin_ghg_tonnes_co2e": total - domestic,
                            "electricity_sector_origin_ghg_tonnes_co2e": float(emissions_total[electricity_mask].sum()),
                            # Output gain.
                            "global_output_change_usd_mn": float(delta_x.sum()),
                            "destination_output_change_usd_mn": float(delta_x[destination_mask].sum()),
                            "foreign_output_change_usd_mn": float(delta_x[~destination_mask].sum()),
                            "global_value_added_change_usd_mn": float(value_added_change.sum()),
                            "destination_value_added_change_usd_mn": float(value_added_change[destination_mask].sum()),
                            "cross_border_intermediates_usd_mn": float(exports.sum()),
                            "intermediate_imports_to_destination_usd_mn": float(exports_to_destination.sum()),
                            "baseline_global_ghg_tonnes_co2e": baseline_total,
                            "baseline_us_origin_ghg_tonnes_co2e": baseline_us,
                            "baseline_destination_country_origin_ghg_tonnes_co2e": destination_baseline,
                            "baseline_foreign_origin_ghg_tonnes_co2e": baseline_total - destination_baseline,
                            "baseline_electricity_sector_origin_ghg_tonnes_co2e": baseline_electricity,
                        })

                        # ---------------- Downstream ---------------------
                        # No electricity target and no on-site generation:
                        # downstream is the supply push from the data-centre
                        # industries to their purchasers.
                        downstream = delta_x_down * intensity
                        down_direct = float(downstream[target_dc_indices].sum())
                        down_total = float(downstream.sum())
                        down_domestic = float(downstream[destination_mask].sum())

                        downstream_detail_rows.append(pd.DataFrame({
                            **common,
                            "io_key": keys,
                            "output_change_usd_mn": delta_x_down,
                            "primary_input_change_usd_mn": delta_v_down,
                            "ghg_tonnes_co2e": downstream,
                            "emissions_intensity_tonnes_per_usd_mn": intensity,
                            "emissions_source": emissions_source,
                            "is_datacentre_industry": np.isin(np.arange(n), target_dc_indices),
                        }).merge(sector_map, on="io_key", how="left", validate="one_to_one"))

                        downstream_summary_rows.append({
                            **common,
                            "shock_basis_note": SHOCK_BASIS_NOTES[shock_basis],
                            "itc_output_change_usd_mn": shock,
                            "attribution_share": share,
                            "downstream_excl_direct_ghg_tonnes_co2e": down_total - down_direct,
                            "direct_datacentre_industry_ghg_tonnes_co2e": down_direct,
                            "global_downstream_ghg_tonnes_co2e": down_total,
                            "destination_country_origin_ghg_tonnes_co2e": down_domestic,
                            "us_origin_ghg_tonnes_co2e": float(downstream[country_by_key == "USA"].sum()),
                            "foreign_origin_ghg_tonnes_co2e": down_total - down_domestic,
                            "electricity_sector_origin_ghg_tonnes_co2e": float(downstream[electricity_mask].sum()),
                            "global_output_change_usd_mn": float(delta_x_down.sum()),
                            "baseline_global_ghg_tonnes_co2e": baseline_total,
                            "baseline_us_origin_ghg_tonnes_co2e": baseline_us,
                            "baseline_destination_country_origin_ghg_tonnes_co2e": destination_baseline,
                            "baseline_foreign_origin_ghg_tonnes_co2e": baseline_total - destination_baseline,
                            "baseline_electricity_sector_origin_ghg_tonnes_co2e": baseline_electricity,
                        })

    if not detail_rows:
        raise ValueError(
            "No Leontief results were generated. Check the output scenario "
            "and whether any data-centre output change is non-zero."
        )

    detail = pd.concat(detail_rows, ignore_index=True)
    summary = _add_summary_ratios(
        pd.DataFrame(summary_rows),
        total_column="global_upstream_ghg_tonnes_co2e",
        extra_columns=[
            "direct_ghg_tonnes_co2e",
            "upstream_excl_direct_ghg_tonnes_co2e",
        ],
    )
    final_demand = pd.concat(final_demand_rows, ignore_index=True)

    downstream_detail = pd.concat(downstream_detail_rows, ignore_index=True)
    downstream_summary = _add_summary_ratios(
        pd.DataFrame(downstream_summary_rows),
        total_column="global_downstream_ghg_tonnes_co2e",
        extra_columns=["downstream_excl_direct_ghg_tonnes_co2e"],
    )

    return detail, summary, final_demand, downstream_detail, downstream_summary


def _add_summary_ratios(
    summary: pd.DataFrame,
    total_column: str,
    extra_columns: list[str],
) -> pd.DataFrame:
    """Origin shares, % change vs average grid and % of the 2022 baseline."""
    summary = summary.copy()
    total = summary[total_column].replace(0, np.nan)

    for name, column in [
        ("us_origin_share_pct", "us_origin_ghg_tonnes_co2e"),
        ("destination_origin_share_pct", "destination_country_origin_ghg_tonnes_co2e"),
        ("foreign_origin_share_pct", "foreign_origin_ghg_tonnes_co2e"),
        ("electricity_sector_origin_share_pct", "electricity_sector_origin_ghg_tonnes_co2e"),
    ]:
        summary[name] = 100.0 * summary[column] / total

    emission_columns = [
        total_column,
        *extra_columns,
        "us_origin_ghg_tonnes_co2e",
        "destination_country_origin_ghg_tonnes_co2e",
        "foreign_origin_ghg_tonnes_co2e",
        "electricity_sector_origin_ghg_tonnes_co2e",
    ]

    keys = [
        "year",
        "shock_basis",
        "shock_destination",
        "attribution",
        "electricity_demand_sensitivity",
    ]

    reference = (
        summary.loc[summary["grid_emissions_sensitivity"] == "average_grid", keys + emission_columns]
        .set_index(keys)
    )
    indexed = summary.set_index(keys)

    for column in emission_columns:
        base = reference[column].reindex(indexed.index).to_numpy()
        summary[f"{column}_pct_change_vs_average_grid"] = (
            (indexed[column].to_numpy() / np.where(base == 0, np.nan, base) - 1.0) * 100.0
        )

    baseline_map = {
        total_column: "baseline_global_ghg_tonnes_co2e",
        **{c: "baseline_global_ghg_tonnes_co2e" for c in extra_columns},
        "us_origin_ghg_tonnes_co2e": "baseline_us_origin_ghg_tonnes_co2e",
        "destination_country_origin_ghg_tonnes_co2e": "baseline_destination_country_origin_ghg_tonnes_co2e",
        "foreign_origin_ghg_tonnes_co2e": "baseline_foreign_origin_ghg_tonnes_co2e",
        "electricity_sector_origin_ghg_tonnes_co2e": "baseline_electricity_sector_origin_ghg_tonnes_co2e",
    }

    for column, baseline_column in baseline_map.items():
        summary[f"{column}_pct_change_vs_2022_baseline"] = (
            100.0 * summary[column] / summary[baseline_column].replace(0, np.nan)
        )

    return summary


# ---------------------------------------------------------------------
# Optional: embodied emissions of data-centre construction and equipment
# ---------------------------------------------------------------------

CAPEX_FILE = "dc_capex.csv"
CAPEX_COMPOSITION_FILE = "dc_capex_composition.csv"


def write_capex_templates(out_dir: Path) -> list[Path]:
    """Write empty templates for the optional capital-expenditure module."""
    out_dir.mkdir(parents=True, exist_ok=True)
    capex = out_dir / f"{Path(CAPEX_FILE).stem}_TEMPLATE.csv"
    composition = out_dir / f"{Path(CAPEX_COMPOSITION_FILE).stem}_TEMPLATE.csv"
    pd.DataFrame(
        columns=["country", "year", "capex_usd_mn_per_mw_2022", "source"]
    ).to_csv(capex, index=False)
    pd.DataFrame(
        columns=["country", "icio_sector_code", "share", "source"]
    ).to_csv(composition, index=False)
    return [capex, composition]


def load_capex_inputs(data_dir: Path):
    """Load dc_capex.csv and dc_capex_composition.csv (both required)."""
    capex = read_table(data_dir / CAPEX_FILE)
    require_columns(capex, ["country", "year", "capex_usd_mn_per_mw_2022"], CAPEX_FILE)
    capex["country"] = capex["country"].astype(str).str.strip().str.upper()
    capex["year"] = pd.to_numeric(capex["year"], errors="raise").astype(int)
    capex["capex_usd_mn_per_mw_2022"] = numeric_column(capex, "capex_usd_mn_per_mw_2022", CAPEX_FILE)
    check_unique(capex, ["country", "year"], CAPEX_FILE)

    composition = read_table(data_dir / CAPEX_COMPOSITION_FILE)
    require_columns(composition, ["country", "icio_sector_code", "share"], CAPEX_COMPOSITION_FILE)
    composition["country"] = composition["country"].astype(str).str.strip().str.upper()
    composition["icio_sector_code"] = composition["icio_sector_code"].astype(str).str.strip()
    composition["share"] = numeric_column(composition, "share", CAPEX_COMPOSITION_FILE)
    sums = composition.groupby("country")["share"].sum()
    if not np.allclose(sums, 1.0, atol=1e-6):
        raise ValueError(f"{CAPEX_COMPOSITION_FILE}: shares must sum to one per country: {sums.to_dict()}")
    return capex, composition


def run_capex(
    A: pd.DataFrame,
    FD: pd.DataFrame,
    extension: pd.DataFrame,
    sector_map: pd.DataFrame,
    scenarios: pd.DataFrame,
    capex: pd.DataFrame,
    composition: pd.DataFrame,
):
    """
    Embodied emissions of building and equipping new capacity.

    New capacity (MW) x capex per MW gives a gross fixed capital formation
    shock in the destination country. It is split across ICIO products with
    dc_capex_composition.csv and across countries of origin with the
    destination's 2022 GFCF purchases of each product (FD column
    <COUNTRY>_GFCF). The shock is propagated with the Leontief inverse:

        delta_x = L delta_y_GFCF,   emissions = f' delta_x

    These are one-off emissions in the year the capacity is built, not
    annual operating emissions; they are reported separately.
    """
    keys = A.index.tolist()
    n = len(keys)
    lu, piv = lu_factor(np.eye(n) - A.to_numpy(dtype=float))

    intensity = extension.set_index("io_key").loc[keys, "ghg_tonnes_per_usd_mn"].to_numpy(float)
    lookup = sector_map.set_index("io_key").loc[keys]
    country_by_key = lookup["country"].to_numpy()
    sector_by_key = lookup["sector_code"].astype(str).to_numpy()

    capex_lookup = capex.set_index(["country", "year"])["capex_usd_mn_per_mw_2022"]

    rows, detail = [], []
    scen = scenarios.sort_values(["country", "year"]).copy()
    scen["new_mw_yoy"] = scen.groupby("country")["dc_capacity_mw"].diff().fillna(0.0)
    scen["new_mw_cumulative"] = (
        scen["dc_capacity_mw"] - scen.groupby("country")["dc_capacity_mw"].transform("first")
    )

    for row in scen.itertuples(index=False):
        country, year = row.country, int(row.year)
        if (country, year) not in capex_lookup.index:
            continue
        gfcf_column = f"{country}_GFCF"
        if gfcf_column not in FD.columns:
            raise ValueError(
                f"FD has no column {gfcf_column}; cannot allocate capex across origins. "
                f"Available columns include {list(FD.columns[:8])}."
            )
        mix = composition.loc[composition["country"] == country]
        if mix.empty:
            raise ValueError(f"{CAPEX_COMPOSITION_FILE} has no rows for {country}.")

        per_mw = float(capex_lookup.loc[(country, year)])

        for basis, new_mw in [("annual_increment", row.new_mw_yoy), ("cumulative_vs_first_year", row.new_mw_cumulative)]:
            capex_usd_mn = float(new_mw) * per_mw
            if abs(capex_usd_mn) < 1e-12:
                continue
            delta_y = np.zeros(n)
            for item in mix.itertuples(index=False):
                product = sector_by_key == item.icio_sector_code
                if not product.any():
                    raise ValueError(f"Unknown ICIO sector in capex composition: {item.icio_sector_code}")
                purchases = FD.loc[np.array(keys)[product], gfcf_column].to_numpy(float).clip(min=0)
                if purchases.sum() <= 0:
                    raise ValueError(f"{country} has no 2022 GFCF purchases of sector {item.icio_sector_code}.")
                delta_y[product] += capex_usd_mn * item.share * purchases / purchases.sum()
            delta_x = lu_solve((lu, piv), delta_y)
            emissions = delta_x * intensity
            domestic = country_by_key == country
            rows.append({
                "year": year,
                "shock_basis": basis,
                "shock_destination": country,
                "new_capacity_mw": float(new_mw),
                "capex_usd_mn_per_mw_2022": per_mw,
                "capex_usd_mn": capex_usd_mn,
                "capex_imports_usd_mn": float(delta_y[~domestic].sum()),
                "global_output_change_usd_mn": float(delta_x.sum()),
                "embodied_capex_ghg_tonnes_co2e": float(emissions.sum()),
                "destination_country_origin_ghg_tonnes_co2e": float(emissions[domestic].sum()),
                "foreign_origin_ghg_tonnes_co2e": float(emissions[~domestic].sum()),
            })
            detail.append(pd.DataFrame({
                "year": year,
                "shock_basis": basis,
                "shock_destination": country,
                "io_key": keys,
                "gfcf_change_usd_mn": delta_y,
                "output_change_usd_mn": delta_x,
                "ghg_tonnes_co2e": emissions,
            }))

    return pd.DataFrame(rows), (pd.concat(detail, ignore_index=True) if detail else pd.DataFrame())

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

    # --------------------------------------------------------------
    # Electricity, behind-the-meter gas and emissions options
    # --------------------------------------------------------------
    elec = parser.add_argument_group("Electricity and emissions")
    elec.add_argument(
        "--btm-natgas-share",
        type=float,
        default=0.92,
        help="Share of facility electricity generated behind the meter from "
             "natural gas in the btm_natgas sensitivity. The default 0.92 is "
             "the share of planned Canadian capacity located in Alberta; it is "
             "an upper bound, not an observed BTM share. Source it or vary it.",
    )
    elec.add_argument(
        "--btm-countries",
        nargs="+",
        default=["CAN"],
        help="Countries for which the btm_natgas sensitivity is calculated.",
    )
    elec.add_argument(
        "--natgas-upstream-kgco2e-per-kwh",
        type=float,
        default=0.0,
        help="Upstream (extraction, processing, transport, methane leakage) "
             "emissions of gas-fired generation, kg CO2e per kWh generated. "
             "Default 0 (combustion only).",
    )
    elec.add_argument(
        "--no-rescale-other-inputs",
        action="store_true",
        help="Do not scale the other data-centre inputs down when the "
             "electricity coefficient is replaced by the physical cost share.",
    )
    elec.add_argument(
        "--it-utilisation",
        type=float,
        default=1.0,
        help="Average share of capacity drawn over the year (0-1]. Scales the "
             "electricity demand file; use when it assumes full load.",
    )
    elec.add_argument(
        "--output-includes-power-passthrough",
        action="store_true",
        help="Treat price_mn_MW_2022 as rent excluding power (wholesale colocation) "
             "and add the facility electricity bill to data-centre output.",
    )
    elec.add_argument(
        "--emissions-extension-year",
        type=int,
        default=2022,
        help="Reference year of the hybrid emissions extension. A warning is "
             "logged when it differs from --icio-year.",
    )
    elec.add_argument(
        "--capex",
        action="store_true",
        help="Also estimate embodied emissions of construction and equipment "
             "from dc_capex.csv and dc_capex_composition.csv in --data-dir.",
    )
    ew = parser.add_argument_group("Server e-waste (ewaste.py)")
    ew.add_argument("--racks-per-mw", type=float, default=ewaste.RACKS_PER_MW,
                    help="AI racks per MW of IT capacity (GB300 NVL72: 7.6).")
    ew.add_argument("--servers-per-rack", type=float, default=ewaste.SERVERS_PER_RACK,
                    help="Servers (compute trays) per rack (GB300 NVL72: 18).")
    ew.add_argument("--server-kg", type=float, default=ewaste.SERVER_KG,
                    help="Mass per server in kg (default 130).")
    ew.add_argument("--server-lifetime-years", type=float, default=None,
                    help="If given, also report the steady-state annual e-waste flow "
                         "(installed server mass / lifetime).")
    elec.add_argument(
        "--no-headline-summary",
        action="store_true",
        help="Skip the headline summary (summarize_results.py).",
    )

    # --------------------------------------------------------------
    # Canadian DSUT split of CAN_J62_63
    # --------------------------------------------------------------
    split = parser.add_argument_group("Canadian DSUT split")
    split.add_argument(
        "--split-canada-dsut",
        action="store_true",
        help="Split CAN_J62_63 into cloud (CLD), other NAICS 51821 (DPH) and rest (OTH).",
    )
    split.add_argument(
        "--canada-dsut-file",
        type=Path,
        default=None,
        help="Filled DSUT template (python canada_dsut_split.py write-templates). "
             "Default: <data-dir>/canada_dsut_2020.csv",
    )
    split.add_argument(
        "--statcan-sut-file",
        type=Path,
        default=None,
        help="StatCan 36-10-0478-01 extract (CSV) or full-table ZIP. "
             "Default: downloaded and cached in <data-dir>/statcan_cache.",
    )
    split.add_argument("--sut-year", type=int, default=2022)
    split.add_argument(
        "--canada-cloud-share-override",
        type=float,
        default=None,
        help="Cloud share of NAICS 51821 output in the SUT year, projected from the "
             "2020 DSUT with E3MC. Without it the 2020 DSUT share is used.",
    )
    split.add_argument(
        "--canada-dc-target",
        choices=["cloud_and_hosting", "cloud"],
        default="cloud_and_hosting",
        help="Canadian subsectors receiving the data-centre output shock.",
    )
    split.add_argument(
        "--canada-split-emissions-basis",
        choices=["output", "electricity_input"],
        default="output",
        help="How CAN_J62_63 direct emissions are split across subsectors.",
    )
    split.add_argument(
        "--sut-member-overrides",
        type=Path,
        default=None,
        help="CSV (member, category) correcting the SUT column classification.",
    )
    split.add_argument(
        "--dc-origin-shares",
        default=None,
        help="Country-of-origin shares for the imported inputs of CLD and DPH "
             "(dc_origin_override.py); 'default' uses the built-in C26 variant.",
    )
    split.add_argument(
        "--canada-dc-sales-structure",
        choices=["sut", "parent"],
        default="sut",
        help="How CLD and DPH sales to Canadian industries are allocated: 'sut' (default) "
             "by SUT purchasers of NAICS 51821 products; 'parent' like all of J62_63/62T63.",
    )
    split.add_argument(
        "--no-split-rebalance",
        action="store_true",
        help="Do not RAS-balance the split columns to source-table value-added shares "
             "(Canada SUT, India IOT).",
    )
    split.add_argument("--sut-pattern-518", default=cds.DEFAULT_518_PATTERN)
    split.add_argument("--sut-pattern-other-j", default=cds.DEFAULT_OTHER_J_PATTERN)

    # --------------------------------------------------------------
    # Indian split of IND_62T63 (2023-24 India IOT with AI-DC)
    # --------------------------------------------------------------
    india = parser.add_argument_group("Indian IOT split")
    india.add_argument(
        "--split-india-iot",
        type=Path,
        default=None,
        help="India IOT workbook with an AI-DC sector (e.g. IOT_2023-24_RAS.xlsx). "
             "Splits IND_62T63 into IND_62T63_DC and IND_62T63_OTH.",
    )
    india.add_argument("--india-iot-sheet", default="Z RAS Final")
    india.add_argument(
        "--india-dc-share",
        type=float,
        default=None,
        help="Data-centre share of IND_62T63 output. Default: AI-DC output / "
             "(AI-DC + Computer related services) in the IOT.",
    )
    india.add_argument(
        "--india-split-emissions-basis",
        choices=["output", "electricity_input"],
        default="output",
    )

    # --------------------------------------------------------------
    # TiVA and GHG footprint indicators
    # --------------------------------------------------------------
    ind = parser.add_argument_group("TiVA and GHG footprint indicators")
    ind.add_argument(
        "--tiva-ghg-indicators",
        action="store_true",
        help="Compute TiVA and GHG footprint indicators on the (split) ICIO.",
    )
    ind.add_argument(
        "--indicators-only",
        action="store_true",
        help="Stop after the indicators (skip scenario inputs and Leontief/Ghosh shocks).",
    )

    return parser.parse_args()


def apply_canada_split(args, A, X, FD):
    """Build the DSUT split plan and split CAN_J62_63 in A, X and FD."""
    dsut_file = args.canada_dsut_file or (args.data_dir / "canada_dsut_2020.csv")
    if not dsut_file.exists():
        template = cds.write_dsut_template(args.output_dir / "canada_dsut_2020_TEMPLATE.csv")
        raise FileNotFoundError(
            f"DSUT file not found: {dsut_file}. A template was written to {template}; "
            "fill it from the 2020 Canadian DSUT and pass it with --canada-dsut-file."
        )

    sut_file = args.statcan_sut_file
    if sut_file is None:
        sut_file = cds.fetch_statcan_sut(args.data_dir / "statcan_cache", args.sut_year)

    plan = cds.prepare_canada_split(
        sut_file=sut_file,
        dsut_file=dsut_file,
        sut_year=args.sut_year,
        cloud_share_override=args.canada_cloud_share_override,
        member_overrides=args.sut_member_overrides,
        pattern_518=args.sut_pattern_518,
        pattern_other_j=args.sut_pattern_other_j,
        # J62_63 (ICIO 2023 edition) or 62T63 (2025 edition).
        parent_sector=cds.resolve_parent_sector(A.index),
    )

    A, X, FD, split_diag = cds.split_icio(
        A, X, FD, plan,
        rebalance=not args.no_split_rebalance,
        dc_sales_structure=args.canada_dc_sales_structure,
    )
    cds.write_split_diagnostics(args.output_dir / "canada_dsut_split", plan, split_diag)

    # Country-of-origin override for the data-centre sectors (CLD, DPH).
    if getattr(args, "dc_origin_shares", None):
        import dc_origin_override as dco
        if str(args.dc_origin_shares).lower() == "default":
            shares_path = dco.write_defaults(args.output_dir / "canada_dsut_split" / "dc_origin_shares_default.csv")
        else:
            shares_path = Path(args.dc_origin_shares)
        shares = dco.load_shares(shares_path)
        A, X, FD, odiag = dco.apply_origin_override(A, X, FD, [plan.cloud_key, plan.hosting_key], shares)
        out = args.output_dir / "canada_dsut_split"
        out.mkdir(parents=True, exist_ok=True)
        odiag.to_csv(out / "dc_origin_override.csv", index=False)
        LOG.info("Applied data-centre origin override from %s", shares_path)
    return A, X, FD, plan


def apply_india_split(args, A, X, FD):
    """Split IND_62T63 into data centres (DC) and the rest (OTH) with the India IOT."""
    import india_iot_split as iis

    plan = iis.prepare_india_split(
        args.split_india_iot,
        parent_sector=cds.resolve_parent_sector(A.index, home_country="IND"),
        sheet=args.india_iot_sheet,
        dc_share_override=args.india_dc_share,
    )
    A, X, FD, diag = iis.split_icio_india(A, X, FD, plan, rebalance=not args.no_split_rebalance)
    iis.write_split_diagnostics(args.output_dir / "india_iot_split", plan, diag)
    return A, X, FD, plan


def main_indicators(args, A, X, FD, extension, plan):
    focus = list(plan.sub_keys) if plan is not None else None
    results = tgi.compute_indicators(
        A=A,
        X=X,
        FD=FD,
        intensity=extension.set_index("io_key")["ghg_tonnes_per_usd_mn"],
        focus_country="CAN",
        focus_keys=focus,
    )
    tgi.write_indicators(results, args.output_dir / "tiva_ghg_indicators")
    return results


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

    if args.emissions_extension_year != args.icio_year:
        LOG.warning(
            "The emissions extension refers to %s but the ICIO table to %s. "
            "Intensities are extension emissions divided by ICIO output, so "
            "they mix years. Rebuild the extension for %s or treat results "
            "as indicative.",
            args.emissions_extension_year,
            args.icio_year,
            args.icio_year,
        )

    # 1. Load ICIO.
    A, X, FD = load_icio(args)

    LOG.info("ICIO A dimensions: %s x %s", *A.shape)
    LOG.info("ICIO X industries: %s", len(X))
    LOG.info("ICIO FD dimensions: %s", FD.shape)

    # 1b. Optionally split Canadian data centres out of CAN_J62_63 / CAN_62T63.
    plan = None
    dc_overrides = {}
    transforms = []

    if args.split_canada_dsut:
        A, X, FD, plan = apply_canada_split(args, A, X, FD)
        dc_overrides["CAN"] = (plan.parent_key, plan.dc_targets(args.canada_dc_target))

        def canada_transform(df, _plan=plan, _A=A, _X=X):
            return cds.split_emissions_frame(
                df, _plan, A_split=_A, X_split=_X,
                basis=args.canada_split_emissions_basis,
            )

        transforms.append(canada_transform)
        LOG.info("Split ICIO dimensions: %s x %s", *A.shape)

    # 1c. Optionally split Indian data centres out of IND_62T63 (India IOT).
    if args.split_india_iot:
        A, X, FD, india_plan = apply_india_split(args, A, X, FD)
        dc_overrides["IND"] = (india_plan.parent_key, [(india_plan.dc_key, 1.0)])

        def india_transform(df, _plan=india_plan, _A=A, _X=X):
            import split_rebalance as sr

            electricity = [k for k in _A.index if k.startswith("IND_") and cds.normalise_sector_code(cds.sector_of(k)) == "35"]
            return sr.split_emissions_by_basis(
                df, _plan.parent_key, _plan.sub_keys, _X, A_split=_A,
                electricity_keys=electricity, basis=args.india_split_emissions_basis,
                label="split_india_iot",
            )

        transforms.append(india_transform)
        LOG.info("Split ICIO dimensions: %s x %s", *A.shape)

    def emissions_transform(df):
        for transform in transforms:
            df = transform(df)
        return df

    if not transforms:
        emissions_transform = None  # noqa: F811
    dc_overrides = dc_overrides or None

    # 1d. Balance check of the (split) table: row identity and value added.
    import split_rebalance as sr

    balance, negative_va = sr.balance_report(A, X, FD, tolerance=args.icio_identity_tolerance)
    sr.log_balance(balance, negative_va)
    negative_va.to_csv(args.output_dir / "icio_negative_value_added.csv", index=False)
    balance["row_errors"].to_csv(args.output_dir / "icio_row_identity_errors.csv", index=False)

    # 2. Load sector maps.
    sector_map, dc_map = load_sector_maps(A, args.data_dir, dc_overrides)

    if args.indicators_only:
        extension, coverage_summary = load_emissions_extension(
            emissions_file=args.emissions_extension,
            X=X,
            sector_map=sector_map,
            transform=emissions_transform,
        )
        extension.to_csv(args.output_dir / "icio_emissions_extension.csv", index=False)
        main_indicators(args, A, X, FD, extension, plan)
        LOG.info("Indicators-only run completed.")
        return

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
        transform=emissions_transform,
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
        it_utilisation=args.it_utilisation,
        output_includes_power_passthrough=args.output_includes_power_passthrough,
    )

    physical.to_csv(
        args.output_dir / "physical_operational_metrics.csv",
        index=False,
    )

    # End-of-life server e-waste from installed IT capacity.
    ewaste_table = ewaste.ewaste_metrics(
        output,
        racks_per_mw=args.racks_per_mw,
        servers_per_rack=args.servers_per_rack,
        server_kg=args.server_kg,
        server_lifetime_years=args.server_lifetime_years,
        coefficients=ewaste.load_coefficients(args.data_dir),
    )
    ewaste_table.to_csv(args.output_dir / "ewaste_metrics.csv", index=False)

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

    # 5b. TiVA and GHG footprint indicators on the (split) ICIO.
    if args.tiva_ghg_indicators:
        main_indicators(args, A, X, FD, extension, plan)

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
        btm_natgas_share=args.btm_natgas_share,
        btm_countries=args.btm_countries,
        natgas_upstream_kgco2e_per_kwh=args.natgas_upstream_kgco2e_per_kwh,
        rescale_other_inputs=not args.no_rescale_other_inputs,
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
                "is_datacentre_industry",
            ],
            as_index=False,
        )[
            [
                "output_change_usd_mn",
                "value_added_change_usd_mn",
                "intermediate_exports_usd_mn",
                "intermediate_exports_to_destination_usd_mn",
                "ghg_tonnes_co2e",
                "direct_btm_adjustment_tonnes_co2e",
            ]
        ].sum()
    )

    by_origin.to_csv(
        args.output_dir / "upstream_emissions_by_origin_country_sector.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Optional: embodied emissions of construction and equipment.
    # ------------------------------------------------------------
    if args.capex:
        missing = [
            f for f in (CAPEX_FILE, CAPEX_COMPOSITION_FILE)
            if not (args.data_dir / f).exists()
        ]
        if missing:
            templates = write_capex_templates(args.output_dir)
            raise FileNotFoundError(
                f"--capex needs {missing} in {args.data_dir}. Templates were "
                f"written to {[str(t) for t in templates]}."
            )
        capex, composition = load_capex_inputs(args.data_dir)
        capex_summary, capex_detail = run_capex(
            A=A,
            FD=FD,
            extension=extension,
            sector_map=sector_map,
            scenarios=scenarios,
            capex=capex,
            composition=composition,
        )
        capex_summary.to_csv(
            args.output_dir / "capex_embodied_emissions_summary.csv",
            index=False,
        )
        if not capex_detail.empty:
            capex_detail.merge(sector_map, on="io_key", how="left").groupby(
                ["year", "shock_basis", "shock_destination", "country", "sector_code"],
                as_index=False,
            )[["gfcf_change_usd_mn", "output_change_usd_mn", "ghg_tonnes_co2e"]].sum().to_csv(
                args.output_dir / "capex_embodied_emissions_by_origin_country_sector.csv",
                index=False,
            )

    # ------------------------------------------------------------
    # Headline "summary of a summary".
    # ------------------------------------------------------------
    if not args.no_headline_summary:
        import summarize_results

        summarize_results.write_headline(
            output_dir=args.output_dir,
            physical=physical,
        )

    LOG.info("Model completed successfully.")
    LOG.info("Results written to %s", args.output_dir.resolve())


if __name__ == "__main__":
    main()
