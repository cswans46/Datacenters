
"""
Build AEA-to-ICIO emissions allocation weights.

AEA emissions observations are obtained from OECD_AEA.csv.
When an AEA activity maps to multiple ICIO industries, its emissions
are allocated using the relative 2022 ICIO gross output of those
industries within each country.

Output:
    aea_weights.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================================
# CONFIGURATION
# =====================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ICIO_X_FILE = Path(r"D:\Research\OECD Research\ICIO2025econX.RData")

AEA_FILE = SCRIPT_DIR / "OECD_AEA.csv"
CROSSWALK_FILE = SCRIPT_DIR / "aea_crosswalk.csv"
OUTPUT_FILE = SCRIPT_DIR / "aea_weights.csv"

REFERENCE_YEAR = 2022

AEA_COUNTRY_COLUMN = "REF_AREA"
AEA_YEAR_COLUMN = "TIME_PERIOD"
AEA_ACTIVITY_COLUMN = "ACTIVITY"
AEA_VALUE_COLUMN = "OBS_VALUE"

POLLUTANT_COLUMN = "POLLUTANT"
POLLUTANT_VALUE = "GHG"

UNIT_COLUMN = "UNIT_MEASURE"
UNIT_VALUE = "T_CO2E"

SCOPE_COLUMN = "ACTIVITY_SCOPE"
SCOPE_VALUE = "RES"
UNRESOLVED_FILE = SCRIPT_DIR / "aea_weights_unresolved.csv"

# =====================================================================
# DETAILED AEA ACTIVITIES USED TO BUILD ALLOCATION WEIGHTS
# =====================================================================

# Each key is an aggregate AEA activity being allocated.
# Each value maps the destination ICIO sector to the detailed AEA
# activity whose emissions should be used as its allocation weight.
#
# IMPORTANT:
# These are candidate mappings. Confirm them against the actual AEA
# activity codes and their definitions before using the weights.

DETAILED_WEIGHT_MAPPING = {
    "B": {
        "05": None,
        "06": None,
        "07": None,
        "08": None,
        "09": None,
    },
    "C24": {
        "24A": None,
        "24B": None,
    },
    "C30": {
        "301": None,
        "302T309": None,
    },
}


def load_icio_X_2022() -> pd.Series:
    import pyreadr

    if not ICIO_X_FILE.exists():
        raise FileNotFoundError(ICIO_X_FILE)

    result = pyreadr.read_r(str(ICIO_X_FILE))

    X_raw = result["ICIO2025econX"]

    # The ICIO X file contains annual observations from 1995 to 2022.
    expected_years = list(range(1995, 2023))

    if len(X_raw) != len(expected_years):
        raise ValueError(
            f"Expected {len(expected_years)} annual observations, "
            f"but found {len(X_raw)}."
        )

    # Assign the known annual sequence to the rows.
    X_raw = X_raw.copy()
    X_raw.index = expected_years

    # Select 2022.
    X_2022 = X_raw.loc[REFERENCE_YEAR]

    X_2022.index = X_raw.columns.astype(str)
    X_2022.name = "output_usd_mn"

    X_2022 = pd.to_numeric(X_2022, errors="coerce")

    if X_2022.isna().any():
        raise ValueError("2022 ICIO gross output contains missing values.")

    print(f"Loaded ICIO gross output for {REFERENCE_YEAR}.")
    print(f"Industries: {len(X_2022):,}")
    print(f"Total gross output: {X_2022.sum():,.2f}")

    return X_2022


# =====================================================================
# LOAD AEA DATA
# =====================================================================

def load_aea():

    if not AEA_FILE.exists():
        raise FileNotFoundError(AEA_FILE)

    aea = pd.read_csv(
        AEA_FILE,
        low_memory=False,
    )
    
    print(sorted(aea["ACTIVITY"].dropna().astype(str).unique()))

    required = [
        AEA_COUNTRY_COLUMN,
        AEA_YEAR_COLUMN,
        AEA_ACTIVITY_COLUMN,
        AEA_VALUE_COLUMN,
        POLLUTANT_COLUMN,
        UNIT_COLUMN,
        SCOPE_COLUMN,
    ]

    missing = set(required) - set(aea.columns)

    if missing:
        raise ValueError(
            f"OECD_AEA.csv is missing columns: {sorted(missing)}"
        )

    aea = aea.copy()

    aea["REF_AREA"] = (
        aea[AEA_COUNTRY_COLUMN].astype("string").str.strip()
    )

    aea["TIME_PERIOD"] = pd.to_numeric(
        aea[AEA_YEAR_COLUMN],
        errors="coerce",
    )

    aea["ACTIVITY"] = (
        aea[AEA_ACTIVITY_COLUMN].astype("string").str.strip()
    )

    aea["emissions_tonnes"] = pd.to_numeric(
        aea[AEA_VALUE_COLUMN],
        errors="coerce",
    )

    # Filter to the reference year and selected AEA dimensions.
    aea = aea.loc[
        aea["TIME_PERIOD"].eq(REFERENCE_YEAR)
        & aea[POLLUTANT_COLUMN].astype(str).eq(POLLUTANT_VALUE)
        & aea[UNIT_COLUMN].astype(str).eq(UNIT_VALUE)
        & aea[SCOPE_COLUMN].astype(str).eq(SCOPE_VALUE)
    ].copy()

    if aea.empty:
        raise ValueError(
            "No AEA observations remain after filtering. "
            "Check the year, pollutant, unit, and activity scope."
        )

    if aea["emissions_tonnes"].isna().any():
        raise ValueError(
            "Missing or non-numeric AEA emissions values found."
        )

    if not np.isfinite(aea["emissions_tonnes"]).all():
        raise ValueError("Non-finite AEA emissions values found.")

    if (aea["emissions_tonnes"] < 0).any():
        raise ValueError(
            "Negative emissions found. Check the selected measure."
        )

    # A country-year-activity must have one observation after filtering.
    keys = ["REF_AREA", "TIME_PERIOD", "ACTIVITY"]

    if aea.duplicated(keys).any():
        duplicate_rows = aea.loc[
            aea.duplicated(keys, keep=False), keys
        ].drop_duplicates()

        raise ValueError(
            "Duplicate AEA observations remain after filtering. "
            "Check additional dimensions before proceeding:\n"
            f"{duplicate_rows.head(20).to_string(index=False)}"
        )

    return aea


# =====================================================================
# BUILD AEA EMISSIONS WEIGHTS
# =====================================================================


def build_aea_weights(
    aea: pd.DataFrame,
    *,
    X: pd.Series,
    crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate country-specific AEA-to-ICIO allocation weights using
    2022 ICIO gross output.

    An AEA activity mapped to multiple ICIO industries is allocated
    according to the relative gross output of those industries.

    Parameters
    ----------
    aea : pd.DataFrame
        Filtered OECD AEA observations. Must contain country and activity
        identifiers, either as REF_AREA / ACTIVITY or country /
        aea_activity_code.

    X : pd.Series
        2022 ICIO gross output, indexed by country-industry codes
        such as CAN_05 or CAN_24A.

    crosswalk : pd.DataFrame
        AEA-to-ICIO crosswalk containing:
        aea_activity_code, icio_sector_code.

    Returns
    -------
    pd.DataFrame
        Country-specific allocation weights.
    """

    # ---------------------------------------------------------
    # 1. Standardize AEA identifiers
    # ---------------------------------------------------------

    aea = aea.copy()

    if "country" not in aea.columns:
        aea = aea.rename(columns={"REF_AREA": "country"})

    if "aea_activity_code" not in aea.columns:
        aea = aea.rename(columns={"ACTIVITY": "aea_activity_code"})

    required_aea = {"country", "aea_activity_code"}

    if not required_aea.issubset(aea.columns):
        raise ValueError(
            f"AEA data must contain {required_aea}. "
            f"Found: {aea.columns.tolist()}"
        )

    aea["country"] = aea["country"].astype(str).str.strip()
    aea["aea_activity_code"] = (
        aea["aea_activity_code"].astype(str).str.strip()
    )

    # Only generate weights for activities actually present in AEA.
    aea_pairs = (
        aea[["country", "aea_activity_code"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # 2. Standardize the crosswalk
    # ---------------------------------------------------------

    cw = crosswalk[
        ["aea_activity_code", "icio_sector_code"]
    ].copy()

    cw["aea_activity_code"] = (
        cw["aea_activity_code"].astype(str).str.strip()
    )
    cw["icio_sector_code"] = (
        cw["icio_sector_code"].astype(str).str.strip()
    )

    cw = cw.drop_duplicates()

    # ---------------------------------------------------------
    # 3. Identify which AEA activities have ICIO mappings
    # ---------------------------------------------------------

    mapped_activities = set(cw["aea_activity_code"])
    observed_activities = set(aea_pairs["aea_activity_code"])

    unmapped = sorted(observed_activities - mapped_activities)

    if unmapped:
        print(
            "AEA activities without an ICIO crosswalk "
            f"(not included in weights): {unmapped}"
        )

    pairs = aea_pairs.merge(
        cw,
        on="aea_activity_code",
        how="inner",
    )

    if pairs.empty:
        raise ValueError("No AEA activities could be matched to the crosswalk.")

    # ---------------------------------------------------------
    # 4. Convert 2022 ICIO output into a country-sector table
    # ---------------------------------------------------------

    if not isinstance(X, pd.Series):
        raise TypeError(
            "X must be a pandas Series indexed by ICIO industry codes."
        )

    output = X.rename("output_usd_mn").rename_axis("io_key").reset_index()

    output["io_key"] = output["io_key"].astype(str)

    output["country"] = (
        output["io_key"].str.split("_", n=1).str[0]
    )

    output["icio_sector_code"] = (
        output["io_key"].str.split("_", n=1).str[1]
    )

    output["output_usd_mn"] = pd.to_numeric(
        output["output_usd_mn"],
        errors="coerce",
    )

    if output["output_usd_mn"].isna().any():
        raise ValueError("ICIO gross output contains missing or non-numeric values.")

    if (output["output_usd_mn"] < 0).any():
        raise ValueError("ICIO gross output contains negative values.")


    # ---------------------------------------------------------
    # 5. Match AEA activities to available ICIO industries
    # ---------------------------------------------------------

    allocation = pairs.merge(
        output,
        on=["country", "icio_sector_code"],
        how="left",
        validate="many_to_one",
    )

    # Identify country-activity combinations with incomplete
    # ICIO industry coverage.
    missing = allocation["output_usd_mn"].isna()

    if missing.any():

        missing_detail = allocation.loc[
            missing,
            ["country", "aea_activity_code", "icio_sector_code"],
        ].copy()

        # Report countries that are not represented in ICIO.
        available_countries = set(output["country"].unique())

        missing_countries = sorted(
            set(missing_detail["country"]) - available_countries
        )

        if missing_countries:
            print(
                "\nAEA reporting areas not represented in ICIO "
                "(excluded):"
            )
            print(missing_countries)

        # Report unavailable industries within otherwise represented
        # countries.
        missing_industries = missing_detail.loc[
            missing_detail["country"].isin(available_countries)
        ]

        if not missing_industries.empty:
            print(
                "\nMapped industries unavailable in ICIO "
                "(affected country-activity groups excluded):"
            )
            print(
                missing_industries.to_string(index=False)
            )

        # Exclude the entire country-activity group if even one
        # mapped industry is unavailable. Do not renormalize
        # incomplete groups.
        incomplete_groups = (
            allocation.loc[
                missing,
                ["country", "aea_activity_code"],
            ]
            .drop_duplicates()
        )
        
        incomplete_groups.to_csv(
            UNRESOLVED_FILE,
            index=False,
        )

        print(
            "\nIncomplete country-activity combinations excluded: "
            f"{len(incomplete_groups):,}"
        )

        allocation = allocation.merge(
            incomplete_groups.assign(_exclude=True),
            on=["country", "aea_activity_code"],
            how="left",
        )

        allocation = allocation.loc[
            allocation["_exclude"].isna()
        ].drop(columns="_exclude")

    if allocation.empty:
        raise ValueError(
            "No complete country-activity combinations remain "
            "after matching AEA observations to ICIO output."
        )


    # ---------------------------------------------------------
    # 6. Calculate output-based allocation weights
    # ---------------------------------------------------------

    allocation["mapped_output_total"] = (
        allocation.groupby(
            ["country", "aea_activity_code"]
        )["output_usd_mn"]
        .transform("sum")
    )

    
    zero_output = allocation["mapped_output_total"] <= 0

    if zero_output.any():

        zero_groups = (
            allocation.loc[
                zero_output,
                ["country", "aea_activity_code"],
            ]
            .drop_duplicates()
            .copy()
        )

        zero_groups["exclusion_reason"] = (
            "Zero total ICIO gross output across mapped industries"
        )

        # Attach the original AEA emissions for audit purposes.
        aea_values = (
            aea.rename(
                columns={
                    "REF_AREA": "country",
                    "ACTIVITY": "aea_activity_code",
                }
            )
            .groupby(
                ["country", "aea_activity_code"],
                as_index=False,
            )["emissions_tonnes"]
            .sum()
        )

        zero_groups = zero_groups.merge(
            aea_values,
            on=["country", "aea_activity_code"],
            how="left",
        )

        zero_output_file = SCRIPT_DIR / "aea_weights_zero_output.csv"

        zero_groups.to_csv(
            zero_output_file,
            index=False,
        )

        print(
            "\nCountry-activity combinations excluded because "
            "their mapped ICIO gross output is zero:"
        )

        print(zero_groups.to_string(index=False))

        print(f"\nZero-output audit saved to: {zero_output_file}")

        # Exclude the entire country-activity combination.
        allocation = allocation.loc[~zero_output].copy()

    if allocation.empty:
        raise ValueError(
            "No valid country-activity combinations remain "
            "for allocation."
        )


    allocation["allocation_weight"] = (
        allocation["output_usd_mn"]
        / allocation["mapped_output_total"]
    )

    allocation["allocation_method"] = "icio_gross_output_2022"

    # ---------------------------------------------------------
    # 7. Validate weights
    # ---------------------------------------------------------

    weight_sums = (
        allocation.groupby(
            ["country", "aea_activity_code"]
        )["allocation_weight"]
        .sum()
    )

    if not np.allclose(weight_sums.values, 1.0, atol=1e-10):
        raise ValueError("Allocation weights do not sum to one.")

    # ---------------------------------------------------------
    # 8. Return results
    # ---------------------------------------------------------

    columns = [
        "country",
        "aea_activity_code",
        "icio_sector_code",
        "io_key",
        "output_usd_mn",
        "mapped_output_total",
        "allocation_weight",
        "allocation_method",
    ]

    return (
        allocation[columns]
        .sort_values(["country", "aea_activity_code", "icio_sector_code"])
        .reset_index(drop=True)
    )



# =====================================================================
# MAIN
# =====================================================================


def main(X_2022: pd.Series):

    print("\nLoading OECD AEA observations...")

    aea = load_aea()

    print(f"AEA observations retained: {len(aea):,}")
    print(
        "Available AEA activities:",
        sorted(aea["ACTIVITY"].dropna().unique()),
    )

    print("\nLoading AEA-to-ICIO crosswalk...")

    if not CROSSWALK_FILE.exists():
        raise FileNotFoundError(CROSSWALK_FILE)

    crosswalk = pd.read_csv(CROSSWALK_FILE)

    print(f"Crosswalk rows: {len(crosswalk):,}")

    print("\nBuilding ICIO gross-output allocation weights...")

    weights = build_aea_weights(
        aea,
        X=X_2022,
        crosswalk=crosswalk,
    )

    # Save the weights once.
    weights.to_csv(OUTPUT_FILE, index=False)

    print("\nAEA allocation weights created successfully.")

    print(f"Countries: {weights['country'].nunique()}")
    print(f"AEA activities: {weights['aea_activity_code'].nunique()}")
    print(f"Weight observations: {len(weights):,}")

    print("\nWeights by AEA activity:")

    print(
        weights.groupby("aea_activity_code")["allocation_weight"]
        .agg(["count", "min", "max", "mean"])
        .to_string()
    )

    print("\nValidation: allocation weights sum to one")

    weight_check = (
        weights.groupby(["country", "aea_activity_code"])
        ["allocation_weight"]
        .sum()
    )

    print(
        f"Maximum deviation from one: "
        f"{(weight_check - 1).abs().max():.12f}"
    )

    print(f"\nOutput: {OUTPUT_FILE}")

if __name__ == "__main__":
    X_2022 = load_icio_X_2022()
    print("\nChecking ICIO coverage...")

    icio_keys = X_2022.index.astype(str)

    icio_countries = sorted({
        key.split("_", 1)[0]
        for key in icio_keys
    })

    print(f"ICIO countries: {len(icio_countries)}")
    print("SRB included:", "SRB" in icio_countries)
    print("EU27_2020 included:", "EU27_2020" in icio_countries)
    print("OECD included:", "OECD" in icio_countries)

    if "SRB" in icio_countries:
        print(
            "SRB sectors:",
            sorted(
                key.split("_", 1)[1]
                for key in icio_keys
                if key.startswith("SRB_")
            )
        )
    main(X_2022)
