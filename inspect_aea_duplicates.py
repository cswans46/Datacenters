
from pathlib import Path

import pandas as pd

path = Path("OECD_AEA.csv")
df = pd.read_csv(path, low_memory=False)

# Apply the same core filters intended for the model.
df["TIME_PERIOD"] = pd.to_numeric(df["TIME_PERIOD"], errors="coerce")

for col in ["REF_AREA", "ACTIVITY", "POLLUTANT",
            "MEASURE", "METHODOLOGY", "ACTIVITY_SCOPE",
            "UNIT_MEASURE"]:
    df[col] = df[col].astype("string").str.strip().str.upper()

df = df.loc[
    df["TIME_PERIOD"].eq(2022)
    & df["MEASURE"].eq("EMISSIONS")
    & df["METHODOLOGY"].eq("EMISSIONS_SEEA")
    & df["ACTIVITY_SCOPE"].eq("RES")
].copy()

print("\n=== UNIT COUNTS BY POLLUTANT ===")
print(
    df.groupby(["POLLUTANT", "UNIT_MEASURE"], dropna=False)
    .size()
    .unstack(fill_value=0)
    .to_string()
)

keys = ["REF_AREA", "ACTIVITY", "POLLUTANT"]

print("\n=== DUPLICATES AFTER RESIDENT-SCOPE FILTER ===")

duplicates = df.loc[
    df.duplicated(keys, keep=False)
].sort_values(keys + ["UNIT_MEASURE"])

print("Rows:", len(duplicates))
print("Duplicate groups:", duplicates.groupby(keys).ngroups)

if not duplicates.empty:
    print("\nUnit combinations:")
    print(
        duplicates.groupby(
            keys + ["UNIT_MEASURE"], dropna=False
        )["OBS_VALUE"]
        .agg(["count", "min", "max"])
        .head(40)
        .to_string()
    )

    # Compare values across unit representations.
    wide = duplicates.pivot_table(
        index=keys,
        columns="UNIT_MEASURE",
        values="OBS_VALUE",
        aggfunc="first",
    ).reset_index()

    print("\n=== COMPARISON OF T AND T_CO2E ===")
    if "T" in wide.columns and "T_CO2E" in wide.columns:
        wide["absolute_difference"] = (
            wide["T"] - wide["T_CO2E"]
        ).abs()

        print(
            wide[
                keys + ["T", "T_CO2E", "absolute_difference"]
            ].head(40).to_string(index=False)
        )

        print(
            "\nMaximum absolute difference:",
            wide["absolute_difference"].max()
        )
        print(
            "Groups with unequal values:",
            int((wide["absolute_difference"] > 1e-8).sum())
        )

    duplicates.to_csv(
        "aea_resident_scope_duplicates.csv", index=False
    )
    wide.to_csv(
        "aea_resident_scope_unit_comparison.csv", index=False
    )

    print("\nSaved detailed duplicate and unit-comparison files.")
else:
    print("No duplicates remain after the RES filter.")
