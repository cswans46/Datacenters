
from pathlib import Path

import pandas as pd

path = Path("OECD_AEA.csv")

raw = pd.read_csv(path, low_memory=False)

# Normalize column names used by the diagnostic.
for col in [
    "TIME_PERIOD",
    "REF_AREA",
    "ACTIVITY",
    "POLLUTANT",
    "UNIT_MEASURE",
    "ACTIVITY_SCOPE",
    "METHODOLOGY",
    "SOURCE",
]:
    if col in raw.columns:
        raw[col] = raw[col].astype("string").str.strip()

# Restrict to 2022.
raw = raw.loc[raw["TIME_PERIOD"].eq("2022")].copy()

print("\n=== AEA 2022 DIMENSION COMBINATIONS ===")

for col in [
    "POLLUTANT",
    "UNIT_MEASURE",
    "ACTIVITY_SCOPE",
    "METHODOLOGY",
    "SOURCE",
]:
    print(f"\n{col}:")
    print(raw[col].value_counts(dropna=False).to_string())

# Check the actual Canadian observations.
print("\n=== CANADA 2022 OBSERVATIONS ===")

can = raw.loc[raw["REF_AREA"].eq("CAN")]

print(
    can.groupby(
        [
            "POLLUTANT",
            "UNIT_MEASURE",
            "ACTIVITY_SCOPE",
            "METHODOLOGY",
        ],
        dropna=False,
    )
    .size()
    .to_string()
)

# Identify duplicate country/activity/pollutant records.
keys = ["REF_AREA", "ACTIVITY", "POLLUTANT"]

duplicates = raw.loc[
    raw.duplicated(keys, keep=False)
].copy()

print("\n=== DUPLICATE OBSERVATION DIMENSIONS ===")

if duplicates.empty:
    print("No duplicates.")
else:
    print(
        duplicates.groupby(
            [
                "UNIT_MEASURE",
                "ACTIVITY_SCOPE",
                "METHODOLOGY",
                "SOURCE",
            ],
            dropna=False,
        )
        .size()
        .to_string()
    )

    print("\n=== EXAMPLE DUPLICATE RECORDS ===")

    print(
        duplicates.loc[
            duplicates["REF_AREA"].isin(["CAN", "CHE", "TUR", "SRB"])
        ][
            [
                "REF_AREA",
                "ACTIVITY",
                "POLLUTANT",
                "UNIT_MEASURE",
                "ACTIVITY_SCOPE",
                "METHODOLOGY",
                "SOURCE",
                "OBS_VALUE",
            ]
        ]
        .sort_values(
            ["REF_AREA", "ACTIVITY", "POLLUTANT"]
        )
        .head(100)
        .to_string(index=False)
    )

duplicates.to_csv(
    "aea_2022_duplicate_diagnostics.csv",
    index=False,
)

print("\nSaved aea_2022_duplicate_diagnostics.csv")

raw = pd.read_csv("OECD_AEA.csv", low_memory=False)

raw["TIME_PERIOD"] = raw["TIME_PERIOD"].astype(str).str.strip()

# Switzerland: inspect all dimensions for the duplicate A activity.
che = raw.loc[
    raw["TIME_PERIOD"].eq("2022")
    & raw["REF_AREA"].eq("CHE")
    & raw["ACTIVITY"].eq("A")
    & raw["POLLUTANT"].eq("CO2")
    & raw["UNIT_MEASURE"].eq("T_CO2E")
    & raw["ACTIVITY_SCOPE"].eq("RES")
    & raw["METHODOLOGY"].eq("EMISSIONS_SEEA")
]

print("\n=== SWITZERLAND: ALL DIMENSIONS ===")
print(che.to_string(index=False))

# Canada: check the availability of the components needed
# to construct non-CO2 GHG emissions.
can = raw.loc[
    raw["TIME_PERIOD"].eq("2022")
    & raw["REF_AREA"].eq("CAN")
    & raw["UNIT_MEASURE"].eq("T_CO2E")
    & raw["ACTIVITY_SCOPE"].eq("RES")
    & raw["METHODOLOGY"].eq("EMISSIONS_SEEA")
    & raw["POLLUTANT"].isin(["CO2", "CO2_BIO", "GHG", "GHG_BIO"])
]

print("\n=== CANADA: POLLUTANT COVERAGE ===")
print(
    can.groupby("POLLUTANT")["OBS_VALUE"]
    .agg(["count", "sum"])
    .to_string()
)

print("\n=== CANADA: SAMPLE ACTIVITY RECORDS ===")
print(
    can.loc[
        can["ACTIVITY"].isin(["A", "A01", "C", "D", "J"])
    ][
        ["ACTIVITY", "POLLUTANT", "OBS_VALUE"]
    ].to_string(index=False)
)

