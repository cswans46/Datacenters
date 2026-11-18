
import pandas as pd

a = pd.read_csv("OECD_AEA.csv", low_memory=False)

a = a[
    (a["MEASURE"] == "EMISSIONS")
    & (a["POLLUTANT"] == "GHG")
    & (a["ACTIVITY_SCOPE"] == "RES")
    & (a["UNIT_MEASURE"] == "T_CO2E")
    & (a["TIME_PERIOD"] == 2022)
].copy()

# Parent activity relevant to each uncovered country-sector.
targets = {
    "COL": {"C", "S"},
    "LUX": {"A", "C"},
    "MLT": {"A", "C"},
    "NOR": {"C", "J"},
    "NZL": {"C"},
    "SVN": {"C"},
}

for country, parents in targets.items():
    print(f"\n{'=' * 75}")
    print(country)

    for parent in sorted(parents):
        x = a[
            (a["REF_AREA"] == country)
            & (a["ACTIVITY"].astype(str).str.startswith(parent))
        ].copy()

        if x.empty:
            continue

        x = x.sort_values("ACTIVITY")

        parent_value = x.loc[
            x["ACTIVITY"] == parent, "OBS_VALUE"
        ].sum()

        children = x[x["ACTIVITY"] != parent]
        child_total = children["OBS_VALUE"].sum()

        print(f"\nParent: {parent}")
        print(f"Parent emissions: {parent_value:,.2f}")
        print(f"Sum of reported detailed activities: {child_total:,.2f}")
        print(f"Difference: {parent_value - child_total:,.2f}")
        print("Reported activities:")

        print(
            x[["ACTIVITY", "OBS_VALUE"]].to_string(index=False)
        )
