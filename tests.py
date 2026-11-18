
import pandas as pd

crosswalk = pd.read_csv("aea_crosswalk.csv", dtype=str)
sector_map = pd.read_csv("sector_map.csv", dtype=str)
io = pd.read_csv("icio_output.csv")

io["output_usd_mn"] = pd.to_numeric(
    io["output_usd_mn"], errors="coerce"
)

# Normalize sector codes for matching.
crosswalk["icio_sector_code"] = (
    crosswalk["icio_sector_code"].str.strip()
)
sector_map["sector_code"] = sector_map["sector_code"].str.strip()
sector_map["country"] = sector_map["country"].str.strip()

cases = {
    "BRN": ["C29"],
    "CHL": ["C29"],
    "SGP": ["B", "A02"],
}

for country, activities in cases.items():
    print(f"\n{'=' * 80}")
    print(f"{country}: ACTUAL CROSSWALK TARGETS AND ICIO OUTPUT")
    print(f"{'=' * 80}")

    country_map = sector_map.loc[
        sector_map["country"].eq(country),
        ["io_key", "sector_code", "sector_label"],
    ].drop_duplicates()

    country_io = io[["io_key", "output_usd_mn"]]

    country_map = country_map.merge(
        country_io,
        on="io_key",
        how="left",
    )

    for activity in activities:
        print(f"\nAEA activity: {activity}")

        targets = crosswalk.loc[
            crosswalk["aea_activity_code"].eq(activity),
            [
                "aea_activity_code",
                "icio_sector_code",
                "mapping_type",
                "hierarchy_group",
                "hierarchy_level",
                "mapping_note",
            ],
        ].drop_duplicates()

        if targets.empty:
            print("No crosswalk targets found.")
            continue

        result = targets.merge(
            country_map,
            left_on="icio_sector_code",
            right_on="sector_code",
            how="left",
        )

        print(
            result.sort_values(
                ["hierarchy_level", "output_usd_mn"],
                na_position="last",
            ).to_string(index=False)
        )
