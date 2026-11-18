
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent

crosswalk = pd.read_csv(ROOT / "aea_crosswalk.csv", dtype=str)
sector_map = pd.read_csv(ROOT / "sector_map.csv", dtype=str)
output = pd.read_csv(ROOT / "icio_output.csv")

targets = [
    ("MLT", "A02", "02"),
    ("LUX", "C19", "19"),
    ("LUX", "A03", "03"),
]

# Attach gross output to the country-specific ICIO sector map.
io = sector_map.merge(
    output[["io_key", "output_usd_mn"]],
    on="io_key",
    how="left",
    validate="one_to_one",
)

print("\n=== MISSING AEA ALLOCATION DIAGNOSTICS ===")

for country, activity, sector in targets:
    print(f"\n--- {country} | {activity} | ICIO {sector} ---")

    # Country-independent AEA-to-ICIO concordance.
    cw = crosswalk.loc[
        crosswalk["aea_activity_code"].str.strip().eq(activity)
        & crosswalk["icio_sector_code"].str.strip().eq(sector)
    ]

    print("\nCrosswalk rows:")
    print(cw.to_string(index=False))

    # Country-specific ICIO industry and output.
    matched = io.loc[
        io["country"].str.strip().eq(country)
        & io["sector_code"].str.strip().eq(sector)
    ]

    print("\nICIO sector rows:")
    print(matched.to_string(index=False))

    if matched.empty:
        print("\nRESULT: Industry is absent from the ICIO sector map.")
    else:
        total_output = matched["output_usd_mn"].sum()
        print(f"\nTotal output (USD million): {total_output:,.10f}")

        if pd.isna(total_output) or total_output <= 0:
            print("RESULT: Industry has zero or missing gross output.")
        else:
            print("RESULT: Industry has positive gross output.")

    # Also show all industries in the country mapped to this AEA activity.
    all_targets = crosswalk.loc[
        crosswalk["aea_activity_code"].str.strip().eq(activity),
        "icio_sector_code",
    ].dropna().unique()

    country_targets = io.loc[
        io["country"].str.strip().eq(country)
        & io["sector_code"].str.strip().isin(all_targets)
    ].copy()

    print("\nAll ICIO industries mapped to this AEA activity:")
    print(
        country_targets[
            ["io_key", "sector_code", "sector_label", "output_usd_mn"]
        ].to_string(index=False)
    )

    print(
        "\nTotal output across mapped industries:",
        f"{country_targets['output_usd_mn'].sum():,.10f}",
    )
