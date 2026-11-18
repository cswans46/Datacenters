
import pandas as pd

u = pd.read_csv("hybrid_emissions/uncovered_icio_industries.csv")
u = u[u["sector_code"].astype(str) != "97T98"]

c = pd.read_csv("aea_crosswalk.csv")
w = pd.read_csv("aea_weights.csv")
e = pd.read_csv("hybrid_emissions/aea_aggregate_activity_exclusions.csv")

a = pd.read_csv("OECD_AEA.csv", low_memory=False)
a = a[
    (a["MEASURE"] == "EMISSIONS")
    & (a["POLLUTANT"] == "GHG")
    & (a["ACTIVITY_SCOPE"] == "RES")
    & (a["UNIT_MEASURE"] == "T_CO2E")
    & (a["TIME_PERIOD"] == 2022)
]

print("COUNTRY / SECTOR DIAGNOSTICS")
print("=" * 100)

for _, r in u.iterrows():
    country = r["country"]
    sector = str(r["sector_code"])

    acts = (
        c.loc[
            c["icio_sector_code"].astype(str) == sector,
            "aea_activity_code",
        ]
        .dropna()
        .unique()
        .tolist()
    )

    ww = w[
        (w["country"] == country)
        & (w["icio_sector_code"].astype(str) == sector)
    ]

    ee = e[
        (e["country"] == country)
        & (e["aea_activity_code"].isin(acts))
    ]

    aa = a[
        (a["REF_AREA"] == country)
        & (a["ACTIVITY"].isin(acts))
    ]

    print(f"\n{country}_{sector}")
    print("  Crosswalk activities:", acts)
    print(
        "  Weights:",
        ww[["aea_activity_code", "allocation_weight"]].to_dict("records"),
    )
    print(
        "  Exclusions:",
        ee[["aea_activity_code", "ghg_tonnes_co2e", "reason"]].to_dict("records"),
    )
    print(
        "  Raw AEA observations:",
        aa[["ACTIVITY", "OBS_VALUE"]].to_dict("records"),
    )
