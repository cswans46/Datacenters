#!/usr/bin/env python3
"""
Public-data estimate of the cloud computing share of NAICS 51821 output,
in the format of the Canadian DSUT input (canada_dsut_2020.csv).

Why: the 2020 Canadian digital supply-use tables (DSUT) are available
from Statistics Canada only on request. This script reproduces, from
published sources, the one DSUT quantity the split needs:

    s_cloud = cloud computing services produced by NAICS 51821
              / total output of NAICS 51821

StatCan DSUT method (Statistics Canada, Digital SUTs, Nov 2022):
    "Split output of software and data processing and hosting into cloud
    vs non-cloud based on review of annual reports of largest firms.
    Assume imports and uses are equivalently split."

Firm-level annual reports with Canadian detail are not public, so the
split is built in two published steps:

    s_cloud(t) = h_CA(t) * c

    h_CA(t)  share of NAICS 51821 sales from "hosting and information
             technology (IT) infrastructure provisioning services" (HITIP),
             Annual Survey of Service Industries: Software Development and
             Computer Services (StatCan table 21-10-0210-01, The Daily).
             HITIP covers website hosting, application service provisioning
             (ASP), business process management (BPM), collocation, data
             storage and data management (2021 questionnaire).

    c        cloud share of the same NAPCS product family, from the 2012 US
             Economic Census product lines for NAICS 518210 (Canada and the
             US share NAPCS). Which lines count as cloud follows the OECD
             Handbook on Compiling Digital SUTs (2023): include application
             service provisioning and other hosting / IT infrastructure
             provisioning (IaaS, PaaS, storage, data management); exclude
             single-server website hosting and collocation; BPM is excluded
             (bundled with labour; DZ.8 leaves BPaaS unsettled).

Sensitivity:
    low      ASP + data storage + other IT infrastructure only
    central  low + data management (OECD Handbook definition)
    high     central + website hosting (DZ.8: hosting "appropriate to
             include in the cloud computing component of the digital SUTs")

StatCan's equivalence assumption is kept: cloud and non-cloud 51821 output
share the same imports and uses, so the optional DSUT use-table rows are
left blank and the model applies identical cost and sales structures.

Usage:
    python estimate_canada_cloud_share.py --out-dir <data dir>
writes canada_dsut_2020.csv (central case, DSUT template format) and
canada_cloud_share_estimates.csv (all years and cases).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import canada_dsut_split as cds


# ---------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------

SOURCES = {
    "statcan_daily_2019": "Statistics Canada, The Daily, 2021-03-04, Software development and computer service industries, 2019. https://www150.statcan.gc.ca/n1/daily-quotidien/210304/dq210304d-eng.htm",
    "statcan_daily_2020": "Statistics Canada, The Daily, 2022-03-14, Software development and computer services, 2020. https://www150.statcan.gc.ca/n1/daily-quotidien/220314/dq220314c-eng.htm",
    "statcan_daily_2021": "Statistics Canada, The Daily, 2023-03-14, Software development and computer services, 2021. https://www150.statcan.gc.ca/n1/daily-quotidien/230314/dq230314e-eng.htm",
    "statcan_daily_2022": "Statistics Canada, The Daily, 2024-03-04, Software development and computer services, 2022. https://www150.statcan.gc.ca/n1/daily-quotidien/240304/dq240304c-eng.htm",
    "statcan_2023": "Statistics Canada, Software development and computer services, 2023. https://www.statcan.gc.ca/o1/en/app/7882-software-development-and-computer-services-2023",
    "statcan_daily_2024": "Statistics Canada, The Daily, 2026-03-11, Software development and computer services, 2024. https://www150.statcan.gc.ca/n1/daily-quotidien/260311/dq260311c-eng.pdf",
    "oecd_2021_dep304": "OECD (2021), Measuring cloud services use by businesses, OECD Digital Economy Papers No. 304, Table 6 (StatCan 21-10-0210-01). https://one.oecd.org/document/DSTI/CDEP/MADE(2019)1/FINAL/en/pdf",
    "voorburg_2018": "Baer, Hill, Ponikowski (US Census Bureau), Murphy, Santiago, Carnival (BLS), ISIC 6311 Data processing, hosting, and related services, Voorburg Group, Rome 2018, Table 3. https://voorburggroup.org/Documents/2018%20Rome/Papers/1014.pdf",
    "statcan_questionnaire_2021": "Statistics Canada, 2021 Annual Survey of Service Industries: Software Development and Computer Services, questionnaire item 1(d). https://www.statcan.gc.ca/en/statistical-programs/instrument/2410_Q4_V5",
    "oecd_handbook_2023": "OECD (2023), OECD Handbook on Compiling Digital Supply and Use Tables, ch. 4 and Table 4.1. https://unstats.un.org/unsD/nationalaccount/docs/OECD_Digital_SUT_HB.pdf",
    "dz8_2023": "ISWGNA Digitalization Task Team, DZ.8 Measurement of Cloud Computing in National Accounts (endorsed, rev. 2023-03-28). https://unstats.un.org/unsd/nationalaccount/RAdocs/ENDORSED_DZ8_Cloud_Computing.pdf",
    "statcan_dsut_2022": "Statistics Canada, Digital Supply and Use Tables (presentation, Nov 2022). https://www.cepal.org/sites/default/files/events/files/cs2_digital_suts_2017_canada_eng.pdf",
}

# HITIP share of NAICS 51821 sales (percent) and 51821 operating revenue (CAD bn).
# 2013-2018 from OECD (2021) Table 6 column (b); 2019-2024 from StatCan releases.
HITIP_SHARE_PCT = {
    2013: (43.2, "oecd_2021_dep304"),
    2014: (45.3, "oecd_2021_dep304"),
    2015: (53.8, "oecd_2021_dep304"),
    2016: (55.1, "oecd_2021_dep304"),
    2017: (52.9, "oecd_2021_dep304"),
    2018: (54.4, "oecd_2021_dep304"),
    2019: (60.1, "statcan_daily_2019"),
    2020: (63.6, "statcan_daily_2020"),
    2021: (63.9, "statcan_daily_2021"),
    2022: (53.4, "statcan_daily_2022"),
    2023: (36.5, "statcan_2023"),
    2024: (33.3, "statcan_daily_2024"),
}

OPERATING_REVENUE_CAD_BN = {
    2020: (10.0, "statcan_daily_2020"),
    2021: (12.1, "statcan_daily_2021"),
    2022: (14.8, "statcan_daily_2022"),
    2023: (18.7, "statcan_2023"),
    2024: (20.9, "statcan_daily_2024"),
}

# US 2012 Economic Census, NAICS 518210 receipts (USD bn).
US_2012 = {
    "application_service_provisioning": (24.294678, "voorburg_2018"),
    "website_hosting": (12.081543, "voorburg_2018"),
    "data_management": (15.240362, "voorburg_2018"),
    "business_process_management": (18.385487, "voorburg_2018"),
    # Web hosting + ASP + data storage + other data processing/IT
    # infrastructure provisioning ("specific cloud-containing categories").
    "specific_cloud_containing": (48.0, "oecd_2021_dep304"),
    # Narrow product category "Data processing, hosting, and related services".
    "narrow_hosting_category": (70.0, "oecd_2021_dep304"),
    "industry_total": (107.994265, "voorburg_2018"),
}


def us_cloud_share_of_hitip() -> dict[str, float]:
    """Cloud share c of the HITIP-equivalent US product family, by case."""
    v = {k: val for k, (val, _) in US_2012.items()}
    storage_and_other = v["specific_cloud_containing"] - v["application_service_provisioning"] - v["website_hosting"]
    # HITIP (Canada) = narrow hosting category + BPM (BPM is listed under HITIP
    # in the StatCan questionnaire but is a separate US product family).
    hitip_equiv = v["narrow_hosting_category"] + v["business_process_management"]
    low = v["application_service_provisioning"] + storage_and_other
    central = low + v["data_management"]
    high = central + v["website_hosting"]
    return {
        "storage_and_other_it_usd_bn": storage_and_other,
        "hitip_equivalent_usd_bn": hitip_equiv,
        "low": low / hitip_equiv,
        "central": central / hitip_equiv,
        "high": high / hitip_equiv,
    }


def estimates() -> pd.DataFrame:
    c = us_cloud_share_of_hitip()
    rows = []
    for year, (pct, src) in sorted(HITIP_SHARE_PCT.items()):
        h = pct / 100.0
        rev = OPERATING_REVENUE_CAD_BN.get(year, (None, ""))[0]
        for case in ("low", "central", "high"):
            rows.append({
                "year": year,
                "case": case,
                "hitip_share_of_51821_sales": h,
                "hitip_source": src,
                "us_cloud_share_of_hitip": c[case],
                "cloud_share_of_51821_output": h * c[case],
                "operating_revenue_51821_cad_bn": rev,
                "cloud_output_51821_cad_bn": rev * h * c[case] if rev else None,
            })
    return pd.DataFrame(rows)


def write_dsut_file(path: Path, case: str = "central", year: int = 2020) -> Path:
    est = estimates()
    row = est.loc[est["year"].eq(year) & est["case"].eq(case)].iloc[0]
    total_k = row["operating_revenue_51821_cad_bn"] * 1e6   # CAD thousands
    cloud_k = total_k * row["cloud_share_of_51821_output"]

    cds.write_dsut_template(path, year=year)
    df = pd.read_csv(path, dtype=str)
    c = us_cloud_share_of_hitip()
    notes = {
        "cloud_output_industry_51821": (
            f"PUBLIC-DATA ESTIMATE ({case}), not the official DSUT. "
            f"= 51821 revenue x HITIP share {row['hitip_share_of_51821_sales']:.3f} "
            f"[{SOURCES[HITIP_SHARE_PCT[year][1]].split('. http')[0]}] x US cloud share of HITIP {c[case]:.3f} "
            "[2012 US Economic Census 518210 product lines via Voorburg 2018 and OECD DEP 304; OECD Handbook cloud definition]. "
            "See estimate_canada_cloud_share.py."
        ),
        "output_industry_51821": (
            f"NAICS 51821 operating revenue {year}, CAD {row['operating_revenue_51821_cad_bn']} bn "
            f"[{SOURCES[OPERATING_REVENUE_CAD_BN[year][1]].split('. http')[0]}]; survey sales shares used as output shares."
        ),
    }
    values = {"cloud_output_industry_51821": int(round(cloud_k)), "output_industry_51821": int(round(total_k))}
    df["value"] = df["measure"].map(values).astype("string")
    df["unit"] = "CAD thousands"
    df["source_note"] = df["measure"].map(notes).fillna(
        "Left blank: StatCan DSUT assumption that imports and uses of cloud are split in the same proportion as output."
    )
    df.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    parser.add_argument("--case", choices=["low", "central", "high"], default="central")
    parser.add_argument("--year", type=int, default=2020)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    est = estimates()
    est.to_csv(args.out_dir / "canada_cloud_share_estimates.csv", index=False)
    path = write_dsut_file(args.out_dir / f"canada_dsut_{args.year}.csv", case=args.case, year=args.year)

    c = us_cloud_share_of_hitip()
    print(f"US 2012 cloud share of HITIP-equivalent family: low {c['low']:.3f}, central {c['central']:.3f}, high {c['high']:.3f}")
    print(est.pivot(index="year", columns="case", values="cloud_share_of_51821_output").round(3).to_string())
    print(f"Wrote {path} ({args.case}) and canada_cloud_share_estimates.csv")


if __name__ == "__main__":
    main()
