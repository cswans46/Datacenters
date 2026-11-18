
from pathlib import Path
import pandas as pd
import re

ROOT = Path(__file__).resolve().parent

# 1. Find the code that creates or writes icio_output.csv
script = ROOT / "build_icio_inputs.R"

print("\n=== R CODE: ICIO OUTPUT CONSTRUCTION ===")

if script.exists():
    lines = script.read_text(encoding="utf-8", errors="replace").splitlines()

    hits = [
        i for i, line in enumerate(lines)
        if re.search(
            r"icio_output|output_usd_mn|gross_output|write\.csv",
            line,
            re.IGNORECASE
        )
    ]

    shown = set()
    for i in hits:
        for j in range(max(0, i - 3), min(len(lines), i + 4)):
            if j not in shown:
                print(f"{j + 1:4}: {lines[j]}")
                shown.add(j)
        print("----")
else:
    print("build_icio_inputs.R not found.")

# 2. Inspect only the seven affected ICIO records
print("\n=== AFFECTED ICIO OUTPUT RECORDS ===")

output = pd.read_csv(ROOT / "icio_output.csv")
sector_map = pd.read_csv(ROOT / "sector_map.csv")

targets = [
    ("BRN", "29"),
    ("CHL", "29"),
    ("SGP", "02"),
    ("SGP", "05"),
    ("SGP", "06"),
    ("SGP", "07"),
    ("SGP", "08"),
    ("SGP", "09"),
]

for country, sector in targets:
    key = f"{country}_{sector}"

    row = output.loc[output["io_key"] == key]
    mapping = sector_map.loc[sector_map["io_key"] == key]

    print(f"\n{key}")
    print(row.to_string(index=False))
    if not mapping.empty:
        print("Sector mapping:")
        print(mapping.to_string(index=False))

# 3. Inspect industry-code descriptions, if available
print("\n=== RELEVANT INDUSTRY CODE DEFINITIONS ===")

codes_file = ROOT / "icio_industry_codes.csv"

if codes_file.exists():
    codes = pd.read_csv(codes_file)

    code_columns = [
        c for c in codes.columns
        if re.search(r"sector|code|industry", c, re.IGNORECASE)
    ]

    print("Columns:", codes.columns.tolist())

    for col in code_columns:
        mask = codes[col].astype(str).isin(
            [sector for _, sector in targets]
        )
        if mask.any():
            print(codes.loc[mask].to_string(index=False))
