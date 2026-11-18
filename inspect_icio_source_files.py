
from pathlib import Path

ROOT = Path(".")

print("=" * 90)
print("POTENTIAL ICIO SOURCE FILES")
print("=" * 90)

extensions = {".csv", ".xlsx", ".xls", ".parquet", ".feather", ".txt", ".zip", ".rdata"}

for path in sorted(ROOT.iterdir()):
    if path.is_file() and (
        path.suffix.lower() in extensions
        or "icio" in path.name.lower()
        or "output" in path.name.lower()
    ):
        try:
            size_mb = path.stat().st_size / (1024 ** 2)
        except OSError:
            size_mb = float("nan")

        print(f"{path.name:<65} {size_mb:>10.2f} MB")

print("\n" + "=" * 90)
print("PYTHON SCRIPTS THAT MAY CONSTRUCT ICIO OUTPUT")
print("=" * 90)

keywords = [
    "icio_output",
    "output_usd_mn",
    "sector_map",
    "ICIO",
    "gross_output",
]

for path in sorted(ROOT.glob("*.py")):
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        continue

    matches = [keyword for keyword in keywords if keyword.lower() in content.lower()]

    if matches:
        print(f"\n{path.name}")
        print("  Matching terms:", ", ".join(matches))
