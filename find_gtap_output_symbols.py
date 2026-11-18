
from pathlib import Path
import gams.transfer as gt

ROOT = Path(r"D:\Research\Datacenters")
SYSTEM_DIRECTORY = r"C:\GAMS\51"

# Symbols potentially relevant to production and supply.
KEYWORDS = (
    "VOM", "VOA", "VOS", "VFOB",
    "VDF", "VMF", "VFA", "VFM",
    "VST", "VKB", "VDEP",
    "ACTS", "COMM", "REG",
    "OUTPUT", "PROD", "SUPPLY",
    "QO",
)

gdx_files = sorted(ROOT.rglob("*.gdx"))

print("=" * 90)
print("GTAP GDX FILE INVENTORY")
print("=" * 90)
print(f"Search directory: {ROOT}")
print(f"GDX files found: {len(gdx_files)}")

if not gdx_files:
    raise FileNotFoundError(
        f"No GDX files found under {ROOT}"
    )

read_errors = []

for path in gdx_files:

    print("\n" + "=" * 90)
    print(f"FILE: {path}")
    print("=" * 90)

    try:
        container = gt.Container(
            system_directory=SYSTEM_DIRECTORY
        )
        container.read(str(path))

        symbols = list(container.data.keys())

        print(f"Number of symbols: {len(symbols)}")
        print("\nCOMPLETE SYMBOL LIST:")
        print(", ".join(symbols))

        relevant = [
            name for name in symbols
            if any(
                keyword in name.upper()
                for keyword in KEYWORDS
            )
        ]

        print("\nPOTENTIALLY RELEVANT SYMBOLS:")
        print(relevant if relevant else "None found")

        for name in relevant:

            symbol = container[name]
            records = symbol.records

            print("\n" + "-" * 75)
            print(f"SYMBOL: {name}")
            print(f"TYPE: {type(symbol).__name__}")

            if records is None:
                print("No records")
                continue

            print(f"SHAPE: {records.shape}")
            print(f"COLUMNS: {records.columns.tolist()}")
            print("SAMPLE RECORDS:")
            print(records.head(8).to_string(index=False))

    except Exception as exc:
        read_errors.append((str(path), str(exc)))
        print(f"ERROR READING FILE: {exc}")

print("\n" + "=" * 90)
print("SCAN SUMMARY")
print("=" * 90)
print(f"GDX files found: {len(gdx_files)}")
print(f"Files with read errors: {len(read_errors)}")

for path, error in read_errors:
    print(f"\n{path}\n{error}")
