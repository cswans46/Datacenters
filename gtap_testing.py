
from pathlib import Path
import pandas as pd
import argparse
import csv

from gams import GamsWorkspace


GAMS_SYSTEM_DIRECTORY = r"C:\GAMS\51"
SAMPLE_RECORDS = 15


def inspect_file(ws, gdx_file, label, output_dir):

    if not gdx_file.exists():
        raise FileNotFoundError(gdx_file)

    db = ws.add_database_from_gdx(str(gdx_file))

    symbols = []
    samples = []

    print(f"\n{'=' * 80}")
    print(f"{label}: {gdx_file.name}")
    print(f"{'=' * 80}")

    for symbol in db:

        try:
            nrecords = symbol.number_records
        except Exception:
            nrecords = None

        symbols.append({
            "file_type": label,
            "file": gdx_file.name,
            "symbol": symbol.name,
            "type": type(symbol).__name__,
            "dimension": symbol.dimension,
            "number_records": nrecords,
            "description": symbol.text,
        })

        print(
            f"{symbol.name:30s} "
            f"dim={symbol.dimension} "
            f"records={nrecords} "
            f"description={symbol.text}"
        )

        for i, record in enumerate(symbol):

            if i >= SAMPLE_RECORDS:
                break

            samples.append({
                "file_type": label,
                "file": gdx_file.name,
                "symbol": symbol.name,
                "record_number": i,
                "keys": "|".join(map(str, record.keys)),
                "value": getattr(record, "value", None),
            })

    return symbols, samples


def main(co2_file, nco2_file, output_dir):

    output_dir.mkdir(parents=True, exist_ok=True)

    ws = GamsWorkspace(
        system_directory=GAMS_SYSTEM_DIRECTORY
    )

    all_symbols = []
    all_samples = []

    for label, file in [
        ("CO2", co2_file),
        ("NCO2", nco2_file),
    ]:

        symbols, samples = inspect_file(
            ws,
            file,
            label,
            output_dir,
        )

        all_symbols.extend(symbols)
        all_samples.extend(samples)

    # Symbol inventory
    with (
        output_dir / "gtap_emissions_symbols.csv"
    ).open("w", newline="", encoding="utf-8-sig") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(all_symbols[0].keys()),
        )

        writer.writeheader()
        writer.writerows(all_symbols)

    # Sample records
    with (
        output_dir / "gtap_emissions_samples.csv"
    ).open("w", newline="", encoding="utf-8-sig") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(all_samples[0].keys()),
        )

        writer.writeheader()
        writer.writerows(all_samples)

    print("\nInspection complete.")
    print("gtap_emissions_symbols.csv")
    print("gtap_emissions_samples.csv")
    

def print_inspection_results(symbols_file, samples_file):
    symbols = pd.read_csv(symbols_file)
    samples = pd.read_csv(samples_file)

    print("\nSYMBOLS")
    print(symbols.to_string(index=False))

    print("\nSAMPLE RECORDS")
    print(samples.to_string(index=False))
    

def print_csv_gtap():
    BASE_DIR = Path(__file__).resolve().parent
    GAMS_DIR = r"C:\GAMS\51"

    FILES = {
        "CO2": BASE_DIR / "GDX12_23" / "GSDFEMISS.GDX",
        "NCO2": BASE_DIR / "GDX12_23" / "GSDFNCO2.GDX",
    }

    TARGET_SYMBOLS = {
        "CO2": ["MDF", "MMF"],
        "NCO2": ["EMI_IO", "EMI_ENDW", "EMI_QO", "EMI_IOP", "EMI_LU"],
    }

    SET_NAMES = {
        "CO2": ["REG", "ACTS", "COMM", "FUEL", "ENDW"],
        "NCO2": ["REG", "ACTS", "COMM", "EM", "EMN", "GHG",
                "FGAS", "NGHG", "ENDW", "AR"],
    }

    ws = GamsWorkspace(system_directory=GAMS_DIR)

    all_records = []
    domain_records = []

    for file_type, filepath in FILES.items():
        db = ws.add_database_from_gdx(str(filepath))

        print(f"\nInspecting {file_type}: {filepath.name}")

        # Export set members.
        for set_name in SET_NAMES[file_type]:
            if set_name not in [s.name for s in db]:
                continue

            symbol = db[set_name]
            members = [record.keys[0] for record in symbol]

            for member in members:
                domain_records.append({
                    "file_type": file_type,
                    "symbol": set_name,
                    "dimension": 1,
                    "member": member,
                })

        # Export all emissions records.
        for symbol_name in TARGET_SYMBOLS[file_type]:
            symbol = db[symbol_name]
            rows = []

            for record in symbol:
                keys = list(record.keys)
                value = record.value

                row = {
                    "file_type": file_type,
                    "symbol": symbol_name,
                    "value": value,
                }

                for i, key in enumerate(keys, start=1):
                    row[f"dim_{i}"] = key
                    domain_records.append({
                        "file_type": file_type,
                        "symbol": symbol_name,
                        "dimension": i,
                        "member": key,
                    })

                rows.append(row)

            df = pd.DataFrame(rows)

            # Keep all records for Canada.
            if not df.empty:
                dimension_cols = [
                    c for c in df.columns if c.startswith("dim_")
                ]
                canada = df[
                    df[dimension_cols].eq("CAN").any(axis=1)
                ]
                all_records.extend(canada.to_dict("records"))

                print(
                    f"{symbol_name}: total={len(df):,}, "
                    f"Canada={len(canada):,}, "
                    f"sum={df['value'].sum():,.6f}"
                )

    # Save outputs.
    pd.DataFrame(all_records).to_csv(
        BASE_DIR / "gtap_canada_emissions_records.csv",
        index=False,
    )

    domains = pd.DataFrame(domain_records).drop_duplicates()
    domains.to_csv(
        BASE_DIR / "gtap_emissions_domains.csv",
        index=False,
    )

    print("\nCreated:")
    print("gtap_canada_emissions_records.csv")
    print("gtap_emissions_domains.csv")

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--co2", type=Path, required=True)
    parser.add_argument("--nco2", type=Path, required=True)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )

    args = parser.parse_args()

    args.co2 = Path(args.co2).resolve()
    args.nco2 = Path(args.nco2).resolve()

    print(f"CO2 file: {args.co2}")
    print(f"CO2 exists: {args.co2.exists()}")

    print(f"NCO2 file: {args.nco2}")
    print(f"NCO2 exists: {args.nco2.exists()}")
    
    main(
        args.co2,
        args.nco2,
        args.output_dir,
    )
    
    print_inspection_results(
        args.output_dir / "gtap_emissions_symbols.csv",
        args.output_dir / "gtap_emissions_samples.csv"
    )
    
    print_csv_gtap()
