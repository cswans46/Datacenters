
import numpy as np
import pandas as pd
import pyreadr

from pathlib import Path


# ============================================================
# 1. USER CONFIGURATION
# ============================================================

ICIO_DIR = Path(r"D:\Research\OECD ICIO 2025")

OUTPUT_DIR = Path("cloud_emissions_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# OECD ICIO input files
# ------------------------------------------------------------

FILES = {
    "A": ICIO_DIR / "A.Rdata",
    "Z": ICIO_DIR / "Z.Rdata",
    "X": ICIO_DIR / "OTHER.Rdata",
    "FD": ICIO_DIR / "OTHER.Rdata",
}

# Object names inside each RData file.
# Adjust these if the objects have different names.

OBJECT_NAMES = {
    "A": "A",
    "Z": "Z",
    "X": "X",
    "FD": "FD",
}

# ------------------------------------------------------------
# Environmental extension
# ------------------------------------------------------------

GHG_FILE = ICIO_DIR / "ghg_scope1.csv"

# ------------------------------------------------------------
# Reference year
# ------------------------------------------------------------

YEAR = 2022

# ------------------------------------------------------------
# Country and sector configuration
# ------------------------------------------------------------

IMPORTER = "USA"

EXPORTERS = {
    "Canada": "CAN",
    "India": "IND",
}

# OECD ICIO information services sector.
# This is a proxy for cloud computing services.

SERVICE_SECTOR = "J62T63"

# Monetary unit in ICIO tables.
# Assumed USD millions.

MONETARY_UNIT = "USD million"

# Emissions input unit.
# Must be tonnes CO2-equivalent.

EMISSIONS_UNIT = "tonnes CO2e"


# ============================================================
# 2. LOAD RDATA OBJECTS
# ============================================================

def load_rdata_object(path, object_name):

    """
    Load a specified object from an RData file.
    """

    objects = pyreadr.read_r(str(path))

    if object_name not in objects:

        raise KeyError(
            f"Object '{object_name}' not found in {path}.\n"
            f"Available objects: {list(objects.keys())}"
        )

    return objects[object_name]


def load_inputs():

    A = load_rdata_object(
        FILES["A"], OBJECT_NAMES["A"]
    )

    Z = load_rdata_object(
        FILES["Z"], OBJECT_NAMES["Z"]
    )

    X = load_rdata_object(
        FILES["X"], OBJECT_NAMES["X"]
    )

    FD = load_rdata_object(
        FILES["FD"], OBJECT_NAMES["FD"]
    )

    return A, Z, X, FD


# ============================================================
# 3. NORMALIZE ICIO MATRICES
# ============================================================

def normalize_matrix(matrix, name):

    """
    Ensure matrix is a labelled pandas DataFrame.
    """

    if not isinstance(matrix, pd.DataFrame):

        raise TypeError(
            f"{name} must be a pandas DataFrame. "
            f"Received {type(matrix)}."
        )

    matrix = matrix.copy()

    matrix.index = matrix.index.map(str)
    matrix.columns = matrix.columns.map(str)

    matrix = matrix.apply(
        pd.to_numeric, errors="raise"
    )

    return matrix


def normalize_output(X):

    """
    Convert gross output into a Series indexed by
    country-industry codes.
    """

    if isinstance(X, pd.Series):

        output = X.copy()

    elif isinstance(X, pd.DataFrame):

        if X.shape[1] == 1:

            output = X.iloc[:, 0].copy()

        elif X.shape[0] == 1:

            output = X.iloc[0, :].copy()

        else:

            raise ValueError(
                "X has multiple columns. "
                "Select the appropriate 2022 output vector."
            )

    else:

        raise TypeError(
            "X must be a pandas Series or DataFrame."
        )

    output.index = output.index.map(str)

    return pd.to_numeric(output)


# ============================================================
# 4. LOAD AND VALIDATE EMISSIONS DATA
# ============================================================

def load_emissions():

    emissions = pd.read_csv(GHG_FILE)

    required_columns = {
        "country",
        "industry",
        "year",
        "ghg_tonnes",
    }

    missing = required_columns - set(emissions.columns)

    if missing:

        raise ValueError(
            f"Missing emissions columns: {missing}"
        )

    emissions["year"] = pd.to_numeric(
        emissions["year"]
    )

    emissions = emissions[
        emissions["year"] == YEAR
    ].copy()

    emissions["code"] = (
        emissions["country"].astype(str)
        + "_"
        + emissions["industry"].astype(str)
    )

    emissions["ghg_tonnes"] = pd.to_numeric(
        emissions["ghg_tonnes"]
    )

    if emissions["code"].duplicated().any():

        raise ValueError(
            "Duplicate country-industry observations "
            "in emissions data."
        )

    return emissions.set_index("code")["ghg_tonnes"]


# ============================================================
# 5. ALIGN MATRICES
# ============================================================

def align_matrices(A, Z, FD, X):

    """
    Ensure all ICIO objects use the same industry ordering.
    """

    A = normalize_matrix(A, "A")
    Z = normalize_matrix(Z, "Z")
    FD = normalize_matrix(FD, "FD")

    X = normalize_output(X)

    sectors = A.index.tolist()

    if len(sectors) != len(set(sectors)):

        raise ValueError("Duplicate sector labels in A.")

    if set(A.index) != set(A.columns):

        raise ValueError(
            "A row and column labels do not match."
        )

    A = A.loc[sectors, sectors]

    if set(Z.index) != set(sectors):

        raise ValueError(
            "Z row labels do not match A."
        )

    if set(Z.columns) != set(sectors):

        raise ValueError(
            "Z column labels do not match A."
        )

    Z = Z.loc[sectors, sectors]

    if set(FD.index) != set(sectors):

        raise ValueError(
            "FD row labels do not match A."
        )

    FD = FD.loc[sectors]

    if set(X.index) != set(sectors):

        raise ValueError(
            "X index does not match A."
        )

    X = X.reindex(sectors)

    return A, Z, FD, X


# ============================================================
# 6. CONSTRUCT DIRECT EMISSIONS INTENSITIES
# ============================================================

def calculate_direct_intensity(X, emissions):

    """
    Calculate direct emissions per USD million of output.

    g_i = Direct emissions_i / Gross output_i
    """

    missing = X.index.difference(emissions.index)

    if len(missing):

        raise ValueError(
            "Missing emissions observations for "
            f"{len(missing)} ICIO sectors.\n"
            f"Examples: {missing[:10].tolist()}"
        )

    emissions = emissions.reindex(X.index)

    if (X <= 0).any():

        raise ValueError(
            "Non-positive gross output detected."
        )

    if (emissions < 0).any():

        raise ValueError(
            "Negative direct emissions detected."
        )

    g = emissions / X

    return g


# ============================================================
# 7. CALCULATE GLOBAL LEONTIEF MULTIPLIERS
# ============================================================

def calculate_leontief_intensity(A, g):

    """
    Calculate total direct and indirect emissions intensity.

    m = g * (I - A)^(-1)

    Uses a linear system rather than explicitly
    calculating the matrix inverse.
    """

    n = len(A)

    I = np.eye(n)

    A_array = A.to_numpy(dtype=float)

    g_array = g.to_numpy(dtype=float)

    # Solve:
    # (I - A)' * m' = g'

    m_array = np.linalg.solve(
        (I - A_array).T,
        g_array
    )

    return pd.Series(
        m_array,
        index=A.index,
        name="Total_Embodied_Intensity"
    )


# ============================================================
# 8. IDENTIFY US IMPORT DEMAND
# ============================================================

def calculate_bilateral_imports(
    Z,
    FD,
    exporter,
    importer,
    sector
):

    """
    Calculate imports of a specific service sector
    from exporter to importer.

    Includes:

    1. Intermediate imports:
       Exporter service industry -> importer industries.

    2. Final demand imports:
       Exporter service industry -> importer final demand.

    Assumes labels follow COUNTRY_INDUSTRY format.
    """

    export_code = f"{exporter}_{sector}"

    if export_code not in Z.index:

        raise KeyError(
            f"{export_code} not found in Z."
        )

    # --------------------------------------------------------
    # Intermediate demand
    # --------------------------------------------------------

    importer_columns = [
        col for col in Z.columns
        if col.startswith(importer + "_")
    ]

    if not importer_columns:

        raise ValueError(
            f"No intermediate demand columns found "
            f"for {importer}."
        )

    intermediate_imports = Z.loc[
        export_code, importer_columns
    ].sum()

    # --------------------------------------------------------
    # Final demand
    # --------------------------------------------------------

    final_demand_columns = [
        col for col in FD.columns
        if col.startswith(importer + "_")
    ]

    if not final_demand_columns:

        raise ValueError(
            f"No final demand columns found for {importer}."
        )

    final_imports = FD.loc[
        export_code, final_demand_columns
    ].sum()

    total_imports = (
        intermediate_imports + final_imports
    )

    return {

        "Intermediate_Imports_USD_million":
            intermediate_imports,

        "Final_Imports_USD_million":
            final_imports,

        "Total_Imports_USD_million":
            total_imports,

    }


# ============================================================
# 9. CALCULATE EMBODIED EMISSIONS
# ============================================================

def calculate_export_footprint(
    Z,
    FD,
    intensity,
    exporter_name,
    exporter_code
):

    """
    Calculate emissions embodied in bilateral
    service exports to the United States.
    """

    code = f"{exporter_code}_{SERVICE_SECTOR}"

    if code not in intensity.index:

        raise KeyError(
            f"{code} not found in emissions intensity vector."
        )

    imports = calculate_bilateral_imports(
        Z=Z,
        FD=FD,
        exporter=exporter_code,
        importer=IMPORTER,
        sector=SERVICE_SECTOR,
    )

    # Total direct and indirect emissions intensity.
    embodied_intensity = intensity.loc[code]

    # Emissions in tonnes CO2e.
    total_emissions = (
        embodied_intensity
        * imports["Total_Imports_USD_million"]
    )

    # Convert tonnes / USD million to kg / USD.
    intensity_kg_per_usd = (
        embodied_intensity / 1000
    )

    return {

        "Year": YEAR,

        "Exporter": exporter_name,

        "Importer": IMPORTER,

        "Sector": SERVICE_SECTOR,

        **imports,

        "Direct_Emissions_Intensity_t_per_USD_million":
            direct_intensity.loc[code],

        "Total_Embodied_Intensity_t_per_USD_million":
            embodied_intensity,

        "Total_Embodied_Intensity_kg_per_USD":
            intensity_kg_per_usd,

        "Embodied_Emissions_tCO2e":
            total_emissions,

        "Embodied_Emissions_ktCO2e":
            total_emissions / 1000,

    }


# ============================================================
# 10. MAIN
# ============================================================

def main():

    print("Loading OECD ICIO data...")

    A, Z, X, FD = load_inputs()

    print("Loading GHG emissions data...")

    emissions = load_emissions()

    print("Aligning ICIO matrices...")

    A, Z, FD, X = align_matrices(
        A, Z, FD, X
    )

    print("Calculating direct emissions intensities...")

    global direct_intensity

    direct_intensity = calculate_direct_intensity(
        X, emissions
    )

    print("Calculating global Leontief multipliers...")

    total_intensity = calculate_leontief_intensity(
        A, direct_intensity
    )

    print("Calculating bilateral service footprints...")

    results = []

    for exporter_name, exporter_code in EXPORTERS.items():

        result = calculate_export_footprint(
            Z=Z,
            FD=FD,
            intensity=total_intensity,
            exporter_name=exporter_name,
            exporter_code=exporter_code,
        )

        results.append(result)

    results = pd.DataFrame(results)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results.to_csv(
        OUTPUT_DIR / "bilateral_cloud_emissions.csv",
        index=False
    )

    total_intensity.to_csv(
        OUTPUT_DIR / "icio_embodied_emissions_intensity.csv"
    )

    direct_intensity.to_csv(
        OUTPUT_DIR / "icio_direct_emissions_intensity.csv"
    )

    print("\nBILATERAL CLOUD SERVICES FOOTPRINT")
    print("=" * 75)

    print(
        results.to_string(index=False)
    )

    print("\nResults saved to:")
    print(OUTPUT_DIR.resolve())


if __name__ == "__main__":

    main()
