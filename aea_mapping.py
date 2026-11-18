
"""
Build an OECD AEA-to-ICIO concordance.

Classification basis:
    - OECD (2024), Greenhouse Gas Emissions Data, Annex B, Table B.4
    - OECD (2024), Annex B, Tables B.6 and B.7
    - OECD 2025 ICIO industry classification
    - JRC (2019), Environmental and Labour Accounts for OECD ICIO Tables

Inputs:
    OECD_AEA.csv
    icio_sector_map.csv

Output:
    aea_crosswalk.csv

Output fields:
    aea_activity_code
    icio_sector_code
    mapping_type
    hierarchy_group
    hierarchy_level
    mapping_note

Mapping types:
    direct
    emission_weight
    aggregate
    separate_account

Important:
    This script creates a candidate concordance. It does not sum
    overlapping AEA aggregates and detailed observations.

    Country/year-specific selection of non-overlapping observations
    must take place when constructing the emissions extension.
"""

from pathlib import Path

import pandas as pd


# =====================================================================
# PATHS
# =====================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

AEA_FILE = Path(
    r"D:\Research\OECD Research\OECD_AEA.csv"
)

ICIO_SECTOR_FILE = SCRIPT_DIR / "icio_sector_map.csv"

OUTPUT_FILE = SCRIPT_DIR / "aea_crosswalk.csv"

UNMAPPED_FILE = SCRIPT_DIR / "aea_unmapped_activities.csv"


# =====================================================================
# ICIO TARGET CLASSIFICATION
# =====================================================================

# Canonical ICIO sector identifiers.
#
# The aliases accommodate differences between the older ICIO
# classification and the 2025 extended ICIO classification.
#
# The script resolves these aliases against the actual sector
# identifiers supplied in icio_sector_map.csv.

TARGET_ALIASES = {

    # Agriculture
    "01": ["01", "A01"],
    "02": ["02", "A02"],
    "03": ["03", "A03"],

    # Mining
    "05": ["05", "B05"],
    "06": ["06", "B06"],
    "07": ["07", "B07"],
    "08": ["08", "B08"],
    "09": ["09", "B09"],

    # Manufacturing
    "10T12": ["10T12", "C10T12"],
    "13T15": ["13T15", "C13T15"],
    "16": ["16", "C16"],
    "17T18": ["17T18", "C17T18"],
    "19": ["19", "C19"],
    "20": ["20", "C20"],
    "21": ["21", "C21"],
    "22": ["22", "C22"],
    "23": ["23", "C23"],
    "24A": ["24A", "C24A"],
    "24B": ["24B", "C24B"],
    "25": ["25", "C25"],
    "26": ["26", "C26"],
    "27": ["27", "C27"],
    "28": ["28", "C28"],
    "29": ["29", "C29"],
    "301": ["301", "C301"],
    "302T309": ["302T309", "C302T309"],
    "31T33": ["31T33", "C31T33"],

    # Utilities and construction
    "35": ["35", "D35"],
    "36T39": ["36T39", "E36T39"],
    "41T43": ["41T43", "F41T43"],

    # Trade
    "45T47": ["45T47", "G45T47"],

    # Transport
    "49": ["49", "H49"],
    "50": ["50", "H50"],
    "51": ["51", "H51"],
    "52": ["52", "H52"],
    "53": ["53", "H53"],

    # Accommodation and food
    "55T56": ["55T56", "I55T56"],

    # Information and communication
    "58T60": ["58T60", "J58T60"],
    "61": ["61", "J61"],
    "62T63": ["62T63", "J62T63", "J62_63"],

    # Financial services
    "64T66": ["64T66", "K64T66"],

    # Real estate
    "68": ["68", "L68"],

    # Professional services
    "69T75": ["69T75", "M69T75"],

    # Administrative services
    "77T82": ["77T82", "N77T82"],

    # Public services
    "84": ["84", "O84"],
    "85": ["85", "P85"],
    "86T88": ["86T88", "Q86T88"],
    "90T93": ["90T93", "R90T93"],
    "94T96": ["94T96", "S94T96"],

    # Household own-use production
    "97T98": ["97T98", "T97T98"],
}


# =====================================================================
# AEA ACTIVITY MAPPINGS
# =====================================================================

# Each entry contains:
#
#     AEA activity: (target ICIO sectors, mapping type, hierarchy group,
#                    hierarchy level, explanatory note)
#
# hierarchy_level:
#     1 = broad section aggregate
#     2 = intermediate aggregate
#     3 = detailed activity
#     4 = residual / unspecified activity
#
# The hierarchy is used later to avoid double-counting.

MAPPINGS = {}


def add_mapping(
    activity,
    targets,
    mapping_type="direct",
    hierarchy_group=None,
    hierarchy_level=3,
    note="",
):
    MAPPINGS[activity] = {
        "targets": targets,
        "mapping_type": mapping_type,
        "hierarchy_group": hierarchy_group,
        "hierarchy_level": hierarchy_level,
        "mapping_note": note,
    }


# ---------------------------------------------------------------------
# AGRICULTURE, FORESTRY AND FISHING
# ---------------------------------------------------------------------

add_mapping(
    "A",
    ["01", "02", "03"],
    "aggregate",
    "A",
    1,
    "Aggregate agriculture, forestry and fishing.",
)

add_mapping("A01", ["01"], hierarchy_group="A")
add_mapping("A02", ["02"], hierarchy_group="A")
add_mapping("A03", ["03"], hierarchy_group="A")


# ---------------------------------------------------------------------
# MINING AND QUARRYING
# ---------------------------------------------------------------------

add_mapping(
    "B",
    ["05", "06", "07", "08", "09"],
    "aggregate",
    "B",
    1,
    "Aggregate mining and quarrying; allocate across mining industries.",
)


# ---------------------------------------------------------------------
# MANUFACTURING
# ---------------------------------------------------------------------

add_mapping(
    "C",
    [
        "10T12",
        "13T15",
        "16",
        "17T18",
        "19",
        "20",
        "21",
        "22",
        "23",
        "24A",
        "24B",
        "25",
        "26",
        "27",
        "28",
        "29",
        "301",
        "302T309",
        "31T33",
    ],
    "aggregate",
    "C",
    1,
    "Aggregate manufacturing; allocate across manufacturing industries.",
)

# Food, textiles and related products
add_mapping("C10T12", ["10T12"], hierarchy_group="C")
add_mapping("C13T15", ["13T15"], hierarchy_group="C")

# Wood, paper and printing
add_mapping("C16", ["16"], hierarchy_group="C")
add_mapping("C17", ["17T18"], hierarchy_group="C")
add_mapping("C18", ["17T18"], hierarchy_group="C")

add_mapping(
    "C16T18",
    ["16", "17T18"],
    "aggregate",
    "C",
    2,
    "Aggregate wood, paper and printing activities.",
)

# Other manufacturing
for activity, target in {
    "C19": "19",
    "C20": "20",
    "C21": "21",
    "C22": "22",
    "C23": "23",
    "C25": "25",
    "C26": "26",
    "C27": "27",
    "C28": "28",
    "C29": "29",
}.items():

    add_mapping(
        activity,
        [target],
        hierarchy_group="C",
    )

# Basic metals: finer 2025 ICIO split
add_mapping(
    "C24",
    ["24A", "24B"],
    "emission_weight",
    "C",
    3,
    "Basic metals; allocate across the two ICIO metal industries.",
)

# Other transport equipment: finer 2025 ICIO split
add_mapping(
    "C30",
    ["301", "302T309"],
    "emission_weight",
    "C",
    3,
    "Other transport equipment; allocate across ICIO transport equipment industries.",
)

# Furniture, other manufacturing, repair and installation
add_mapping("C31_32", ["31T33"], hierarchy_group="C")
add_mapping("C33", ["31T33"], hierarchy_group="C")

add_mapping(
    "C31T33",
    ["31T33"],
    "aggregate",
    "C",
    2,
    "Aggregate furniture, other manufacturing, repair and installation.",
)


# ---------------------------------------------------------------------
# UTILITIES AND CONSTRUCTION
# ---------------------------------------------------------------------

add_mapping("D", ["35"])

add_mapping("E", ["36T39"], "aggregate", "E", 1)
add_mapping("E36", ["36T39"], hierarchy_group="E")
add_mapping("E37T39", ["36T39"], hierarchy_group="E")

add_mapping("F", ["41T43"])


# ---------------------------------------------------------------------
# WHOLESALE AND RETAIL TRADE
# ---------------------------------------------------------------------

add_mapping(
    "G",
    ["45T47"],
    "aggregate",
    "G",
    1,
    "Aggregate wholesale and retail trade.",
)

add_mapping("G45", ["45T47"], hierarchy_group="G")
add_mapping("G46", ["45T47"], hierarchy_group="G")
add_mapping("G47", ["45T47"], hierarchy_group="G")


# ---------------------------------------------------------------------
# TRANSPORT AND STORAGE
# ---------------------------------------------------------------------

add_mapping(
    "H",
    ["49", "50", "51", "52", "53"],
    "aggregate",
    "H",
    1,
    "Aggregate transportation and storage.",
)

for activity, target in {
    "H49": "49",
    "H50": "50",
    "H51": "51",
    "H52": "52",
    "H53": "53",
}.items():

    add_mapping(
        activity,
        [target],
        hierarchy_group="H",
    )


# ---------------------------------------------------------------------
# ACCOMMODATION AND FOOD SERVICES
# ---------------------------------------------------------------------

add_mapping("I", ["55T56"])


# ---------------------------------------------------------------------
# INFORMATION AND COMMUNICATION
# ---------------------------------------------------------------------

add_mapping(
    "J",
    ["58T60", "61", "62T63"],
    "aggregate",
    "J",
    1,
    "Aggregate information and communication.",
)

add_mapping("J58", ["58T60"], hierarchy_group="J")
add_mapping("J59_60", ["58T60"], hierarchy_group="J")
add_mapping("J61", ["61"], hierarchy_group="J")
add_mapping("J62_63", ["62T63"], hierarchy_group="J")

add_mapping(
    "J58T60",
    ["58T60"],
    "aggregate",
    "J",
    2,
    "Aggregate publishing, audiovisual and broadcasting activities.",
)


# ---------------------------------------------------------------------
# FINANCIAL AND INSURANCE ACTIVITIES
# ---------------------------------------------------------------------

add_mapping(
    "K",
    ["64T66"],
    "aggregate",
    "K",
    1,
    "Aggregate financial and insurance activities.",
)

for activity in ["K64", "K65", "K66"]:
    add_mapping(
        activity,
        ["64T66"],
        hierarchy_group="K",
    )


# ---------------------------------------------------------------------
# REAL ESTATE
# ---------------------------------------------------------------------

add_mapping("L", ["68"])

add_mapping(
    "L68A",
    ["68"],
    "direct",
    "L",
    3,
    "Imputed rents of owner-occupied dwellings; map to real estate.",
)


# ---------------------------------------------------------------------
# PROFESSIONAL, SCIENTIFIC AND TECHNICAL ACTIVITIES
# ---------------------------------------------------------------------

add_mapping(
    "M",
    ["69T75"],
    "aggregate",
    "M",
    1,
    "Aggregate professional, scientific and technical activities.",
)

for activity in [
    "M69_70",
    "M71",
    "M72",
    "M73",
    "M74_75",
]:

    add_mapping(
        activity,
        ["69T75"],
        hierarchy_group="M",
    )

add_mapping(
    "M73T75",
    ["69T75"],
    "aggregate",
    "M",
    2,
    "Aggregate advertising and other professional activities.",
)


# ---------------------------------------------------------------------
# ADMINISTRATIVE AND SUPPORT SERVICES
# ---------------------------------------------------------------------

add_mapping(
    "N",
    ["77T82"],
    "aggregate",
    "N",
    1,
    "Aggregate administrative and support service activities.",
)

for activity in [
    "N77",
    "N78",
    "N79",
    "N80T82",
]:

    add_mapping(
        activity,
        ["77T82"],
        hierarchy_group="N",
    )


# ---------------------------------------------------------------------
# PUBLIC SERVICES
# ---------------------------------------------------------------------

add_mapping("O", ["84"])
add_mapping("P", ["85"])

add_mapping(
    "Q",
    ["86T88"],
    "aggregate",
    "Q",
    1,
)

add_mapping("Q86", ["86T88"], hierarchy_group="Q")
add_mapping("Q87_88", ["86T88"], hierarchy_group="Q")

add_mapping(
    "R",
    ["90T93"],
    "aggregate",
    "R",
    1,
)

add_mapping("R90T92", ["90T93"], hierarchy_group="R")
add_mapping("R93", ["90T93"], hierarchy_group="R")

add_mapping(
    "S",
    ["94T96"],
    "aggregate",
    "S",
    1,
)

for activity in ["S94", "S95", "S96"]:
    add_mapping(
        activity,
        ["94T96"],
        hierarchy_group="S",
    )


# ---------------------------------------------------------------------
# HOUSEHOLD OWN-USE PRODUCTION
# ---------------------------------------------------------------------

add_mapping(
    "T",
    ["97T98"],
    "direct",
    "T",
    3,
    "Household employers and own-use production; separate ICIO household sector.",
)


# =====================================================================
# UNSPECIFIED ACTIVITY MAPPINGS
# =====================================================================

# These are residual activities within a broad ISIC section.
# They should be retained, but allocated using emission weights
# across the relevant destination industries.
#
# They are not equivalent to the full section aggregate.
#
# If the full section aggregate is also reported for a country/year,
# the selection procedure must prevent double-counting.

RESIDUAL_MAPPINGS = {

    "A_X": ["01", "02", "03"],
    "B_X": ["05", "06", "07", "08", "09"],

    "C_X": [
        "10T12",
        "13T15",
        "16",
        "17T18",
        "19",
        "20",
        "21",
        "22",
        "23",
        "24A",
        "24B",
        "25",
        "26",
        "27",
        "28",
        "29",
        "301",
        "302T309",
        "31T33",
    ],

    "G_X": ["45T47"],
    "H_X": ["49", "50", "51", "52", "53"],
    "J_X": ["58T60", "61", "62T63"],
    "K_X": ["64T66"],
    "M_X": ["69T75"],
    "N_X": ["77T82"],
    "Q_X": ["86T88"],
    "R_X": ["90T93"],
    "S_X": ["94T96"],
}

for activity, targets in RESIDUAL_MAPPINGS.items():

    section = activity[0]

    add_mapping(
        activity,
        targets,
        "emission_weight",
        section,
        4,
        "Residual unspecified activity within the corresponding ISIC section.",
    )


# =====================================================================
# NON-INDUSTRY AND BROAD AGGREGATE ACTIVITIES
# =====================================================================

SEPARATE_ACCOUNTS = {

    "HH": (
        "Household emissions aggregate.",
    ),

    "HH_OTH": (
        "Other household emissions.",
    ),

    "HH_TR": (
        "Household transport emissions.",
    ),

    "HH_HEAT": (
        "Household heating and cooling emissions.",
    ),

    "U": (
        "Extraterritorial organizations and bodies.",
    ),

    "ATU": (
        "Aggregate of all ISIC activities.",
    ),

    "ATU_HH": (
        "Aggregate of all ISIC activities and households.",
    ),

    "GTUXH": (
        "Broad aggregate spanning activities and households.",
    ),
}


# =====================================================================
# HIERARCHICAL AGGREGATE DEFINITIONS
# =====================================================================

# Parent aggregate -> activities that it supersedes.
#
# These definitions are used by the downstream emissions-extension
# procedure to select non-overlapping AEA observations by country/year.

HIERARCHY = {

    "A": [
        "A01", "A02", "A03", "A_X",
    ],

    "B": [
        "B_X",
    ],

    "C": [
        "C10T12",
        "C13T15",
        "C16",
        "C17",
        "C18",
        "C16T18",
        "C19",
        "C20",
        "C21",
        "C22",
        "C23",
        "C24",
        "C25",
        "C26",
        "C27",
        "C28",
        "C29",
        "C30",
        "C31_32",
        "C33",
        "C31T33",
        "C_X",
    ],

    "G": ["G45", "G46", "G47", "G_X"],

    "H": ["H49", "H50", "H51", "H52", "H53", "H_X"],

    "J": [
        "J58",
        "J59_60",
        "J61",
        "J62_63",
        "J58T60",
        "J_X",
    ],

    "K": ["K64", "K65", "K66", "K_X"],

    "M": [
        "M69_70",
        "M71",
        "M72",
        "M73",
        "M74_75",
        "M73T75",
        "M_X",
    ],

    "N": ["N77", "N78", "N79", "N80T82", "N_X"],

    "Q": ["Q86", "Q87_88", "Q_X"],

    "R": ["R90T92", "R93", "R_X"],

    "S": ["S94", "S95", "S96", "S_X"],
}


# =====================================================================
# LOAD ICIO SECTOR METADATA
# =====================================================================

def load_icio_sectors():

    sectors = pd.read_csv(
        ICIO_SECTOR_FILE,
        dtype=str,
    )

    required = {
        "icio_industry_code",
        "country",
        "icio_sector_code",
    }

    missing = required.difference(sectors.columns)

    if missing:
        raise ValueError(
            f"ICIO sector file is missing columns: {sorted(missing)}"
        )

    sectors = sectors.dropna(
        subset=["icio_industry_code", "icio_sector_code"]
    ).copy()

    for column in required:
        sectors[column] = (
            sectors[column]
            .astype(str)
            .str.strip()
        )

    if sectors["icio_industry_code"].duplicated().any():

        raise ValueError(
            "ICIO industry codes are not unique."
        )

    return sectors


# =====================================================================
# RESOLVE CANONICAL TARGETS AGAINST ACTUAL ICIO CODES
# =====================================================================

def resolve_targets(canonical_targets, available_codes):

    resolved = []

    for canonical in canonical_targets:

        aliases = TARGET_ALIASES.get(
            canonical,
            [canonical],
        )

        matches = [
            alias
            for alias in aliases
            if alias in available_codes
        ]

        if len(matches) == 0:

            raise ValueError(
                f"Could not resolve canonical ICIO target "
                f"{canonical}. Tried aliases: {aliases}"
            )

        if len(matches) > 1:

            raise ValueError(
                f"Ambiguous ICIO target {canonical}. "
                f"Multiple aliases found: {matches}"
            )

        resolved.append(matches[0])

    return resolved


# =====================================================================
# BUILD CROSSWALK
# =====================================================================

def build_crosswalk():

    aea = pd.read_csv(
        AEA_FILE,
        low_memory=False,
        usecols=["ACTIVITY"],
    )

    aea_codes = set(
        aea["ACTIVITY"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    icio = load_icio_sectors()

    available_codes = set(
        icio["icio_sector_code"].unique()
    )

    rows = []
    unmapped = []
    separate = []

    for activity in sorted(aea_codes):

        if activity in SEPARATE_ACCOUNTS:

            separate.append({
                "aea_activity_code": activity,
                "mapping_type": "separate_account",
                "mapping_note": SEPARATE_ACCOUNTS[activity][0],
            })

            continue

        if activity not in MAPPINGS:

            unmapped.append(activity)

            continue

        mapping = MAPPINGS[activity]

        targets = resolve_targets(
            mapping["targets"],
            available_codes,
        )

        for target in targets:

            rows.append({
                "aea_activity_code": activity,
                "icio_sector_code": target,
                "mapping_type": mapping["mapping_type"],
                "hierarchy_group": mapping["hierarchy_group"],
                "hierarchy_level": mapping["hierarchy_level"],
                "mapping_note": mapping["mapping_note"],
            })

    crosswalk = pd.DataFrame(rows)

    if not crosswalk.empty:

        crosswalk = crosswalk.sort_values(
            [
                "hierarchy_group",
                "hierarchy_level",
                "aea_activity_code",
                "icio_sector_code",
            ],
            na_position="last",
        ).reset_index(drop=True)

    crosswalk.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    pd.DataFrame(separate).to_csv(
        SCRIPT_DIR / "aea_separate_accounts.csv",
        index=False,
    )

    pd.DataFrame({
        "aea_activity_code": unmapped,
    }).to_csv(
        UNMAPPED_FILE,
        index=False,
    )

    print("\nAEA-to-ICIO crosswalk created.")

    print(f"Mapped activities: {crosswalk['aea_activity_code'].nunique()}")
    print(f"Crosswalk rows: {len(crosswalk)}")
    print(f"Separate accounts: {len(separate)}")
    print(f"Unmapped activities: {len(unmapped)}")

    print("\nMapping types:")

    print(
        crosswalk["mapping_type"]
        .value_counts()
        .to_string()
    )

    if unmapped:

        print("\nUnmapped AEA activities:")

        print(unmapped)

    print("\nFiles written:")

    print(OUTPUT_FILE)

    print(SCRIPT_DIR / "aea_separate_accounts.csv")

    print(UNMAPPED_FILE)

    return crosswalk


# =====================================================================
# EXECUTION
# =====================================================================

if __name__ == "__main__":

    crosswalk = build_crosswalk()
