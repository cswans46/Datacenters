from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# =============================================================================
# GTAP 12 -> OECD AEA ACTIVITY CONCORDANCE
#
# PURPOSE
# -------
# Map GTAP production activities to OECD AEA activity codes.
#
# This script does NOT construct a GTAP-to-ICIO concordance.
# It does NOT allocate emissions to ICIO sectors.
# The separate AEA-to-ICIO crosswalk is read only to:
#   1. identify valid AEA activity codes;
#   2. distinguish direct activities from aggregate activities; and
#   3. exclude residual emission-weight codes from activity matching.
#
# The downstream hybrid-emissions script is responsible for:
#   GTAP activity -> AEA activity -> ICIO sector.
#
# MAPPING RULES
# -------------
# Candidate tiers are ordered from most specific to least specific.
# The script selects ALL available codes in the first tier containing
# at least one eligible match.
#
# For example:
#   PPP -> C17, C18 if both are available.
#   If neither is available, it tries the next candidate tier.
#
# MATCH STATUS
# ------------
# DIRECT_MATCH:
#   The selected AEA code is classified as direct in aea_crosswalk.csv.
#
# AGGREGATE_MATCH:
#   The selected AEA code is classified as aggregate in aea_crosswalk.csv.
#
# METRO_CONCORDANCE:
#   Explicit GTAP-to-AEA correspondence supplied from the METRO mapping.
#
# SEPARATE_NOT_ALLOCATED:
#   GTAP DWE is retained separately and is not mapped to an AEA activity.
#
# MANUAL_REVIEW:
#   No eligible AEA activity code was found.
#
# ALLOCATION WEIGHT
# -----------------
# allocation_weight = 1.0 is a placeholder, NOT an emissions share.
# Pollutant-specific allocation shares are calculated separately in the
# hybrid-emissions extension using AEA emissions observations.
#
# METRO CORRESPONDENCE
# --------------------
#   CMN -> H53, J58, J59_60, J61, J62_63
#   RSA -> L
#   DWE -> separate; no AEA mapping
#
# IMPORTANT
# ---------
# AEA activity T -> ICIO sector 97T98 is an AEA-to-ICIO correspondence.
# It has no role in this GTAP-to-AEA activity concordance and is therefore
# deliberately not generated or audited by this script.
# =============================================================================


# =============================================================================
# 1. GTAP ACTIVITY MAPPING RULES
# =============================================================================

# Each rule contains:
#   description,
#   candidate tiers,
#   configured status.
#
# The first candidate tier with at least one eligible AEA code is selected.
# All eligible codes in that tier are retained.
#
# Candidate codes must exist in the AEA crosswalk and must not be classified
# as emission_weight residual codes.

# These are valid AEA production activities, not residual codes.
#
# Their "emission_weight" classification describes how their emissions
# are allocated across ICIO industries in the separate AEA-to-ICIO
# crosswalk. It does not make them ineligible for GTAP-to-AEA matching.
#
# Other emission_weight codes (e.g. C_X, H_X, J_X) remain excluded.
GTAP_ELIGIBLE_EMISSION_WEIGHT_CODES = {
    "C24",
    "C30",
}

GTAP_RULES = {

    # -------------------------------------------------------------------------
    # Agriculture, forestry and fishing
    # -------------------------------------------------------------------------

    "pdr": (
        "Paddy rice",
        [["A01"], ["A"]],
        "CANDIDATE_TIER",
    ),
    "wht": (
        "Wheat",
        [["A01"], ["A"]],
        "CANDIDATE_TIER",
    ),
    "gro": (
        "Cereal grains",
        [["A01"], ["A"]],
        "CANDIDATE_TIER",
    ),
    "v_f": (
        "Vegetables, fruit and nuts",
        [["A01"], ["A"]],
        "CANDIDATE_TIER",
    ),
    "osd": (
        "Oil seeds",
        [["A01"], ["A"]],
        "CANDIDATE_TIER",
    ),
    "c_b": (
        "Sugar crops",
        [["A01"], ["A"]],
        "CANDIDATE_TIER",
    ),
    "pfb": (
        "Plant-based fibres",
        [["A01"], ["A"]],
        "CANDIDATE_TIER",
    ),
    "ocr": (
        "Other crops",
        [["A01"], ["A"]],
        "CANDIDATE_TIER",
    ),
    "ctl": (
        "Livestock",
        [["A01"], ["A"]],
        "CANDIDATE_TIER",
    ),
    "oap": (
        "Other animal products",
        [["A01"], ["A"]],
        "CANDIDATE_TIER",
    ),
    "rmk": (
        "Raw milk",
        [["A01"], ["A"]],
        "CANDIDATE_TIER",
    ),
    "wol": (
        "Wool and silk",
        [["A01"], ["A"]],
        "CANDIDATE_TIER",
    ),
    "frs": (
        "Forestry",
        [["A02"], ["A"]],
        "CANDIDATE_TIER",
    ),
    "fsh": (
        "Fishing",
        [["A03"], ["A"]],
        "CANDIDATE_TIER",
    ),

    # -------------------------------------------------------------------------
    # Mining and extraction
    # -------------------------------------------------------------------------

    "coa": (
        "Coal",
        [["B"]],
        "CANDIDATE_TIER",
    ),
    "oil": (
        "Oil extraction",
        [["B"]],
        "CANDIDATE_TIER",
    ),
    "gas": (
        "Gas extraction",
        [["B"]],
        "CANDIDATE_TIER",
    ),
    "oxt": (
        "Other extraction",
        [["B"]],
        "CANDIDATE_TIER",
    ),

    # -------------------------------------------------------------------------
    # Food, beverages and tobacco
    # -------------------------------------------------------------------------

    "cmt": (
        "Bovine meat products",
        [["C10T12"], ["C"]],
        "CANDIDATE_TIER",
    ),
    "omt": (
        "Other meat products",
        [["C10T12"], ["C"]],
        "CANDIDATE_TIER",
    ),
    "vol": (
        "Vegetable oils and fats",
        [["C10T12"], ["C"]],
        "CANDIDATE_TIER",
    ),
    "mil": (
        "Dairy products",
        [["C10T12"], ["C"]],
        "CANDIDATE_TIER",
    ),
    "pcr": (
        "Processed rice",
        [["C10T12"], ["C"]],
        "CANDIDATE_TIER",
    ),
    "sgr": (
        "Sugar",
        [["C10T12"], ["C"]],
        "CANDIDATE_TIER",
    ),
    "ofd": (
        "Other food products",
        [["C10T12"], ["C"]],
        "CANDIDATE_TIER",
    ),
    "b_t": (
        "Beverages and tobacco",
        [["C10T12"], ["C"]],
        "CANDIDATE_TIER",
    ),

    # -------------------------------------------------------------------------
    # Other manufacturing
    # -------------------------------------------------------------------------

    "tex": (
        "Textiles",
        [["C13T15"], ["C"]],
        "CANDIDATE_TIER",
    ),
    "wap": (
        "Wearing apparel",
        [["C13T15"], ["C"]],
        "CANDIDATE_TIER",
    ),
    "lea": (
        "Leather products",
        [["C13T15"], ["C"]],
        "CANDIDATE_TIER",
    ),
    "lum": (
        "Wood products",
        [["C16"], ["C16T18"], ["C"]],
        "CANDIDATE_TIER",
    ),
    "ppp": (
        "Paper products",
        [["C17", "C18"], ["C16T18"], ["C"]],
        "CANDIDATE_TIER",
    ),
    "p_c": (
        "Petroleum and coal products",
        [["C19"], ["C"]],
        "CANDIDATE_TIER",
    ),
    "chm": (
        "Chemical products",
        [["C20"], ["C"]],
        "CANDIDATE_TIER",
    ),
    "bph": (
        "Pharmaceutical products",
        [["C21"], ["C"]],
        "CANDIDATE_TIER",
    ),
    "rpp": (
        "Rubber and plastic products",
        [["C22"], ["C22", "C23"], ["C"]],
        "CANDIDATE_TIER",
    ),
    "nmm": (
        "Non-metallic mineral products",
        [["C23"], ["C22", "C23"], ["C"]],
        "CANDIDATE_TIER",
    ),
    "i_s": (
        "Ferrous metals",
        [["C24"], ["C24", "C25"], ["C"]],
        "CANDIDATE_TIER",
    ),
    "nfm": (
        "Non-ferrous metals",
        [["C24"], ["C24", "C25"], ["C"]],
        "CANDIDATE_TIER",
    ),
    "fmp": (
        "Fabricated metal products",
        [["C25"], ["C24", "C25"], ["C"]],
        "CANDIDATE_TIER",
    ),
    "ele": (
        "Computer and electronic products",
        [["C26"], ["C"]],
        "CANDIDATE_TIER",
    ),
    "eeq": (
        "Electrical equipment",
        [["C27"], ["C"]],
        "CANDIDATE_TIER",
    ),
    "ome": (
        "Machinery and equipment",
        [["C28"], ["C"]],
        "CANDIDATE_TIER",
    ),
    "mvh": (
        "Motor vehicles",
        [["C29"], ["C29", "C30"], ["C"]],
        "CANDIDATE_TIER",
    ),
    "otn": (
        "Other transport equipment",
        [["C30"], ["C29", "C30"], ["C"]],
        "CANDIDATE_TIER",
    ),
    "omf": (
        "Other manufacturing",
        [["C31T33"], ["C31_32", "C33"], ["C"]],
        "CANDIDATE_TIER",
    ),

    # -------------------------------------------------------------------------
    # Utilities and construction
    # -------------------------------------------------------------------------

    "ely": (
        "Electricity",
        [["D"]],
        "CANDIDATE_TIER",
    ),
    "gdt": (
        "Gas distribution",
        [["D"]],
        "CANDIDATE_TIER",
    ),
    "wtr": (
        "Water and waste services",
        [["E36", "E37T39"], ["E"]],
        "CANDIDATE_TIER",
    ),
    "cns": (
        "Construction",
        [["F"]],
        "CANDIDATE_TIER",
    ),

    # -------------------------------------------------------------------------
    # Market services
    # -------------------------------------------------------------------------

    "trd": (
        "Trade",
        [["G45", "G46", "G47"], ["G"]],
        "CANDIDATE_TIER",
    ),
    "afs": (
        "Accommodation and food services",
        [["I"]],
        "CANDIDATE_TIER",
    ),
    "otp": (
        "Land transport",
        [["H49"], ["H"]],
        "CANDIDATE_TIER",
    ),
    "wtp": (
        "Water transport",
        [["H50"], ["H"]],
        "CANDIDATE_TIER",
    ),
    "atp": (
        "Air transport",
        [["H51"], ["H"]],
        "CANDIDATE_TIER",
    ),
    "whs": (
        "Warehousing",
        [["H52"], ["H"]],
        "CANDIDATE_TIER",
    ),

    # METRO correspondence: communication and information services.
    # CMN includes postal/courier, publishing, audiovisual, broadcasting,
    # telecommunications and IT services.
    #
    # Do not replace these detailed activities with the broad J aggregate.
    "cmn": (
        "Communication and information services",
        [["H53", "J58", "J59_60", "J61", "J62_63"]],
        "METRO_CONCORDANCE",
    ),

    "ofi": (
        "Financial services",
        [["K64"], ["K"]],
        "CANDIDATE_TIER",
    ),
    "ins": (
        "Insurance",
        [["K65"], ["K"]],
        "CANDIDATE_TIER",
    ),

    # METRO correspondence: real estate (ISIC 68).
    # L is the direct real-estate activity.
    # L68A is imputed rents of owner-occupied dwellings and is not used.
    "rsa": (
        "Real estate",
        [["L"]],
        "METRO_CONCORDANCE",
    ),

    # Dwellings are retained separately.
    # Do not map DWE to L68A, L, or another AEA production activity.
    "dwe": (
        "Dwellings",
        [],
        "SEPARATE_NOT_ALLOCATED",
    ),

    "obs": (
        "Other business services",
        [["M", "N"]],
        "CANDIDATE_TIER",
    ),
    "ros": (
        "Recreational and other services",
        [["R", "S"]],
        "CANDIDATE_TIER",
    ),

    # -------------------------------------------------------------------------
    # Non-market services
    # -------------------------------------------------------------------------

    "osg": (
        "Public administration",
        [["O"]],
        "CANDIDATE_TIER",
    ),
    "edu": (
        "Education",
        [["P"]],
        "CANDIDATE_TIER",
    ),
    "hht": (
        "Human health and social work",
        [["Q86", "Q87_88"], ["Q"]],
        "CANDIDATE_TIER",
    ),
}


# Explicit METRO mappings that must be present whenever the corresponding
# GTAP activity is present in the GTAP activity set.
METRO_REQUIRED = {
    "CMN": {"H53", "J58", "J59_60", "J61", "J62_63"},
    "RSA": {"L"},
}


OUTPUT_COLUMNS = [
    "gtap_activity_code",
    "aea_activity_code",
    "allocation_weight",
    "gtap_description",
    "match_status",
    "notes",
]


# =============================================================================
# 2. NORMALIZATION AND INPUT READING
# =============================================================================

def norm(series: pd.Series) -> pd.Series:
    """Normalize classification codes for case-insensitive matching."""
    return (
        series.astype("string")
        .str.strip()
        .str.upper()
    )


def read_gtap_activities(args) -> list[str]:
    """
    Read the GTAP ACTS set from the exported domains CSV.

    If the domains CSV is missing or contains no ACTS members, read ACTS
    directly from the GTAP GDX using the GAMS Python API.
    """

    if args.domains_file.exists():
        domains = pd.read_csv(args.domains_file)

        required = {"symbol", "dimension", "member"}
        if not required.issubset(domains.columns):
            raise ValueError(
                f"{args.domains_file} must contain {sorted(required)}"
            )

        acts = domains.loc[
            (
                domains["symbol"].astype(str).str.strip().str.upper()
                == "ACTS"
            )
            & (
                pd.to_numeric(
                    domains["dimension"],
                    errors="coerce",
                )
                == 1
            ),
            "member",
        ]

        acts = norm(acts).dropna()
        acts = acts[acts.ne("")]

        if not acts.empty:
            return sorted(acts.unique().tolist())

    try:
        from gams import GamsWorkspace
    except ImportError as exc:
        raise ImportError(
            "Could not read GTAP activities. Provide "
            "gtap_emissions_domains.csv or install the GAMS Python API."
        ) from exc

    gdx_path = args.gtap_co2

    if not gdx_path.is_absolute():
        gdx_path = args.data_dir / gdx_path

    if not gdx_path.exists():
        raise FileNotFoundError(gdx_path)

    ws = GamsWorkspace(system_directory=args.gams_system_dir)
    db = ws.add_database_from_gdx(str(gdx_path.resolve()))

    if "ACTS" not in {symbol.name for symbol in db}:
        raise ValueError("ACTS set was not found in the GTAP CO2 GDX.")

    return sorted({
        str(record.keys[0]).strip().upper()
        for record in db["ACTS"]
    })



def prepare_aea_classification(
    crosswalk: pd.DataFrame,
) -> tuple[set[str], dict[str, str], dict[str, set[str]]]:
    """
    Prepare eligible AEA activity codes and their classification metadata.

    Eligible activity codes:
      - mapping_type = direct
      - mapping_type = aggregate
      - explicitly approved production activities C24 and C30, which
        have mapping_type = emission_weight.

    Important distinction:
      C24 and C30 are valid AEA production activities. Their
      emission_weight classification specifies how their emissions
      are allocated across ICIO industries. It does not indicate that
      these are residual activities.

    Other emission_weight codes, such as C_X, H_X and J_X, remain
    excluded from GTAP activity matching.

    Returns
    -------
    available_aea:
        Set of eligible AEA activity codes.

    aea_granularity:
        Dictionary mapping AEA activity code to DIRECT or AGGREGATE.

    aea_mapping_types:
        Dictionary mapping AEA activity code to all mapping_type values
        appearing in the AEA-to-ICIO crosswalk.
    """

    required = {
        "aea_activity_code",
        "icio_sector_code",
        "mapping_type",
    }

    if not required.issubset(crosswalk.columns):
        raise ValueError(
            "aea_crosswalk.csv must contain "
            f"{sorted(required)}. The mapping_type column is needed "
            "to distinguish direct activities, aggregates and "
            "emission-weight activities."
        )

    data = crosswalk.copy()

    data["aea_activity_code"] = norm(data["aea_activity_code"])
    data["mapping_type"] = norm(data["mapping_type"])

    data = data.dropna(
        subset=["aea_activity_code", "mapping_type"]
    )

    data = data[
        data["aea_activity_code"].ne("")
        & data["mapping_type"].ne("")
    ]

    aea_mapping_types = (
        data.groupby("aea_activity_code")["mapping_type"]
        .agg(lambda values: set(values))
        .to_dict()
    )

    aea_granularity = {}

    for code, mapping_types in aea_mapping_types.items():

        # Ordinary directly corresponding AEA production activities.
        if "DIRECT" in mapping_types:
            aea_granularity[code] = "DIRECT"

        # Ordinary aggregate AEA production activities.
        elif "AGGREGATE" in mapping_types:
            aea_granularity[code] = "AGGREGATE"

        # Explicit exception: valid AEA production activities whose
        # emissions are allocated across ICIO industries using weights.
        elif (
            code in GTAP_ELIGIBLE_EMISSION_WEIGHT_CODES
            and "EMISSION_WEIGHT" in mapping_types
        ):
            aea_granularity[code] = "DIRECT"

    available_aea = set(aea_granularity)

    return (
        available_aea,
        aea_granularity,
        aea_mapping_types,
    )



# =============================================================================
# 3. CONCORDANCE CONSTRUCTION
# =============================================================================

def build_concordance(
    gtap_activities: list[str],
    available_aea: set[str],
    aea_granularity: dict[str, str],
) -> pd.DataFrame:
    """
    Build a potentially one-to-many GTAP-to-AEA activity concordance.

    Candidate tiers are evaluated in order. All eligible AEA codes in the
    first tier with at least one match are selected.

    Match status is determined using the AEA classification metadata:
      - DIRECT_MATCH for direct AEA activities;
      - AGGREGATE_MATCH for aggregate AEA activities;
      - METRO_CONCORDANCE for explicit METRO mappings.
    """

    rows = []

    normalized_rules = {
        code.strip().upper(): rule
        for code, rule in GTAP_RULES.items()
    }

    for raw_code in gtap_activities:

        code = str(raw_code).strip().upper()

        if code not in normalized_rules:
            rows.append({
                "gtap_activity_code": code,
                "aea_activity_code": "",
                "allocation_weight": 1.0,
                "gtap_description": "Description not defined",
                "match_status": "MANUAL_REVIEW",
                "notes": (
                    "GTAP activity not included in concordance rules."
                ),
            })
            continue

        description, candidate_tiers, configured_status = (
            normalized_rules[code]
        )

        # -----------------------------------------------------------------
        # DWE: explicitly retained outside the AEA production mapping.
        # -----------------------------------------------------------------

        if configured_status == "SEPARATE_NOT_ALLOCATED":
            rows.append({
                "gtap_activity_code": code,
                "aea_activity_code": "",
                "allocation_weight": 0.0,
                "gtap_description": description,
                "match_status": "SEPARATE_NOT_ALLOCATED",
                "notes": (
                    "Dwellings are retained separately. They are not "
                    "mapped to L68A, L or another AEA production activity."
                ),
            })
            continue

        # -----------------------------------------------------------------
        # Select the first candidate tier with at least one eligible code.
        # -----------------------------------------------------------------

        selected = None
        selected_tier_number = None

        for tier_number, tier in enumerate(candidate_tiers):

            matches = [
                candidate.strip().upper()
                for candidate in tier
                if candidate.strip().upper() in available_aea
            ]

            if matches:
                selected = matches
                selected_tier_number = tier_number
                break

        if selected is None:
            rows.append({
                "gtap_activity_code": code,
                "aea_activity_code": "",
                "allocation_weight": 1.0,
                "gtap_description": description,
                "match_status": "MANUAL_REVIEW",
                "notes": (
                    "No eligible AEA activity code found. "
                    f"Candidates considered: {candidate_tiers}"
                ),
            })
            continue

        # -----------------------------------------------------------------
        # Determine the match status.
        # -----------------------------------------------------------------

        if configured_status == "METRO_CONCORDANCE":
            status = "METRO_CONCORDANCE"

            note = (
                "Explicit METRO correspondence. Pollutant-specific "
                "allocation shares are calculated separately using "
                "AEA emissions observations."
            )

        else:
            selected_granularities = {
                aea_granularity[aea_code]
                for aea_code in selected
            }

            # If any selected activity is an aggregate, flag the GTAP
            # mapping as an aggregate match. This makes the limitation
            # visible when a GTAP activity maps to multiple AEA codes.
            if "AGGREGATE" in selected_granularities:
                status = "AGGREGATE_MATCH"

                note = (
                    "Candidate-tier mapping to one or more aggregate "
                    "AEA activities. The AEA classification does not "
                    "provide a fully disaggregated counterpart for "
                    "this GTAP activity. Pollutant-specific allocation "
                    "shares are calculated separately using AEA "
                    "emissions observations."
                )

            else:
                status = "DIRECT_MATCH"

                note = (
                    "Candidate-tier mapping to direct AEA activity "
                    "code(s). Pollutant-specific allocation shares "
                    "are calculated separately using AEA emissions "
                    "observations."
                )

            if selected_tier_number > 0:
                note += (
                    f" Candidate tier {selected_tier_number + 1} "
                    "was used because no eligible code was available "
                    "in the preceding tier(s)."
                )

        # -----------------------------------------------------------------
        # Write one row per GTAP-AEA correspondence.
        # -----------------------------------------------------------------

        for aea_code in selected:
            rows.append({
                "gtap_activity_code": code,
                "aea_activity_code": aea_code,
                "allocation_weight": 1.0,
                "gtap_description": description,
                "match_status": status,
                "notes": note,
            })

    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    return result


# =============================================================================
# 4. VALIDATION
# =============================================================================

def validate_metro_concordance(
    result: pd.DataFrame,
    gtap_activities: list[str],
) -> None:
    """
    Validate the explicit METRO mappings and DWE treatment.

    This validates only the GTAP-to-AEA concordance. It does not validate
    or construct an AEA-to-ICIO correspondence.
    """

    present_activities = {
        str(code).strip().upper()
        for code in gtap_activities
    }

    for gtap_activity, required_codes in METRO_REQUIRED.items():

        if gtap_activity not in present_activities:
            continue

        actual_codes = set(
            result.loc[
                result["gtap_activity_code"].eq(gtap_activity),
                "aea_activity_code",
            ]
            .replace("", pd.NA)
            .dropna()
        )

        missing = required_codes - actual_codes

        if missing:
            raise ValueError(
                f"METRO mapping for {gtap_activity} is incomplete. "
                f"Missing AEA activity codes: {sorted(missing)}"
            )

    # DWE must remain separate.
    if "DWE" in present_activities:

        dwe = result.loc[
            result["gtap_activity_code"].eq("DWE")
        ]

        if dwe.empty:
            raise ValueError(
                "DWE is present in GTAP but missing from the concordance."
            )

        if not dwe["match_status"].eq(
            "SEPARATE_NOT_ALLOCATED"
        ).all():
            raise ValueError(
                "DWE must be marked SEPARATE_NOT_ALLOCATED."
            )

        if (
            dwe["aea_activity_code"]
            .replace("", pd.NA)
            .notna()
            .any()
        ):
            raise ValueError(
                "DWE must not have an AEA activity mapping."
            )

    # No DWE mapping to real estate or imputed rents is permitted.
    forbidden_dwe_codes = {"L", "L68A"}

    actual_dwe_codes = set(
        result.loc[
            result["gtap_activity_code"].eq("DWE"),
            "aea_activity_code",
        ]
        .replace("", pd.NA)
        .dropna()
    )

    invalid_dwe_codes = actual_dwe_codes & forbidden_dwe_codes

    if invalid_dwe_codes:
        raise ValueError(
            "DWE has an invalid real-estate mapping: "
            f"{sorted(invalid_dwe_codes)}"
        )


def validate_concordance(
    result: pd.DataFrame,
    gtap_activities: list[str],
    available_aea: set[str],
) -> None:
    """Run general integrity checks on the generated concordance."""

    if result.empty:
        raise ValueError("The generated GTAP-to-AEA concordance is empty.")

    # Every GTAP activity should appear at least once.
    expected_activities = {
        str(code).strip().upper()
        for code in gtap_activities
    }

    actual_activities = set(result["gtap_activity_code"])

    missing_activities = expected_activities - actual_activities

    if missing_activities:
        raise ValueError(
            "GTAP activities missing from the output: "
            f"{sorted(missing_activities)}"
        )

    # Every nonblank AEA code must exist in the eligible AEA activity set.
    mapped = (
        result["aea_activity_code"]
        .replace("", pd.NA)
        .dropna()
    )

    invalid_codes = set(mapped) - available_aea

    if invalid_codes:
        raise ValueError(
            "The concordance contains AEA codes that are not eligible "
            f"activities in the AEA crosswalk: {sorted(invalid_codes)}"
        )

    # No duplicate GTAP-AEA pairs.
    duplicates = result.loc[
        result.duplicated(
            ["gtap_activity_code", "aea_activity_code"],
            keep=False,
        )
    ]

    if not duplicates.empty:
        raise ValueError(
            "Duplicate GTAP-to-AEA correspondence rows found:\n"
            + duplicates.to_string(index=False)
        )

    # All mapped rows must have positive placeholder weights.
    mapped_rows = result[
        result["aea_activity_code"].ne("")
    ]

    if (mapped_rows["allocation_weight"] <= 0).any():
        raise ValueError(
            "Mapped GTAP-AEA rows must have positive placeholder weights."
        )

    # Run the METRO and DWE checks.
    validate_metro_concordance(
        result=result,
        gtap_activities=gtap_activities,
    )


# =============================================================================
# 5. MAPPING DIAGNOSTICS
# =============================================================================

def build_mapping_diagnostics(
    result: pd.DataFrame,
    aea_granularity: dict[str, str],
) -> pd.DataFrame:
    """
    Build a row-level diagnostic file showing:
      - whether the selected AEA code is direct or aggregate;
      - how many GTAP activities share that AEA code;
      - which GTAP activities share it;
      - whether the AEA code is used by multiple GTAP activities.

    This is an audit of the GTAP-to-AEA mapping, not an emissions allocation.
    """

    mapped = result.loc[
        result["aea_activity_code"].ne("")
    ].copy()

    if mapped.empty:
        return pd.DataFrame(
            columns=[
                "gtap_activity_code",
                "aea_activity_code",
                "aea_granularity",
                "gtap_activities_using_aea_code",
                "number_of_gtap_activities_using_aea_code",
                "shared_aea_code",
                "is_aggregate_mapping",
            ]
        )

    usage = (
        mapped.groupby("aea_activity_code")["gtap_activity_code"]
        .agg(lambda values: sorted(set(values)))
        .to_dict()
    )

    rows = []

    for _, row in mapped.iterrows():

        aea_code = row["aea_activity_code"]
        gtap_users = usage[aea_code]
        granularity = aea_granularity.get(aea_code, "UNKNOWN")

        rows.append({
            "gtap_activity_code": row["gtap_activity_code"],
            "aea_activity_code": aea_code,
            "aea_granularity": granularity,
            "gtap_activities_using_aea_code": ";".join(gtap_users),
            "number_of_gtap_activities_using_aea_code": len(gtap_users),
            "shared_aea_code": len(gtap_users) > 1,
            "is_aggregate_mapping": granularity == "AGGREGATE",
        })

    return pd.DataFrame(rows)


def build_shared_aea_code_audit(
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Return one row per AEA activity code shared by multiple GTAP activities."""

    if diagnostics.empty:
        return diagnostics.copy()

    shared = diagnostics.loc[
        diagnostics["shared_aea_code"]
    ].copy()

    if shared.empty:
        return shared

    return (
        shared.drop_duplicates("aea_activity_code")
        .sort_values(
            [
                "number_of_gtap_activities_using_aea_code",
                "aea_activity_code",
            ],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


def build_aggregate_mapping_audit(
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Return mappings that use aggregate AEA activity codes."""

    if diagnostics.empty:
        return diagnostics.copy()

    aggregate = diagnostics.loc[
        diagnostics["is_aggregate_mapping"]
    ].copy()

    return aggregate.sort_values(
        ["aea_activity_code", "gtap_activity_code"]
    ).reset_index(drop=True)


# =============================================================================
# 6. MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build a GTAP 12 to OECD AEA activity concordance."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(r"D:\Research\Datacenters"),
    )

    parser.add_argument(
        "--crosswalk",
        type=str,
        default="aea_crosswalk.csv",
    )

    parser.add_argument(
        "--domains-file",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--gtap-co2",
        type=Path,
        default=Path(r"GDX12_23\GSDFEMISS.GDX"),
    )

    parser.add_argument(
        "--gams-system-dir",
        type=str,
        default=r"C:\GAMS\51",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="gtap_activity_to_aea.csv",
    )

    args = parser.parse_args()

    args.data_dir = args.data_dir.resolve()

    if args.domains_file is None:
        args.domains_file = (
            args.data_dir / "gtap_emissions_domains.csv"
        )

    elif not args.domains_file.is_absolute():
        args.domains_file = (
            args.data_dir / args.domains_file
        )

    # -------------------------------------------------------------------------
    # Read AEA-to-ICIO crosswalk as AEA classification metadata only.
    # -------------------------------------------------------------------------

    crosswalk_path = args.data_dir / args.crosswalk

    if not crosswalk_path.exists():
        raise FileNotFoundError(crosswalk_path)

    crosswalk = pd.read_csv(
        crosswalk_path,
        dtype=str,
    )

    (
        available_aea,
        aea_granularity,
        aea_mapping_types,
    ) = prepare_aea_classification(crosswalk)

    if not available_aea:
        raise ValueError(
            "No eligible AEA activity codes were found. Check the "
            "mapping_type column in aea_crosswalk.csv."
        )
        
    # Confirm that the two explicitly approved emission-weight activities
    # are available and that residual codes have not become eligible.
    required_weighted_activities = {"C24", "C30"}

    missing_weighted_activities = (
        required_weighted_activities - available_aea
    )

    if missing_weighted_activities:
        raise ValueError(
            "Required AEA production activities are missing or have "
            "unexpected mapping types: "
            f"{sorted(missing_weighted_activities)}"
        )

    for residual_code in ["C_X", "H_X", "J_X"]:
        if residual_code in available_aea:
            raise ValueError(
                f"Residual AEA code {residual_code} must not be eligible "
                "for GTAP activity matching."
            )

    # -------------------------------------------------------------------------
    # Read GTAP production activities.
    # -------------------------------------------------------------------------

    gtap_activities = read_gtap_activities(args)

    gtap_activities = sorted({
        str(code).strip().upper()
        for code in gtap_activities
        if str(code).strip()
    })

    if not gtap_activities:
        raise ValueError("No GTAP activities were found.")

    # -------------------------------------------------------------------------
    # Build and validate the GTAP-to-AEA concordance.
    # -------------------------------------------------------------------------

    result = build_concordance(
        gtap_activities=gtap_activities,
        available_aea=available_aea,
        aea_granularity=aea_granularity,
    )

    validate_concordance(
        result=result,
        gtap_activities=gtap_activities,
        available_aea=available_aea,
    )

    # -------------------------------------------------------------------------
    # Identify rules not represented in the GTAP activity set.
    # -------------------------------------------------------------------------

    normalized_rules = {
        code.strip().upper(): rule
        for code, rule in GTAP_RULES.items()
    }

    unknown = sorted(
        set(gtap_activities) - set(normalized_rules)
    )

    missing = sorted(
        set(normalized_rules) - set(gtap_activities)
    )

    if unknown:
        print(
            "\nWARNING: GTAP activities without mapping rules:",
            unknown,
        )

    if missing:
        print(
            "\nNOTE: Mapping rules not present in this GTAP activity set:",
            missing,
        )

    # -------------------------------------------------------------------------
    # Write the main concordance.
    # -------------------------------------------------------------------------

    output_path = args.data_dir / args.output

    result.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # Write review file.
    #
    # Include:
    #   - manual review;
    #   - aggregate mappings;
    #   - separately retained DWE.
    #
    # Direct matches are omitted from this review file.
    # -------------------------------------------------------------------------

    review_path = (
        args.data_dir / "gtap_activity_to_aea_review.csv"
    )

    review = result.loc[
        result["match_status"].isin(
            [
                "MANUAL_REVIEW",
                "AGGREGATE_MATCH",
                "SEPARATE_NOT_ALLOCATED",
            ]
        )
    ].copy()

    review.to_csv(
        review_path,
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # Write mapping diagnostics.
    # -------------------------------------------------------------------------

    diagnostics = build_mapping_diagnostics(
        result=result,
        aea_granularity=aea_granularity,
    )

    diagnostics_path = (
        args.data_dir / "gtap_activity_to_aea_diagnostics.csv"
    )

    diagnostics.to_csv(
        diagnostics_path,
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # Write audit of AEA codes shared by multiple GTAP activities.
    # -------------------------------------------------------------------------

    shared_audit = build_shared_aea_code_audit(
        diagnostics=diagnostics,
    )

    shared_path = (
        args.data_dir / "gtap_activities_sharing_aea_codes.csv"
    )

    shared_audit.to_csv(
        shared_path,
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # Write audit of aggregate AEA mappings.
    # -------------------------------------------------------------------------

    aggregate_audit = build_aggregate_mapping_audit(
        diagnostics=diagnostics,
    )

    aggregate_path = (
        args.data_dir / "gtap_aggregate_aea_mappings.csv"
    )

    aggregate_audit.to_csv(
        aggregate_path,
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # Summary by unique GTAP activity.
    # -------------------------------------------------------------------------

    activity_status = (
        result[
            ["gtap_activity_code", "match_status"]
        ]
        .drop_duplicates()
    )

    print("\nGTAP-to-AEA concordance created.")

    print(
        f"GTAP activities: "
        f"{activity_status['gtap_activity_code'].nunique()}"
    )

    for status in [
        "METRO_CONCORDANCE",
        "DIRECT_MATCH",
        "AGGREGATE_MATCH",
        "MANUAL_REVIEW",
        "SEPARATE_NOT_ALLOCATED",
    ]:
        count = (
            activity_status["match_status"].eq(status).sum()
        )
        print(f"{status}: {count}")

    print(f"Concordance rows: {len(result)}")

    # -------------------------------------------------------------------------
    # Mapping diagnostics summary.
    # -------------------------------------------------------------------------

    if not diagnostics.empty:

        shared_codes = (
            diagnostics.loc[
                diagnostics["shared_aea_code"],
                "aea_activity_code",
            ]
            .nunique()
        )

        aggregate_codes = (
            diagnostics.loc[
                diagnostics["is_aggregate_mapping"],
                "aea_activity_code",
            ]
            .nunique()
        )

        print("\nMapping diagnostics:")
        print(
            "AEA codes shared by multiple GTAP activities: "
            f"{shared_codes}"
        )
        print(
            "Aggregate AEA activity codes used: "
            f"{aggregate_codes}"
        )

    # -------------------------------------------------------------------------
    # Display important mappings.
    # -------------------------------------------------------------------------

    print("\nCMN and RSA mappings:")

    print(
        result.loc[
            result["gtap_activity_code"].isin(["CMN", "RSA"])
        ].to_string(index=False)
    )

    print("\nDWE treatment:")

    print(
        result.loc[
            result["gtap_activity_code"].eq("DWE")
        ].to_string(index=False)
    )

    print("\nGTAP activities sharing AEA activity codes:")

    if shared_audit.empty:
        print("No shared AEA activity codes.")
    else:
        print(shared_audit.to_string(index=False))

    print("\nAggregate AEA mappings:")

    if aggregate_audit.empty:
        print("No aggregate AEA mappings.")
    else:
        print(
            aggregate_audit[
                [
                    "gtap_activity_code",
                    "aea_activity_code",
                    "aea_granularity",
                ]
            ].to_string(index=False)
        )

    # -------------------------------------------------------------------------
    # Output paths.
    # -------------------------------------------------------------------------

    print(f"\nMain output: {output_path}")
    print(f"Review output: {review_path}")
    print(f"Diagnostics output: {diagnostics_path}")
    print(f"Shared-code audit: {shared_path}")
    print(f"Aggregate-mapping audit: {aggregate_path}")


if __name__ == "__main__":
    main()