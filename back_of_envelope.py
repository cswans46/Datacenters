
import pandas as pd
from pathlib import Path


# ============================================================
# 1. USER INPUTS
# ============================================================

# Electricity demand must be supplied in GWh.
#
# You can supply as many years as you want.
# The model will calculate results for every year supplied.

COUNTRIES = {

    "Canada": {
        "demand_gwh": {
            2024: 20_000,
            2025: 23_000,
            2026: 26_000,
            2027: 29_000,
            2028: 32_000,
            2029: 35_000,
            2030: 38_000,
        },

        # Grid emissions intensity in kg CO2/kWh.
        # Can be a single constant or a year-specific dictionary.

        "grid_emissions_factor": {
            2024: 0.120,
            2025: 0.115,
            2026: 0.110,
            2027: 0.105,
            2028: 0.100,
            2029: 0.095,
            2030: 0.090,
        },

        # Electricity demand basis:
        # "facility" = total data-centre electricity consumption
        # "IT"       = electricity consumed by IT equipment

        "demand_basis": "facility",

        # Power Usage Effectiveness.
        # Only used to convert between facility and IT demand.

        "PUE": 1.40,

        # Water Usage Effectiveness (litres per kWh of IT energy).

        "WUE": 1.80,
    },


    "India": {
        "demand_gwh": {
            2024: 10_000,
            2025: 13_000,
            2026: 16_000,
            2027: 20_000,
            2028: 24_000,
            2029: 29_000,
            2030: 35_000,
        },

        "grid_emissions_factor": {
            2024: 0.710,
            2025: 0.675,
            2026: 0.650,
            2027: 0.620,
            2028: 0.590,
            2029: 0.560,
            2030: 0.530,
        },

        "demand_basis": "facility",

        "PUE": 1.50,

        "WUE": 1.80,
    },

}


# Output directory
OUTPUT_DIR = Path("datacenter_results")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def get_annual_value(value, year):
    """
    Retrieve either a constant value or a year-specific value.
    """

    if isinstance(value, dict):
        if year not in value:
            raise ValueError(
                f"Missing input for year {year}."
            )
        return value[year]

    return value


# ============================================================
# 3. ENVIRONMENTAL ACCOUNTING
# ============================================================

def calculate_country_footprint(country, inputs):

    results = []

    demand = inputs["demand_gwh"]
    emissions_factor = inputs["grid_emissions_factor"]

    demand_basis = inputs["demand_basis"].lower()
    pue = inputs["PUE"]
    wue = inputs["WUE"]

    if demand_basis not in ["facility", "it"]:
        raise ValueError(
            f"{country}: demand_basis must be 'facility' or 'IT'."
        )

    if pue < 1:
        raise ValueError(
            f"{country}: PUE must be greater than or equal to 1."
        )

    if wue < 0:
        raise ValueError(
            f"{country}: WUE cannot be negative."
        )

    for year in sorted(demand):

        # ----------------------------------------------------
        # A. Retrieve annual inputs
        # ----------------------------------------------------

        electricity_demand = demand[year]

        grid_factor = get_annual_value(
            emissions_factor, year
        )

        if electricity_demand < 0 or grid_factor < 0:
            raise ValueError(
                f"{country}: Negative input in {year}."
            )

        # ----------------------------------------------------
        # B. Convert IT demand and facility demand
        # ----------------------------------------------------

        if demand_basis == "facility":

            facility_demand = electricity_demand

            it_demand = facility_demand / pue

        else:

            it_demand = electricity_demand

            facility_demand = it_demand * pue

        # ----------------------------------------------------
        # C. Electricity-related emissions
        # ----------------------------------------------------

        # GWh * kg CO2/kWh = kt CO2

        electricity_emissions = (
            facility_demand * grid_factor
        )

        # ----------------------------------------------------
        # D. Water consumption
        # ----------------------------------------------------

        # GWh IT * L/kWh = million litres = ML

        water_consumption = (
            it_demand * wue
        )

        # ----------------------------------------------------
        # E. Store annual results
        # ----------------------------------------------------

        results.append({

            "Country": country,

            "Year": year,

            "Electricity_Demand_GWh": electricity_demand,

            "Facility_Electricity_GWh": facility_demand,

            "IT_Electricity_GWh": it_demand,

            "Grid_Emissions_Factor_kg_CO2_kWh": grid_factor,

            "Electricity_Emissions_kt_CO2": electricity_emissions,

            "PUE": pue,

            "WUE_L_per_kWh_IT": wue,

            "Water_Consumption_ML": water_consumption,

        })

    return pd.DataFrame(results)


# ============================================================
# 4. RUN MODEL FOR ALL COUNTRIES
# ============================================================

annual_results = pd.concat(

    [
        calculate_country_footprint(country, inputs)

        for country, inputs in COUNTRIES.items()
    ],

    ignore_index=True

)


# ============================================================
# 5. CALCULATE CUMULATIVE RESULTS
# ============================================================

annual_results = annual_results.sort_values(
    ["Country", "Year"]
)

annual_results["Cumulative_Emissions_kt_CO2"] = (

    annual_results.groupby("Country")[
        "Electricity_Emissions_kt_CO2"
    ].cumsum()

)

annual_results["Cumulative_Water_ML"] = (

    annual_results.groupby("Country")[
        "Water_Consumption_ML"
    ].cumsum()

)


# ============================================================
# 6. CREATE COUNTRY SUMMARY
# ============================================================

summary = (

    annual_results.groupby("Country")

    .agg(

        First_Year=("Year", "min"),

        Last_Year=("Year", "max"),

        Total_Emissions_kt_CO2=(
            "Electricity_Emissions_kt_CO2", "sum"
        ),

        Total_Water_ML=(
            "Water_Consumption_ML", "sum"
        ),

    )

    .reset_index()

)


# Extract final-year results for each country.

final_year_results = (

    annual_results.sort_values("Year")

    .groupby("Country")

    .tail(1)

    [[

        "Country",

        "Year",

        "Electricity_Demand_GWh",

        "Electricity_Emissions_kt_CO2",

        "Water_Consumption_ML",

    ]]

    .rename(columns={

        "Year": "Final_Year",

        "Electricity_Demand_GWh": "Final_Demand_GWh",

        "Electricity_Emissions_kt_CO2": "Final_Emissions_kt_CO2",

        "Water_Consumption_ML": "Final_Water_ML",

    })

)

summary = summary.merge(
    final_year_results,
    on="Country"
)


# ============================================================
# 7. EXPORT RESULTS
# ============================================================

annual_results.to_csv(
    OUTPUT_DIR / "annual_datacenter_footprint.csv",
    index=False
)

summary.to_csv(
    OUTPUT_DIR / "datacenter_footprint_summary.csv",
    index=False
)


# ============================================================
# 8. DISPLAY RESULTS
# ============================================================

print("\nANNUAL ENVIRONMENTAL FOOTPRINT")
print("=" * 70)

print(
    annual_results.to_string(index=False)
)

print("\nCOUNTRY SUMMARY")
print("=" * 70)

print(
    summary.to_string(index=False)
)

print("\nResults saved to:", OUTPUT_DIR.resolve())
