# Data-centre expansion footprint model: mathematical description

This document describes, step by step, how `global_datacentre_icio_model_v2.py`
and its helper scripts turn a data-centre capacity scenario into output,
emissions, trade and water estimates. Equations use GitHub math notation.
Section 15 lists what changed in the November 2026 methodology fixes and
how it affects the numbers.

Contents

1. Pipeline and files
2. Notation and the ICIO table
3. Direct-emissions extension (AEA / GTAP 12)
4. Canadian data-centre split (DSUT / SUT)
5. Country-of-origin override for servers
6. From capacity to an output shock
7. Physical metrics: electricity, grid emissions, water
8. Upstream: Leontief model with exogenous data-centre output
9. Electricity: ICIO-endogenous and hybrid (physical) columns
10. Emission intensities and grid sensitivities
11. Emissions components: direct, upstream, downstream
12. Output gain, value added and cross-border intermediates
13. Downstream: Ghosh supply-driven model
14. Baselines, shares and the headline summary
15. Optional: construction and equipment (capex)
16. Other modules (TiVA indicators, projection)
17. Assumptions and limitations
18. Change log (methodology fixes)

---

## 1. Pipeline and files

| Step | Script | Output |
|---|---|---|
| Emissions extension | `build_hybrid_emissions_extension.py` (+ `build_aea_crosswalk.py`, `build_gtap_aea_concordance.py`) | `hybrid_emissions/direct_emissions.csv` |
| Canadian split (optional) | `canada_dsut_split.py`, `dc_origin_override.py` | split A, X, FD |
| Main model | `global_datacentre_icio_model_v2.py` | upstream / downstream summaries and details |
| Headline summary | `summarize_results.py` (run automatically) | `headline_summary.csv/.md`, `headline_top_origin_countries.csv` |
| Projection (optional) | `project_icio.py`, `icio_projection/` | projected ICIO tables |

Scenario inputs in `--data-dir`: `dc_output_scenarios.csv`,
`dc_electricity_demand.csv`, `electricity_prices.csv`,
`grid_emissions_factors.csv`, `water_coefficients.csv`,
`us_customer_shares.csv`, optional `us_parent_operator_shares.csv`,
`dc_sector_map.csv`, and for `--capex` `dc_capex.csv` and
`dc_capex_composition.csv`.

## 2. Notation and the ICIO table

The OECD Inter-Country Input-Output table (2025 edition, reference year
2022, current USD million) has $n$ country-industries (81 economies × 50
industries; China and Mexico are each split into two blocks, CN1/CN2 and
MX1/MX2).

| Symbol | Meaning |
|---|---|
| $\mathbf{A}$ | technical coefficients, $a_{ij}$ = purchases from $i$ per USD of output of $j$ |
| $\mathbf{x}$ | gross output (2022) |
| $\mathbf{Z} = \mathbf{A}\hat{\mathbf{x}}$ | intermediate transactions |
| $\mathbf{F}$ | final demand by category and country (columns such as `CAN_GFCF`) |
| $\mathbf{L} = (\mathbf{I}-\mathbf{A})^{-1}$ | Leontief inverse |
| $\mathbf{f}$ | direct emission intensity, t CO2e per USD million of output |
| $\mathbf{v}$ | value-added coefficients, $v_j = 1 - \sum_i a_{ij}$ (value added plus net taxes on products) |
| $S$ | the data-centre industries of the destination country (one, or CLD and DPH after the Canadian split) |
| $E$ | the destination country's electricity industry (ICIO 35) |
| $c(i)$ | the country of industry $i$ |

The loader checks the accounting identity $\mathbf{x} = \mathbf{A}\mathbf{x} + \mathbf{F}\mathbf{1}$ to a relative tolerance (default 2%).

All monetary scenario values are in constant 2022 USD so that they are
consistent with the 2022 coefficients.

## 3. Direct-emissions extension

`build_hybrid_emissions_extension.py` produces one row per ICIO industry
with direct (production-based, residence-principle) GHG emissions in
t CO2e, using AR6 100-year GWPs.

**Source selection is country-wide.** A country uses OECD Air Emission
Accounts (AEA, `ACTIVITY_SCOPE=RES`, SEEA methodology) if usable
resident-industry GHG data exist; otherwise GTAP 12. The two are never
added for the same country. In the current extension AEA covers 33
economies (EU members, the UK, Switzerland, Norway, Iceland, New Zealand
and Türkiye); all other economies, including Canada, the United States,
India, China and Australia, use GTAP 12.

**AEA to ICIO.** Broad AEA aggregates are dropped when their detailed
components are reported. An AEA activity $a$ that maps to several ICIO
industries is allocated by 2022 gross output:

$$E_i = E_a \frac{x_i}{\sum_{j \in a} x_j}, \qquad i \in a .$$

**GTAP to ICIO.** GTAP CO2 covers fuel combustion by producing industries
(domestic and imported fuels: `MDF + MMF`); non-CO2 gases are the
production accounts `EMI_IO + EMI_ENDW + EMI_QO + EMI_IOP` (households
`EMI_HH` and land use `EMI_LU` excluded). GTAP activities are mapped to
AEA activities (weights, with pooled AEA pollutant shares where a GTAP
activity covers several AEA activities) and then to ICIO industries as
above, so both sources end in the same industry classification.
Industries that are structural zeros in the ICIO are flagged rather than
receiving emissions; ICIO 97T98 is out of scope.

**Intensities.** In the main model

$$f_i = \frac{E_i}{x_i}\quad (x_i>0), \qquad f_i = 0 \text{ if } x_i = 0 \text{ and } E_i = 0 .$$

Electricity-generation emissions stay with ICIO 35 (no reallocation to
users), so electricity use enters footprints through the supply chain.

## 4. Canadian data-centre split (optional)

ICIO has no data-centre industry; data centres sit in J62_63 (IT and other
information services). With `--split-canada-dsut`, `CAN_J62_63` is split
into CLD (cloud computing services), DPH (other NAICS 51821: hosting and
data processing) and OTH (the rest) using the 2020 StatCan digital SUT and
the 2022 SUT (table 36-10-0478-01):

$$w_{518} = \frac{x_{51821}}{x_{5415}+x_{518}+x_{519}}, \quad
s_{cloud} = \frac{\text{cloud output of } 51821}{\text{output of } 51821},$$

$$w_{CLD} = w_{518}s_{cloud}, \quad w_{DPH} = w_{518}(1-s_{cloud}), \quad w_{OTH} = 1-w_{518}.$$

Columns (cost structures): each input cell is split in proportion to
$w_k r_k(g)$, where $r_k(g)$ is the SUT input intensity of subsector $k$
for input group $g$ relative to J62_63. Rows (sales structures) are split
by destination block (domestic/foreign × intermediate/final) and
RAS-balanced. The columns are then RAS-balanced to SUT value-added
targets (section 4c). The split is exact in aggregate. Direct emissions of
`CAN_J62_63` are split by output (equal intensity, default) or by
electricity purchases. The data-centre shock then goes to CLD and DPH in
proportion to their output. Sector labels of both ICIO editions are
accepted (`J62_63` in 2023, `62T63` in 2025).

### 4b. Indian data-centre split (optional, `--split-india-iot`)

Source: the 2023–24 India input-output table (70 sectors, INR crore) with
an added AI-DC sector (data-centre services, NIC 6311), RAS-balanced in
sheet `Z RAS Final` of `IOT_2023-24_RAS.xlsx`. Only ratios are used.
`IND_62T63` is split into `IND_62T63_DC` and `IND_62T63_OTH`:

$$w_{DC} = \frac{x_{AIDC}}{x_{AIDC} + x_{CS}}$$

with $x_{CS}$ the output of Computer related services before the AI-DC
purchases were added (0.449% in the current workbook; override with
`--india-dc-share`).

*Columns.* AI-DC input coefficients $a^{IOT}_{g,DC}$ are mapped from IOT
sectors $g$ to ICIO sectors $s$ (`IOT_TO_ICIO` in `india_iot_split.py`;
one-to-many mappings are shared by Indian ICIO output). The IOT records
total (domestic + imported) use, so each ICIO sector's coefficient is
shared across countries of origin $o$ with the origin mix of
`IND_62T63`'s 2022 purchases:

$$z^{seed}_{(o,s),DC} = a^{IOT}_{s,DC}\, x_{DC}\, \frac{z_{(o,s),62T63}}{\sum_{o'} z_{(o',s),62T63}} .$$

The rest of the sector gets $z^{seed}_{\cdot,OTH} = z_{\cdot,62T63} - z^{seed}_{\cdot,DC}$.
The $n\times 2$ block is then RAS-balanced (section 4c) to the AI-DC and
Computer related services value-added shares of the IOT (12.0% and
72.0%).

*Rows.* AI-DC sales shares by purchasing IOT sector are mapped to Indian
ICIO purchasers (sales to Computer related services go to `IND_62T63_OTH`);
data centres have no final-demand sales or exports, as in the IOT. The
$2\times(n+1+n_{FD})$ row block is RAS-balanced so that rows add up to
subsector output and columns to the parent's row cells.

*Emissions.* `IND_62T63` direct emissions are split by output (default) or
electricity purchases (`--india-split-emissions-basis`). The data-centre
shock for India goes to `IND_62T63_DC` (map India to `IND_62T63` in
`dc_sector_map.csv`).

Note: the IOT's AI-DC column spends 1.18% of output on electricity and
treats servers, machinery and steel as intermediate inputs. In the
`icio_endogenous` sensitivity this column is used as is; in the physical
sensitivity the electricity coefficient is replaced (section 9). If
`--capex` is also used, equipment would be counted in both the column and
the capex shock.

### 4c. RAS re-balancing of split sectors

Splitting a parent industry $p$ into subsectors $k$ shares its input
column out across the subsector columns. To make each subsector's value
added consistent with its source table, the $n\times K$ block of subsector
columns is RAS-balanced (`split_rebalance.ras_split_columns`, signed RAS
from cswans46/baltic):

$$\sum_k z_{ik} = z_{ip}\ \ \forall i, \qquad
\sum_i z_{ik} = x_k - VA_k^{*}\ \ \forall k, \qquad
VA_k^{*} = VA_p\,\frac{x_k s_k}{\sum_m x_m s_m},$$

where $s_k$ is the value-added share of output in the source table
(Canada: $1-\sum_g u_{gk}/x_k$ from the StatCan SUT for NAICS 51821 and the
rest of J62_63; India: the IOT shares above). Every other cell of the ICIO
is untouched, so the table remains balanced in both directions and the
parent's value added is preserved. `--no-split-rebalance` turns the
column RAS off.

After all splits and the origin override, `split_rebalance.balance_report`
checks the full table: the row identity $\mathbf{x} = \mathbf{Z}\mathbf{1} + \mathbf{f}$
and industries with negative value added, which are listed in
`icio_negative_value_added.csv` and in the log.

## 5. Country-of-origin override (optional)

`dc_origin_override.py` re-weights the imported origins of selected inputs
of CLD and DPH (for example ICIO 26, computers and electronics, using server
import shares for HS 847150), keeps the domestic share and the import
total, and re-balances the table with signed RAS so all rows still sum to
output.

## 6. From capacity to an output shock

For country $c$ and year $t$, data-centre output in 2022 USD million is

$$Q_{c,t} = MW_{c,t}\, p_{c} ,$$

where $MW$ is installed (IT) capacity and $p$ is output (revenue) per MW
per year (`price_mn_MW_2022`). Three shock bases are calculated:

| Basis | Shock $\Delta Q$ | Electricity $\Delta G$ (GWh) | Use |
|---|---|---|---|
| `annual_increment` | $Q_t - Q_{t-1}$ | $G_t - G_{t-1}$ | year-on-year |
| `cumulative_vs_first_year` | $Q_t - Q_{t_0}$ | $G_t - G_{t_0}$ | **headline** |
| `2030_additional_output` | $Q_{2030}$ | $G_{2030}$ | upper bound: treats all 2030 output as new, although the 2022 ICIO already contains existing data centres |

Two options change the output and electricity definitions:

* `--output-includes-power-passthrough`: `price_mn_MW_2022` is treated as
  rent that excludes power (wholesale colocation leases are quoted
  excluding power pass-through), and the facility electricity bill is
  added to output: $Q = MW\,p + G^{fac} P / 1000$ (columns
  `dc_rent_usd_mn` and `power_passthrough_usd_mn`).
* `--it-utilisation u`: electricity demand is multiplied by $u \in (0,1]$.
  Use it when `dc_electricity_demand.csv` assumes full load
  (electricity MW = capacity × PUE). It also lowers grid emissions and
  water; e-waste (capacity-based) is unchanged.

The resulting electricity cost share of output is

$$e = \frac{u\, PUE \times 8760 \times P / 10^6}{p + \mathbb{1}_{pt}\, u\, PUE \times 8760 \times P/10^6}.$$

**Attribution lenses** multiply both output and electricity by a share
$\alpha$: all expansion ($\alpha = 1$), US customer demand
($\alpha$ = `us_customer_share`), US-parent operators
($\alpha$ = `us_parent_operator_share`). They are alternative views and
must not be added. A lens with $\alpha = 0$ produces no results.

## 7. Physical metrics

From `dc_electricity_demand.csv` (facility or IT basis, GWh or average MW
× 8,760 h) and the power usage effectiveness (PUE):

$$G^{fac} = G^{IT}\,PUE, \qquad G^{IT} = G^{fac}/PUE .$$

Location-based grid emissions of the facility (reported in
`physical_operational_metrics.csv`, kt CO2e):

$$E^{grid} = G^{fac}\,\varepsilon^{grid}_{c,t} \quad (\text{GWh} \times \text{kg/kWh} = \text{kt}).$$

Operational (direct) cooling water, million litres:

$$W = G^{IT}\,WUE \quad (\text{GWh} \times \text{L/kWh} = \text{ML}).$$

Indirect water (in the electricity and goods supply chains) is not
calculated.

**End-of-life server e-waste** (`ewaste.py`, written to
`ewaste_metrics.csv`). From installed IT capacity $MW_{c,t}$:

$$N^{rack} = MW\,\rho, \qquad N^{srv} = N^{rack}\,\sigma, \qquad
W^{EOL} = N^{srv}\,m/1000 = MW\,\rho\,\sigma\,m/1000 \ \text{(tonnes)},$$

with $\rho$ = 7.6 AI racks per MW, $\sigma$ = 18 servers (compute trays)
per rack and $m$ = 130 kg per server for an NVIDIA GB300 NVL72 AI factory
(NVIDIA Enterprise Reference Architecture, NVL72 AI factory with GB300
NVL72, dual-plane networking), i.e. 17.784 t per MW. Check: 275 MW gives
2,090 racks, 37,620 servers and 4,890.6 t.

Reported: the stock $W^{EOL}_{c,t}$ (all servers installed in year $t$,
which become e-waste at end of life), the part from capacity added since
the first scenario year $W^{EOL}_{c,t}-W^{EOL}_{c,t_0}$, and, if
`--server-lifetime-years` $L$ is given, the steady-state annual flow
$W^{EOL}/L$. Parameters can be changed with `--racks-per-mw`,
`--servers-per-rack`, `--server-kg`, or per country-year in
`ewaste_coefficients.csv`. Scope: servers only (no networking, storage,
power and cooling equipment, or batteries), and all capacity is treated as
GB300-class AI capacity, which overstates mass for conventional
cloud and colocation capacity.

**Electricity cross-check** (`output_shock_comparison.csv`): the output
change implied by the ICIO electricity coefficient,

$$\Delta Q^{elec} = \frac{\Delta G \, P/1000}{a_{ES}},$$

with $P$ the electricity price (USD/MWh). In the current inputs it is
roughly 140–210 times the capacity-based shock (Canada 144–185×, India
191–205×), i.e. the ICIO J62_63 column spends 0.36% (Canada) and 0.49%
(India) of output on electricity whereas
the physical data imply 59% (Canada) and 98% (India). This is why the
physical (hybrid) sensitivity is the headline.

## 8. Upstream: Leontief model with exogenous data-centre output

The data-centre industries' output change is imposed, not derived from
final demand. With $o$ the other industries, the mixed (exogenous-output)
model is

$$\Delta\mathbf{x}_S = \Delta\mathbf{q}, \qquad
\Delta\mathbf{x}_o = (\mathbf{I}-\mathbf{A}_{oo})^{-1}\mathbf{A}_{oS}\,\Delta\mathbf{q},$$

where $\Delta\mathbf{q} = \alpha\,\Delta Q\,\mathbf{w}$ distributes the
shock over the target industries with weights $\mathbf{w}$.

It is computed with the full Leontief factorisation: find the final
demand $\Delta\mathbf{y}_S$ at the targets such that

$$\Delta\mathbf{x} = \mathbf{L}\begin{bmatrix}\Delta\mathbf{y}_S\\ \mathbf{0}\end{bmatrix},
\qquad \Delta\mathbf{y}_S = \mathbf{L}_{SS}^{-1}\Delta\mathbf{q},$$

which is algebraically identical to the mixed model (the partitioned
inverse gives $\mathbf{L}_{oS}\mathbf{L}_{SS}^{-1} = (\mathbf{I}-\mathbf{A}_{oo})^{-1}\mathbf{A}_{oS}$).
It reproduces the targets exactly and does not double count the data
centres' purchases from themselves.

## 9. Electricity: ICIO-endogenous and hybrid columns

**`icio_endogenous`.** Data-centre purchases follow the ICIO column
$\mathbf{A}_{\cdot S}$, including its (small) electricity coefficient.

**`physical_facility_demand` (hybrid).** The data-centre column is
replaced by $\mathbf{A}'_{\cdot S}$. The electricity cost share of the
output change is

$$e = \frac{\Delta G\,P/1000}{\Delta Q}$$

(the same for every attribution lens, because both numerator and
denominator scale with $\alpha$). For target $k$:

$$a'_{Ek} = e\,(1-\beta), \qquad
a'_{ik} = a_{ik}\,\frac{1-e}{1-a_{Ek}} \quad (i \ne E),$$

where $\beta$ is the behind-the-meter (BTM) share (zero except in the
`btm_natgas` sensitivity). Scaling the other inputs (optional,
`--no-rescale-other-inputs` disables it) keeps the column sum at or below
one: when a larger part of each dollar of output pays for electricity, less
is left for other inputs and value added. BTM electricity is not bought
from the grid; its cost stays in the data-centre column as own-account
generation.

The model raises an error if $e \ge 1$ and logs a warning if $e > 0.5$,
because then output per MW is low relative to electricity cost.

The hybrid column is implemented without refactorising $(\mathbf{I}-\mathbf{A})$:
the change in purchases is passed as extra intermediate demand on the
non-target industries,

$$\mathbf{d} = (\mathbf{A}'_{\cdot S} - \mathbf{A}_{\cdot S})\,\Delta\mathbf{q}, \qquad d_S = 0,$$

$$\Delta\mathbf{y}_S = \mathbf{L}_{SS}^{-1}\big(\Delta\mathbf{q} - (\mathbf{L}\mathbf{d})_S\big), \qquad
\Delta\mathbf{x} = \mathbf{L}_{\cdot S}\Delta\mathbf{y}_S + \mathbf{L}\mathbf{d},$$

which gives $\Delta\mathbf{x}_o = (\mathbf{I}-\mathbf{A}_{oo})^{-1}\mathbf{A}'_{oS}\Delta\mathbf{q}$
and $\Delta\mathbf{x}_S = \Delta\mathbf{q}$. Electricity output therefore
rises by the data centres' purchases **plus** the electricity used
indirectly by their suppliers (including the electricity sector's own
supply chain). The previous version fixed total electricity output at the
physical expenditure, which removed that indirect electricity and forced a
negative final demand for electricity.

## 10. Emission intensities and grid sensitivities

| Sensitivity | Destination electricity intensity $f_E$ | Other industries |
|---|---|---|
| `average_grid` | 2022 extension, $E_E / x_E$ | 2022 extension |
| `projected_grid` | $\varepsilon^{grid}_{c,t}\times 10^6 / P_{c,t}$ | 2022 extension |
| `btm_natgas` | as `projected_grid` | 2022 extension |

$\varepsilon^{grid}_{c,t}$ is the scenario-year grid factor
(`kgco2_per_kwh`, kg/kWh = t/MWh) and $P$ the data-centre electricity price
(USD/MWh), so $f_E$ is in t per USD million and
$f_E \times \text{(purchased electricity, USD mn)} = \text{GWh} \times \varepsilon$.
The conversion uses the data-centre price for all electricity-sector
output of the destination country, which is an approximation.

In the current run the 2022 extension implies 0.117 kg/kWh for Canada
(1,923 t/USD mn × 60.6 USD/MWh) and 0.83 kg/kWh for India
(10,249 t/USD mn × 81.3 USD/MWh), against projected 2030 factors of 0.038
and 0.675.

**BTM natural gas** (`btm_natgas`, physical sensitivity, `--btm-countries`,
default Canada; share $\beta$ = `--btm-natgas-share`, default 0.92):

$$E^{onsite} = \beta\,\alpha\,\Delta G \times 1000 \times \varepsilon^{gas}, \qquad
E^{gas,up} = \beta\,\alpha\,\Delta G \times 1000 \times \varepsilon^{gas,up},$$

where $\varepsilon^{gas}$ is `kgco2_per_kwh_natgas` (combustion) and
$\varepsilon^{gas,up}$ is `--natgas-upstream-kgco2e-per-kwh` (extraction,
processing, transport and methane leakage; default 0). The grid supplies
only $(1-\beta)$ of facility electricity through $a'_{ES}$. The default
0.92 is the share of planned Canadian capacity located in Alberta; it is an
upper bound for the BTM share, not an observed value.

## 11. Emissions components

For a given scenario, $\mathbf{e} = \hat{\mathbf{f}}\Delta\mathbf{x}$ is the
vector of emissions by origin industry. The components do not overlap:

$$\text{Direct} = \sum_{k\in S} f_k \Delta x_k + E^{onsite}$$

$$\text{Upstream (excl. direct)} = \sum_{i\notin S} f_i \Delta x_i + E^{gas,up}$$

$$\text{of which purchased electricity} = f_E\, a'_{ES}\,\Delta\mathbf{q}$$

$$\text{Footprint} = \text{Direct} + \text{Upstream} \quad (\text{column } \texttt{global\_upstream\_ghg\_tonnes\_co2e}).$$

Emissions are also split by origin: destination country, United States,
foreign (all other), and electricity industries (ICIO 35) worldwide.

**Downstream** is defined in section 13 and is never added to the
footprint.

## 12. Output gain, value added and cross-border intermediates

$$\text{Global output gain} = \mathbf{1}'\Delta\mathbf{x}, \quad
\text{destination} = \sum_{c(i)=c}\Delta x_i, \quad
\text{foreign} = \sum_{c(i)\ne c}\Delta x_i .$$

Value added uses the hybrid coefficients for the data-centre columns:

$$\Delta VA = \mathbf{v}'\Delta\mathbf{x}, \qquad v_k = 1-\sum_i a'_{ik}\ (k\in S).$$

Cross-border intermediate sales (exports of intermediates) of industry $i$
induced by the expansion:

$$\Delta z_i^{exp} = \sum_{j:\,c(j)\ne c(i)} a'_{ij}\Delta x_j ,$$

and intermediates imported directly by the destination country:

$$\Delta z_i^{imp,c} = \sum_{j:\,c(j)=c} a'_{ij}\Delta x_j, \qquad c(i)\ne c .$$

The headline summary ranks foreign supplier countries by the value of
goods and services produced there for the expansion,
$\sum_{c(i)=d}\Delta x_i$ (CN1+CN2 and MX1+MX2 are combined; the
rest-of-world aggregate is reported separately), and reports the
emissions emitted there, $\sum_{c(i)=d} e_i$, their cross-border
intermediate sales and their three largest products by $\Delta x_i$.

## 13. Downstream: Ghosh supply-driven model

Allocation (sales) coefficients and the Ghosh inverse:

$$\mathbf{B} = \hat{\mathbf{x}}^{-1}\mathbf{Z}, \qquad
\mathbf{G} = (\mathbf{I}-\mathbf{B})^{-1}, \qquad
\Delta\mathbf{x}'^{\,G} = \Delta\mathbf{v}'\mathbf{G}.$$

Primary-input shocks are placed at the data-centre industries only and
calibrated so the Ghosh output change at $S$ equals the same
$\Delta\mathbf{q}$:

$$\Delta\mathbf{v}_S = \big(\mathbf{G}^{\top}_{SS}\big)^{-1}\Delta\mathbf{q}, \qquad
\Delta\mathbf{x}^G = \mathbf{G}^{\top}_{\cdot S}\Delta\mathbf{v}_S .$$

$$\text{Downstream (excl. direct)} = \sum_{i\notin S} f_i\,\Delta x_i^G .$$

Interpretation: income-based ("enabled") emissions of the industries that
buy data-centre services, in proportion to their 2022 purchases (Lenzen
and Murray 2010; Marques et al. 2012). As a quantity model the Ghosh model
is implausible (Oosterhaven 1988), so these are an allocation of
responsibility, not emissions caused by the expansion, and must never be
added to upstream.

The electricity industry is not a Ghosh target: the data centres consume
their electricity, they do not sell it on. Forcing electricity output up in
a supply-driven model would push it to all of the electricity sector's 2022
customers (heavy industry), which is spurious. On-site generation is not
added downstream either. Downstream results therefore do not depend on the
electricity sensitivity.

## 14. Baselines, shares and the headline summary

2022 baseline emissions $\mathbf{f}'\mathbf{x}$: global 48.37 Gt CO2e of
production emissions (households and land use excluded), Canada 0.70 Gt,
India 4.11 Gt. Results are reported as % of the global baseline, % of the
destination's baseline, and % change against the `average_grid` result.

`summarize_results.py` picks one headline row per destination: last
scenario year, `cumulative_vs_first_year`, `all_expansion`,
`physical_facility_demand`, and `btm_natgas` where it exists, otherwise
`projected_grid`, otherwise `average_grid` (all overridable). It writes
`headline_summary.csv`, `headline_top_origin_countries.csv` and
`headline_summary.md` (abstract-ready sentences).

For outputs written before the fixes ("legacy"), it reconstructs the
components as: direct = data-centre industry emissions + $0.92\,\Delta G\,\varepsilon^{gas}$;
upstream = total − direct; downstream = the `icio_endogenous` Ghosh run
(which had no electricity target) minus the data-centre industry's
emissions. Value added, cross-border intermediates and purchased
electricity are not available for legacy outputs.

## 15. Optional: construction and equipment (`--capex`)

Requires `dc_capex.csv` (country, year, capex per MW in 2022 USD million)
and `dc_capex_composition.csv` (country, ICIO product, share; shares sum
to one). Templates are written if they are missing; no default values are
supplied. New capacity $\Delta MW$ (annual or cumulative) gives

$$\Delta y_i = \Delta MW \cdot k \cdot s_{g(i)} \cdot
\frac{F_{i,\,c\_GFCF}}{\sum_{j\in g(i)} F_{j,\,c\_GFCF}},$$

i.e. capex $k$ per MW split over products $g$ by $s_g$ and over countries
of origin by the destination's 2022 gross fixed capital formation
purchases of each product. Then $\Delta\mathbf{x} = \mathbf{L}\Delta\mathbf{y}$
and emissions $\mathbf{f}'\Delta\mathbf{x}$. These are one-off emissions
in the build year, reported separately in
`capex_embodied_emissions_summary.csv`.

## 16. Other modules

* `tiva_ghg_indicators.py` (`--tiva-ghg-indicators`): OECD TiVA-style
  value added and GHG content of gross exports, final demand and the Ghosh
  forward multiplier, on the (split) table with $\mathbf{f}$ in place of
  $\mathbf{v}$.
* `project_icio.py` (optional): projects the table 2023–2030 with OECD
  bridge regressions, ARIMA or EIA/IMF drivers and RAS. The headline
  results use the 2022 table and 2022 intensities; if a projected table is
  used, the emissions extension should be rebuilt for that year (a warning
  is logged when `--emissions-extension-year` differs from `--icio-year`).

## 17. Assumptions and limitations

1. Fixed 2022 technology and intensities except the destination
   electricity sector under `projected_grid` / `btm_natgas`.
2. The data-centre proxy is J62_63 / 62T63, or CLD/DPH for Canada and DC
   for India when the splits are used; without the India split, India's
   cost structure is that of IT services in general.
3. Output per MW (`price_mn_MW_2022`) drives the shock size and the
   electricity cost share. Current inputs imply electricity costs of 59%
   (Canada) and 98% (India) of data-centre revenue, with capacity running
   at 100% utilisation (facility MW = IT MW × PUE). With such shares the
   hybrid column leaves little room for other inputs (India: other inputs
   scaled to 2.4% of the ICIO column).
4. Electricity intensity is a national average converted with the
   data-centre price; provincial/state differences are not modelled.
5. BTM share 0.92 is an upper bound; gas upstream emissions default to
   zero.
6. Construction and IT hardware are excluded unless `--capex` is used.
7. Indirect water is not estimated; e-waste covers servers only at
   GB300 NVL72 density (section 7).
8. Downstream (Ghosh) results are an income-based allocation, not
   impacts.
9. Monetary values in 2022 USD; no price effects or substitution.

## 18. Change log (methodology fixes, November 2026)

| Issue in the previous version | Fix | Effect on results |
|---|---|---|
| Physical sensitivity fixed *total* domestic electricity output at the data centres' expenditure (indirect electricity dropped; negative electricity final demand) | Hybrid column: electricity coefficient = measured cost share, other inputs rescaled (section 9) | More domestic electricity (indirect use kept); fewer non-electricity inputs |
| Electricity shock not scaled by the attribution share (a lens with share 0 still produced up to 11.9 Mt in Canada) | Electricity scales with output through the hybrid column; zero shock gives no rows | Attribution lenses now consistent |
| Ghosh run also targeted the electricity industry, pushing electricity to all its customers | Ghosh targets the data-centre industries only | Downstream fell from 12.40 to 0.53 Mt (Canada) and from 53.45 to 0.87 Mt (India), excluding direct |
| BTM adjustment added to upstream and to downstream | On-site generation is direct, reported once | No double counting |
| BTM displaced the 2022 extension grid (0.117 kg/kWh), not the projected grid (0.038) | `projected_grid` intensity; BTM builds on it | Canadian BTM increase over grid supply 9.77 Mt instead of 7.35 Mt (2025–2030 cumulative) |
| Hard-coded BTM share 0.92, combustion only | `--btm-natgas-share`, `--natgas-upstream-kgco2e-per-kwh` | Sensitivities possible |
| Direct emissions counted in both upstream and downstream | Direct / upstream / downstream reported separately | Non-overlapping components |
| No output gain, value added or trade measures | Global/domestic/foreign output, value added, cross-border intermediates | New columns |
| No embodied capex | Optional `--capex` module | Not run by default |
| E-waste calculated outside the model | `ewaste.py`, in `ewaste_metrics.csv` and the headline summary | Same values as before (17.784 t/MW) |
| `2030_additional_output` not flagged | `shock_basis_note` marks it as an upper bound | — |
| Canadian split failed on ICIO 2025 labels (`62T63`) and matched inputs by 2023 letter codes | Both label conventions accepted | Split runs on ICIO 2025 |
| India not split | `--split-india-iot` (section 4b) | India shock goes to a data-centre sector |
| Split columns' value added was a residual | RAS to source-table value-added shares (section 4c); table-wide balance check | Balanced table with sourced value added |
| Electricity cost share fixed by rent per MW at full load | `--output-includes-power-passthrough`, `--it-utilisation` | Lower, sourced electricity shares |

## References

Dietzenbacher, E. (1997). In vindication of the Ghosh model: a
reinterpretation as a price model. *Journal of Regional Science* 37(4).

Lenzen, M. and Murray, J. (2010). Conceptualising environmental
responsibility. *Ecological Economics* 70(2).

Marques, A., Rodrigues, J., Lenzen, M. and Domingos, T. (2012).
Income-based environmental responsibility. *Ecological Economics* 84.

Miller, R. E. and Blair, P. D. (2022). *Input-Output Analysis:
Foundations and Extensions*, 3rd ed., Cambridge University Press
(mixed exogenous/endogenous models; supply-side models).

Oosterhaven, J. (1988). On the plausibility of the supply-driven
input-output model. *Journal of Regional Science* 28(2).
