@echo off
setlocal

REM Run from the directory containing this batch file
cd /d "%~dp0"
@REM  Comment this line to disable check-only mode
@REM  set "check_only=--check-only"

REM Create Conda environment if needed
conda env list | findstr /I "datacenter_env" >nul
if errorlevel 1 (
    echo Creating Conda environment...
    call conda create -n datacenter_env python=3.11 -y || exit /b 1
)

REM Activate environment
call conda activate datacenter_env || exit /b 1

REM Install or update the Python dependencies listed in pyproject.toml
REM ([project] dependencies). Packages already installed are skipped.
echo Checking Python dependencies from pyproject.toml...
python -c "import tomllib, subprocess, sys; deps = tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']; subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', *deps])" || exit /b 1

REM Define directories
set "DATA_DIR=%CD%"
set "ICIO_DATA_DIR=D:\Research\OECD Research"
set "OUTPUT_DIR=%DATA_DIR%\datacentre_results"

REM ------------------------------------------------------------------
REM Canadian DSUT split of CAN_J62_63 (cloud / other 51821 / rest)
REM   1. python canada_dsut_split.py write-templates --out-dir "%DATA_DIR%"
REM      and fill canada_dsut_2020.csv from the 2020 Canadian DSUT.
REM   2. The 2022 StatCan SUT (36-10-0478-01) is downloaded to
REM      %DATA_DIR%\statcan_cache on first run (or set --statcan-sut-file).
REM   3. Optional: projected (E3MC) cloud share of NAICS 51821 output for
REM      the SUT year, e.g. set "cloud_share=--canada-cloud-share-override 0.45"
REM   Comment out the next line to run without the split.
set "canada_split=--split-canada-dsut --canada-dsut-file "%DATA_DIR%\canada_dsut_2020.csv" --sut-year 2022 --canada-dc-target cloud_and_hosting --dc-origin-shares default"
@REM set "cloud_share=--canada-cloud-share-override 0.45"

REM ------------------------------------------------------------------
REM Indian split of IND_62T63 (data centres vs rest of IT services) from the
REM 2023-24 India IOT with an AI-DC sector. dc_sector_map.csv must map IND
REM to IND_62T63. Comment out to run without the split.
set "india_split=--split-india-iot "%DATA_DIR%\IOT_2023-24_RAS.xlsx""

REM Electricity cost share options (see docs\methodology.md, section 6):
REM   --output-includes-power-passthrough  rent per MW excludes power; add the bill
REM   --it-utilisation 0.7                 electricity file assumes full load
@REM set "power_opts=--output-includes-power-passthrough --it-utilisation 0.7"

REM TiVA and GHG footprint indicators on the (split) ICIO.
set "indicators=--tiva-ghg-indicators"
@REM set "indicators=--tiva-ghg-indicators --indicators-only"

REM Electricity / behind-the-meter gas options (see docs\methodology.md).
REM 0.92 is the Alberta share of planned capacity: an upper bound for BTM.
set "elec_opts=--btm-natgas-share 0.92 --natgas-upstream-kgco2e-per-kwh 0.0"
REM Optional embodied capex (needs dc_capex.csv and dc_capex_composition.csv):
@REM set "capex=--capex"
REM Server e-waste: defaults are GB300 NVL72 (7.6 racks/MW, 18 servers/rack, 130 kg).
@REM set "ewaste_opts=--server-lifetime-years 5"

REM Run model
echo.
echo Running the data-centre ICIO model...

python global_datacentre_icio_model_v2.py --data-dir "%DATA_DIR%" --output-dir "%OUTPUT_DIR%" --A-file "%ICIO_DATA_DIR%\ICIO2025econA.RData" --X-file "%ICIO_DATA_DIR%\ICIO2025econX.RData" --FD-file "%ICIO_DATA_DIR%\ICIO2025econFD.RData" --emissions-extension "%DATA_DIR%\hybrid_emissions\direct_emissions.csv" --icio-year 2022 %canada_split% %india_split% %cloud_share% %indicators% %power_opts% %elec_opts% %capex% %ewaste_opts% %check_only%

if errorlevel 1 (
    echo.
    echo Model failed. See the error messages above.
    pause
    exit /b 1
)

REM ------------------------------------------------------------------
REM Optional: project the (split) ICIO 2023-2030 with the OECD, ARIMA and
REM external (AEO/IEO/WEO) methods and compare them. Run once:
REM   python project_icio.py build-panel --out data\projection\panel --data-dir "%DATA_DIR%" --A-file "%ICIO_DATA_DIR%\ICIO2025econA.RData" --X-file "%ICIO_DATA_DIR%\ICIO2025econX.RData" --FD-file "%ICIO_DATA_DIR%\ICIO2025econFD.RData" [--pyp-dir D:\ICIO\PYP]
REM   set EIA_API_KEY=...   then   python project_icio.py fetch-external --out data\projection\external.csv
REM Then (set run_projection=1 to enable):
if "%run_projection%"=="1" (
    python project_icio.py run --panel data\projection\panel --external data\projection\external.csv --out "%OUTPUT_DIR%\projection" --rdata --data-dir "%DATA_DIR%" --output-dir "%OUTPUT_DIR%" --A-file "%ICIO_DATA_DIR%\ICIO2025econA.RData" --X-file "%ICIO_DATA_DIR%\ICIO2025econX.RData" --FD-file "%ICIO_DATA_DIR%\ICIO2025econFD.RData" %canada_split%
)

echo.
echo Model completed successfully.
pause