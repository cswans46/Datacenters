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
set "canada_split=--split-canada-dsut --canada-dsut-file "%DATA_DIR%\canada_dsut_2020.csv" --sut-year 2022 --canada-dc-target cloud_and_hosting"
@REM set "cloud_share=--canada-cloud-share-override 0.45"

REM TiVA and GHG footprint indicators on the (split) ICIO.
set "indicators=--tiva-ghg-indicators"
@REM set "indicators=--tiva-ghg-indicators --indicators-only"

REM Run model
echo.
echo Running the data-centre ICIO model...

python global_datacentre_icio_model_v2.py --data-dir "%DATA_DIR%" --output-dir "%OUTPUT_DIR%" --A-file "%ICIO_DATA_DIR%\ICIO2025econA.RData" --X-file "%ICIO_DATA_DIR%\ICIO2025econX.RData" --FD-file "%ICIO_DATA_DIR%\ICIO2025econFD.RData" --emissions-extension "%DATA_DIR%\hybrid_emissions\direct_emissions.csv" --icio-year 2022 %canada_split% %cloud_share% %indicators% %check_only%

if errorlevel 1 (
    echo.
    echo Model failed. See the error messages above.
    pause
    exit /b 1
)

echo.
echo Model completed successfully.
pause