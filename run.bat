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

REM Run model
echo.
echo Running the data-centre ICIO model...

python global_datacentre_icio_model_v2.py --data-dir "%DATA_DIR%" --output-dir "%OUTPUT_DIR%" --A-file "%ICIO_DATA_DIR%\ICIO2025econA.RData" --X-file "%ICIO_DATA_DIR%\ICIO2025econX.RData" --FD-file "%ICIO_DATA_DIR%\ICIO2025econFD.RData" --emissions-extension "%DATA_DIR%\hybrid_emissions\direct_emissions.csv" --icio-year 2022 %check_only%

if errorlevel 1 (
    echo.
    echo Model failed. See the error messages above.
    pause
    exit /b 1
)

echo.
echo Model completed successfully.
pause