
@echo off
setlocal

cd /d "%~dp0"

call conda activate datacenter_env

if errorlevel 1 (
    echo ERROR: Could not activate datacenter_env.
    pause
    exit /b 1
)

python diagnose_gtap12_emissions.py ^
    --co2 "GDX12_23\GSDFEMISS.GDX" ^
    --nco2 "GDX12_23\GSDFNCO2.GDX" ^
    --output-dir "gtap12_diagnostics"

if errorlevel 1 (
    echo.
    echo ERROR: Diagnostic script failed.
    pause
    exit /b 1
)

echo.
echo Diagnostic completed successfully.
pause
