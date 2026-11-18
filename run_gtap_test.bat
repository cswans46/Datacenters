
@echo off
setlocal

cd /d "%~dp0"

call conda activate datacenter_env

python gtap_testing.py ^
    --co2 "GDX12_23\GSDFEMISS.GDX" ^
    --nco2 "GDX12_23\GSDFNCO2.GDX"

pause
