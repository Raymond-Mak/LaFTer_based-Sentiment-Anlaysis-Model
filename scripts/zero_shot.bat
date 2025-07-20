@echo off
rem --- Set variables used by the script ---
set DATA=data
set TRAINER=LaFTer
set CFG=vit_b32
set dset=%1
set txt_cls=zero_shot

rem --- Check if dataset parameter is provided ---
if "%dset%"=="" (
    echo Error: Please provide a dataset name as the first parameter.
    echo Usage: zero_shot.bat [dataset_name]
    goto :eof
)

rem --- Set CUDA visible devices environment variable ---
rem --- Note: The specific effect on Windows depends on your CUDA and driver configuration ---
set CUDA_VISIBLE_DEVICES=0

rem --- Execute Python script ---
echo Running LaFter.py with dataset '%dset%'...
python LaFter.py ^
--root %DATA% ^
--trainer %TRAINER% ^
--dataset-config-file configs\datasets\%dset%.yaml ^
--config-file configs\trainers\text_cls\%CFG%.yaml ^
--output-dir output\%TRAINER%\%CFG%\%dset% ^
--lr 0.0005 ^
--zero_shot ^
--txt_cls %txt_cls%

echo.
echo Script execution completed.

rem --- Optional: Uncomment the next line to pause after script execution to view output ---
rem pause

:eof
