@echo off
:: --- Parameter Definition ---
set DATA=data
set TRAINER=LaFTer
set CFG=vit_b32
set DSET=%1
set SKIP_FT=%2
set TXT_CLS=lafter

:: --- Usage ---
:: LaFTer.bat [dataset] [skip_finetune]
:: Example: LaFTer.bat Emotion6 0  (enable fine-tuning)
:: Example: LaFTer.bat Emotion6 1  (skip fine-tuning)

:: --- Determine whether to skip fine-tuning ---
if "%SKIP_FT%"=="1" (
    echo Skip image fine-tuning, execute text classifier-only mode
    python LaFTer.py ^
        --root %DATA% ^
        --trainer %TRAINER% ^
        --dataset-config-file configs\datasets\%DSET%.yaml ^
        --config-file configs\trainers\text_cls\%CFG%.yaml ^
        --output-dir output\%TRAINER%\%CFG%\%DSET% ^
        --lr 0.001 ^
        --txt_cls %TXT_CLS% ^
        --skip_finetune
) else (
    echo Enable image fine-tuning, execute full training mode
    python LaFTer.py ^
        --root %DATA% ^
        --trainer %TRAINER% ^
        --dataset-config-file configs\datasets\%DSET%.yaml ^
        --config-file configs\trainers\text_cls\%CFG%.yaml ^
        --output-dir output\%TRAINER%\%CFG%\%DSET% ^
        --lr 0.001 ^
        --txt_cls %TXT_CLS%
)

