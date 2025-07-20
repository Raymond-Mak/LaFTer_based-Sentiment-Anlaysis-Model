@echo off
:: --- Dual Task Learning Training Script (Basic LaFTer) ---
set DATA=data
set TRAINER=LaFTer
set CFG=vit_b32
set DSET=Emotion6
set TXT_CLS=lafter

:: --- Distribution Strategy Configuration ---
:: Available strategies: json (read from JSON) or strategy1 (generate using Gaussian function)
set DISTRIBUTION_STRATEGY=strategy1
set SIGMA_CONF=1.5
set EPSILON=0.1

echo Starting dual-task training with basic LaFTer trainer...
echo Note: Using LaFTer trainer (basic version) for dual-task learning
python LaFTer.py ^
    --root %DATA% ^
    --trainer %TRAINER% ^
    --dataset-config-file configs\datasets\%DSET%.yaml ^
    --config-file configs\trainers\text_cls\%CFG%.yaml ^
    --output-dir output\%TRAINER%\%CFG%\%DSET%_dual_task_%DISTRIBUTION_STRATEGY% ^
    --lr 0.001 ^
    --epochs 7 ^
    --txt_cls %TXT_CLS% ^
    --dual_task ^
    --lambda_weight 0.8 ^
    DATASET.DISTRIBUTION_STRATEGY %DISTRIBUTION_STRATEGY% ^
    DATASET.SIGMA_CONF %SIGMA_CONF% ^
    DATASET.EPSILON %EPSILON%

echo ============================================
echo Distribution strategy used: %DISTRIBUTION_STRATEGY%
if "%DISTRIBUTION_STRATEGY%"=="strategy1" (
    echo Gaussian distribution parameters - sigma_conf: %SIGMA_CONF%, epsilon: %EPSILON%
)
echo ============================================
echo Dual task learning training completed with basic LaFTer trainer
echo ============================================
pause 