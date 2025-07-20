@echo off
:: --- Direct Dual Task Learning Training Script ---
:: Skip text classifier training, go directly to dual-task image fine-tuning
set DATA=data
set TRAINER=LaFTer
set CFG=vit_b32
set DSET=Emotion6
set TXT_CLS=lafter

:: --- Distribution Strategy Configuration ---
:: Available strategies: json (read from JSON) or strategy1 (generate using Gaussian function)
set DISTRIBUTION_STRATEGY=strategy1
set SIGMA_CONF=2.5
set EPSILON=0.01

echo ============================================
echo Starting Direct Dual-Task Training Mode
echo ============================================
echo Dataset: %DSET%
echo Architecture: %CFG%
echo Distribution Strategy: %DISTRIBUTION_STRATEGY%
echo Lambda Weight: 0.8
echo Learning Rate: 0.001
echo Epochs: 5
echo ============================================

echo Starting training...
python LaFTer.py ^
    --root %DATA% ^
    --trainer %TRAINER% ^
    --dataset-config-file configs\datasets\%DSET%.yaml ^
    --config-file configs\trainers\text_cls\%CFG%.yaml ^
    --output-dir output\%TRAINER%\%CFG%\%DSET%_direct_dual_task_%DISTRIBUTION_STRATEGY% ^
    --lr 0.001 ^
    --epochs 5 ^
    --txt_cls %TXT_CLS% ^
    --direct_dualtask ^
    --dual_task ^
    --lambda_weight 0.8 ^
    DATASET.DISTRIBUTION_STRATEGY %DISTRIBUTION_STRATEGY% ^
    DATASET.SIGMA_CONF %SIGMA_CONF% ^
    DATASET.EPSILON %EPSILON%

echo ============================================
echo Direct Dual-Task Training Features:
echo - Skips text classifier pre-training
echo - Starts directly with image dual-task fine-tuning
echo - Uses both classification and distribution learning
echo - Lambda weight: 0.8 (favors distribution learning)
if "%DISTRIBUTION_STRATEGY%"=="strategy1" (
    echo - Gaussian distribution parameters: sigma=%SIGMA_CONF%, epsilon=%EPSILON%
)
echo ============================================
echo Direct dual task training completed
echo ============================================
pause 