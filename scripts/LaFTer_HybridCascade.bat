@echo off
echo ========================================
echo HybridCascade-LaFTer Training Script
echo Using HybridCascadeLaFTer trainer
echo ========================================
set TXT_CLS=lafter

set DISTRIBUTION_STRATEGY=strategy1
set SIGMA_CONF=2.5
set EPSILON=0.1
set TXT_CLS=lafter
python LaFTer.py ^
    --root data ^
    --trainer HybridCascadeLaFTer ^
    --dataset-config-file configs/datasets/Emotion6.yaml ^
    --config-file configs/trainers/hybrid_cascade_lafter/vit_b32.yaml ^
    --output-dir output/hybrid_cascade_test ^
    --dual_task ^
    --txt_cls %TXT_CLS% ^
    --scheduler none ^
    --lambda_weight 0.8 ^
    --epochs 10 ^
    --lr 0.001 ^
    DATASET.DISTRIBUTION_STRATEGY %DISTRIBUTION_STRATEGY% ^
    DATASET.SIGMA_CONF %SIGMA_CONF% ^
    DATASET.EPSILON %EPSILON%

echo Training completed!
pause