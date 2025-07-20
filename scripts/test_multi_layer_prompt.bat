@echo off
REM Multi-Layer Prompt Technology Test Script (Windows)

echo === Multi-Layer Prompt Technology Test ===
echo Note: Using MultiLayerLaFTer trainer with multi-layer prompt technology

set DATA=data
set CFG=vit_b32
set TXT_CLS=lafter
REM Basic configuration
set DATASET=Emotion6
set CONFIG_FILE=configs/trainers/text_cls/vit_b32.yaml
set DATASET_CONFIG=configs/datasets/Emotion6.yaml
set OUTPUT_DIR=output/multi_layer_prompt_test

echo.
echo Test 1: Multi-Layer Prompt + Dual Task Learning
echo Using MultiLayerLaFTer trainer...
python LaFTer.py ^
    --trainer MultiLayerLaFTer ^
    --config-file %CONFIG_FILE% ^
    --dataset-config-file %DATASET_CONFIG% ^
    --output-dir %OUTPUT_DIR%/dual_task_multilayer ^
    --dual_task ^
    --lambda_weight 0.8 ^
    --epochs 8 ^
    --batch_size 32 ^
    --txt_cls %TXT_CLS% ^
    --lr 0.001 ^
    --multi_layer_prompt

echo ============================================
echo Multi-layer prompt test completed with MultiLayerLaFTer trainer
echo ============================================
pause

