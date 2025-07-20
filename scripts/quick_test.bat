@echo off
:: --- Quick Test with Multi-Layer LaFTer Trainer ---

set DATA=data
set TRAINER=MultiLayerLaFTer
set CFG=vit_b32
set DSET=Emotion6
set TXT_CLS=lafter

echo Starting quick test with MultiLayerLaFTer trainer...
echo Note: Using MultiLayerLaFTer trainer with multi-layer prompt technology

python LaFTer.py ^
    --root %DATA% ^
    --trainer %TRAINER% ^
    --dataset-config-file configs\datasets\%DSET%.yaml ^
    --config-file configs\trainers\text_cls\%CFG%.yaml ^
    --output-dir output/quick_test_multilayer ^
    --epochs 2 ^
    --batch_size 16 ^
    --lr 0.001 ^
    --txt_cls %TXT_CLS% ^
    --txt_epochs 100 ^
    --weight-decay 0.0001 ^
    --scheduler cosine ^
    --multi_layer_prompt

echo ============================================
echo Quick test completed with MultiLayerLaFTer trainer
echo ============================================
pause
