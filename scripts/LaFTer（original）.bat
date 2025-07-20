@echo off
:: Custom config
set DATA=data
set TRAINER=LaFTer
set CFG=vit_b32
set dset=%1
set txt_cls=lafter

:: Set environment variable for CUDA (Optional, depending on setup)
:: set CUDA_VISIBLE_DEVICES=0

:: Run the python script with the correct parameters
python LaFTer.py ^
--root %DATA% ^
--trainer %TRAINER% ^
--dataset-config-file configs\datasets\%dset%.yaml ^
--config-file configs\trainers\text_cls\%CFG%.yaml ^
--output-dir output\%TRAINER%\%CFG%\%dset% ^
--lr 0.0005 ^
--txt_cls %txt_cls%
