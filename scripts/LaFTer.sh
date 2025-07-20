#!/bin/bash
# custom config
# Usage: ./LaFTer.sh [dataset] [enable_augmentation]
# Example: ./LaFTer.sh Emotion6 1  (enable augmentation)
# Example: ./LaFTer.sh Emotion6 0  (disable augmentation)

DATA=data
TRAINER=LaFTer
CFG=vit_b32
dset="$1"
enable_aug="${2:-0}"  # Default to 0 if not provided
txt_cls=lafter

if [ "$enable_aug" == "1" ]; then
    echo "Starting training with image augmentation enabled..."
    CUDA_VISIBLE_DEVICES=1 python LaFTer.py \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/"${dset}".yaml \
    --config-file configs/trainers/text_cls/${CFG}.yaml \
    --output-dir output/${TRAINER}/${CFG}/"${dset}"_aug \
    --lr 0.0005 \
    --txt_cls ${txt_cls} \
    --enable_augmentation
else
    echo "Starting training without image augmentation..."
    CUDA_VISIBLE_DEVICES=1 python LaFTer.py \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/"${dset}".yaml \
    --config-file configs/trainers/text_cls/${CFG}.yaml \
    --output-dir output/${TRAINER}/${CFG}/"${dset}" \
    --lr 0.0005 \
    --txt_cls ${txt_cls}
fi
