# LaFTer 双任务学习功能说明

## 概述

本项目已经扩展支持双任务学习功能，基于论文 "Joint Image Emotion Classification and Distribution Learning via Deep Convolutional Neural Network" 的方法，在原有的分类任务基础上增加了情感分布学习任务。

## 双任务学习原理

### 损失函数设计

双任务学习使用加权组合的损失函数：

```
L = (1 - λ)L_cls(x, y) + λL_sdl(x, l)
```

其中：
- **L_cls**: 标准的softmax分类损失
- **L_sdl**: 情感分布学习损失（使用KL散度）
- **λ**: 权重参数，控制两种损失的平衡（默认设置为0.8）

### 优化器配置

- 使用 **SGD优化器**（momentum=0.9）
- 学习率：0.001
- 权重衰减：根据args.weight_decay设置

### 分布标签来源

Emotion6数据集中已包含人工标注的情感分布标签，无需使用论文中的"策略1 - 基于蕴含关系"来生成分布。每个图像的情感分布直接从JSON文件中的`emotions`字段读取。

## 使用方法

### 1. 启用双任务学习模式

```bash
python LaFTer.py \
    --root data \
    --trainer LaFTer \
    --dataset-config-file configs/datasets/Emotion6.yaml \
    --config-file configs/trainers/text_cls/vit_b32.yaml \
    --output-dir output/LaFTer/vit_b32/Emotion6_dual_task \
    --lr 0.001 \
    --epochs 50 \
    --txt_cls lafter \
    --dual_task \
    --lambda_weight 0.8
```

### 2. 使用预设脚本

执行双任务学习训练：
```bash
scripts\LaFTer_DualTask.bat
```

### 3. 传统单任务模式（对比）

不添加 `--dual_task` 参数即为传统单任务分类模式：
```bash
python LaFTer.py \
    --root data \
    --trainer LaFTer \
    --dataset-config-file configs/datasets/Emotion6.yaml \
    --config-file configs/trainers/text_cls/vit_b32.yaml \
    --output-dir output/LaFTer/vit_b32/Emotion6_single_task \
    --lr 0.001 \
    --epochs 50 \
    --txt_cls lafter
```

## 主要修改内容

### 1. 数据加载器修改 (`datasets/Emotion6.py`)
- 在 `Datum` 类中添加 `emotion_distribution` 属性
- 修改 `_read_data_from_json` 方法，从JSON文件中读取情感分布标签
- 对分布进行归一化处理

### 2. 双任务损失函数 (`utils/utils.py`)
- 新增 `DualTaskLoss` 类
- 实现分类损失 + 分布学习损失的加权组合
- 使用KL散度计算分布学习损失

### 3. 训练流程修改 (`LaFTer.py`)
- 修改 `setup_lafter_training_utils` 使用SGD优化器
- 更新 `train_lafter` 函数支持双任务学习
- 添加命令行参数：`--dual_task` 和 `--lambda_weight`

### 4. 优化器更改 (`utils/utils.py`)
- 从AdamW更改为SGD优化器
- 添加momentum=0.9参数

## 关键参数说明

- `--dual_task`: 启用双任务学习模式的开关
- `--lambda_weight`: 控制分布学习损失的权重（推荐值：0.8）
- `--lr`: 学习率（推荐值：0.001，配合SGD使用）
- `--epochs`: 训练轮数（推荐值：50）

## 预期效果

双任务学习通过同时优化分类和分布预测任务，能够：
1. 提高特征学习的鲁棒性
2. 更好地捕捉图像情感的复杂性和主观性
3. 利用情感分布的标签歧义性信息
4. 改善模型在情感识别任务上的性能

## 输出信息

训练过程中会显示详细的损失信息：
- `total_loss`: 总损失（加权组合）
- `cls_loss`: 分类损失
- `dist_loss`: 分布学习损失

示例输出：
```
epoch [1/50][1/28]	total_loss 2.1543	cls_loss 1.8923	dist_loss 0.8104	lr 1.000000e-03
Epoch 1 - Avg Total Loss: 2.0234, Avg Cls Loss: 1.7845, Avg Dist Loss: 0.7891
``` 