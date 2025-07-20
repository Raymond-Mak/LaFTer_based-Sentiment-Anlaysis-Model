# LaFTer 直接双任务模式 (Direct Dual-Task Mode)

## 概述

传统的LaFTer训练流程包含两个阶段：
1. **文本分类器预训练阶段**：使用文本描述训练adapter
2. **图像微调阶段**：使用图像数据进行双任务学习

**直接双任务模式**跳过第一阶段，让模型直接从图像数据开始双任务学习，适用于以下场景：
- 希望从头开始训练adapter，不依赖文本预训练
- 想要更纯粹的图像到情感映射学习
- 测试双任务学习的直接效果

## 新增功能

### 1. 新增函数

#### `train_lafter_direct(args, model, tr_loader, val_loader)`
- 跳过文本分类器训练阶段
- 直接进入图像双任务微调
- 自动启用双任务学习模式
- 从随机初始化的adapter开始训练

### 2. 新增命令行参数

```bash
--direct_dualtask
```
启用直接双任务模式的开关参数

### 3. 训练流程对比

#### 传统模式 (train_lafter)
```
1. 文本分类器预训练 (train_txt_cls)
   ↓
2. 图像双任务微调 (DualTaskLoss)
   ↓
3. 输出结果
```

#### 直接双任务模式 (train_lafter_direct)
```
1. 直接图像双任务微调 (DualTaskLoss)
   ↓
2. 输出结果
```

## 使用方法

### 1. 命令行方式

#### Emotion6数据集
```bash
python LaFTer.py \
    --root data \
    --trainer LaFTer \
    --dataset-config-file configs/datasets/Emotion6.yaml \
    --config-file configs/trainers/text_cls/vit_b32.yaml \
    --output-dir output/LaFTer/vit_b32/Emotion6_direct_dual_task \
    --lr 0.001 \
    --epochs 10 \
    --txt_cls lafter \
    --direct_dualtask \
    --dual_task \
    --lambda_weight 0.8
```

#### EmoSet数据集
```bash
python LaFTer.py \
    --root data \
    --trainer LaFTer \
    --dataset-config-file configs/datasets/Emoset.yaml \
    --config-file configs/trainers/text_cls/vit_b32.yaml \
    --output-dir output/LaFTer/vit_b32/Emoset_direct_dual_task \
    --lr 0.0008 \
    --epochs 30 \
    --txt_cls lafter \
    --direct_dualtask \
    --dual_task \
    --lambda_weight 0.8
```

### 2. 批处理脚本

#### Emotion6数据集
```bash
scripts\LaFTer_DirectDualTask.bat
```

#### EmoSet数据集
```bash
scripts\LaFTer_DirectDualTask_EmoSet.bat
```

## 参数配置建议

### Emotion6数据集
- **学习率**: 0.001
- **训练轮数**: 10-15 epochs
- **Lambda权重**: 0.8
- **批次大小**: 50

### EmoSet数据集
- **学习率**: 0.0008
- **训练轮数**: 30 epochs
- **Lambda权重**: 0.8
- **批次大小**: 50

## 核心特性

### 1. 自动双任务启用
如果未设置 `--dual_task`，直接双任务模式会自动启用：
```python
if not use_dual_task:
    print("Warning: --dual_task not enabled. Enabling dual-task mode for direct training.")
    use_dual_task = True
    args.dual_task = True
```

### 2. 情感分布处理
- 自动检测数据集中的情感分布标签
- 如果没有分布标签，自动生成one-hot分布作为fallback
- 支持多种分布生成策略（strategy1等）

### 3. 损失函数
使用与传统模式相同的 `DualTaskLoss`：
- 分类损失 + 分布学习损失
- 协同梯度优化
- 温度缩放的KL散度

### 4. 详细调试信息
- 分布质量统计
- 训练过程监控
- Epoch级别的损失分析

## 预期效果

### 优势
1. **训练时间更短**：跳过文本预训练阶段
2. **更直接的图像学习**：没有文本预训练的先验偏置
3. **纯粹的双任务学习**：从头开始的分类+分布联合优化

### 劣势
1. **可能收敛较慢**：没有文本预训练的良好初始化
2. **需要更仔细的超参数调优**：学习率和训练轮数更关键

## 输出日志示例

```
=== Direct Dual-Task Mode Activated ===
=== Direct Dual-Task Image Fine-tuning Mode ===
Skipping text classifier training, starting direct image fine-tuning...
Detected number of classes: 6
Direct dual-task learning enabled, lambda weight: 0.8
Found 1080 training samples in data source
Building emotion_dist_map for 1080 training samples
Valid distributions found: 1080
Total samples processed: 1080
=== Starting Direct Image Fine-tuning (No Text Classifier Pre-training) ===

epoch [1/10][1/22]	total_loss 2.1234	cls_loss 1.8901	dist_loss 0.7833	lr 1.000000e-03
Epoch 1 - Avg Total Loss: 2.0456, Avg Cls Loss: 1.8234, Avg Dist Loss: 0.8123
TOP-1 Accuracy: 45.67
```

## 文件结构

```
LaFTer-master/
├── LaFTer.py                           # 新增 train_lafter_direct 函数
├── scripts/
│   ├── LaFTer_DirectDualTask.bat       # Emotion6直接双任务脚本
│   └── LaFTer_DirectDualTask_EmoSet.bat # EmoSet直接双任务脚本
└── DirectDualTask_README.md            # 本文档
```

## 与传统模式的对比实验

建议同时运行两种模式进行对比：

### 传统模式
```bash
scripts\LaFTer_DualTask.bat
```

### 直接双任务模式
```bash
scripts\LaFTer_DirectDualTask.bat
```

比较两种模式在相同数据集上的：
- 最终准确率
- 收敛速度
- 训练时间
- 损失曲线

这将帮助您评估文本预训练对双任务学习的影响。 