# 情感分布生成策略说明

## 概述

本项目支持两种情感分布生成策略，可以通过配置参数在不同策略间切换：

1. **JSON策略**：从 `Emotion6.json` 文件读取众包标注的情感分布
2. **策略1**：基于论文的高斯函数和情感距离生成分布

## 策略对比

### JSON策略 (`json`)
- **数据来源**：`data/Emotion6/Emotion6.json` 中的 `emotions` 字段
- **优点**：使用真实的众包标注数据，包含人类的主观性
- **缺点**：
  - 标签不一致问题严重（`true_emotion` ≠ `dominant_emotion`）
  - 分布质量参差不齐，部分分布过于平坦
  - 约30-40%的样本被过滤掉

### 策略1 (`strategy1`) - 推荐
- **数据来源**：基于 `true_emotion` 使用高斯函数生成
- **理论依据**：论文 "Joint Image Emotion Classification and Distribution Learning via Deep Convolutional Neural Network" 的策略1
- **优点**：
  - 完全基于 `true_emotion`，避免标签不一致
  - 分布质量稳定，主导情感明确
  - 符合情感心理学理论（基于Mikels情感轮盘）
  - 可调节参数控制分布形状

## 情感距离矩阵

基于Mikels情感轮盘，我们的6种情感按以下顺序排列：
```
anger(0) -> disgust(1) -> fear(2) -> sadness(4) -> surprise(5) -> joy(3) -> anger(0)
```

距离矩阵：
```
        anger  disgust  fear  joy  sadness  surprise
anger     0      1      2     3      3        2
disgust   1      0      1     2      2        3  
fear      2      1      0     1      1        2
joy       3      2      1     0      2        1
sadness   3      2      1     2      0        1
surprise  2      3      2     1      1        0
```

## 参数配置

### 策略1参数
- **sigma_conf**：控制高斯分布的扩散程度
  - 值越小 → 分布越尖锐，主导情感权重越大
  - 值越大 → 分布越平坦，相邻情感也有较高权重
  - 推荐值：0.5-2.0，默认1.0

- **epsilon**：确保每个情感都有非零概率
  - 避免某些情感完全为0
  - 推荐值：0.05-0.1，默认0.1

### 分布效果示例
以 `anger` 为例，不同 `sigma_conf` 的分布效果：

```
sigma_conf = 0.5:  [0.809, 0.124, 0.017, 0.017, 0.017, 0.017]  # 尖锐分布
sigma_conf = 1.0:  [0.485, 0.302, 0.082, 0.025, 0.025, 0.082]  # 平衡分布  
sigma_conf = 1.5:  [0.325, 0.264, 0.145, 0.061, 0.061, 0.145]  # 平坦分布
```

## 使用方法

### 1. 修改配置文件

编辑 `configs/datasets/Emotion6.yaml`：

```yaml
DATASET:
  NAME: "Emotion6"
  # 策略选择
  DISTRIBUTION_STRATEGY: "strategy1"  # 或 "json"
  # 策略1参数
  SIGMA_CONF: 1.0
  EPSILON: 0.1
```

### 2. 使用批处理脚本

**策略1（推荐）：**
```bash
scripts\LaFTer_DualTask.bat
```
- 默认使用策略1
- 可修改脚本中的 `DISTRIBUTION_STRATEGY`、`SIGMA_CONF`、`EPSILON` 变量

**JSON策略：**
```bash
scripts\LaFTer_DualTask_JSON.bat
```
- 使用JSON文件中的分布数据

### 3. 命令行参数

也可以通过命令行直接指定：

```bash
python LaFTer.py \
    --root data \
    --trainer LaFTer \
    --dataset-config-file configs/datasets/Emotion6.yaml \
    --config-file configs/trainers/text_cls/vit_b32.yaml \
    --output-dir output/LaFTer/vit_b32/Emotion6_dual_task_strategy1 \
    --lr 0.001 \
    --epochs 10 \
    --txt_cls lafter \
    --dual_task \
    --lambda_weight 0.8 \
    DATASET.DISTRIBUTION_STRATEGY strategy1 \
    DATASET.SIGMA_CONF 1.0 \
    DATASET.EPSILON 0.1
```

## 输出目录

不同策略的输出会保存到不同目录：
- 策略1：`output/LaFTer/vit_b32/Emotion6_dual_task_strategy1/`
- JSON策略：`output/LaFTer/vit_b32/Emotion6_dual_task_json/`

## 推荐设置

基于测试结果，推荐使用以下配置：

```yaml
DATASET:
  DISTRIBUTION_STRATEGY: "strategy1"
  SIGMA_CONF: 1.0        # 平衡的分布形状
  EPSILON: 0.1           # 适度的最小概率
```

这个配置能够：
1. 避免标签不一致问题
2. 生成质量稳定的情感分布
3. 保持合理的主导情感权重
4. 符合情感心理学理论

## 参数调优建议

- **提高分类性能**：减小 `sigma_conf` (0.5-0.8)，使分布更尖锐
- **提高分布学习效果**：增大 `sigma_conf` (1.2-1.8)，使分布更平坦
- **平衡两个任务**：使用默认值 `sigma_conf=1.0` 