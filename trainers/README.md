# LaFTer 训练器说明

项目已将两个训练器分离到不同的文件中，避免了冲突并提高了代码的可维护性。

## 文件结构

```
trainers/
├── lafter_common.py      # 共享组件和基类
├── LaFTer_basic.py       # 基础版本的LaFTer训练器 
├── LaFTer_multilayer.py  # 多层Prompt版本的LaFTer训练器
└── LaFTer_trainers.py    # 原始文件（已弃用，建议删除）
```

## 训练器说明

### 1. LaFTer (基础版本)
- **文件**: `trainers/LaFTer_basic.py`
- **注册名**: `LaFTer`
- **特点**: 
  - 使用单层prompt技术
  - 包含dual-head架构（分类头 + 分布学习头）
  - 支持文本分类训练
  - 修复了`forward_backward`方法的缺失问题

### 2. MultiLayerLaFTer (多层版本)
- **文件**: `trainers/LaFTer_multilayer.py`
- **注册名**: `MultiLayerLaFTer`
- **特点**:
  - 使用多层prompt技术（4个stage）
  - 包含NGA聚合器和多段一致性损失
  - 支持级联分类
  - 更复杂的特征融合机制

## 使用方法

### 使用基础版本LaFTer训练器
```bash
python LaFTer.py --trainer LaFTer --config-file your_config.yaml
```

### 使用多层版本LaFTer训练器
```bash
python LaFTer.py --trainer MultiLayerLaFTer --config-file your_config.yaml
```

## 配置文件设置

在您的配置文件中设置相应的训练器：

```yaml
# 对于基础版本
TRAINER:
  NAME: "LaFTer"

# 对于多层版本  
TRAINER:
  NAME: "MultiLayerLaFTer"
```

## 问题解决

原始的`NotImplementedError`问题已通过以下方式解决：

1. **基础版本LaFTer**: 添加了缺失的`forward_backward`方法
2. **多层版本**: 已经包含完整的`forward_backward`实现
3. **代码分离**: 避免了两个训练器之间的冲突

## 建议

1. **删除旧文件**: 可以安全删除`trainers/LaFTer_trainers.py`，因为功能已分离到新文件中
2. **优先使用基础版本**: 如果不需要多层prompt功能，建议使用基础版本，它更稳定且资源消耗更少
3. **测试验证**: 建议先用小数据集测试两个训练器，确保功能正常

## 共享组件

`lafter_common.py`包含：
- `load_clip_to_cpu()`: CLIP模型加载函数
- `BaseLaFTerTrainer`: 训练器基类，包含通用方法
- 减少了代码重复，便于维护