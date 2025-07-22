# 运行脚本修改说明

根据新的训练器分离结构，已对三个运行脚本进行了相应修改：

## 1. LaFTer_DualTask.bat
**用途**: 使用基础LaFTer训练器进行双任务学习
**修改内容**:
- ✅ 保持 `--trainer LaFTer` (使用基础版本)
- ✅ 添加了说明注释，明确使用基础LaFTer训练器
- ✅ 修改输出消息，标明使用的是基础版本
- ✅ 功能保持不变，继续支持双任务学习

**运行命令**: 直接双击 `LaFTer_DualTask.bat` 或在命令行运行

## 2. LaFTer_DirectDualTask.bat
**用途**: 不进行第一阶段的text classifier训练，使用基础LaFTer训练器进行双任务学习
**修改内容**:
- ✅ 保持 `--trainer LaFTer` (使用基础版本)
- ✅ 添加了说明注释，明确使用基础LaFTer训练器
- ✅ 跳过对text classifier的训练，直接开始第二阶段的双任务学习

**运行命令**: 直接双击 `LaFTer_DualTask.bat` 或在命令行运行

## 3. LaFTer_MultiPrompt.bat
**用途**: 使用多层LaFTer训练器测试多层prompt技术
**修改内容**:
- ✅ 保持 `--trainer MultiLayerLaFTer` (使用多层版本)
- ✅ 修改配置文件路径为标准路径 `configs/trainers/text_cls/vit_b32.yaml`
- ✅ 修改输出目录为 `output/multi_layer_prompt_test/dual_task_multilayer`
- ✅ 添加了说明注释，明确使用MultiLayerLaFTer训练器
- ✅ 保持双任务学习和多层prompt参数

## 4. LaFTer.bat
**用途**: 使用基础LaFTer训练器进行单分类任务的监督学习
**修改内容**:
- ✅ 保持 `--trainer LaFTer` (使用基础版本)
- ✅ 添加了说明注释，明确使用基础LaFTer训练器
- ✅ 只支持单任务学习（分类任务，监督学习，无情感分布学习）

## 重要说明

1. **训练器选择**: 
   - 使用 `LaFTer` 进行稳定的双任务学习
   - 使用 `MultiLayerLaFTer` 进行多层prompt实验

2. **参数组合**:
   - `--trainer LaFTer`: 单任务学习
   - `--trainer LaFTer` + `--dual_task`: 基础双任务学习
   - `--trainer MultiLayerLaFTer` + `--multi_layer_prompt`: 多层prompt技术
   - `--trainer MultiLayerLaFTer` + `--dual_task` + `--multi_layer_prompt`: 多层prompt双任务学习

3. **配置文件**: 所有脚本现在使用标准配置文件路径，确保兼容性

4. **输出目录**: 每个脚本使用不同的输出目录，避免结果混淆
