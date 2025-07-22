# 技术细节记录 - 2025年7月22日

## 学习率调度器改进

### 问题背景
用户希望简化`setup_lafter_training_utils`函数中的学习率调度器，移除MultiStepLR和CosineAnnealingLR，统一使用带warmup功能的调度器。

### 具体修改

#### 修改前 (`utils/utils.py` 第164-172行):
```python
elif args.scheduler == 'multistep':
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.mile_stones, 0.60)
    print(f">>> Using MultiStepLR scheduler")
elif args.scheduler == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    print(f">>> Using CosineAnnealingLR scheduler")
else:
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.mile_stones, 0.60)
    print(f">>> Using default MultiStepLR scheduler")
```

#### 修改后:
```python
elif args.scheduler == 'multistep' or args.scheduler == 'cosine':
    # 使用带warmup的恒定学习率调度器替代MultiStepLR和CosineAnnealingLR
    warmup_epochs = 3
    constant_scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=args.epochs)
    scheduler = ConstantWarmupScheduler(optimizer, constant_scheduler, warmup_epochs, args.lr * 0.1)
    print(f">>> Using ConstantWarmupScheduler: warmup_epochs={warmup_epochs}, warmup_lr={args.lr * 0.1:.6f}")
else:
    # 默认使用最基础的学习率调度器（不带warmup）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs, gamma=1.0)
    print(f">>> Using default StepLR scheduler without warmup")
```

### 兼容性修复

#### `setup_text_training_utils`函数修复 (`utils/utils.py` 第65-67行):
```python
# 修改前
else:
    raise NotImplementedError

# 修改后
else:
    # 默认使用最基础的学习率调度器（不带warmup）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs, gamma=1.0)
```

## MultiLayer Prompt技术实现细节

### CLIP模型架构扩展 (`clip/model.py`)

#### 新增MultiLayerTransformer类:
```python
class MultiLayerTransformer(nn.Module):
    """支持多层prompt注入的Transformer"""
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        
        # 定义分段策略：{1-4}, {5-8}, {9-11}, {12}（索引从0开始）
        self.stage_boundaries = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10], [11]]
        self.num_stages = len(self.stage_boundaries)
```

#### 关键方法实现:
```python
def _inject_prompt_at_stage(self, x, stage_prompt):
    """在指定stage前注入prompt"""
    # 确保数据类型和设备一致性
    if stage_prompt.dtype != x.dtype:
        stage_prompt = stage_prompt.to(dtype=x.dtype)
    if stage_prompt.device != x.device:
        stage_prompt = stage_prompt.to(device=x.device)
    
    # 转换维度并拼接
    prompt_tokens = stage_prompt.permute(1, 0, 2)
    cls_token = x[:1]
    patch_tokens = x[1:]
    x_with_prompt = torch.cat([cls_token, prompt_tokens, patch_tokens], dim=0)
    return x_with_prompt
```

### VisualTransformer扩展

#### 新增多层支持方法:
```python
def forward_multi_stage_prompts(self, x: torch.Tensor, multi_prompts=None):
    """支持多层prompt的前向传播"""
    x = x.permute(1, 0, 2)  # NLD -> LND for transformer
    
    # 确保数据类型一致性
    if multi_prompts is not None:
        for i, prompt in enumerate(multi_prompts):
            if prompt is not None and prompt.dtype != x.dtype:
                multi_prompts[i] = prompt.to(dtype=x.dtype)
    
    final_x, stage_outputs = self.multi_transformer(x, multi_prompts)
    
    final_x = final_x.permute(1, 0, 2)  # LND -> NLD
    final_features = self.ln_post(final_x[:, 0, :])
    if self.proj is not None:
        final_features = final_features @ self.proj
    
    return final_features, stage_outputs
```

### 数据类型一致性处理

#### ResidualAttentionBlock改进:
```python
def forward(self, x: torch.Tensor):
    # 确保输入数据类型一致性
    target_dtype = x.dtype
    
    # Attention分支
    attn_input = self.ln_1(x)
    if attn_input.dtype != target_dtype:
        attn_input = attn_input.to(dtype=target_dtype)
    
    attn_output = self.attention(attn_input)
    if attn_output.dtype != target_dtype:
        attn_output = attn_output.to(dtype=target_dtype)
    
    x = x + attn_output
    
    # MLP分支类似处理...
```

### 权重加载优化

#### build_model函数改进:
```python
# 复制transformer权重到multi_transformer
new_state_dict = {}
for key, value in state_dict.items():
    new_state_dict[key] = value
    if key.startswith("visual.transformer."):
        multi_key = key.replace("visual.transformer.", "visual.multi_transformer.")
        new_state_dict[multi_key] = value.clone().detach()

# 数据类型一致性检查
model_dtype = next(model.parameters()).dtype
for key, value in new_state_dict.items():
    if value.dtype != model_dtype:
        new_state_dict[key] = value.to(dtype=model_dtype)
```

## ViT架构层数分析

### 层数确定方法
```python
# 在build_model函数中
vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
```

### 各架构详细规格

#### ViT-B/32:
- Vision Layers: 12
- Vision Width: 768
- Vision Heads: 12 (768 // 64)
- Patch Size: 32×32
- Grid Size: 7×7 (224÷32)
- Positional Embeddings: [50, 768] (7×7+1)

#### ViT-B/16:
- Vision Layers: 12
- Vision Width: 768
- Vision Heads: 12
- Patch Size: 16×16
- Grid Size: 14×14 (224÷16)
- Positional Embeddings: [197, 768] (14×14+1)

#### ViT-L/14:
- Vision Layers: 24
- Vision Width: 1024
- Vision Heads: 16 (1024 // 64)
- Patch Size: 14×14
- Grid Size: 16×16 (224÷14)
- Positional Embeddings: [257, 1024] (16×16+1)

### 分段策略设计

#### 当前12层分段（适用于ViT-B系列）:
```python
self.stage_boundaries = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10], [11]]
```

#### 建议的24层分段（适用于ViT-L/14）:
```python
# 选项1: 4段均匀分割
self.stage_boundaries = [[0,1,2,3,4,5], [6,7,8,9,10,11], [12,13,14,15,16,17], [18,19,20,21,22,23]]

# 选项2: 前浅后深分割
self.stage_boundaries = [[0,1,2,3], [4,5,6,7,8,9], [10,11,12,13,14,15,16,17], [18,19,20,21,22,23]]
```

## 性能优化要点

### 内存管理
- 及时进行数据类型转换，避免混合精度计算
- 合理设计prompt token数量（每段16个token）
- 使用梯度检查点减少显存占用

### 计算效率
- 分段并行处理，减少顺序依赖
- 优化prompt注入位置，减少数据重组开销
- 使用高效的attention实现

### 训练策略
- 渐进式解冻：从深层到浅层逐步训练
- 学习率分层设置：adapter参数使用5倍学习率
- warmup策略：3个epoch线性增长到目标学习率