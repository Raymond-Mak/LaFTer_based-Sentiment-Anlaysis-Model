 ● Update(utils\utils.py)
  ⎿  Updated utils\utils.py with 21 additions and 1 removal
       150        optimizer = optim.AdamW(optimizer_grouped_parameters, betas=(0.9, 0.999), weight_decay=args.weight_decay)
       151        print(f">>> Adapter learning rate: {args.lr * 5.0:.6f}")
       152        print(f">>> Other parameters learning rate: {args.lr:.6f}")
       153 -      scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.mile_stones, 0.60)
       153 +
       154 +      # 调度器选择
       155 +      if args.scheduler == 'coslr':
       156 +          scheduler = CosineLRScheduler(optimizer,
       157 +                                        t_initial=args.epochs,
       158 +                                        lr_min=1e-6,
       159 +                                        warmup_lr_init=args.lr * 0.1,  # warmup起始学习率为正常学习率的0.1倍
       160 +                                        warmup_t=5,                    # warmup 5个epoch
       161 +                                        cycle_limit=1,
       162 +                                        t_in_epochs=True)
       163 +          print(f">>> Using CosineLRScheduler with warmup: warmup_lr_init={args.lr * 0.1:.6f}, warmup_t=5")
       164 +      elif args.scheduler == 'multistep':
       165 +          scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.mile_stones, 0.60)
       166 +          print(f">>> Using MultiStepLR scheduler")
       167 +      elif args.scheduler == 'cosine':
       168 +          scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
       169 +          print(f">>> Using CosineAnnealingLR scheduler")
       170 +      else:
       171 +          scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.mile_stones, 0.60)
       172 +          print(f">>> Using default MultiStepLR scheduler")
       173 +
       174        criteria = LabelSmoothingCrossEntropy()
       175        return optimizer, scheduler, criteria
       176
 
添加参数：
--scheduler coslr ^


  1. MultiStepLR调度器

  - 作用: 在预设的epoch节点突然降低学习率
  - 特点: 阶梯式下降，在特定milestone处乘以gamma系数
  - 适用: 传统训练，需要精确控制学习率下降时机
  - 示例: epoch [10,20]处学习率乘以0.6

  2. CosineAnnealingLR调度器

  - 作用: 学习率按余弦函数平滑下降
  - 特点: 平滑连续变化，避免突变
  - 适用: 现代深度学习，提供更稳定的收敛
  - 公式: lr = lr_min + (lr_max - lr_min) * (1 + cos(π * epoch / max_epoch)) / 2

  3. CosineLRScheduler (with warmup)

  - 作用: 结合warmup和余弦退火的完整训练策略
  - 特点:
    - 前期warmup逐步提升学习率
    - 后期余弦退火平滑下降
  - 适用: 现代Transformer训练的标准做法
  - 阶段: warmup (5 epochs) → cosine annealing (45 epochs)

  推荐使用场景:

  你的LaFTer项目建议用coslr，因为:
  1. 视觉-语言模型: CLIP等预训练模型适合warmup策略
  2. 微调任务: warmup防止大幅参数更新破坏预训练权重
  3. 现代最佳实践: warmup + cosine是当前主流方法
  4. 稳定收敛: 比突变式调度器更稳定

  性能对比 (一般情况):
  - coslr: 收敛最稳定，泛化性能最好
  - cosine: 比multistep平滑，但缺少warmup
  - multistep: 传统方法，可能造成训练震荡