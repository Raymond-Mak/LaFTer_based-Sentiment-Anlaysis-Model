import argparse
import torch
import datetime
import numpy as np
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from utils.utils import *
# custom

import datasets.Emotion6
import datasets.Emoset
import trainers.LaFTer_trainers as lafter_uft
from utils.utils import *
import os


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head
    


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    
    # Add distribution strategy configuration
    cfg.DATASET.DISTRIBUTION_STRATEGY = "strategy1"  # Options: "json" or "strategy1"
    cfg.DATASET.SIGMA_CONF = 1.0  # Controls Gaussian distribution spread
    cfg.DATASET.EPSILON = 0.1     # Parameter to ensure non-zero probabilities


    cfg.txt_cls = args.txt_cls
    cfg.gpt_prompts = args.gpt_prompts


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    return cfg


class lossmeter:
    """Compute and store the average and current value.

    Examples::
        >>> # 1. Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # 2. Update meter after every mini-batch update
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self, ema=False):
        """
        Args:
            ema (bool, optional): apply exponential moving average.
        """
        self.ema = ema
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()

        self.val = val
        self.sum += val * n
        self.count += n

        if self.ema:
            self.avg = self.avg * 0.9 + self.val * 0.1
        else:
            self.avg = self.sum / self.count


def test(args, teloader, model):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top1_pl = AverageMeter('Acc@1', ':6.2f')
    one_hot = []
    one_hot_pl = []

    for i, (inputs) in enumerate(tqdm(teloader)):
        img = inputs["img"]
        labels = inputs["label"]

        if args.zero_shot:
            with torch.no_grad():
                output_pseudo_label = model(inputs.cuda(), zero_shot=True)
                _, predicted_pl = output_pseudo_label.max(1)
                one_hot_pl.append(predicted_pl.eq(labels.cuda()).cpu())
                acc1_pl = one_hot_pl[-1].sum().item() / len(labels)
                top1_pl.update(acc1_pl, len(labels))

        else:
            with torch.no_grad():
                inputs, labels = img.cuda(), labels.cuda()
                outputs = model(inputs, clip_eval=True)
                _, predicted = outputs.max(1)
                one_hot.append(predicted.eq(labels).cpu())
                acc1 = one_hot[-1].sum().item() / len(labels)
                top1.update(acc1, len(labels))

    if not args.zero_shot:
        return top1.avg * 100, top1_pl.avg * 100
    else:
        return top1_pl.avg * 100


def train_txt_cls(args, model):
    optimizer, _, _ = setup_text_training_utils(args, model)
    criteria = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    for i in tqdm(range(args.txt_epochs)):
        loss = model.train_txt_clas(criteria)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.txt_cls_init()

def train_lafter(args, model, tr_loader, val_loader):
    # 第一步：训练文本分类器（保持不变）
    train_txt_cls(args, model)

    # 检查是否启用双任务学习
    use_dual_task = getattr(args, 'dual_task', False)
    lambda_weight = getattr(args, 'lambda_weight', 0.8)
    
    # 动态确定类别数量
    train_items = None
    if hasattr(tr_loader.dataset, 'data_source') and hasattr(tr_loader.dataset.data_source, 'train_x'):
        train_items = tr_loader.dataset.data_source.train_x
    elif hasattr(tr_loader.dataset, 'data_source') and isinstance(tr_loader.dataset.data_source, (list, tuple)):
        train_items = tr_loader.dataset.data_source
    elif hasattr(tr_loader.dataset, 'train_x'):
        train_items = tr_loader.dataset.train_x
    
    num_classes = 6  # 默认值
    if train_items:
        max_label = max(item.label for item in train_items)
        num_classes = max_label + 1
        print(f"Detected number of classes: {num_classes}")
    
    if use_dual_task:
        print(f"Dual-task learning enabled, lambda weight: {lambda_weight}")
        # 创建从图像索引到分布标签的映射
        # 从训练数据集中提取分布信息
        emotion_dist_map = {}
        valid_dist_count = 0
        fallback_count = 0
        
        if train_items is not None:
            print(f"Found {len(train_items)} training samples in data source")
            print(f"Building emotion_dist_map for {len(train_items)} training samples")
            
            for i, item in enumerate(train_items):
                if hasattr(item, 'emotion_distribution') and item.emotion_distribution is not None:
                    # 归一化处理，确保分布和为1
                    dist = np.array(item.emotion_distribution, dtype=np.float32)
                    sum_dist = dist.sum()
                    if sum_dist > 0:
                        dist = dist / sum_dist
                    else:
                        # 若出现异常，回退到one-hot
                        dist = np.zeros(num_classes, dtype=np.float32)
                        dist[item.label] = 1.0
                    emotion_dist_map[i] = dist
                    valid_dist_count += 1
                    
                    # 打印前几个有效分布作为调试
                    if valid_dist_count <= 5:
                        print(f"Sample {i}: found valid distribution (normalized): {emotion_dist_map[i]}")
                        print(f"  Class: {item.classname}, Label: {item.label}")
                        print(f"  Image path: {item.impath}")
                else:
                    # 如果没有分布标签，创建one-hot分布作为fallback
                    one_hot_dist = np.zeros(num_classes, dtype=np.float32)
                    one_hot_dist[item.label] = 1.0
                    emotion_dist_map[i] = one_hot_dist
                    fallback_count += 1
                    
                    # 打印前几个fallback情况作为调试
                    if fallback_count <= 5:
                        print(f"Sample {i}: using fallback one-hot distribution")
                        print(f"  hasattr(item, 'emotion_distribution'): {hasattr(item, 'emotion_distribution')}")
                        if hasattr(item, 'emotion_distribution'):
                            print(f"  item.emotion_distribution is None: {item.emotion_distribution is None}")
                            print(f"  item.emotion_distribution: {item.emotion_distribution}")
                        print(f"  Class: {item.classname}, Label: {item.label}")
                        print(f"  Created one-hot distribution: {one_hot_dist}")
            
            print("emotion_dist_map built:")
            print(f"  Total samples: {len(train_items)}")
            print(f"  Valid distributions: {valid_dist_count}")
            print(f"  Fallback count: {fallback_count}")
            print(f"  emotion_dist_map size: {len(emotion_dist_map)}")
        else:
            print("Could not access training data, creating empty emotion_dist_map")
            emotion_dist_map = {}
        
        # 调试：检查emotion_distributions的统计信息
        if emotion_dist_map:
            dist_samples = list(emotion_dist_map.values())[:10]  # 取前10个样本
            print("=== First 10 distributions in emotion_dist_map ===")
            for i, dist in enumerate(dist_samples):
                max_val = np.max(dist)
                entropy = -np.sum(dist * np.log(dist + 1e-8))
                is_one_hot = np.sum(dist == 1.0) == 1 and np.sum(dist == 0.0) == 5
                print(f"Map sample {i}: max={max_val:.3f}, entropy={entropy:.3f}, one_hot={is_one_hot}, dist={dist}")
        else:
            print("emotion_dist_map is empty, distribution learning skipped")
        
        # 使用双任务损失函数，添加温度参数放大KL梯度
        dual_task_loss = DualTaskLoss(lambda_weight=lambda_weight, num_classes=num_classes, temperature=16.0).cuda()
    else:
        print("Using traditional single-task classification mode")

    all_acc = list()
    optimizer, scheduler, criteria = setup_lafter_training_utils(args, model)
    
    for epoch in range(args.epochs):
        print(f'Epoch: {epoch}')
        model.eval()  # 视觉编码器不训练，仅训练适配器和提示
        model.adapter.train()  # 适配器参与训练

        # 渐进式λ权重调整：前几个epoch降低分布学习权重
        if use_dual_task:
            # 计算当前epoch的λ权重：前10个epoch线性增长到目标值
            warmup_epochs = min(0, args.epochs // 2)
            if epoch < warmup_epochs:
                current_lambda = lambda_weight * (epoch + 1) / warmup_epochs
                dual_task_loss.lambda_weight = current_lambda
                print(f"Warmup phase: epoch {epoch+1}, lambda={current_lambda:.4f}")
            else:
                dual_task_loss.lambda_weight = lambda_weight

        epoch_total_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_dist_loss = 0.0
        num_batches = 0

        for i, batch in enumerate(tr_loader):
            inputs = batch["img"].cuda()
            labels = batch["label"].cuda()
            
            optimizer.zero_grad()

            # 使用监督学习前向方法
            outputs = model.forward_supervised(inputs)
            
            if use_dual_task:
                # 双任务学习模式
                # 根据当前batch的索引获取对应的emotion_distribution
                batch_size = inputs.size(0)
                emotion_distributions = []
                
                # 计算当前batch在整个数据集中的起始索引
                start_idx = i * tr_loader.batch_size
                
                for j in range(batch_size):
                    sample_idx = start_idx + j
                    if sample_idx in emotion_dist_map:
                        emotion_distributions.append(emotion_dist_map[sample_idx])
                    else:
                        # fallback: 创建one-hot分布
                        one_hot_dist = np.zeros(num_classes, dtype=np.float32)
                        one_hot_dist[labels[j].cpu().item()] = 1.0
                        emotion_distributions.append(one_hot_dist)
                
                emotion_distributions = torch.tensor(np.stack(emotion_distributions), dtype=torch.float32).cuda()
                
                # 调试：在第一个batch打印更多信息
                if epoch == 0 and i == 0:
                    print("=== Distribution info for first batch ===")
                    print(f"emotion_distributions shape: {emotion_distributions.shape}")
                    print("First 3 sample distributions:")
                    for k in range(min(3, batch_size)):
                        dist = emotion_distributions[k].cpu().numpy()
                        is_one_hot = torch.sum(emotion_distributions[k] == 1.0) == 1 and torch.sum(emotion_distributions[k] == 0.0) == 5
                        print(f"  Sample {k}: {dist}, one-hot={is_one_hot}")
                    
                    # 添加数据类型和设备检查
                    print("=== Data type and device check ===")
                    print(f"outputs dtype: {outputs.dtype}, device: {outputs.device}")
                    print(f"labels dtype: {labels.dtype}, device: {labels.device}")
                    print(f"emotion_distributions dtype: {emotion_distributions.dtype}, device: {emotion_distributions.device}")
                
                # 计算双任务损失
                total_loss, cls_loss, dist_loss = dual_task_loss(outputs, labels, emotion_distributions)
                
                # 累计损失用于监控
                epoch_total_loss += total_loss.item()
                epoch_cls_loss += cls_loss.item()
                epoch_dist_loss += dist_loss.item()
                
                if i % args.print_freq == 0:
                    print(
                        "epoch [{0}/{1}][{2}/{3}]\t"
                        "total_loss {total_loss:.4f}\t"
                        "cls_loss {cls_loss:.4f}\t"
                        "dist_loss {dist_loss:.4f}\t"
                        "lr {lr:.6e}".format(
                            epoch + 1,
                            args.epochs,
                            i + 1,
                            len(tr_loader),
                            total_loss=total_loss.item(),
                            cls_loss=cls_loss.item(),
                            dist_loss=dist_loss.item(),
                            lr=optimizer.param_groups[0]["lr"],
                        ))
            else:
                # 传统的单任务学习模式
                total_loss = criteria(outputs, labels)
                epoch_total_loss += total_loss.item()
                
                if i % args.print_freq == 0:
                    print(
                        "epoch [{0}/{1}][{2}/{3}]\t"
                        "loss {losses:.4f}\t"
                        "lr {lr:.6e}".format(
                            epoch + 1,
                            args.epochs,
                            i + 1,
                            len(tr_loader),
                            losses=total_loss.item(),
                            lr=optimizer.param_groups[0]["lr"],
                        ))

            num_batches += 1
            total_loss.backward()
            optimizer.step()
        
        # 打印epoch统计信息
        if use_dual_task:
            avg_total_loss = epoch_total_loss / num_batches
            avg_cls_loss = epoch_cls_loss / num_batches
            avg_dist_loss = epoch_dist_loss / num_batches
            print(f"Epoch {epoch + 1} - Avg Total Loss: {avg_total_loss:.4f}, "
                  f"Avg Cls Loss: {avg_cls_loss:.4f}, Avg Dist Loss: {avg_dist_loss:.4f}")
        else:
            avg_total_loss = epoch_total_loss / num_batches
            print(f"Epoch {epoch + 1} - Avg Single-Task Loss: {avg_total_loss:.4f}")
        
        scheduler.step()
        print(f'Evaluation: {epoch}')
        acc = test_prompting(val_loader, model)
        print(f'TOP-1 Accuracy: {acc}')
        all_acc.append(acc)
    print(f'-------------------------------- Best Accuracy: {max(all_acc)} --------------------------------')

def train_lafter_direct(args, model, tr_loader, val_loader):
    """
    直接双任务图像微调模式：跳过文本分类器训练，直接进入图像微调阶段
    适用于想要从头开始训练adapter的场景
    """
    print("=== Direct Dual-Task Image Fine-tuning Mode ===")
    print("Skipping text classifier training, starting direct image fine-tuning...")
    
    # 检查是否启用双任务学习
    use_dual_task = getattr(args, 'dual_task', False)
    lambda_weight = getattr(args, 'lambda_weight', 0.8)
    
    if not use_dual_task:
        print("Warning: --dual_task not enabled. Enabling dual-task mode for direct training.")
        use_dual_task = True
        args.dual_task = True
    
    # 动态确定类别数量
    train_items = None
    if hasattr(tr_loader.dataset, 'data_source') and hasattr(tr_loader.dataset.data_source, 'train_x'):
        train_items = tr_loader.dataset.data_source.train_x
    elif hasattr(tr_loader.dataset, 'data_source') and isinstance(tr_loader.dataset.data_source, (list, tuple)):
        train_items = tr_loader.dataset.data_source
    elif hasattr(tr_loader.dataset, 'train_x'):
        train_items = tr_loader.dataset.train_x
    
    num_classes = 6  # 默认值
    if train_items:
        max_label = max(item.label for item in train_items)
        num_classes = max_label + 1
        print(f"Detected number of classes: {num_classes}")
    
    print(f"Direct dual-task learning enabled, lambda weight: {lambda_weight}")
    
    # 创建从图像索引到分布标签的映射
    emotion_dist_map = {}
    valid_dist_count = 0
    fallback_count = 0
    
    if train_items is not None:
        print(f"Found {len(train_items)} training samples in data source")
        print(f"Building emotion_dist_map for {len(train_items)} training samples")
        
        for i, item in enumerate(train_items):
            if hasattr(item, 'emotion_distribution') and item.emotion_distribution is not None:
                # 归一化处理，确保分布和为1
                dist = np.array(item.emotion_distribution, dtype=np.float32)
                sum_dist = dist.sum()
                if sum_dist > 0:
                    dist = dist / sum_dist
                else:
                    # 若出现异常，回退到one-hot
                    dist = np.zeros(num_classes, dtype=np.float32)
                    dist[item.label] = 1.0
                emotion_dist_map[i] = dist
                valid_dist_count += 1
                
                # 打印前几个有效分布作为调试
                if valid_dist_count <= 5:
                    print(f"Sample {i}: found valid distribution (normalized): {emotion_dist_map[i]}")
                    print(f"  Class: {item.classname}, Label: {item.label}")
                    print(f"  Image path: {item.impath}")
            else:
                # 如果没有情感分布，使用one-hot编码
                one_hot_dist = np.zeros(num_classes, dtype=np.float32)
                one_hot_dist[item.label] = 1.0
                emotion_dist_map[i] = one_hot_dist
                fallback_count += 1
                
                # 打印前几个fallback样本作为调试
                if fallback_count <= 5:
                    print(f"Sample {i}: using fallback one-hot distribution")
                    print(f"  hasattr(item, 'emotion_distribution'): {hasattr(item, 'emotion_distribution')}")
                    if hasattr(item, 'emotion_distribution'):
                        print(f"  item.emotion_distribution is None: {item.emotion_distribution is None}")
                        print(f"  item.emotion_distribution: {item.emotion_distribution}")
                    print(f"  Class: {item.classname}, Label: {item.label}")
                    print(f"  Image path: {item.impath}")
        
        print(f"Valid distributions found: {valid_dist_count}")
        print(f"Fallback one-hot distributions created: {fallback_count}")
        print(f"Total samples processed: {len(train_items)}")
    else:
        print("Warning: No training items found, cannot create emotion distribution map")
    
    # 调试：检查emotion_distributions的统计信息
    if emotion_dist_map:
        dist_samples = list(emotion_dist_map.values())[:10]  # 取前10个样本
        print("=== First 10 distributions in emotion_dist_map ===")
        for i, dist in enumerate(dist_samples):
            max_val = np.max(dist)
            entropy = -np.sum(dist * np.log(dist + 1e-8))
            is_one_hot = np.sum(dist == 1.0) == 1 and np.sum(dist == 0.0) == 5
            print(f"Map sample {i}: max={max_val:.3f}, entropy={entropy:.3f}, one_hot={is_one_hot}, dist={dist}")
    else:
        print("emotion_dist_map is empty, distribution learning skipped")
    
    # 使用双任务损失函数，添加温度参数放大KL梯度
    dual_task_loss = DualTaskLoss(lambda_weight=lambda_weight, num_classes=num_classes, temperature=12.0).cuda()

    all_acc = list()
    optimizer, scheduler, criteria = setup_lafter_training_utils(args, model)
    
    print("=== Starting Direct Image Fine-tuning (No Text Classifier Pre-training) ===")
    
    for epoch in range(args.epochs):
        print(f'Epoch: {epoch}')
        model.eval()  # 视觉编码器不训练，仅训练适配器和提示
        model.adapter.train()  # 适配器参与训练

        # 渐进式λ权重调整：前几个epoch降低分布学习权重
        # 计算当前epoch的λ权重：前10个epoch线性增长到目标值
        warmup_epochs = min(0, args.epochs // 2)
        if epoch < warmup_epochs:
            current_lambda = lambda_weight * (epoch + 1) / warmup_epochs
            dual_task_loss.lambda_weight = current_lambda
            print(f"Warmup phase: epoch {epoch+1}, lambda={current_lambda:.4f}")
        else:
            dual_task_loss.lambda_weight = lambda_weight

        epoch_total_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_dist_loss = 0.0
        num_batches = 0

        for i, batch in enumerate(tr_loader):
            inputs = batch["img"].cuda()
            labels = batch["label"].cuda()
            
            optimizer.zero_grad()

            # 使用监督学习前向方法
            outputs = model.forward_supervised(inputs)
            
            # 双任务学习模式
            # 根据当前batch的索引获取对应的emotion_distribution
            batch_size = inputs.size(0)
            emotion_distributions = []
            
            # 计算当前batch在整个数据集中的起始索引
            start_idx = i * tr_loader.batch_size
            
            for j in range(batch_size):
                sample_idx = start_idx + j
                if sample_idx in emotion_dist_map:
                    emotion_distributions.append(emotion_dist_map[sample_idx])
                else:
                    # fallback: 创建one-hot分布
                    one_hot_dist = np.zeros(num_classes, dtype=np.float32)
                    one_hot_dist[labels[j].cpu().item()] = 1.0
                    emotion_distributions.append(one_hot_dist)
            
            emotion_distributions = torch.tensor(np.stack(emotion_distributions), dtype=torch.float32).cuda()
            
            # 调试：在第一个batch打印更多信息
            if epoch == 0 and i == 0:
                print("=== Distribution info for first batch ===")
                print(f"emotion_distributions shape: {emotion_distributions.shape}")
                print("First 3 sample distributions:")
                for k in range(min(3, batch_size)):
                    dist = emotion_distributions[k].cpu().numpy()
                    is_one_hot = torch.sum(emotion_distributions[k] == 1.0) == 1 and torch.sum(emotion_distributions[k] == 0.0) == 5
                    print(f"  Sample {k}: {dist}, one-hot={is_one_hot}")
            
            # 计算双任务损失
            total_loss, cls_loss, dist_loss = dual_task_loss(outputs, labels, emotion_distributions)
            
            # 累计损失用于监控
            epoch_total_loss += total_loss.item()
            epoch_cls_loss += cls_loss.item()
            epoch_dist_loss += dist_loss.item()
            
            if i % args.print_freq == 0:
                print(
                    "epoch [{0}/{1}][{2}/{3}]\t"
                    "total_loss {total_loss:.4f}\t"
                    "cls_loss {cls_loss:.4f}\t"
                    "dist_loss {dist_loss:.4f}\t"
                    "lr {lr:.6e}".format(
                        epoch + 1,
                        args.epochs,
                        i + 1,
                        len(tr_loader),
                        total_loss=total_loss.item(),
                        cls_loss=cls_loss.item(),
                        dist_loss=dist_loss.item(),
                        lr=optimizer.param_groups[0]["lr"],
                    ))

            num_batches += 1
            total_loss.backward()
            optimizer.step()
        
        # 打印epoch统计信息
        avg_total_loss = epoch_total_loss / num_batches
        avg_cls_loss = epoch_cls_loss / num_batches
        avg_dist_loss = epoch_dist_loss / num_batches
        print(f"Epoch {epoch + 1} - Avg Total Loss: {avg_total_loss:.4f}, "
              f"Avg Cls Loss: {avg_cls_loss:.4f}, Avg Dist Loss: {avg_dist_loss:.4f}")
        
        scheduler.step()
        print(f'Evaluation: {epoch}')
        acc = test_prompting(val_loader, model)
        print(f'TOP-1 Accuracy: {acc}')
        all_acc.append(acc)

    print(f'-------------------------------- Best Accuracy: {max(all_acc)} --------------------------------')
    return max(all_acc)


def test_img_classifier(args, model, teloader):
    """
    用已经训练好的图像分类器，在测试集上跑一次测试返回 top-1 精度。
    处理图像输入而不是文本prompt。
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(teloader):
            # 从batch中提取图像和标签
            images = batch["img"].to(model.device)  # 确保图像在正确的设备上
            labels = batch["label"].to(model.device)  # 确保标签在相同的设备上
            
            # 需要确保模型有正确的评估方法
            try:
                # 尝试使用模型特定的评估方法(如果存在)
                outputs = model.forward_normal_for_pl(images)  # 或model.eval_img_clas(images)
            except AttributeError:
                # 针对数据类型不匹配问题的处理
                # 这里我们假设模型有一个特定的forward方法需要调用
                # 您可能需要根据实际模型的实现来调整
                if hasattr(model, 'text_features') and model.text_features.dtype != images.dtype:
                    # 如果text_features是half类型，将images也转为half
                    if model.text_features.dtype == torch.float16:
                        images = images.half()
                    # 如果images是half类型，将text_features转为float
                    elif images.dtype == torch.float16:
                        model.text_features = model.text_features.float()
                
                # 使用常规forward
                outputs = model(images)
            
            # 计算预测结果
            preds = outputs.argmax(dim=1)
            
            # 确保两个张量在同一设备上进行比较
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total



def main(args):
    cfg = setup_cfg(args)
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.batch_size
    cfg.DATALOADER.TEST.BATCH_SIZE = args.batch_size
    cfg.SEED = args.seed

    dataset_name = cfg.DATASET.NAME
    setup_txt_epochs(args, dataset_name)

    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)
    print_args(args, cfg)
    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True
    trainer = build_trainer(cfg)
    model = trainer.model
    model.args = args

    train_loader = trainer.train_loader_x
    test_loader = trainer.test_loader

    # —— 文本分类器训练和测试 —— 
    if args.txt_cls in ['cls_only', 'templates_only'] or (args.txt_cls == 'lafter' and args.skip_finetune):
        mode_desc = "仅文本分类器" if args.skip_finetune else args.txt_cls
        print(f"Only training text classifier (mode={mode_desc}), skipping image fine-tuning")
        # 1) 训练文本分类器
        train_txt_cls(args, model)
        # 2) 在测试集上评估文本分类器
        acc = test_img_classifier(args, model, test_loader)
        print(f"[Text-CLS] Test Top-1 Accuracy: {acc:.2f}%")
        return

    # —— 图像微调流程 —— 
    if hasattr(args, 'direct_dualtask') and args.direct_dualtask:
        # 直接双任务模式：跳过文本分类器训练，直接进入图像双任务微调
        print("=== Direct Dual-Task Mode Activated ===")
        train_lafter_direct(args, model, train_loader, test_loader)
    elif args.txt_cls == 'lafter' and not args.skip_finetune:
        # 传统模式：先训练文本分类器，再进行图像微调
        train_lafter(args, model, train_loader, test_loader)
    elif args.zero_shot:
        zero_shot(model, test_loader)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=7777, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--print_freq", type=int, default=10, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    parser.add_argument('--exp-name', type=str, required=False)
    parser.add_argument('--scheduler', default='cosine')
    parser.add_argument('--scheduler-epochs', type=int, default=15)
    parser.add_argument('--scheduler-gamma', type=float, default=0.3)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--acc-batches', type=int, default=1)
    parser.add_argument('--arch', type=str, default='ViT-B/32', required=False)
    parser.add_argument('--gpt_prompts', action='store_true')
    parser.add_argument('--text_prompts', action='store_true')
    parser.add_argument('--zero_shot', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--txt_cls', type=str, default='tap', required=True, choices=['cls_only',
                                                                                      'templates_only', 'lafter', 'zero_shot'])
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--txt_epochs', type=int, default=1000)
    parser.add_argument('--logfolder', default='logs', type=str)
    parser.add_argument('--skip_finetune', action='store_true', 
                help='只训练文本分类器，跳过图像微调步骤') #new
    parser.add_argument('--dual_task', action='store_true',
                help='启用双任务学习模式（分类+分布学习）')
    parser.add_argument('--lambda_weight', type=float, default=0.8,
                help='双任务损失中分布学习的权重系数（λ）')
    parser.add_argument('--direct_dualtask', action='store_true',
                help='直接双任务模式：跳过文本分类器训练，直接进入图像双任务微调')
    args = parser.parse_args()
    args.mile_stones = None
    main(args)

