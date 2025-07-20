import logging
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from timm.scheduler import CosineLRScheduler
logger_initialized = {}
import getpass
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
import random
from pathlib import Path


lafter_datasets = ['DescribableTextures',  'EuroSAT', 'OxfordFlowers', 'SUN397', 'UCF101', 'ImageNetR', 'ImageNetSketch',
                   'ImageNetA', 'CIFAR10_local', 'CIFAR100_local', 'ImageNet', 'Caltech101', 'FGVCAircraft', 'Emotion6','abstract','Emoset']


def setup_text_training_utils(args, model):
    model = model.cuda()
    model = model.float()
    params = list()
    mile_stones = args.mile_stones

    for key, value in model.named_parameters():
        if 'adapter' in key and 'adapter_pl' not in key:
            value.requires_grad = True
        else:
            value.requires_grad = False

    print('------------------ Learnable Parameters ------------------')
    for key, value in model.named_parameters():
        if value.requires_grad:
            print("\t{}, {}, {}".format(key, value.numel(), value.shape))
            params.append((key, value))
    print('----------------------------------------------------------')

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in params
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in params
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.lr, betas=(0.9, 0.999))
    if args.scheduler == 'coslr':
        scheduler = CosineLRScheduler(optimizer,
                                      t_initial=args.epochs,
                                      lr_min=1e-6,
                                      warmup_lr_init=1e-4,
                                      warmup_t=5,
                                      cycle_limit=1,
                                      t_in_epochs=True)
    elif args.scheduler == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, mile_stones, 0.1)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(args.epochs))
    else:
        raise NotImplementedError

    criteria = LabelSmoothingCrossEntropy()
    return optimizer, scheduler, criteria

def setup_lafter_training_utils(args, model):
    model = model.cuda()
    model = model.float()
    params = list()
    for key, value in model.named_parameters():
        if 'adapter' in key and 'adapter_pl' not in key:
            value.requires_grad = True
        elif 'projector' in key and not args.entropy:
            value.requires_grad = True
        elif 'ln' in key:
            value.requires_grad = True
        else:
            value.requires_grad = False

    for key, value in model.named_parameters():
        if 'visual' in key:
            if 'ln' in key or 'bn' in key:
                value.requires_grad = True
            else:
                value.requires_grad = False

    print('------------------ Learnable Parameters ------------------')
    for key, value in model.named_parameters():
        if value.requires_grad:
            print("\t{}, {}, {}".format(key, value.numel(), value.shape))
            params.append((key, value))
    print('----------------------------------------------------------')

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    
    # 分别设置adapter和其他参数的学习率，adapter用更高学习率来加强分布学习
    adapter_params = []
    other_params = []
    adapter_params_no_decay = []
    other_params_no_decay = []
    
    for n, p in params:
        if 'adapter' in n:
            if any(nd in n for nd in no_decay):
                adapter_params_no_decay.append(p)
            else:
                adapter_params.append(p)
        else:
            if any(nd in n for nd in no_decay):
                other_params_no_decay.append(p)
            else:
                other_params.append(p)
    
    optimizer_grouped_parameters = []
    
    # adapter参数用5倍学习率
    if adapter_params:
        optimizer_grouped_parameters.append({
            'params': adapter_params,
            'lr': args.lr * 5.0,
            'weight_decay': 0.01
        })
    if adapter_params_no_decay:
        optimizer_grouped_parameters.append({
            'params': adapter_params_no_decay,
            'lr': args.lr * 5.0,
            'weight_decay': 0.0
        })
    
    # 其他参数用正常学习率
    if other_params:
        optimizer_grouped_parameters.append({
            'params': other_params,
            'lr': args.lr,
            'weight_decay': 0.01
        })
    if other_params_no_decay:
        optimizer_grouped_parameters.append({
            'params': other_params_no_decay,
            'lr': args.lr,
            'weight_decay': 0.0
        })

    # 使用AdamW优化器
    optimizer = optim.AdamW(optimizer_grouped_parameters, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    print(f">>> Adapter learning rate: {args.lr * 5.0:.6f}")
    print(f">>> Other parameters learning rate: {args.lr:.6f}")
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.mile_stones, 0.60)
    criteria = LabelSmoothingCrossEntropy()
    return optimizer, scheduler, criteria


class DualTaskLoss(nn.Module):
    """
    双任务损失函数：分类任务 + 分布学习任务
    基于论文"Joint Image Emotion Classification and Distribution Learning via Deep Convolutional Neural Network"
    实现协同梯度优化：∂L/∂aj = pj - (1-λ)yj - λlj
    """
    def __init__(self, lambda_weight, num_classes, temperature):
        super(DualTaskLoss, self).__init__()
        self.lambda_weight = lambda_weight  # λ参数，控制两个任务的权重
        self.num_classes = num_classes
        self.temperature = temperature  # 温度参数，放大KL损失梯度
        self.classification_loss = nn.CrossEntropyLoss()
        print(f"DualTaskLoss initialized: lambda={self.lambda_weight}, temperature={self.temperature}")
        
    def forward(self, logits, class_labels, emotion_distributions):
        """
        计算双任务损失 - 使用论文的协同梯度优化方法
        
        Args:
            logits: 模型输出的logits [batch_size, num_classes]
            class_labels: 分类标签 [batch_size]
            emotion_distributions: 情感分布标签 [batch_size, num_classes]
        
        Returns:
            total_loss: 总损失
            cls_loss: 分类损失
            dist_loss: 分布学习损失
        """
        # 添加输入验证和类型转换
        try:
            # 确保所有张量都在同一设备上
            device = logits.device
            if class_labels.device != device:
                class_labels = class_labels.to(device)
            if emotion_distributions.device != device:
                emotion_distributions = emotion_distributions.to(device)
            
            # 确保数据类型正确
            if logits.dtype != torch.float32:
                logits = logits.float()
            if emotion_distributions.dtype != torch.float32:
                emotion_distributions = emotion_distributions.float()
            if class_labels.dtype != torch.long:
                class_labels = class_labels.long()
            
            # 维度检查
            batch_size = logits.size(0)
            assert logits.shape[0] == class_labels.shape[0] == emotion_distributions.shape[0], \
                f"Batch size mismatch: logits {logits.shape[0]}, labels {class_labels.shape[0]}, distributions {emotion_distributions.shape[0]}"
            assert logits.shape[1] == emotion_distributions.shape[1], \
                f"Class number mismatch: logits {logits.shape[1]}, distributions {emotion_distributions.shape[1]}"
            
        except Exception as e:
            print(f"Error in DualTaskLoss input validation: {e}")
            print(f"logits shape: {logits.shape}, dtype: {logits.dtype}, device: {logits.device}")
            print(f"class_labels shape: {class_labels.shape}, dtype: {class_labels.dtype}, device: {class_labels.device}")
            print(f"emotion_distributions shape: {emotion_distributions.shape}, dtype: {emotion_distributions.dtype}, device: {emotion_distributions.device}")
            raise e
        
        # 实现论文的协同梯度优化方法
        # 梯度公式：∂L/∂aj = pj - (1-λ)yj - λlj
        
        # 1. 计算概率分布 pj
        probabilities = F.softmax(logits, dim=1)  # [B, C]
        
        # 2. 构造one-hot编码的分类标签 yj
        one_hot_labels = torch.zeros_like(logits)
        one_hot_labels.scatter_(1, class_labels.unsqueeze(1), 1.0)
        
        # 3. 数据质量检查和自适应λ权重
        dist_quality = torch.max(emotion_distributions, dim=1)[0]
        avg_quality = torch.mean(dist_quality)
        
        # 根据分布质量调整λ权重
        adaptive_lambda = torch.tensor(self.lambda_weight, device=device, dtype=torch.float32)
        
        # 4. 构造协同目标：(1-λ)yj + λlj
        # 这是论文梯度公式中pj应该逼近的目标
        unified_target = (1 - adaptive_lambda) * one_hot_labels + adaptive_lambda * emotion_distributions
        
        # 5. 计算协同损失：让预测概率逼近统一目标
        # 使用带温度的KL散度放大梯度
        unified_loss = F.kl_div(
            F.log_softmax(logits / self.temperature, dim=1), 
            unified_target, 
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # 6. 为了监控，分别计算分类损失和分布损失
        cls_loss = self.classification_loss(logits, class_labels)
        dist_loss = F.kl_div(
            F.log_softmax(logits / self.temperature, dim=1),
            emotion_distributions,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        total_loss = unified_loss
        
        # 调试信息（仅在第一次调用时打印）
        if not hasattr(self, '_debug_printed'):
            print("=== DualTaskLoss debug info (cooperative gradient) ===")
            print(f"Lambda weight: {self.lambda_weight}")
            print(f"Adaptive lambda: {adaptive_lambda.item():.4f}")
            print(f"Average distribution quality: {avg_quality.item():.4f}")
            print("Unified target examples:")
            for i in range(min(2, batch_size)):
                target = unified_target[i].detach().cpu().numpy()
                pred = probabilities[i].detach().cpu().numpy()
                print(f"  Sample {i}: target={target}, prediction={pred}")
            
            # 检查emotion_distributions的统计信息
            print(f"emotion_distributions shape: {emotion_distributions.shape}")
            print("emotion_distributions samples:")
            for i in range(min(3, batch_size)):
                dist = emotion_distributions[i]
                max_val = torch.max(dist).item()
                entropy = -torch.sum(dist * torch.log(dist + 1e-8)).item()
                is_one_hot = (torch.sum(dist == 1.0) == 1 and torch.sum(dist == 0.0) == 5).item()
                print(f"  Sample {i}: max={max_val:.3f}, entropy={entropy:.3f}, one_hot={is_one_hot}")
                print(f"         distribution: {dist.detach().cpu().numpy()}")
             
            self._debug_printed = True
        
        return total_loss, cls_loss, dist_loss

'''
# ============================== Dual-Head Loss ==============================
class DualHeadLoss(nn.Module):
    """
    双头损失函数：分类头采用交叉熵，分布头采用带温度缩放的KL散度，
    两者按照 λ 权重进行线性组合。
    """

    def __init__(self, lambda_weight: float = 0.5, temperature: float = 3.0):
        super(DualHeadLoss, self).__init__()
        self.lambda_weight = lambda_weight
        self.temperature = temperature
        self.classification_loss = nn.CrossEntropyLoss()

    def forward(self, cls_logits: torch.Tensor, dist_logits: torch.Tensor,
                class_labels: torch.Tensor, emotion_distributions: torch.Tensor):
        """
        Args:
            cls_logits: 分类头输出 [B, C]
            dist_logits: 分布头输出 [B, C]
            class_labels: 标签 [B]
            emotion_distributions: 目标分布 [B, C]
        Returns:
            total_loss, cls_loss, dist_loss
        """
        # 分类损失
        cls_loss = self.classification_loss(cls_logits, class_labels)

        # 分布学习损失（温度放大 KL）
        dist_loss = F.kl_div(
            F.log_softmax(dist_logits / self.temperature, dim=1),
            emotion_distributions,
            reduction='batchmean'
        ) * (self.temperature ** 2)

        total_loss = (1 - self.lambda_weight) * cls_loss + self.lambda_weight * dist_loss

        return total_loss, cls_loss, dist_loss
'''

def test_prompting(teloader, model):
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    one_hot = []
    losses = []
    criterion = torch.nn.CrossEntropyLoss(reduction='mean').cuda()
    end = time.time()
    for i, inputs in enumerate(tqdm(teloader)):
        labels = inputs['label']
        inputs = inputs['img']
        if isinstance(inputs, list):
            inputs = inputs[0]
        with torch.no_grad():
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model.forward_supervised(inputs)  # 更新为监督学习前向方法
            _, predicted = outputs.max(1)
            losses.append(criterion(outputs, labels).cpu())
            one_hot.append(predicted.eq(labels).cpu())
        acc1 = one_hot[-1].sum().item() / len(labels)
        top1.update(acc1, len(labels))
        batch_time.update(time.time() - end)
        end = time.time()
    model.eval()
    return top1.avg * 100

text_cls_epochs = {
    'DescribableTextures': 400, # 5.5k for txt_cls
    'EuroSAT': 400,
    'FGVCAircraft': 500,
    'Food101': 400,
    'CIFAR10_local': 400,
    'CIFAR100_local': 400, # 4k for txt_cls
    'ImageNet': 500,
    'OxfordFlowers': 600, # 600 for txt_cls
    'SUN397': 500, # 2k for txt_cls
    'UCF101': 400,
    'ImageNetR': 500, # 4k for txt_cls
    'ImageNetA': 500, # 4k for txt_cls
    'ImageNetSketch': 500, # 4k for txt_cls
    'Caltech101': 500, # 4k for txt_cls
    'Emotion6': 500,
    'abstract': 500,
    'Emoset':600
}

def setup_txt_epochs(args, dataset):
    args.txt_epochs = text_cls_epochs[dataset]


def get_env_id():
    return getpass.getuser()


def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class DINOLoss(nn.Module):
    def __init__(self, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax(teacher_output / temp, dim=-1)
        # teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            loss = torch.sum(-q * F.log_softmax(student_out, dim=-1), dim=-1)
            total_loss += loss.mean()
            n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss


def setup_log_folder(args):
    Path(args.logfolder).mkdir(exist_ok=True, parents=True)
    args.logfile = args.logfolder + f'/{time.strftime("%Y%m%d_%H%M%S")}.txt'


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def zero_shot(model, loader):
    print('-------------- ZERO SHOT INFERENCE --------------')
    total = 0.
    correct_base = 0.
    model.eval()
    with torch.no_grad():
        for i, inputs in enumerate(tqdm(loader)):
            target = inputs['label']
            images = inputs['img']
            if isinstance(images, list):
                images = images[0]
            images = images.cuda()
            target = target.cuda()
            out = model.forward_supervised(images)  # 更新为监督学习前向方法
            logits_base = out
            pred_base = torch.argmax(logits_base, dim=1)
            for j in range(len(target)):
                total += 1.
                if pred_base[j] == target[j]:
                    correct_base += 1.
    top1 = (correct_base / total) * 100
    print(f"Top-1 accuracy standard: {top1:.2f}")
    return top1


def test(teloader, model):
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    one_hot = []
    losses = []
    criterion = torch.nn.CrossEntropyLoss(reduction='mean').cuda()
    end = time.time()
    for i, (inputs, labels) in enumerate(tqdm(teloader)):
        if isinstance(inputs, list):
            inputs = inputs[0]

        with torch.no_grad():
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model.eval_clip(inputs)
            _, predicted = outputs.max(1)
            losses.append(criterion(outputs, labels).cpu())
            one_hot.append(predicted.eq(labels).cpu())
        acc1 = one_hot[-1].sum().item() / len(labels)
        top1.update(acc1, len(labels))
        batch_time.update(time.time() - end)
        end = time.time()
    model.eval()
    return top1.avg * 100


def update_ema_variables(model, alpha_teacher):
    teacher_prompt_param = []
    student_prompt_param = []

    for key, value in model.named_parameters():
        if key == 'prompt_embeddings':
            student_prompt_param.append(value)

        elif key == 'prompt_embeddings_teacher':
            teacher_prompt_param.append(value)

    for ema_param, param in zip(teacher_prompt_param, student_prompt_param):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[
                                                                                         :]  # alpha * teacher_weights + (1 - alpha) * student_weights

    for k, v in model.named_parameters():
        if k == 'prompt_embeddings_teacher':
            v = ema_param

    # return ema_model


def update_ema_variables_sanity(ema_model, model, alpha_teacher):
    for kv_ema, kv_student in zip(ema_model.named_parameters(), model.named_parameters()):
        if 'ln' in kv_ema[0] and 'ln' in kv_student[0]:
            kv_ema[1].data[:] = alpha_teacher * kv_ema[1][:].data[:] + (1 - alpha_teacher) * kv_student[1][:].data[:]
    return ema_model


class _BaseWarmupScheduler(_LRScheduler):

    def __init__(
            self,
            optimizer,
            successor,
            warmup_epoch,
            last_epoch=-1,
            verbose=False
    ):
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.successor.step(epoch)
            self._last_lr = self.successor.get_last_lr()
        else:
            super().step(epoch)


class ConstantWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
            self,
            optimizer,
            successor,
            warmup_epoch,
            cons_lr,
            last_epoch=-1,
            verbose=False
    ):
        self.cons_lr = cons_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        return [self.cons_lr for _ in self.base_lrs]


def get_root_logger(log_file=None, log_level=logging.INFO, name='main'):
    """Get root logger and add a keyword filter to it.
    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name,
    e.g., "mmdet3d".
    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.
        name (str, optional): The name of the root logger, also used as a
            filter keyword. Defaults to 'mmdet3d'.
    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    logger = get_logger(name=name, log_file=log_file, log_level=log_level)
    # add a logging filter
    logging_filter = logging.Filter(name)
    logging_filter.filter = lambda record: record.find(name) != -1

    return logger


def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    """Initialize and get a logger by name.
    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.
    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
        file_mode (str): The file mode used in opening log file.
            Defaults to 'w'.
    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    # handle duplicate logs to the console
    # Starting in 1.8.0, PyTorch DDP attaches a StreamHandler <stderr> (NOTSET)
    # to the root logger. As logger.propagate is True by default, this root
    # level handler causes logging messages from rank>0 processes to
    # unexpectedly show up on the console, creating much unwanted clutter.
    # To fix this issue, we set the root logger's StreamHandler, if any, to log
    # at the ERROR level.
    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        # Here, the default behaviour of the official logger is 'a'. Thus, we
        # provide an interface to change the file mode to the default
        # behaviour.
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True

    return logger


