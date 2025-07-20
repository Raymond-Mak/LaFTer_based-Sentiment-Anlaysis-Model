from torch.nn import Conv2d, Dropout
import math
import os
import os.path as osp
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_checkpoint
from dassl.data.data_manager import DataManager
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.clip import tokenize
from dassl.metrics import compute_accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
_tokenizer = _Tokenizer()
from utils.model_utils import *
from utils.utils import *
_tokenizer = _Tokenizer()
from functools import reduce
from operator import mul
from utils.data_utils import ds_specific_templates
from tqdm import tqdm

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, root='all_weights')

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class MultiStageConsistencyLoss(nn.Module):
    """多段一致性损失函数"""
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, stage_outputs):
        """
        Args:
            stage_outputs: 列表，包含各段的logits输出
        """
        if len(stage_outputs) < 2:
            return torch.tensor(0.0, device=stage_outputs[0].device)
        
        consistency_loss = 0.0
        num_pairs = 0
        
        # 计算相邻段之间的KL散度
        for i in range(len(stage_outputs) - 1):
            current_stage = F.log_softmax(stage_outputs[i], dim=-1)
            next_stage = F.softmax(stage_outputs[i + 1], dim=-1)
            
            kl_loss = F.kl_div(current_stage, next_stage, reduction='batchmean')
            consistency_loss += kl_loss
            num_pairs += 1
        
        return self.alpha * consistency_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0)


class NGAAggregator(nn.Module):
    """邻域高斯聚合模块"""
    def __init__(self, num_stages=4, init_sigma=1.0):
        super().__init__()
        self.num_stages = num_stages
        # 每个stage的sigma参数，可学习
        self.sigmas = nn.Parameter(torch.ones(num_stages) * init_sigma)
        
    def forward(self, stage_features):
        """
        Args:
            stage_features: 列表，每个元素是该stage内所有层的特征列表
        Returns:
            aggregated_features: 每个stage聚合后的特征
        """
        aggregated_features = []
        
        for stage_idx, features_in_stage in enumerate(stage_features):
            if len(features_in_stage) == 1:
                # 如果该stage只有一层，直接使用
                aggregated_features.append(features_in_stage[0])
            else:
                # 计算高斯权重
                num_layers = len(features_in_stage)
                sigma = self.sigmas[stage_idx]
                
                weights = []
                for l in range(num_layers):
                    # 权重计算：距离stage末尾越近权重越大
                    weight = torch.exp(-((num_layers - l) ** 2) / (2 * sigma ** 2))
                    weights.append(weight)
                
                # 归一化权重
                weights = torch.stack(weights)
                weights = weights / weights.sum()
                
                # 加权聚合
                aggregated = sum(w * feat for w, feat in zip(weights, features_in_stage))
                aggregated_features.append(aggregated)
        
        return aggregated_features


class MultiLayerLaFTerUFT(nn.Module):
    """多层Prompt版本的LaFTer模型"""
    
    def __init__(self, model, classes, templates, ds_templates, dataset_name, txt_cls, cfg, device="cuda", log=False):
        super().__init__()
        
        # 基础配置
        self.device = device
        self.cfg = cfg
        self.dataset_templates = ds_templates
        self.classes = classes
        self.dataset_name = dataset_name
        self.txt_cls = txt_cls
        self.log = log
        patch_size = (32, 32)  # ViT-B/32使用32x32 patch
        self.model = model.to(device)
        self.templates = templates
        self.backbone_out_size = 512
        self.hidden_size = 768  # ViT-B/32使用768维
        
        # 冻结CLIP主干网络
        print("Freezing CLIP backbone...")
        for param in self.model.parameters():
            param.requires_grad = False
        print("CLIP backbone frozen successfully")
        
        # 多层prompt配置 - 与CLIP的MultiLayerTransformer对齐
        self.num_stages = 4  # 4个分段
        self.stage_layers = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10], [11]]  # 与CLIP一致的分段
        self.prompt_tokens_per_stage = 16  # 每段16个prompt token
        
        # 多层prompt参数
        self.multi_prompts = nn.ParameterList([
            nn.Parameter(torch.zeros(1, self.prompt_tokens_per_stage, self.hidden_size))
            for _ in range(self.num_stages)
        ])
        
        # 初始化prompt参数
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + self.hidden_size))
        for prompt in self.multi_prompts:
            nn.init.uniform_(prompt.data, -val, val)
        
        # NGA聚合器
        self.nga_aggregator = NGAAggregator(num_stages=self.num_stages, init_sigma=1.0)
        
        # 段间融合权重（可学习）
        self.stage_weights = nn.Parameter(torch.ones(self.num_stages) / self.num_stages)
        
        # Dropout和投影层
        self.prompt_dropout = Dropout(0.0)
        self.prompt_proj = nn.Identity()
        
        # 参考Cascade-CLIP：为每个stage添加独立的LayerNorm和Projection层
        self.stage_ln_layers = nn.ModuleList([
            nn.LayerNorm(self.hidden_size) for _ in range(self.num_stages)
        ])
        self.stage_proj_layers = nn.ModuleList([
            nn.Linear(self.hidden_size, self.backbone_out_size) for _ in range(self.num_stages)
        ])
        
        # 分类器和分布头
        self.adapter = nn.Sequential(nn.Linear(int(self.backbone_out_size), len(classes), bias=False)).to(device)
        self.dist_head = nn.Sequential(nn.Linear(int(self.backbone_out_size), len(classes), bias=False)).to(device)
        
        # 多段一致性损失
        self.consistency_loss = MultiStageConsistencyLoss(alpha=0.1)
        
        # 文本特征（保持与原版相同）
        self.txt_features_for_text_cls, self.labels_for_text_cls = self.txt_features_for_text_cls()
        self.text_features = self.txt_features()
        
        # 确保所有参数都在正确的设备上
        self.to(device)


    def incorporate_multi_prompts(self, x):
        """在patch embedding后添加第一段prompt"""
        B = x.shape[0]
        x = self.patch_embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
        
        # 添加第一段prompt
        first_stage_prompt = self.prompt_dropout(
            self.prompt_proj(self.multi_prompts[0]).expand(B, -1, -1)
        )
        
        x = torch.cat((
            x[:, :1, :],  # CLS token
            first_stage_prompt,  # 第一段prompt
            x[:, 1:, :]   # patch tokens
        ), dim=1)
        
        return x

    def forward_multi_stage(self, x):
        """多层prompt前向传播 - 修复版本"""
        # 第一步：添加patch embedding和第一段prompt
        x_with_first_prompt = self.incorporate_multi_prompts(x)
        
        # 第二步：准备其他stages的prompt列表
        B = x_with_first_prompt.shape[0]
        
        # 根据CLIP的MultiLayerTransformer期望，构建 multi_prompts 列表
        # 第一个stage的prompt已经在incorporate_multi_prompts中处理，所以设为None
        multi_prompts = [None]  # stage 0: 已在x_with_first_prompt中
        
        # 添加其余stages的prompts
        for i in range(1, self.num_stages):
            if i < len(self.multi_prompts):
                prompt = self.prompt_dropout(self.prompt_proj(self.multi_prompts[i]).expand(B, -1, -1))
                multi_prompts.append(prompt)
            else:
                multi_prompts.append(None)
        
        # 第三步：调用CLIP的多层transformer
        try:
            final_features, stage_outputs = self.model.visual.forward_multi_stage_prompts(
                x_with_first_prompt, multi_prompts
            )
        except Exception as e:
            print(f"Error in forward_multi_stage_prompts: {e}")
            # 如果多层方法失败，回退到传统方法
            features = self.model.visual.forward_after_patch_embeddings(x_with_first_prompt)
            # 模拟 stage_outputs 结构
            stage_outputs = [[features] for _ in range(self.num_stages)]
            return features, stage_outputs
        
        return final_features, stage_outputs

    def multi_stage_classification(self, stage_outputs):
        """多段级联分类 - 对应CLIP的stage_outputs格式"""
        # 检查stage_outputs格式
        if not isinstance(stage_outputs, list) or len(stage_outputs) == 0:
            print(f"Warning: Invalid stage_outputs format: {type(stage_outputs)}")
            # 回退到默认处理
            return None, []
        
        # NGA聚合各段特征
        try:
            aggregated_features = self.nga_aggregator(stage_outputs)
        except Exception as e:
            print(f"Warning: NGA aggregation failed: {e}")
            # 回退方案：只使用最后一个stage
            if stage_outputs and len(stage_outputs[-1]) > 0:
                aggregated_features = [stage_outputs[-1][-1]]  # 最后一个stage的最后一层
            else:
                return None, []
        
        # 为每个stage进行独立的normalization和projection
        normalized_projected_features = []
        for idx, stage_feat in enumerate(aggregated_features):
            if idx < len(self.stage_ln_layers) and idx < len(self.stage_proj_layers):
                # LayerNorm
                normed_feat = self.stage_ln_layers[idx](stage_feat)
                # 投影到统一维度 (768 -> 512)
                projected_feat = self.stage_proj_layers[idx](normed_feat)
                # L2归一化，确保特征在相同尺度
                projected_feat = F.normalize(projected_feat, p=2, dim=-1)
                normalized_projected_features.append(projected_feat)
        
        if not normalized_projected_features:
            print("Warning: No valid projected features")
            return None, []
        
        # 对每段特征进行分类
        stage_logits = []
        for stage_feat in normalized_projected_features:
            logits = self.adapter(stage_feat)
            stage_logits.append(logits)
        
        # 可学习权重融合多段logits
        if len(stage_logits) > 0:
            weights = F.softmax(self.stage_weights[:len(stage_logits)], dim=0)
            final_logits = sum(w * logits for w, logits in zip(weights, stage_logits))
        else:
            final_logits = None
        
        return final_logits, stage_logits

    def eval_clip_multi_stage(self, x):
        """多层prompt的CLIP评估 - 添加错误处理"""
        with torch.no_grad():
            try:
                # 多层prompt前向传播
                final_features, stage_outputs = self.forward_multi_stage(x)
                
                # 多段分类
                final_logits, stage_logits = self.multi_stage_classification(stage_outputs)
                
                if final_logits is not None:
                    return final_logits
                else:
                    # 回退到传统方法
                    print("Warning: Multi-stage classification failed, falling back to traditional method")
                    img_features = self.model.visual.forward_after_patch_embeddings(
                        self.incorporate_prompt(x)
                    )
                    return self.adapter(img_features)
                    
            except Exception as e:
                print(f"Error in eval_clip_multi_stage: {e}")
                # 完全回退到传统方法
                img_features = self.model.visual.forward_after_patch_embeddings(
                    self.incorporate_prompt(x)
                )
                return self.adapter(img_features)

    def incorporate_prompt(self, x, teacher=False):
        """传统单层prompt方法 - 保持与原版LaFTer兼容"""
        B = x.shape[0]
        x = self.patch_embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
        # 使用第一个prompt作为默认prompt
        prompt_emb = self.prompt_dropout(self.prompt_proj(self.multi_prompts[0]).expand(B, -1, -1))
        x = torch.cat((
            x[:, :1, :],  # CLS token
            prompt_emb,   # prompt tokens
            x[:, 1:, :]   # patch tokens
        ), dim=1)
        return x

    def patch_embeddings(self, x: torch.tensor):
        """Wrapper for visual patch embeddings"""
        # Fix dtype mismatch - convert input to same dtype as model weights
        x = x.to(self.model.visual.conv1.weight.dtype)
        return self.model.visual.embeddings_patch(x)

    def train_txt_clas(self, criteria):
        noise_std = 0.1
        noise = torch.randn(self.txt_features_for_text_cls.shape) * noise_std
        txt_feas = self.txt_features_for_text_cls
        txt_label = self.labels_for_text_cls
        feas = (self.adapter(txt_feas.to(torch.float32) + noise.cuda()))
        loss = criteria(feas, txt_label)
        return loss

    def txt_features_for_text_cls(self):
        """生成用于文本分类的特征和标签（保持原版逻辑）"""
        if self.txt_cls == 'cls_only':
            gpt3_prompts = None
            desc, labels_for_descriptions = gen_labels_with_classes(self.classes, descriptions=gpt3_prompts)
        elif self.txt_cls == 'templates_only':
            gpt3_prompts = self.templates
            desc, labels_for_descriptions = gen_labels_with_templates(self.classes, descriptions=gpt3_prompts)
        elif self.txt_cls == 'full':
            gpt3_prompts = None  
            desc, labels_for_descriptions = gen_labels_with_classes_and_simple_template(self.classes, descriptions=gpt3_prompts)
        else:
            # 其他模式保持原有逻辑
            desc, labels_for_descriptions = gen_labels_with_classes(self.classes)
        
        # 生成嵌入
        cache_filename = f"cache/{self.dataset_name}_{self.txt_cls}_embeddings.pth"
        os.makedirs(os.path.dirname(cache_filename), exist_ok=True)
        
        if os.path.exists(cache_filename):
            print(f"Loading cached embeddings from: {cache_filename}")
            cache_data = torch.load(cache_filename)
            return cache_data['features'], cache_data['labels']
        else:
            return self._generate_new_embeddings(desc, labels_for_descriptions, cache_filename)

    def _generate_new_embeddings(self, desc, labels_for_descriptions, cache_filename):
        """生成新的文本嵌入（保持原版逻辑）"""
        print(f"Processing {len(desc)} text descriptions...")
        labels_for_descriptions = torch.tensor(labels_for_descriptions).cuda()
        
        zeroshot_weights = []
        processed_labels = []
        
        with torch.no_grad():
            for i, (text_desc, label) in enumerate(zip(tqdm(desc, desc="处理文本"), labels_for_descriptions)):
                try:
                    text_tokens = tokenize([text_desc]).cuda()
                    class_embeddings = self.model.encode_text(text_tokens)
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                    single_embedding = class_embeddings.squeeze(0)
                    
                    zeroshot_weights.append(single_embedding)
                    processed_labels.append(label)
                except Exception as e:
                    print(f"Error processing text #{i}: {text_desc[:50]}... - {e}")
                    continue
            
            if len(zeroshot_weights) > 0:
                zeroshot_weights = torch.stack(zeroshot_weights).cuda()
                processed_labels = torch.stack(processed_labels).cuda()
            else:
                raise RuntimeError("没有成功处理任何文本嵌入")
            
            # 保存缓存
            cache_data = {
                'features': zeroshot_weights,
                'labels': processed_labels,
                'metadata': {
                    'dataset_name': self.dataset_name,
                    'txt_cls_mode': self.txt_cls,
                    'total_texts': len(desc),
                    'num_classes': len(self.classes),
                    'template_count': len(self.dataset_templates) if self.dataset_templates else 0
                }
            }
            torch.save(cache_data, cache_filename)
            print(f"Embeddings saved to: {cache_filename}")
            
            return zeroshot_weights, processed_labels

    def txt_features(self):
        """生成文本特征（保持原版逻辑）"""
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(self.classes):
                texts = [template.format(classname) for template in self.templates]
                texts = tokenize(texts).cuda()
                class_embeddings = self.model.encode_text(texts)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights

    def image_features(self, images):
        """提取图像特征（使用多层prompt）"""
        with torch.no_grad():
            final_features, _ = self.forward_multi_stage(images)
            final_features /= final_features.norm(dim=-1, keepdim=True)
            return final_features

    def eval_text_clas(self, batch):
        """评估文本分类性能"""
        with torch.no_grad():
            txt_feas = self.txt_features_for_text_cls
            txt_label = self.labels_for_text_cls
            logits = self.adapter(txt_feas.to(torch.float32))
        return logits, txt_label
    
    def forward(self, x):
        """标准前向方法 - 兼容dassl"""
        # 对于训练，使用多层prompt分类
        if self.training:
            try:
                final_features, stage_outputs = self.forward_multi_stage(x)
                final_logits, stage_logits = self.multi_stage_classification(stage_outputs)
                if final_logits is not None:
                    return final_logits
            except Exception as e:
                print(f"Multi-stage forward failed: {e}, falling back to traditional")
        
        # 回退到传统方法
        img_features = self.incorporate_prompt(x)
        img_features = self.model.visual.forward_after_patch_embeddings(img_features)
        return self.adapter(img_features)
    
    def txt_cls_init(self):
        import copy
        self.adapter_pl = copy.deepcopy(self.adapter)
    
    def forward_supervised(self, x):
        """监督学习前向方法"""
        return self.forward(x)


@TRAINER_REGISTRY.register()
class MultiLayerLaFTer(TrainerX):
    """使用多层Prompt技术的LaFTer训练器 - 完全按照原版LaFTer结构"""

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            clip_model.float()
        print("Building Multi-Layer Prompt LaFTer Model")
        self.model = MultiLayerLaFTerUFT(
            model=clip_model, 
            classes=classnames,
            templates=['a photo of a {}'], 
            ds_templates=ds_specific_templates[cfg.DATASET.NAME], 
            dataset_name=cfg.DATASET.NAME, 
            txt_cls=cfg.txt_cls, 
            cfg=cfg
        )
        
        # 在register_model之前打印可训练参数信息用于调试
        print("=== Trainable Parameters Before Registration ===")
        total_params = 0
        trainable_params = 0
        trainable_param_names = []
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                trainable_param_names.append(name)
                print(f"  {name}: {param.shape} ({param.numel()} params)")
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Number of trainable parameters: {len(trainable_param_names)}")
        print("===========================")
        
        # 确保模型有可训练参数
        if trainable_params == 0:
            print("ERROR: No trainable parameters found!")
            raise RuntimeError("Model has no trainable parameters")
        
        self.register_model("adapt", self.model)
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
            
        # 验证模型注册后的状态
        print(f"Registered models: {list(self._models.keys())}")
        print(f"Model 'adapt' parameters: {sum(p.numel() for p in self._models['adapt'].parameters() if p.requires_grad)}")
        
        # 立即构建优化器
        self.build_optim()

    def build_optim(self):
        """构建优化器 - 解决dassl优化器初始化问题"""
        cfg = self.cfg
        print("Building optimizer...")
        
        # 获取模型参数
        params = list(self.model.parameters())
        trainable_params = [p for p in params if p.requires_grad]
        
        print(f"Found {len(trainable_params)} trainable parameter groups")
        
        if len(trainable_params) == 0:
            raise RuntimeError("No trainable parameters found for optimizer")
        
        # 创建优化器
        if cfg.OPTIM.NAME == "adam":
            optimizer = torch.optim.Adam(
                trainable_params,
                lr=cfg.OPTIM.LR,
                weight_decay=cfg.OPTIM.WEIGHT_DECAY,
                betas=(cfg.OPTIM.ADAM_BETA1, cfg.OPTIM.ADAM_BETA2)
            )
        elif cfg.OPTIM.NAME == "sgd":
            optimizer = torch.optim.SGD(
                trainable_params,
                lr=cfg.OPTIM.LR,
                weight_decay=cfg.OPTIM.WEIGHT_DECAY,
                momentum=cfg.OPTIM.MOMENTUM,
                nesterov=cfg.OPTIM.SGD_NESTEROV
            )
        else:
            raise NotImplementedError(f"Optimizer {cfg.OPTIM.NAME} not implemented")
        
        # 直接设置优化器而不使用register方法
        if not hasattr(self, '_optims'):
            self._optims = {}
        if not hasattr(self, '_scheds'):
            self._scheds = {}
            
        self._optims["adapt"] = optimizer
        
        # 创建学习率调度器
        if cfg.OPTIM.LR_SCHEDULER == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.OPTIM.MAX_EPOCH
            )
        elif cfg.OPTIM.LR_SCHEDULER == "single_step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=cfg.OPTIM.STEPSIZE[0], gamma=cfg.OPTIM.GAMMA
            )
        else:
            scheduler = None
        
        if scheduler is not None:
            self._scheds["adapt"] = scheduler
        
        print(f"Optimizer created: {optimizer}")
        print(f"Scheduler created: {scheduler}")
        print(f"_optims keys: {list(self._optims.keys())}")
        print(f"_scheds keys: {list(self._scheds.keys()) if hasattr(self, '_scheds') else 'None'}")

    def build_data_loader(self):
        """Create essential data-related attributes."""
        # 直接使用原始的te_transform，不引入复杂的条件逻辑
        from utils.model_utils import te_transform
        dm = DataManager(self.cfg, custom_tfm_test=te_transform, custom_tfm_train=te_transform)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

        self.dm = dm

    # 更新 parse_batch_train，确保只处理单一图像和标签
    def parse_batch_train(self, batch):
        input = batch["img"].to(self.device)  # 不再是列表，直接是单一图像张量
        label = batch["label"].to(self.device)
        return input, label

    def forward_backward(self, batch):
        """前向传播和反向传播 - dassl训练循环必需的方法"""
        input, label = self.parse_batch_train(batch)
        
        # 使用多层prompt前向传播
        try:
            output = self.model.forward(input)
        except Exception as e:
            print(f"Warning: Multi-stage forward failed ({e}), using fallback")
            # 回退到传统方法
            output = self.model.forward_supervised(input)
        
        # 计算损失
        loss = F.cross_entropy(output, label)
        
        # 反向传播
        self.model_backward_and_update(loss)
        
        loss_summary = {
            "loss": loss.item()
        }
        
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        
        return loss_summary

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)