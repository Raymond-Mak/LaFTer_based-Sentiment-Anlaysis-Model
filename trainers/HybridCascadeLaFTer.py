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
import json
import clip
from pathlib import Path
# 导入HybridCascade专用的CLIP模型
from clip.hybrid_cascade_model import build_hybrid_cascade_model


def load_hybrid_cascade_clip_to_cpu(cfg):
    """加载HybridCascade专用的CLIP模型到CPU"""
    from clip import clip
    
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, root='all_weights')

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    # 使用HybridCascade版本的模型构建器
    model = build_hybrid_cascade_model(state_dict or model.state_dict())
    return model


class NGAAggregator(nn.Module):
    """邻域高斯聚合模块"""
    def __init__(self, num_layers, init_sigma=1.0):
        super().__init__()
        self.num_layers = num_layers
        # 每个stage的sigma参数，可学习
        self.sigma = nn.Parameter(torch.tensor(init_sigma))
        
    def forward(self, layer_features):
        """
        Args:
            layer_features: 列表，每个元素是该层的特征 [B, 768]
        Returns:
            aggregated_features: 聚合后的特征 [B, 768]
        """
        if len(layer_features) == 1:
            return layer_features[0]
            
        # 计算高斯权重
        num_layers = len(layer_features)
        weights = []
        for l in range(num_layers):
            # 权重计算：距离stage末尾越近权重越大
            weight = torch.exp(-((num_layers - l) ** 2) / (2 * self.sigma ** 2))
            weights.append(weight)
        
        # 归一化权重
        weights = torch.stack(weights)
        weights = weights / weights.sum()
        
        # 加权聚合特征
        stacked_features = torch.stack(layer_features, dim=0)  # [num_layers, B, 768]
        weights = weights.view(-1, 1, 1)  # [num_layers, 1, 1]
        aggregated = (weights * stacked_features).sum(dim=0)  # [B, 768]
        
        return aggregated


class StageConsistencyLoss(nn.Module):
    """多段一致性损失函数"""
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, stage_logits):
        """
        Args:
            stage_logits: 列表，包含各段的logits输出
        """
        if len(stage_logits) < 2:
            return torch.tensor(0.0, device=stage_logits[0].device)
        
        consistency_loss = 0.0
        num_pairs = 0
        
        # 计算各段之间的KL散度
        for i in range(len(stage_logits)):
            for j in range(i + 1, len(stage_logits)):
                p_i = F.softmax(stage_logits[i], dim=-1)
                p_j = F.softmax(stage_logits[j], dim=-1)
                kl_loss = F.kl_div(p_i.log(), p_j, reduction='batchmean')
                consistency_loss += kl_loss
                num_pairs += 1
        
        return self.alpha * consistency_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0)


class HybridCascadeLaFTerUFT(nn.Module):
    """
    HybridCascade-LaFTer混合架构：
    结合CascadeCLIP稳定特征提取与LaFTer文本分类优势
    """
    def __init__(self, model, classes, templates, ds_templates=None, device='cuda', 
                 log=None, dataset_name=None, txt_cls=None, cfg=None):
        super().__init__()
        
        # 基础配置 - 保持与LaFTer一致的接口
        self.adapter_pl = None
        self.device = device
        self.cfg = cfg
        self.dataset_templates = ds_templates
        self.classes = classes
        self.dataset_name = dataset_name
        self.txt_cls = txt_cls
        self.log = log
        self.model = model.to(device)
        self.templates = templates
        self.backbone_out_size = 512
        self.hidden_size = 768  # ViT-B: 768, ViT-L: 1024
        self.num_tokens = 50
        self.prompt_proj = nn.Identity()
        self.hidden_size_text = 512
        self.num_tokens_text = 77
        
        # Stage配置：将transformer layers分成3个stage进行特征提取
        self.extraction_indices = [4, 5, 6, 7, 8, 9, 10, 11]  # 提取第5-12层
        self.stage_boundaries = [
            [0, 1, 2],      # layers 5,6,7 -> stage 0 (中层特征)
            [3, 4],         # layers 8,9 -> stage 1 (高层特征)  
            [5, 6, 7]       # layers 10,11,12 -> stage 2 (最终特征)
        ]
        
        # NGA聚合层
        self.nga_layers = nn.ModuleList([
            NGAAggregator(len(boundary), init_sigma=1.0) 
            for boundary in self.stage_boundaries
        ])
        
        # Stage融合权重（可学习）
        self.stage_fusion_weights = nn.Parameter(torch.ones(len(self.stage_boundaries)))
        
        # 一致性损失
        self.consistency_loss_fn = StageConsistencyLoss(alpha=0.1)
        
        # LaFTer的adapter结构（关键！）
        self.adapter = nn.Sequential(nn.Linear(self.backbone_out_size, len(classes), bias=False)).to(device)
        # 双头架构：分布学习头部
        self.dist_head = nn.Sequential(nn.Linear(self.backbone_out_size, len(classes), bias=False)).to(device)
        
        # 初始化prompt参数
        self._init_prompt_parameters()
        
        # 生成文本特征用于分类器训练 - 完全复用LaFTer逻辑
        try:
            self.txt_features_for_text_cls_data, self.labels_for_text_cls = self.txt_features_for_text_cls()
            self.text_features = self.txt_features()
            print(f"Successfully loaded text features for {len(self.classes)} classes")
        except Exception as e:
            print(f"Warning: Error loading text features: {e}")
            # 回退到简单文本特征
            self.txt_features_for_text_cls_data = None
            self.labels_for_text_cls = None
            self.text_features = self.txt_features()
        
        print(f"HybridCascade-LaFTer initialized with {len(self.stage_boundaries)} stages")
        print(f"Stage boundaries: {self.stage_boundaries}")
        print(f"Extraction indices: {self.extraction_indices}")

    def _init_prompt_parameters(self):
        """初始化prompt参数 - 复用LaFTer逻辑"""
        patch_size = (16, 16)
        prompt_dim = self.hidden_size
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))
        
        # Prompt相关参数
        self.prompt_dropout = Dropout(0.0)
        self.prompt_embeddings = nn.Parameter(torch.zeros(
            1, self.num_tokens, self.hidden_size), requires_grad=True)
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)

    def txt_features_for_text_cls(self):
        """
        生成用于文本分类的特征和标签 - 完全复用LaFTer逻辑
        """
        if self.txt_cls == 'cls_only':
            gpt3_prompts = None
            desc, labels_for_descriptions = gen_labels_with_classes(self.classes, descriptions=gpt3_prompts)
        elif self.txt_cls == 'templates_only':
            gpt3_prompts = self.templates
            desc, labels_for_descriptions = gen_labels_with_templates(self.classes, descriptions=gpt3_prompts)
        elif self.txt_cls == 'lafter':
            if self.dataset_name not in lafter_datasets:
                raise ValueError(f'Invalid dataset name for LaFTer: {self.dataset_name}')

            # 加载GPT生成的描述
            path_to_file = f'./descriptions/generic/{self.dataset_name}.json'
            if not os.path.exists(path_to_file):
                print(f"Warning: Description file not found: {path_to_file}")
                # 回退到简单模板
                desc, labels_for_descriptions = gen_labels_with_templates(self.classes, self.dataset_templates)
            else:
                with open(path_to_file) as f:
                    gpt3_prompts = json.load(f)

                # 生成基于描述的文本和标签
                desc, labels_for_descriptions = gen_labels_with_descrptions(self.classes, descriptions=gpt3_prompts)
                
                # 生成基于模板的文本和标签
                templates, labels_for_templates = gen_labels_with_templates(self.classes, descriptions=self.dataset_templates)

                # 合并所有文本和标签
                desc += templates
                labels_for_descriptions += labels_for_templates
                
                # 调试信息
                print("=== HybridCascade LaFTer data statistics ===")
                print(f"Number of GPT descriptions: {len(desc) - len(templates)}")
                print(f"Number of template texts: {len(templates)}")
                print(f"Total description texts: {len(desc)}")
                print(f"Total labels: {len(labels_for_descriptions)}")
                print(f"Number of classes: {len(self.classes)}")
                print(f"Average texts per class: {len(desc) / len(self.classes):.1f}")
        elif self.txt_cls == 'zero_shot':
            return None, None
        else:
            raise ValueError(f'Invalid txt_cls argument: {self.txt_cls}')

        if self.txt_cls in ['cls_only', 'templates_only', 'lafter']:
            Path(f'embeddings').mkdir(parents=True, exist_ok=True)
            
            # 构建缓存文件名，包含模板数量信息以避免缓存冲突
            template_count = len(self.dataset_templates) if self.dataset_templates else 0
            total_text_count = len(desc)
            cache_filename = f'embeddings/hybrid_cascade_{self.txt_cls}_{self.dataset_name}_templates{template_count}_total{total_text_count}_embeddings.pt'
            
            if os.path.isfile(cache_filename):
                print(f"******** Loading cached embeddings: {cache_filename} *********")
                try:
                    cached_data = torch.load(cache_filename)
                    if isinstance(cached_data, dict) and 'features' in cached_data and 'labels' in cached_data:
                        zeroshot_weights = cached_data['features']
                        labels_for_descriptions = cached_data['labels']
                        print(f"Loaded from cache - feature shape: {zeroshot_weights.shape}, label shape: {labels_for_descriptions.shape}")
                        
                        # 验证缓存数据的一致性
                        if zeroshot_weights.shape[0] != len(labels_for_descriptions):
                            print(f"Cache inconsistency: features {zeroshot_weights.shape[0]} != labels {len(labels_for_descriptions)}")
                            raise FileNotFoundError("缓存数据不一致，需要重新生成")
                    else:
                        raise FileNotFoundError("需要重新生成缓存")
                except Exception as e:
                    print(f"Failed to load cache: {e}")
                    print("Regenerating embeddings...")
                    zeroshot_weights, labels_for_descriptions = self._generate_new_embeddings(desc, labels_for_descriptions, cache_filename)
            else:
                print('******** No embedding cache found --- generating new embeddings *********')
                zeroshot_weights, labels_for_descriptions = self._generate_new_embeddings(desc, labels_for_descriptions, cache_filename)

            # 最终的一致性检查
            print("=== Final consistency check ===")
            print(f"Feature shape: {zeroshot_weights.shape}")
            print(f"Label shape: {labels_for_descriptions.shape}")
            
            return zeroshot_weights.squeeze(), labels_for_descriptions
        else:
            return None, None

    def _generate_new_embeddings(self, desc, labels_for_descriptions, cache_filename):
        """生成新的文本嵌入，确保特征和标签数量一致"""
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
                
                # 保存到缓存
                cache_data = {
                    'features': zeroshot_weights,
                    'labels': processed_labels
                }
                torch.save(cache_data, cache_filename)
                print(f"Saved embeddings to cache: {cache_filename}")
                
                return zeroshot_weights, processed_labels
            else:
                raise RuntimeError("没有成功处理任何文本嵌入")

    def train_txt_clas(self, criteria):
        """训练文本分类器 - 复用LaFTer逻辑"""
        if self.txt_features_for_text_cls_data is None:
            return torch.tensor(0.0).cuda()
            
        noise_std = 0.1
        noise = torch.randn(self.txt_features_for_text_cls_data.shape) * noise_std
        txt_feas = self.txt_features_for_text_cls_data
        txt_label = self.labels_for_text_cls
        feas = (self.adapter(txt_feas.to(torch.float32) + noise.cuda()))
        loss = criteria(feas, txt_label)
        return loss

    def txt_features(self):
        """生成文本特征 - 保持LaFTer接口"""
        prompt_prefix = " ".join(["X"] * 16)  # 默认使用16个context tokens
        prompts = [prompt_prefix + " " + name + "." for name in self.classes]
        
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        return text_features

    def incorporate_prompt(self, x):
        """
        保持LaFTer原有的prompt注入方式，但适配HybridCascade CLIP
        """
        B = x.shape[0]
        # 使用HybridCascade CLIP的embeddings_patch方法
        x = self.model.visual.embeddings_patch(x)  # [B, N+1, 768]
        
        # 注入prompt
        prompt_emb = self.prompt_dropout(
            self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)
        )
        
        # 在CLS token后插入prompt tokens
        x = torch.cat([x[:, :1, :], prompt_emb, x[:, 1:, :]], dim=1)  # [B, N+1+50, 768]
        
        return x

    def embeddings_after_prompts_with_intermediates(self, x):
        """
        HybridCascade核心：使用HybridCascade专用的CLIP进行中间特征提取
        """
        # 使用HybridCascade专用的forward_with_intermediates方法
        final_features, intermediate_features = self.model.visual.forward_with_intermediates(x)
        
        # intermediate_features现在包含第5-12层的CLS token特征
        # 对应extraction_indices = [4, 5, 6, 7, 8, 9, 10, 11]
        
        return final_features, intermediate_features

    def aggregate_features_by_stages(self, intermediate_features):
        """按stage分组并聚合特征"""
        stage_features = []
        for stage_idx, boundaries in enumerate(self.stage_boundaries):
            # 取该stage对应的层特征
            stage_feats = [intermediate_features[i] for i in boundaries]
            # NGA聚合该stage内的多层特征
            aggregated_feat = self.nga_layers[stage_idx](stage_feats)
            stage_features.append(aggregated_feat)
        
        return stage_features

    def forward_supervised(self, image):
        """监督学习前向传播 - 保持与LaFTer接口一致"""
        logits = self.forward(image)  # 不传入label，只返回logits
        return logits

    def forward(self, image, label=None):
        """HybridCascade-LaFTer主要前向传播"""
        B = image.shape[0]
        
        # 1. 保持LaFTer的prompt注入方式
        x_with_prompt = self.incorporate_prompt(image)
        
        # 2. 通过修改的transformer提取中间特征
        final_features, intermediate_features = self.embeddings_after_prompts_with_intermediates(x_with_prompt)
        
        # 3. 按stage分组并聚合特征
        stage_features = self.aggregate_features_by_stages(intermediate_features)
        
        # 4. 使用LaFTer的text classifier对每个stage进行分类
        stage_logits = []
        for stage_feat in stage_features:
            # 关键：使用LaFTer经过LLM文本训练的adapter进行分类
            logits = self.adapter(stage_feat)  # nn.Linear(512, num_classes)
            stage_logits.append(logits)
        
        # 5. 融合多stage结果
        stage_weights = F.softmax(self.stage_fusion_weights, dim=0)
        final_logits = sum(w * logits for w, logits in zip(stage_weights, stage_logits))
        
        # 6. 一致性损失计算
        consistency_loss = self.consistency_loss_fn(stage_logits)
        
        if label is not None:
            return final_logits, consistency_loss
        
        return final_logits

    def txt_cls_init(self):
        """初始化文本分类器的伪标签适配器 - 与LaFTer_basic保持一致"""
        import copy
        self.adapter_pl = copy.deepcopy(self.adapter)

    def forward_dual(self, image):
        """双任务前向传播 - 兼容LaFTer接口"""
        final_logits, consistency_loss = self.forward(image)
        # 返回分类logits和分布logits（这里使用相同的logits）
        return final_logits, final_logits, consistency_loss


class BaseLaFTerTrainer(TrainerX):
    """LaFTer训练器的基类 - 从LaFTer_basic.py复用"""

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_data_loader(self):
        """Create essential data-related attributes."""
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

    def parse_batch_train(self, batch):
        """解析训练批次数据"""
        input = batch["img"].to(self.device)
        label = batch["label"].to(self.device)
        return input, label

    def forward_backward(self, batch):
        """前向传播和反向传播"""
        input, label = self.parse_batch_train(batch)
        output = self.model.forward_supervised(input)
        loss = F.cross_entropy(output, label)
        self.model_backward_and_update(loss)
        
        loss_summary = {
            "loss": loss.item()
        }
        
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        
        return loss_summary

    def load_model(self, directory, epoch=None):
        """加载预训练模型"""
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
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
            self._models[name].load_state_dict(state_dict, strict=False)


@TRAINER_REGISTRY.register()
class HybridCascadeLaFTer(BaseLaFTerTrainer):
    """
    HybridCascade-LaFTer训练器：
    结合CascadeCLIP稳定特征提取与LaFTer文本分类优势的混合架构
    """
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_hybrid_cascade_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            clip_model.float()

        print("Building HybridCascade-LaFTer model")
        
        # 获取数据集模板
        dataset_templates = ds_specific_templates.get(cfg.DATASET.NAME, ['a photo of a {}'])
        
        self.model = HybridCascadeLaFTerUFT(
            model=clip_model,
            classes=classnames,
            templates=['a photo of a {}'],
            ds_templates=dataset_templates,
            dataset_name=cfg.DATASET.NAME,
            txt_cls=getattr(cfg, 'txt_cls', 'lafter'),
            cfg=cfg
        )

        print("# params: {:,}".format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))
        
        self.register_model("adapt", self.model)
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        """前向传播和反向传播 - 支持HybridCascade特殊损失"""
        input, label = self.parse_batch_train(batch)
        
        # HybridCascade前向传播
        final_logits, consistency_loss = self.model(input, label)
        
        # 分类损失
        classification_loss = F.cross_entropy(final_logits, label)
        
        # 总损失：分类损失 + 一致性损失
        total_loss = classification_loss + consistency_loss
        
        self.model_backward_and_update(total_loss)
        
        loss_summary = {
            "loss": total_loss.item(),
            "cls_loss": classification_loss.item(),
            "consistency_loss": consistency_loss.item()
        }
        
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        
        return loss_summary

    @torch.no_grad()
    def test(self, split=None):
        """测试方法"""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def model_inference(self, input):
        """模型推理"""
        return self.model.forward_supervised(input)

    def parse_batch_test(self, batch):
        input = batch["img"].to(self.device)
        label = batch["label"].to(self.device)
        return input, label