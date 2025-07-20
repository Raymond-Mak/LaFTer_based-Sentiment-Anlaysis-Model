"""
LaFTer训练器的共享组件
包含公共的函数和配置
"""
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
    """加载CLIP模型到CPU"""
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


class BaseLaFTerTrainer(TrainerX):
    """LaFTer训练器的基类，包含共同的方法"""

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

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

    def parse_batch_train(self, batch):
        """解析训练批次数据"""
        input = batch["img"].to(self.device)  # 不再是列表，直接是单一图像张量
        label = batch["label"].to(self.device)
        return input, label

    def forward_backward(self, batch):
        """前向传播和反向传播 - 基础实现"""
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