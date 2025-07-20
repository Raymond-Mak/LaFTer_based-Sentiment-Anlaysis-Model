from collections import OrderedDict
from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        
        # 确保MultiHeadAttention的所有权重参数与输入x的数据类型一致
        if hasattr(self.attn, 'in_proj_weight') and self.attn.in_proj_weight is not None:
            if self.attn.in_proj_weight.dtype != x.dtype:
                # 将注意力层的权重转换为与输入相同的数据类型
                self.attn = self.attn.to(dtype=x.dtype)
        
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

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
        
        # MLP分支
        mlp_input = self.ln_2(x)
        if mlp_input.dtype != target_dtype:
            mlp_input = mlp_input.to(dtype=target_dtype)
        
        # 确保MLP层权重与输入数据类型一致
        if hasattr(self.mlp[0], 'weight') and self.mlp[0].weight.dtype != target_dtype:
            self.mlp = self.mlp.to(dtype=target_dtype)
        
        mlp_output = self.mlp(mlp_input)
        if mlp_output.dtype != target_dtype:
            mlp_output = mlp_output.to(dtype=target_dtype)
        
        x = x + mlp_output
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


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

    def forward(self, x: torch.Tensor, multi_prompts=None):
        """
        Args:
            x: 输入特征 [seq_len, batch_size, dim]
            multi_prompts: 多层prompts列表，每个元素形状为 [batch_size, num_tokens, dim]
        """
        stage_outputs = []
        current_x = x
        
        for stage_idx, layer_indices in enumerate(self.stage_boundaries):
            # 在每个stage开始前注入prompt（除了第一个stage，已在外部处理）
            if stage_idx > 0 and multi_prompts is not None:
                current_x = self._inject_prompt_at_stage(current_x, multi_prompts[stage_idx])
            
            # 确保current_x的数据类型在整个stage过程中保持一致
            target_dtype = current_x.dtype
            
            # 通过该stage的所有层
            stage_features = []
            for layer_idx in layer_indices:
                # 在每层前检查数据类型
                if current_x.dtype != target_dtype:
                    current_x = current_x.to(dtype=target_dtype)
                
                current_x = self.resblocks[layer_idx](current_x)
                
                # 确保输出的数据类型与输入保持一致
                if current_x.dtype != target_dtype:
                    current_x = current_x.to(dtype=target_dtype)
                
                # 收集该stage内每层的CLS token特征用于NGA
                cls_feature = current_x[0]  # [batch_size, dim]
                stage_features.append(cls_feature)
            
            stage_outputs.append(stage_features)
        
        return current_x, stage_outputs
    
    def _inject_prompt_at_stage(self, x, stage_prompt):
        """在指定stage前注入prompt"""
        # x: [seq_len, batch_size, dim]
        # stage_prompt: [batch_size, num_tokens, dim]
        
        # 确保stage_prompt与x的数据类型和设备一致
        if stage_prompt.dtype != x.dtype:
            stage_prompt = stage_prompt.to(dtype=x.dtype)
        if stage_prompt.device != x.device:
            stage_prompt = stage_prompt.to(device=x.device)
        
        # 转换维度：[batch_size, num_tokens, dim] -> [num_tokens, batch_size, dim]
        prompt_tokens = stage_prompt.permute(1, 0, 2)
        
        # 在CLS token后，patch tokens前插入prompt
        cls_token = x[:1]  # [1, batch_size, dim]
        patch_tokens = x[1:]  # [patch_len, batch_size, dim]
        
        # 拼接：CLS + prompt + patches
        x_with_prompt = torch.cat([cls_token, prompt_tokens, patch_tokens], dim=0)
        
        return x_with_prompt


class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)
        # 添加多层transformer支持
        self.multi_transformer = MultiLayerTransformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        # xavier_uniform initialization
        # nn.init.uniform_(self.prompt_embeddings.data, -val, val)

    def embeddings_patch(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        return x

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj

        return x

    def forward_after_patch_embeddings(self, x: torch.Tensor):
        # seems like the prompt should be here!
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x

    def forward_multi_stage_prompts(self, x: torch.Tensor, multi_prompts=None):
        """支持多层prompt的前向传播"""
        # x: [batch_size, seq_len, dim] 从 embeddings_patch 输出
        x = x.permute(1, 0, 2)  # NLD -> LND for transformer
        
        # 确保multi_prompts与x的数据类型一致
        if multi_prompts is not None:
            for i, prompt in enumerate(multi_prompts):
                if prompt is not None and prompt.dtype != x.dtype:
                    multi_prompts[i] = prompt.to(dtype=x.dtype)
        
        # 使用多层transformer
        final_x, stage_outputs = self.multi_transformer(x, multi_prompts)
        
        # 确保输出数据类型一致
        if final_x.dtype != x.dtype:
            final_x = final_x.to(dtype=x.dtype)
        
        final_x = final_x.permute(1, 0, 2)  # LND -> NLD
        final_features = self.ln_post(final_x[:, 0, :])  # 只取CLS token
        if self.proj is not None:
            # 确保projection权重与特征的数据类型一致
            if self.proj.dtype != final_features.dtype:
                final_features = final_features @ self.proj.to(dtype=final_features.dtype)
            else:
                final_features = final_features @ self.proj
            
        return final_features, stage_outputs

    def forward_after_transformer(self, x: torch.Tensor):
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]))

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)

        # prompt should be here!

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def positional_embeddings(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        # print(x.shape)
        # quit()
        return x

    def embeddings_after_prompting(self, x):

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), x.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module, target_dtype=None):
    """
    Convert applicable model parameters to specified dtype.
    If target_dtype is None, skip conversion to maintain original precision.
    """
    
    # 如果没有指定目标数据类型，跳过转换
    if target_dtype is None:
        return
    
    def _convert_weights(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            if l.weight.dtype != target_dtype:
                l.weight.data = l.weight.data.to(dtype=target_dtype)
            if l.bias is not None and l.bias.dtype != target_dtype:
                l.bias.data = l.bias.data.to(dtype=target_dtype)

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None and tensor.dtype != target_dtype:
                    tensor.data = tensor.data.to(dtype=target_dtype)

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None and attr.dtype != target_dtype:
                    attr.data = attr.data.to(dtype=target_dtype)

    model.apply(_convert_weights)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        del state_dict[key]

    # 不强制转换为fp16，保持原始精度以避免dtype不匹配
    # convert_weights(model, target_dtype=torch.float16)  # 注释掉强制fp16转换
    
    # Fix multi_transformer state_dict loading issue
    # Copy transformer weights to multi_transformer
    new_state_dict = {}
    for key, value in state_dict.items():
        new_state_dict[key] = value
        # If it's a visual transformer weight, also create multi_transformer version
        if key.startswith("visual.transformer."):
            multi_key = key.replace("visual.transformer.", "visual.multi_transformer.")
            # 确保复制的权重保持相同的数据类型
            new_state_dict[multi_key] = value.clone().detach()
    
    # 在加载权重前，确保模型和权重的数据类型一致
    model_dtype = next(model.parameters()).dtype
    for key, value in new_state_dict.items():
        if value.dtype != model_dtype:
            new_state_dict[key] = value.to(dtype=model_dtype)
    
    model.load_state_dict(new_state_dict, strict=False)
    return model.eval()
