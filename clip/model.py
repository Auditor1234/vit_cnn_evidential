from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float16))
        return ret.type(orig_type)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head) # 768 12
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
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class EMGBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, (3, 1), padding=(1, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d((stride, 1)) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * EMGBottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d((stride, 1))),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class EMGAttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC shape(26,16,2048)
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
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
        ) # shape(1,16,2048)
        return x.squeeze(0)


class EMGModifiedResNet1D(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    # input shape(B, 8, 400, 1)
    def __init__(self, layers, output_dim, width=64):
        super().__init__()
        self.output_dim = output_dim

        # the 3-layer stem
        self.conv1 = nn.Conv2d(8, width // 2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), bias=False) # shape(B,32,200,1)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), bias=False) # shape(B,32,200,1)
        self.bn2 = nn.BatchNorm2d(width)
        self.relu2 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d((2, 1)) # shape(B,64,100,1)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0]) # shape(B,256,100,1)
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2) # shape(B,512,50,1)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=1) # shape(B,1024,25,1)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=1) # shape(B,2048,25,1)

        embed_dim = width * 8  # the ResNet feature dimension
        self.attnpool = EMGAttentionPool2d(25, embed_dim, 16, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [EMGBottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * EMGBottleneck.expansion
        for _ in range(1, blocks):
            layers.append(EMGBottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x))) # shape(B,32,200,1)
            x = self.relu2(self.bn2(self.conv2(x))) # shape(B,32,200,1)
            x = self.avgpool(x) # shape(B,64,100,1)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x) # shape(B,256,100,1)
        x = self.layer2(x) # shape(B,512,50,1)
        x = self.attnpool(x) # shape(B,512)

        return x


class EMGModifiedResNet2D(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    # input shape(B, 1, 400, 8)
    def __init__(self, layers, output_dim, width=64):
        super().__init__()
        self.output_dim = output_dim

        # the 3-layer stem
        self.conv1 = nn.Conv2d(1, width // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False) # shape(B,32,200,4)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False) # shape(B,64,100,2)
        self.bn2 = nn.BatchNorm2d(width)
        self.relu2 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d((2, 2)) # shape(B,64,50,1)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0]) # shape(B,256,50,1)
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2) # shape(B,512,50,1)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=1) # shape(B,1024,25,1)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=1) # shape(B,2048,25,1)

        embed_dim = width * 8  # the ResNet feature dimension
        self.attnpool = EMGAttentionPool2d(25, embed_dim, 32, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [EMGBottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * EMGBottleneck.expansion
        for _ in range(1, blocks):
            layers.append(EMGBottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x))) # shape(B,32,200,4)
            x = self.relu2(self.bn2(self.conv2(x))) # shape(B,32,200,4)
            x = self.avgpool(x) # shape(B,64,100,2)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x) # shape(B,256,100,1)
        x = self.layer2(x) # shape(B,512,50,1)
        x = self.attnpool(x) # shape(B,1024)

        return x


class EMGVisionTransformer1D(nn.Module): # 400      8             512         12           12            512
    # input shape(B,8,400,1)
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=8, out_channels=width, kernel_size=(patch_size, 1), stride=(patch_size, 1), bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        # shape(B,8,400,1)
        x = self.conv1(x)  # shape(B,width,50,1)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape(B,width,50)
        x = x.permute(0, 2, 1)  # shape(B,50,width)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape(B,50 + 1,width)
        x = x + self.positional_embedding.to(x.dtype) # shape(B,51,width)
        x = self.ln_pre(x) # shape(B,51,width)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x) # shape(51,B,width)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :]) # shape(B,width)

        if self.proj is not None:
            x = x @ self.proj

        return x # shape(B,512)


class EMGVisionTransformer2D(nn.Module): # 400      8             512         12           12            512
    # input shape(B,1,400,8)
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=width, kernel_size=(patch_size, patch_size), stride=(patch_size, 1), bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        # shape(B,8,400,1)
        x = self.conv1(x)  # shape(B,width,50,1)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape(B,width,50)
        x = x.permute(0, 2, 1)  # shape(B,50,width)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape(B,50 + 1,width)
        x = x + self.positional_embedding.to(x.dtype) # shape(B,51,width)
        x = self.ln_pre(x) # shape(B,51,width)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x) # shape(51,B,width)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :]) # shape(B,width)

        if self.proj is not None:
            x = x @ self.proj

        return x # shape(1,512)


class EMGCLIP(nn.Module):
    def __init__(self,
                 embed_dim: int, # 512
                 # text
                 context_length: int, # 77
                 vocab_size: int, # 49408
                 transformer_width: int, # 512
                 transformer_heads: int, # 8
                 transformer_layers: int, # 12
                 classification: bool,
                 model_dim=2
                 ):
        super().__init__()

        self.context_length = context_length
        self.classification = classification

        # signal
        if model_dim == 1:
            self.visual1 = EMGVisionTransformer1D(
                input_resolution=400,
                patch_size=8, # 8
                width=64,
                layers=12, # 12
                heads=32, # 32
                output_dim=embed_dim
            )
            self.visual2 = EMGModifiedResNet1D(
                layers=(3,4,6,4), # (3, 4, 6, 4)
                output_dim=embed_dim,
                width=64
            )
        else:
            self.visual1 = EMGVisionTransformer2D(
                input_resolution=400,
                patch_size=8,
                width=64,
                layers=12,
                heads=32,
                output_dim=embed_dim
            )
            self.visual2 = EMGModifiedResNet2D(
                layers=(3,4,6,4), # (3, 4, 6, 4)
                output_dim=embed_dim,
                width=64
            )
            

        # text
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width) # shape(49408,512)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width)) # shape(77,512)
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.dropout = nn.Dropout()
        self.softplus = nn.Softplus()
        self.class_projection1 = nn.Linear(embed_dim, 10)
        self.class_projection2 = nn.Linear(embed_dim, 10)

        self.softmax = nn.Softmax(dim=1)
        self.weight = nn.Parameter(torch.empty(1, 2))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02) # 完成初始化，随机赋值
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.weight, std=0.01)

        if isinstance(self.visual2, (EMGModifiedResNet1D, EMGModifiedResNet2D)):
            if self.visual2.attnpool is not None:
                std = self.visual2.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual2.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual2.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual2.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual2.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual2.layer1, self.visual2.layer2, self.visual2.layer3, self.visual2.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual2.conv1.weight.dtype

    def encode_image(self, image):
        image = image.type(self.dtype)
        return self.visual1(image), self.visual2(image)

    def encode_text(self, text): # shape(3,77)
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model] shape(3,77,512)

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype) # shape(3,77,512)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection # (3,512) -> (3,512)

        return x

    def forward(self, image, text):
        features1, features2 = self.encode_image(image) # shape(B,8,400,1)
        evidence1 = self.softplus(self.class_projection1(self.dropout(features1)))
        evidence2 = self.softplus(self.class_projection2(self.dropout(features2)))
        if self.classification:
            return evidence1, evidence2
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True) # shape(1,512)
        text_features = text_features / text_features.norm(dim=1, keepdim=True) # shape(3,512)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text
    
    def forward_moe(self, image):
        features1, features2 = self.encode_image(image) # shape(B,8,400,1)
        evidence1 = self.softplus(self.class_projection1(self.dropout(features1)))
        evidence2 = self.softplus(self.class_projection2(self.dropout(features2)))
        
        prediction1 = self.softmax(evidence1)
        prediction2 = self.softmax(evidence2)
        weight = self.softmax(self.weight)

        prediction = prediction1 * weight[0][0] + prediction2 * weight[0][1]
        return prediction
