# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2020 Ross Wightman
# Modified Model definition

import torch
import torch.nn as nn
from functools import partial
import math
import warnings
import torch.nn.functional as F
import numpy as np

from timesformer.models.vit_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timesformer.models.helpers import load_pretrained
from timesformer.models.vit_utils import DropPath, to_2tuple, trunc_normal_

from .build import MODEL_REGISTRY
from torch import einsum
from einops import rearrange, reduce, repeat

# 加载预训练
def _cfg(url='', **kwargs):
    '''
    url: 预训练权重文件下载地址
    num_classes: 模型分类头的默认类别数
    input_size: 模型标准输入张量的尺寸:(C,H,W)
    pool_size: 全局池化层的输出尺寸
    crop_pct: 推理阶段的中心裁剪比例，先将图像短边resize到224/0.9=249，再中心裁剪出224*224输入模型
    interpolation: 'bicubic': resize时的插值算法
    mean/std: ImageNet数据集的均值和标准差
    first_conv: 'patch_embed.proj': 模型第一个卷积层的属性路径
    classifier:head :模型分类头的属性路径
    '''
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

# 为 ViT-B/16 模型注册预训练配置
default_cfgs = {
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}

# FFN模块
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
# 多头注意力
class Attention(nn.Module):
    '''
    dim: 输入特征的总通道维度
    num_heads: 注意力头数量
    qkv_bias: QKV线性投影层是否添加偏置项
    qk_scale: Q、K点积后的自定义缩放因子，等于None时自动使用标准缩放系数(1/sqrt(head_dim))
    attn_drop: 注意力权重矩阵的Dropout概率
    proj_drop: 输出投影层后的Dropout概率
    with_qkv: 是否包含QKV线性投影层和输出投影层；False时仅保留纯注意力计算逻辑
    '''
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        '''
        B: 批次大小
        N: token总数
        C: 特征总维度
        '''
        B, N, C = x.shape
        if self.with_qkv:
           # 完整自注意力
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)      #调整顺序为[3, B, num_heads, N, head_dim]
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           # 纯注意力计算
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v  = qkv, qkv, qkv
        # 计算注意力权重
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # 加权求和与输出还原
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x
# 核心模块
class Block(nn.Module):
    # 输入输出尺寸： [B, 1+T*H*W, C]
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type='divided_space_time'):
        '''
        dim: 特征嵌入总维度
        num_heads: 注意力头数
        mlp_ratio: MLP 隐藏层维度相对输入的倍率
        attention_type: 时空注意力模式三选一
        '''
        super().__init__()
        self.attention_type = attention_type
        assert(attention_type in ['divided_space_time', 'space_only','joint_space_time'])

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
           dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        ## Temporal Attention Parameters--时间注意力分支
        if self.attention_type == 'divided_space_time':
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = Attention(
              dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.temporal_fc = nn.Linear(dim, dim)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x, B, T, W):
        # [B, 1+T*H*W, C]
        num_spatial_tokens = (x.size(1) - 1) // T
        H = num_spatial_tokens // W

        if self.attention_type in ['space_only', 'joint_space_time']:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        elif self.attention_type == 'divided_space_time':
            ## Temporal--对每个空间位置，让它在所有时间帧的 token 之间做自注意力，实现帧间信息融合。
            xt = x[:,1:,:]    #去掉 cls token,仅保留空间 patch token,[B, T*H*W, C]
            xt = rearrange(xt, 'b (h w t) m -> (b h w) t m',b=B,h=H,w=W,t=T)      #[B*H*W, T, C]
            res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
            res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m',b=B,h=H,w=W,t=T)     #[B, T*H*W, C]
            res_temporal = self.temporal_fc(res_temporal)
            xt = x[:,1:,:] + res_temporal

            ## Spatial--对每一帧，让帧内所有空间 token 做自注意力，实现帧内空间信息融合。
            init_cls_token = x[:,0,:].unsqueeze(1)      # 取出原始 cls token  [B, 1, C]
            cls_token = init_cls_token.repeat(1, T, 1)  # 将cls token复制T份,每帧对应一个cls token [B, T, C]
            cls_token = rearrange(cls_token, 'b t m -> (b t) m',b=B,t=T).unsqueeze(1)    #合并batch和时间维度，得到每帧的cls token [B*T, 1, C]
            xs = xt
            xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m',b=B,h=H,w=W,t=T)   #对时间注意力后的patch token做维度重排：合并batch和时间维度，空间维度作为token维度 [B*T, H*W, C]
            xs = torch.cat((cls_token, xs), 1)      # 拼接cls token和空间token，得到标准ViT输入格式 [B*T, 1+H*W, C]
            res_spatial = self.drop_path(self.attn(self.norm1(xs)))

            ### Taking care of CLS token
            cls_token = res_spatial[:,0,:]     # 取出每帧的 cls token
            cls_token = rearrange(cls_token, '(b t) m -> b t m',b=B,t=T)       # 拆分回 [B, T, C]
            cls_token = torch.mean(cls_token,1,True) # 对T帧的cls token取平均,得到全局cls token [B, 1, C]
            res_spatial = res_spatial[:,1:,:]
            res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m',b=B,h=H,w=W,t=T)    # 去掉空间输出中的cls token，重排回原始时空格式 [B, T*H*W, C]
            res = res_spatial
            x = xt

            ## Mlp
            x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)   #分别拼接原始cls+原始patch、处理后cls+处理后patch，做残差相加，形状还原 [B, 1+T*H*W, C]
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)         # 转换成(H,W)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.proj(x)
        W = x.size(-1)
        x = x.flatten(2).transpose(1, 2)
        return x, T, W


class VisionTransformer(nn.Module):
    """ Vision Transformere
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, hybrid_backbone=None, norm_layer=nn.LayerNorm, num_frames=8, attention_type='divided_space_time', dropout=0.):
        super().__init__()
        self.attention_type = attention_type
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        ## Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        if self.attention_type != 'space_only':
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=drop_rate)

        ## Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, attention_type=self.attention_type)
            for i in range(self.depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        ## initialization of temporal attention weights
        if self.attention_type == 'divided_space_time':
            i = 0
            for m in self.blocks.modules():
                m_str = str(m)
                if 'Block' in m_str:
                    if i > 0:
                      nn.init.constant_(m.temporal_fc.weight, 0)
                      nn.init.constant_(m.temporal_fc.bias, 0)
                    i += 1

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x, T, W = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        ## 当输入token数量 和 预设位置编码长度不一致时，执行缩放逻辑
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed      # 原始位置编码: shape [1, num_tokens, embed_dim]
            # 1. 单独提取 CLS token 对应的位置编码（固定不变，不插值）
            # pos_embed[0,0,:] 取batch=0、cls位置、全维度
            # unsqueeze 两次补维度 -> [1, 1, embed_dim]
            cls_pos_embed = pos_embed[0,0,:].unsqueeze(0).unsqueeze(1)
            # 2. 提取除cls外的图像块位置编码: [1, num_patches, embed_dim]
            other_pos_embed = pos_embed[0,1:,:].unsqueeze(0).transpose(1, 2)
            # 3. 计算原始图像块的边长 P (训练时是标准正方形分块)
            P = int(other_pos_embed.size(2) ** 0.5)
            # 新分辨率下特征图高、宽
            H = x.size(1) // W
            # 4.一维位置编码 reshape 回二维特征图 [1, embed_dim, P, P]
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            # 5. 插值缩放至新尺寸 (H, W)，mode='nearest' 最近邻插值
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
            # 6. 展平空间维度：[1, embed_dim, H*W]
            new_pos_embed = new_pos_embed.flatten(2)
            # 7. 转回 Transformer 格式: [1, H*W, embed_dim]
            new_pos_embed = new_pos_embed.transpose(1, 2)
            # 8. 拼接 cls 位置编码 + 新图像块位置编码
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            # 特征 + 缩放后的位置编码
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)


        ## 仅在不是纯空间注意力时，启用时间编码（时序建模）
        if self.attention_type != 'space_only':
            # 1. 单独剥离 CLS token：全局分类标记，不参与时序变换
            cls_tokens = x[:B, 0, :].unsqueeze(1)          # shape: [B, 1, m]
            # 剔除 cls，只保留所有空间patch特征
            x = x[:,1:]                                    # shape: [B*T, n, m]
            # 2. 维度重排：把「帧维度」单独拆出来，适配时间编码
            x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)
            ## Resizing time embeddings in case they don't match
            if T != self.time_embed.size(1):
                time_embed = self.time_embed.transpose(1, 2)
                new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
                new_time_embed = new_time_embed.transpose(1, 2)
                x = x + new_time_embed
            else:
                x = x + self.time_embed
            x = self.time_drop(x)
            x = rearrange(x, '(b n) t m -> b (n t) m',b=B,t=T)
            x = torch.cat((cls_tokens, x), dim=1)

        ## Attention blocks
        for blk in self.blocks:
            x = blk(x, B, T, W)

        ### Predictions for space-only baseline
        if self.attention_type == 'space_only':
            x = rearrange(x, '(b t) n m -> b t n m',b=B,t=T)
            x = torch.mean(x, 1) # averaging predictions for every frame

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            if v.shape[-1] != patch_size:
                patch_size = v.shape[-1]
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

@MODEL_REGISTRY.register()
class vit_base_patch16_224(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(vit_base_patch16_224, self).__init__()
        self.pretrained=True
        patch_size = 16
        self.model = VisionTransformer(img_size=cfg.DATA.TRAIN_CROP_SIZE, num_classes=cfg.MODEL.NUM_CLASSES, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=cfg.DATA.NUM_FRAMES, attention_type=cfg.TIMESFORMER.ATTENTION_TYPE, **kwargs)

        self.attention_type = cfg.TIMESFORMER.ATTENTION_TYPE
        self.model.default_cfg = default_cfgs['vit_base_patch16_224']
        self.num_patches = (cfg.DATA.TRAIN_CROP_SIZE // patch_size) * (cfg.DATA.TRAIN_CROP_SIZE // patch_size)
        pretrained_model=cfg.TIMESFORMER.PRETRAINED_MODEL
        if self.pretrained:
            load_pretrained(self.model, num_classes=self.model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter, img_size=cfg.DATA.TRAIN_CROP_SIZE, num_patches=self.num_patches, attention_type=self.attention_type, pretrained_model=pretrained_model)

    def forward(self, x):
        x = self.model(x)
        return x

@MODEL_REGISTRY.register()
class TimeSformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=400, num_frames=8, attention_type='divided_space_time',  pretrained_model='', **kwargs):
        super(TimeSformer, self).__init__()
        self.pretrained=True
        self.model = VisionTransformer(img_size=img_size, num_classes=num_classes, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=num_frames, attention_type=attention_type, **kwargs)

        self.attention_type = attention_type
        self.model.default_cfg = default_cfgs['vit_base_patch'+str(patch_size)+'_224']
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        if self.pretrained:
            load_pretrained(self.model, num_classes=self.model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter, img_size=img_size, num_frames=num_frames, num_patches=self.num_patches, attention_type=self.attention_type, pretrained_model=pretrained_model)
    def forward(self, x):
        x = self.model(x)
        return x
