import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
import numpy as np
# from pytorch_wavelets import DWTForward, DWTInverse # (or import DWT, IDWT)
from mmdet.utils import get_root_logger
from mmcv.runner import load_checkpoint
from mmcv.cnn import build_norm_layer
from mmdet.models.builder import BACKBONES

class SpectralGatingNetwork(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # this weights are valid for h=14 and w=8
        if dim in [64, 96]: #96 for large model, 64 for small and base model (1024,1024)
            # self.h = 256#56 #H
            # self.w = 129#29 #(W/2)+1      
            self.h = 208
            self.w = 105        
            self.complex_weight = nn.Parameter(torch.randn(self.h, self.w, dim, 2, dtype=torch.float32) * 0.02)
        if dim in [128, 192]:
            # self.h = 128#28 #H
            # self.w = 65#15 #(W/2)+1, this is due to rfft2
            self.h = 104
            self.w = 53  
            self.complex_weight = nn.Parameter(torch.randn(self.h, self.w, dim, 2, dtype=torch.float32) * 0.02)
        if dim in [256, 384]: #96 for large model, 64 for small and base model (832,832)
            # self.h = 64 #H
            # self.w = 33 #(W/2)+1        
            self.h = 52
            self.w = 27      
            self.complex_weight = nn.Parameter(torch.randn(self.h, self.w, dim, 2, dtype=torch.float32) * 0.02)
        if dim in [512, 768]:
            # self.h = 32 #H
            # self.w = 17 #(W/2)+1, this is due to rfft2
            self.h = 26
            self.w = 14  
            self.complex_weight = nn.Parameter(torch.randn(self.h, self.w, dim, 2, dtype=torch.float32) * 0.02)

    def forward(self, x, H, W):
        # print('wno',x.shape) #CIFAR100 image :[128, 196, 384]
        B, N, C = x.shape 
        # print('wno B, N, C',B, N, C) #CIFAR100 image : 128 196 384
        x = x.view(B, H, W, C)
        # print("x.shape:", x.shape)
        # B, H, W, C=x.shape
        x = x.to(torch.float32) 
        # print(x.dtype)
        # Add above for this error, RuntimeError: Input type (torch.cuda.HalfTensor) and weight type (torch.cuda.FloatTensor) should be the same
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        # x = torch.fft.rfft2(x, dim=(1, 2))
        # print('wno',x.shape)
        weight = torch.view_as_complex(self.complex_weight)
        # print('weight',weight.shape)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')
        # print('wno',x.shape)
        x = x.reshape(B, N, C)# permute is not same as reshape or view
        return x


# class ClassAttention(nn.Module):
#     def __init__(self, dim, num_heads):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.head_dim = head_dim
#         self.scale = head_dim**-0.5
#         self.kv = nn.Linear(dim, dim * 2)
#         self.q = nn.Linear(dim, dim)
#         self.proj = nn.Linear(dim, dim)
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     def forward(self, x):
#         B, N, C = x.shape
#         kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         k, v = kv[0], kv[1]
#         q = self.q(x[:, :1, :]).reshape(B, self.num_heads, 1, self.head_dim)
#         attn = ((q * self.scale) @ k.transpose(-2, -1))
#         attn = attn.softmax(dim=-1)
#         cls_embed = (attn @ v).transpose(1, 2).reshape(B, 1, self.head_dim * self.num_heads)
#         cls_embed = self.proj(cls_embed)
#         return cls_embed

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

# class ClassBlock(nn.Module):
#     def __init__(self, dim, num_heads, mlp_ratio, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.norm2 = norm_layer(dim)
#         self.attn = ClassAttention(dim, num_heads)
#         self.mlp = FFN(dim, int(dim * mlp_ratio))
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     def forward(self, x):
#         cls_embed = x[:, :1]
#         cls_embed = cls_embed + self.attn(self.norm1(x))
#         cls_embed = cls_embed + self.mlp(self.norm2(cls_embed))
#         return torch.cat([cls_embed, x[:, 1:]], dim=1)

class PVT2FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.fc2(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self, 
        dim, 
        # num_heads, 
        mlp_ratio,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        # sr_ratio=1, 
        block_type = 'wave'
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        # if block_type == 'std_att':
        #     self.attn = Attention(dim, num_heads)
        # else:
        self.attn = SpectralGatingNetwork (dim)
        self.mlp = PVT2FFN(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

class DownSamples(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(out_channels)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class Stem(nn.Module):
    def __init__(self, in_channels, stem_hidden_dim, out_channels):
        super().__init__()
        hidden_dim = stem_hidden_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=7, stride=2,
                      padding=3, bias=False),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                      padding=1, bias=False),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                      padding=1, bias=False),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(hidden_dim,
                              out_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1)
        self.norm = nn.LayerNorm(out_channels)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class GFNet(nn.Module):
    def __init__(self, 
        in_chans=3, 
        num_classes=1, #########################################################
        stem_hidden_dim = 32,
        embed_dims=[64, 128, 320, 448],
        # num_heads=[2, 4, 10, 14], 
        mlp_ratios=[8, 8, 4, 4], 
        drop_path_rate=0., 
        norm_layer=nn.LayerNorm,
        depths=[3, 4, 6, 3], 
        # sr_ratios=[4, 2, 1, 1], 
        num_stages=4,
        pretrained=None,
        **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = Stem(in_chans, stem_hidden_dim, embed_dims[i])
            else:
                patch_embed = DownSamples(embed_dims[i - 1], embed_dims[i])

            block = nn.ModuleList([Block(
                dim = embed_dims[i], 
                # num_heads = num_heads[i], 
                mlp_ratio = mlp_ratios[i], 
                drop_path=dpr[cur + j], 
                norm_layer=norm_layer,
                # sr_ratio = sr_ratios[i],
                block_type='wave' if i < 2 else 'std_att')
            for j in range(depths[i])])

            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # post_layers = ['ca']
        # self.post_network = nn.ModuleList([ # 最后一层网络
        #     ClassBlock(
        #         dim = embed_dims[-1], 
        #         num_heads = num_heads[-1], 
        #         mlp_ratio = mlp_ratios[-1],
        #         norm_layer=norm_layer)
        #     for _ in range(len(post_layers))
        # ])

        # classification head
        # self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        ##################################### token_label #####################################

        ##################################### token_label #####################################

        self.apply(self._init_weights)
        self.init_weights(pretrained)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def forward_features(self, x):
        B = x.shape[0]
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            norm = getattr(self, f"norm{i + 1}")
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()            
            # print("xx1.shape:", x.shape)
            outs.append(x)            
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

@BACKBONES.register_module()
class GFNet_t(GFNet):
    def __init__(self, **kwargs):
        super(GFNet_t, self).__init__(
            stem_hidden_dim = 32,
            # embed_dims = [64, 128, 320, 512], 
            embed_dims = [64, 128, 256, 512],
            # num_heads = [2, 4, 10, 16], 
            # num_heads = [2, 4, 8, 16], 
            # mlp_ratios = [8, 8, 4, 4],
            mlp_ratios = [4, 4, 4, 4],
            norm_layer = partial(nn.LayerNorm, eps=1e-6), 
            # depths = [3, 4, 12, 3], 
            depths = [3, 3, 10, 3],
            # sr_ratios = [4, 2, 1, 1], 
            **kwargs)

# @register_model
@BACKBONES.register_module()
class GFNet_s(GFNet):
    def __init__(self, **kwargs):
        super(GFNet_s, self).__init__(
            stem_hidden_dim = 64,
            embed_dims = [96, 192, 384, 768], 
            # num_heads = [3, 6, 12, 24], 
            mlp_ratios = [8, 8, 4, 4],
            norm_layer = partial(nn.LayerNorm, eps=1e-6), 
            depths = [3, 3, 10, 3], 
            # stem_hidden_dim = 64,
            # embed_dims = [64, 128, 320, 448], 
            # num_heads = [2, 4, 10, 14], 
            # mlp_ratios = [8, 8, 4, 4],
            # norm_layer = partial(nn.LayerNorm, eps=1e-6), 
            # depths = [3, 3, 10, 3], 
            # sr_ratios = [4, 2, 1, 1], 
            **kwargs)


# @register_model
@BACKBONES.register_module()
class GFNet_b(GFNet):
    def __init__(self, **kwargs):
        super(GFNet_b, self).__init__(
            stem_hidden_dim = 64,
            embed_dims = [96, 192, 384, 768], 
            # num_heads = [3, 6, 12, 24], 
            mlp_ratios = [8, 8, 4, 4],
            norm_layer = partial(nn.LayerNorm, eps=1e-6), 
            depths = [3, 3, 27, 3],            
            # stem_hidden_dim = 64,
            # embed_dims = [96, 192, 384, 512], 
            # num_heads = [3, 6, 12, 16], 
            # mlp_ratios = [8, 8, 4, 4],
            # norm_layer = partial(nn.LayerNorm, eps=1e-6), 
            # depths = [3, 6, 18, 3], 
            # sr_ratios = [4, 2, 1, 1], 
            **kwargs)   
