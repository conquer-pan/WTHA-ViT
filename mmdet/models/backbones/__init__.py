# Copyright (c) OpenMMLab. All rights reserved.
from .csp_darknet import CSPDarknet
from .darknet import Darknet
from .detectors_resnet import DetectoRS_ResNet
from .detectors_resnext import DetectoRS_ResNeXt
from .efficientnet import EfficientNet
from .hourglass import HourglassNet
from .hrnet import HRNet
from .mobilenet_v2 import MobileNetV2
from .pvt import PyramidVisionTransformer, PyramidVisionTransformerV2
from .regnet import RegNet
from .res2net import Res2Net
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
from .swin import SwinTransformer
from .trident_resnet import TridentResNet
from .wavevit import wavevit_s, wavevit_b, wavevit_l
from .WTHA_ViT import WTHA_ViT_s, WTHA_ViT_m, WTHA_ViT_l, WTHA_ViT_s_all, WTHA_ViT_s_root, WTHA_ViT_s_network, WTHA_ViT_s_network2, WTHA_ViT_s_network3, WTHA_ViT_s_similarity, WTHA_ViT_s_noidwt, WTHA_ViT_s_nolifting_noidwt, WTHA_ViT_s_nolifting_nowavetree_noidwt, WTHA_ViT_s_nolifting, WTHA_ViT_s_nowavetree_noidwt
from .gvt import alt_gvt_small, alt_gvt_base, alt_gvt_large, pcpvt_small, pcpvt_base, pcpvt_large
from .spectformer import spectformer_s, spectformer_b, spectformer_l
from .GFNet import GFNet_t, GFNet_s, GFNet_b
from .AFNO import AFNO_t, AFNO_s, AFNO_b
from .dat import DAT
from .uniformer import UniFormer
from .lit import LIT
from .lit_ti import LIT_Ti

__all__ = [
    'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet',
    'MobileNetV2', 'Res2Net', 'HourglassNet', 'DetectoRS_ResNet',
    'DetectoRS_ResNeXt', 'Darknet', 'ResNeSt', 'TridentResNet', 'CSPDarknet',
    'SwinTransformer', 'PyramidVisionTransformer',
    'PyramidVisionTransformerV2', 'EfficientNet',
    'wavevit_s', 'wavevit_b', 'wavevit_l',
    'WTHA_ViT_s', 'WTHA_ViT_m', 'WTHA_ViT_l',
    'WTHA_ViT_s_all', 'WTHA_ViT_s_network', 'WTHA_ViT_s_similarity', 'WTHA_ViT_s_root',
    'WTHA_ViT_s_noidwt', 'WTHA_ViT_s_nolifting_noidwt', 'WTHA_ViT_s_nolifting_nowavetree_noidwt',
    'WTHA_ViT_s_nolifting', 'WTHA_ViT_s_nowavetree_noidwt', 'WTHA_ViT_s_network2', 'WTHA_ViT_s_network3',
    'alt_gvt_small', 'alt_gvt_base', 'alt_gvt_large', 'pcpvt_small', 'pcpvt_base', 'pcpvt_large', 
    'spectformer_s', 'spectformer_b', 'spectformer_l',
    'GFNet_t', 'GFNet_s', 'GFNet_b',
    'AFNO_t', 'AFNO_s', 'AFNO_b', 
    'DAT', 'UniFormer',
    'LIT', 'LIT_Ti'
]
