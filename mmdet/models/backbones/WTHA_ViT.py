import math
import pywt
import torch
import torch.nn as nn
from torch.autograd import Function
from functools import partial
import numpy as np
import pywt
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mmdet.models.builder import BACKBONES
from mmdet.utils import get_root_logger
from mmcv.runner import load_checkpoint
from mmcv.cnn import build_norm_layer

from mmdet.models.backbones.lifting import LiftingScheme, LiftingScheme2D, LiftingScheme_by_channel, LiftingScheme2D_by_channel
from mmdet.models.backbones.WaveletTransform_filters import IDWT_2D, DWT_2D
# from mmdet.models.backbones.WaveletTree import WaveletTree

""" main code in my paper:《Wavelet Tree-based Head attention in Vision Transformer》.

    The main innovative modification of the article is in WaveTreeHeadAttention:
        1. channel lifting scheme module.
        2. channel lifting scheme in lifting.py including model importance for head att: three methods.
        3. wavelet tree selection use in key/value.
        4. idwt_by_tree

    """

        
class DWConv(nn.Module): #深度可分离卷积（Depthwise separable convolution）
    def __init__(self, dim):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)#深度可分离卷积（Depthwise separable convolution）中的逐通道卷积（Depthwise Convolution）

    def forward(self, x, H, W):
        B, N, C = x.shape#transformer的形状
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class PVT2FFN(nn.Module):#前馈层FFN
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

#by filters but no idwt2D
# class Wavelet(nn.Module):
#     """This module extract wavelet coefficient defined in pywt
#     and create 2D convolution kernels to be able to use GPU"""

#     def _coef_h(self, in_planes, coef):
#         """Construct the weight matrix for horizontal 2D convolution.
#         The weights are repeated on the diagonal"""
#         v = []
#         for i in range(in_planes):
#             l = []
#             for j in range(in_planes):
#                 if i == j:
#                     l.append([[c for c in coef]])
#                 else:
#                     l.append([[0.0 for c in coef]])
#             v.append(l)
#         return v

#     def _coef_v(self, in_planes, coef):
#         """Construct the weight matrix for vertical 2D convolution.
#         The weights are repeated on the diagonal"""
#         v = []
#         for i in range(in_planes):
#             l = []
#             for j in range(in_planes):
#                 if i == j:
#                     l.append([[c] for c in coef])
#                 else:
#                     l.append([[0.0] for c in coef])
#             v.append(l)
#         return v

#     def __init__(self, in_planes, horizontal, name="db2"):
#         super(Wavelet, self).__init__()

#         # Import wavelet coefficients
#         import pywt
#         wavelet = pywt.Wavelet(name)
#         coef_low = wavelet.dec_lo
#         coef_high = wavelet.dec_hi
#         # Determine the kernel 2D shape
#         nb_coeff = len(coef_low)
#         if horizontal:
#             kernel_size = (1, nb_coeff)
#             stride = (1, 2)
#             pad = (nb_coeff // 2, nb_coeff - 1 - nb_coeff // 2, 0, 0)
#             weights_low = self._coef_h(in_planes, coef_low)
#             weights_high = self._coef_h(in_planes, coef_high)
#         else:
#             kernel_size = (nb_coeff, 1)
#             stride = (2, 1)
#             pad = (0, 0, nb_coeff // 2, nb_coeff - 1 - nb_coeff // 2)
#             weights_low = self._coef_v(in_planes, coef_low)
#             weights_high = self._coef_v(in_planes, coef_high)
#         # TODO: Debug prints
#         # print("")
#         # print("Informations: ")
#         # print("- kernel_size: ", kernel_size)
#         # print("- stride     : ", stride)
#         # print("- pad        : ", pad)
#         # print("- low        : ", weights_low)
#         # print("- high       : ", weights_high)

#         # Create the conv2D
#         self.conv_high = nn.Conv2d(
#             in_planes, in_planes, kernel_size=kernel_size, stride=stride, bias=False)
#         self.conv_low = nn.Conv2d(
#             in_planes, in_planes, kernel_size=kernel_size, stride=stride, bias=False)
#         self.padding = nn.ReflectionPad2d(padding=pad)
#         # TODO: Debug prints
#         # print("- low        : ", self.conv_low.weight)
#         # print("- high       : ", self.conv_high.weight)

#         # Replace their weights
#         self.conv_high.weight = torch.nn.Parameter(
#             data=torch.Tensor(weights_high), requires_grad=False)
#         self.conv_low.weight = torch.nn.Parameter(
#             data=torch.Tensor(weights_low), requires_grad=False)

#         # self.apply(self._init_weights)
    
#     # def _init_weights(self, m):
#     #     if isinstance(m, nn.Conv2d):
#     #         fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#     #         fan_out //= m.groups
#     #         m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#     #         if m.bias is not None:
#     #             m.bias.data.zero_()

#     def forward(self, x):
#         '''Returns the approximation and detail part'''
#         x = self.padding(x)
#         return (self.conv_low(x), self.conv_high(x))

# class Wavelet2D(nn.Module):
#     def __init__(self, in_planes, name="db1"):
#         super(Wavelet2D, self).__init__()
#         self.horizontal_wavelet = Wavelet(in_planes, horizontal=True, name=name)
#         self.vertical_wavelet = Wavelet(in_planes, horizontal=False, name=name)

#     def forward(self, x):
#         '''Returns (LL, LH, HL, HH)'''
#         (c, d) = self.horizontal_wavelet(x)
#         (LL, LH) = self.vertical_wavelet(c)
#         (HL, HH) = self.vertical_wavelet(d)
#         return (LL, LH, HL, HH)


# class wav_transform(nn.Module):
#     def __init__(self, wave="haar"):
#         super(wav_transform, self).__init__()
#         self.wave = wave

#     def forward(self, x):
#         x = x.cpu().detach().numpy() 
#         LL, (LH, HL, HH) = pywt.dwt2(x, self.wave)
#         LL = torch.from_numpy(LL)
#         LH = torch.from_numpy(LH)
#         HL = torch.from_numpy(HL)
#         HH = torch.from_numpy(HH)
#         device = torch.device("cuda:0")
#         LL = LL.to(device)
#         LH = LH.to(device)
#         HL = HL.to(device)
#         HH = HH.to(device)
#         return LL, LH, HL, HH

# # by cpu()
# class inverse_wav_transform_full(nn.Module):
#     def __init__(self, wave="db1"):
#         super(inverse_wav_transform_full, self).__init__()
#         self.wave = wave

#     def forward(self, x, LH, HL, HH):
#         # print("X.shape: ", x.shape)
#         x = x.cpu().detach().numpy() 
#         if LH is not None:
#             LH = LH.cpu().detach().numpy()
#         if HL is not None:
#             HL = HL.cpu().detach().numpy()
#         if HH is not None:
#             HH = HH.cpu().detach().numpy()
#         x = pywt.idwt2((x, (LH, HL, HH)), self.wave)   
#         x = torch.from_numpy(x)
#         device = torch.device("cuda:0")
#         x = x.to(device)
#         return x

# class inverse_wav_transform(nn.Module):
#     def __init__(self, wave="db1"):
#         super(inverse_wav_transform, self).__init__()
#         self.wave = wave

#     def forward(self, x):
#         # print("X.shape: ", x.shape)
#         x = x.cpu().detach().numpy() 
#         x = pywt.idwt2((x, (None, None, None)), self.wave)   
#         x = torch.from_numpy(x)
#         device = torch.device("cuda:0")
#         x = x.to(device)
#         return x

class WaveletTree(nn.Module):
    def __init__(self, in_planes, sr_ratio=1, tree_level=8, threshold=1, use_thres = False, method = 'root'):
        super(WaveletTree, self).__init__()
        self.in_planes = in_planes
        self.tree_level = tree_level
        self.sr_ratio = sr_ratio
        self.threshold = threshold
        self.use_thres = use_thres
        self.method = method
        # self.idwt = IDWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave = 'haar')
        # self.idwt = nn.UpsamplingNearest2d(scale_factor=2)
        # self.wavelettree1 = Wavelet2D(in_planes, name="db2")
        # self.wavelettree2 = Wavelet2D(in_planes, name="db2")
        # self.wavelettree3 = Wavelet2D(in_planes, name="db2")
        # self.wavelettree1 = Wavelet2D(in_planes, name="haar")
        # self.wavelettree2 = Wavelet2D(in_planes, name="haar")
        # self.wavelettree3 = Wavelet2D(in_planes, name="haar")
        self.wavelettree1 = DWT_2D(wave="haar")
        self.wavelettree2 = DWT_2D(wave="haar")
        self.wavelettree3 = DWT_2D(wave="haar")


    def forward(self, x):
        # print("sr_ratio: ", self.sr_ratio)
        # print("x: ", x.shape)
        # print("self.method: ", self.method)
        if self.tree_level == 8:
            (LL1, LH1, HL1, HH1) = self.wavelettree1(x)
            # print("LL1: ", LL1.shape)
            (LL2, LH2, HL2, HH2) = self.wavelettree2(LL1)
            # print("LL2: ", LL2.shape)
            (LL3, LH3, HL3, HH3) = self.wavelettree3(LL2)
            # print("LL3: ", LL3.shape)
            # print(LL3.contiguous().view(-1).abs().mean(), LL3.contiguous().view(-1).abs().max())
            # print(LH3.contiguous().view(-1).abs().mean(), LH3.contiguous().view(-1).abs().max())#低频通常要大一些
            # print(HL3.contiguous().view(-1).abs().mean(), HL3.contiguous().view(-1).abs().max())
            # print(HH3.contiguous().view(-1).abs().mean(), HH3.contiguous().view(-1).abs().max())
            # print("LL3")
            
            # print(LL2.contiguous().view(-1).abs().mean(), LL2.contiguous().view(-1).abs().max())
            # print(LH2.contiguous().view(-1).abs().mean(), LH2.contiguous().view(-1).abs().max())#低频通常要大一些
            # print(HL2.contiguous().view(-1).abs().mean(), HL2.contiguous().view(-1).abs().max())
            # print(HH2.contiguous().view(-1).abs().mean(), HH2.contiguous().view(-1).abs().max())
            # print("LL2")
            # print(LL1.contiguous().view(-1).abs().mean(), LL1.contiguous().view(-1).abs().max())
            # print(LH1.contiguous().view(-1).abs().mean(), LH1.contiguous().view(-1).abs().max())#低频通常要大一些
            # print(HL1.contiguous().view(-1).abs().mean(), HL1.contiguous().view(-1).abs().max())
            # print(HH1.contiguous().view(-1).abs().mean(), HH1.contiguous().view(-1).abs().max())
            # print("LL1")
            if self.use_thres == False:
                # std + mean + 小波零树初始化
    
                # threshold =  (LL3.contiguous().view(-1).abs().mean() + pow(2, int(math.log2(LL3.contiguous().view(-1).abs().max()))))/4 # 小波零树初始化阈值
                threshold = 10
            else:
                threshold = self.threshold
            tree = {'LL3': 1, 'LH3': 0, 'HL3': 0, 'HH3': 0, 'LH2': 0, 'HL2': 0, 'HH2': 0, 'LH1': 0, 'HL1': 0, 'HH1': 0} #初始化树
            lh, hl, hh = None, None, None
            if self.method == 'all':
                ## 8
                if self.sr_ratio == 8:
                    # print("thres: ", threshold)
                    if LH3.contiguous().view(-1).abs().max() > threshold:
                        x = LL3 + LH3
                        # print("LH3")
                    else:
                        x = LL3
                    if HL3.contiguous().view(-1).abs().max() > threshold:
                        x = x + HL3
                        # print("HL3")
                    if HH3.contiguous().view(-1).abs().max() > threshold:
                        x = x + HH3
                        # print("HH3")
                    # print("x2: ", x.shape)
                ## 4
                elif self.sr_ratio == 4:
                    # print("thres: ", threshold)
                    # 8
                    if LH3.contiguous().view(-1).abs().max() > threshold:
                        # x = LL3 + LH3
                        # print("LH3")
                        lh = LH3
                    # else:
                    #     x = LL3
                    if HL3.contiguous().view(-1).abs().max() > threshold:
                        # x = x + HL3
                        # print("HL3")
                        hl = HL3
                    if HH3.contiguous().view(-1).abs().max() > threshold:
                        # x = x + HH3
                        # print("HH3")
                        hh = HH3
                    # 4
                    # x = torch.FloatTensor(x)
                    # x = x.half()
                    # x = self.idwt(x) # IDWT
                    x = self.idwt(LL3, lh, hl, hh)
                    threshold = threshold / 2
                    # threshold = (LL2.contiguous().view(-1).abs().mean() * 0.5 + pow(2, int(math.log2(LL2.contiguous().view(-1).abs().max()))))/4 # Self
                    # print("thres: ", threshold)
                    if LH2.contiguous().view(-1).abs().max() > threshold:
                        x = x + LH2
                        # print("LH2")
                    if HL2.contiguous().view(-1).abs().max() > threshold:
                        x = x + HL2
                        # print("HL2")
                    if HH2.contiguous().view(-1).abs().max() > threshold:
                        x = x + HH2    
                        # print("HH2")     
                ## 2
                elif self.sr_ratio == 2:
                    # 8
                    # print("thres: ", threshold)
                    if LH3.contiguous().view(-1).abs().max() > threshold:
                        # x = LL3 + LH3
                        # print("LH3")
                        lh = LH3
                    # else:
                    #     x = LL3
                    if HL3.contiguous().view(-1).abs().max() > threshold:
                        # x = x + HL3
                        # print("HL3")
                        hl = HL3
                    if HH3.contiguous().view(-1).abs().max() > threshold:
                        # x = x + HH3
                        # print("HH3")
                        hh = HH3
                    # 4
                    # x = torch.FloatTensor(x)
                    # x = x.half()
                    # x = self.idwt(x) # IDWT
                    x = self.idwt(LL3, lh, hl, hh)
                    lh, hl, hh = None, None, None
                    threshold = threshold / 2
                    # threshold = LL2.contiguous().view(-1).abs().std() + LL2.contiguous().view(-1).abs().mean() + pow(2, int(math.log2(LL2.contiguous().view(-1).abs().max())))
                    # print("thres: ", threshold)
                    if LH2.contiguous().view(-1).abs().max() > threshold:
                        # x = x + LH2
                        # print("LH2")
                        lh = LH2
                    if HL2.contiguous().view(-1).abs().max() > threshold:
                        # x = x + HL2
                        # print("HL2")
                        hl = HL2
                    if HH2.contiguous().view(-1).abs().max() > threshold:
                        # x = x + HH2   
                        # print("HH2")
                        hh = HH2
                    # 2
                    # x = torch.FloatTensor(x)
                    # x = x.half()
                    # x = self.idwt(x) # IDWT
                    x = self.idwt(x, lh, hl, hh)
                    threshold = threshold / 2
                    # threshold = LL1.contiguous().view(-1).abs().std() + LL1.contiguous().view(-1).abs().mean() + pow(2, int(math.log2(LL1.contiguous().view(-1).abs().max())))
                    # print("thres: ", threshold)
                    if LH1.contiguous().view(-1).abs().max() > threshold:
                        x = x + LH1
                        # print("LH1")
                    if HL1.contiguous().view(-1).abs().max() > threshold:
                        x = x + HL1
                        # print("HL1")
                    if HH1.contiguous().view(-1).abs().max() > threshold:
                        x = x + HH1   
                        # print("HH1")  

                    # if self.sr_ratio == 1:
                    #     x = self.idwt(x) # IDWT
                    #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                else: 
                    x = x                               
                # return (LL1, LH1, HL1, HH1, LH2, HL2, HH2, LH3, HL3, HH3)
                return x
            elif self.method == 'root':
                ## 8
                # threshold =  (LL3.contiguous().view(-1).abs().mean() + pow(2, int(math.log2(LL3.contiguous().view(-1).abs().max()))))/2 # EZW Self
                # threshold = 10 #Fixed
                if self.sr_ratio == 8:
                    # print("thres: ", threshold)
                    if LH3.contiguous().view(-1).abs().max() > threshold:
                        x = LL3 + LH3
                        tree['LH3'] = 1
                        # lh = LH3
                    else:
                        x = LL3
                    if HL3.contiguous().view(-1).abs().max() > threshold:
                        x = x + HL3
                        tree['HL3'] = 1
                        # hl = HL3
                    if HH3.contiguous().view(-1).abs().max() > threshold:
                        x = x + HH3
                        tree['HH3'] = 1
                        # hh = HH3
                    # print("sr=8:", tree)
                ## 4
                elif self.sr_ratio == 4:
                    # 8
                    # print("thres: ", threshold)
                    if LH3.contiguous().view(-1).abs().max() > threshold:
                        # x = LL3 + LH3
                        tree['LH3'] = 1
                        lh = LH3
                    # else:
                    #     x = LL3
                    if HL3.contiguous().view(-1).abs().max() > threshold:
                        # x = x + HL3
                        tree['HL3'] = 1
                        hl = HL3
                    if HH3.contiguous().view(-1).abs().max() > threshold:
                        # x = x + HH3
                        tree['HH3'] = 1
                        hh = HH3
                    # 4
                    x = self.idwt(LL3, lh, hl, hh) # IDWT
                    # x = x.half()
                    # threshold = 7 # Fixed
                    threshold = threshold / 2 #EZW
                    # threshold = (LL2.contiguous().view(-1).abs().mean() * 0.5 + pow(2, int(math.log2(LL2.contiguous().view(-1).abs().max()))))/4 # Self
                    # print("thres2: ", threshold)
                    if LH2.contiguous().view(-1).abs().max() > threshold and tree['LH3'] == 1:
                        x = x + LH2
                        tree['LH2'] = 1
                    if HL2.contiguous().view(-1).abs().max() > threshold and tree['HL3'] == 1:
                        x = x + HL2
                        tree['HL2'] = 1
                    if HH2.contiguous().view(-1).abs().max() > threshold and tree['HH3'] == 1:
                        x = x + HH2   
                        tree['HH2'] = 1
                    # print("sr=4:", tree)
                elif self.sr_ratio == 2:
                    # 8
                    # print("thres: ", threshold)
                    if LH3.contiguous().view(-1).abs().max() > threshold:
                        # x = LL3 + LH3
                        tree['LH3'] = 1
                        lh = LH3
                    # else:
                    #     x = LL3
                    if HL3.contiguous().view(-1).abs().max() > threshold:
                        # x = x + HL3
                        tree['HL3'] = 1
                        hl = HL3
                    if HH3.contiguous().view(-1).abs().max() > threshold:
                        # x = x + HH3
                        tree['HH3'] = 1
                        hh = HH3
                    # 4
                    # x = self.idwt(x) # IDWT
                    x = self.idwt(LL3, lh, hl, hh) # IDWT
                    # x = x.half()
                    lh, hl, hh = None, None, None
                    threshold = threshold / 2 # EZW
                    # threshold = LL2.view(-1).abs().std() + LL2.view(-1).abs().mean() + pow(2, int(math.log2(LL2.view(-1).abs().max()))) #Self
                    # print("thres2: ", threshold)
                    if LH2.contiguous().view(-1).abs().max() > threshold and tree['LH3'] == 1:
                        # x = x + LH2
                        tree['LH2'] = 1
                        lh = LH2
                    if HL2.contiguous().view(-1).abs().max() > threshold and tree['HL3'] == 1:
                        # x = x + HL2
                        tree['HL2'] = 1
                        hl = HL2
                    if HH2.contiguous().view(-1).abs().max() > threshold and tree['HH3'] == 1:
                        # x = x + HH2  
                        tree['HH2'] = 1
                        hh = HH2
                    # 2
                    # x = self.idwt(x) # IDWT
                    x = self.idwt(x, lh, hl, hh) # IDWT
                    # x = x.half()
                    threshold = threshold / 2 # EZW
                    # threshold = LL1.contiguous().view(-1).abs().std() + LL1.contiguous().view(-1).abs().mean() + pow(2, int(math.log2(LL1.contiguous().view(-1).abs().max())))#Self
                    # print("thres3: ", threshold)
                    if LH1.contiguous().view(-1).abs().max() > threshold and tree['LH2'] == 1:
                        x = x + LH1
                        tree['LH1'] = 1
                    if HL1.contiguous().view(-1).abs().max() > threshold and tree['HL2'] == 1:
                        x = x + HL1
                        tree['HL1'] = 1
                    if HH1.contiguous().view(-1).abs().max() > threshold and tree['HH2'] == 1:
                        x = x + HH1    
                        tree['HH1'] = 1  
                    # print(tree)                                                                        
                else: 
                    x = x                               
                return x
        elif self.tree_level == 4:    
            (LL1, LH1, HL1, HH1) = self.wavelettree1(x)
            (LL2, LH2, HL2, HH2) = self.wavelettree2(LL1)
            # print(LL2.contiguous().view(-1).abs().mean(), LL2.contiguous().view(-1).abs().max())
            # print(LH2.contiguous().view(-1).abs().mean(), LH2.contiguous().view(-1).abs().max())#低频通常要大一些
            # print(HL2.contiguous().view(-1).abs().mean(), HL2.contiguous().view(-1).abs().max())
            # print(HH2.contiguous().view(-1).abs().mean(), HH2.contiguous().view(-1).abs().max())

            # print(LL1.contiguous().view(-1).abs().mean(), LL1.contiguous().view(-1).abs().max())
            # print(LH1.contiguous().view(-1).abs().mean(), LH1.contiguous().view(-1).abs().max())#低频通常要大一些
            # print(HL1.contiguous().view(-1).abs().mean(), HL1.contiguous().view(-1).abs().max())
            # print(HH1.contiguous().view(-1).abs().mean(), HH1.contiguous().view(-1).abs().max())
            if self.use_thres == False:
                # std + mean + 小波零树初始化
            
                # threshold = (LL2.reshape(-1).abs().std() + pow(2, int(math.log2(LL2.reshape(-1).abs().max()))))/2
                threshold = 5
            else:
                threshold = self.threshold
            tree = {'LL2': 1, 'LH2': 0, 'HL2': 0, 'HH2': 0, 'LH1': 0, 'HL1': 0, 'HH1': 0} #初始化树   
            lh, hl, hh = None, None, None  
            if self.method == 'all':
                ## 4
                if self.sr_ratio == 4:
                    # 4
                    # threshold = LL2.contiguous().view(-1).abs().std() + LL2.contiguous().view(-1).abs().mean() + pow(2, int(math.log2(LL2.contiguous().view(-1).abs().max())))
                    # print("thres: ", threshold)
                    if LH2.contiguous().view(-1).abs().max() > threshold:
                        x = LL2 + LH2
                    else:
                        x = LL2
                    if HL2.contiguous().view(-1).abs().max() > threshold:
                        x = x + HL2
                        # print("HL2")
                    if HH2.contiguous().view(-1).abs().max() > threshold:
                        x = x + HH2    
                        # print("HH2")     
                ## 2
                elif self.sr_ratio == 2:
                    # print("!!!!!!!!")
                    # 4
                    # threshold = LL2.contiguous().view(-1).abs().std() + LL2.contiguous().view(-1).abs().mean() + pow(2, int(math.log2(LL2.contiguous().view(-1).abs().max())))
                    # print("thres: ", threshold)
                    if LH2.contiguous().view(-1).abs().max() > threshold:
                        # x = LL2 + LH2
                        lh = LH2
                    # else:
                    #     x = LL2
                    if HL2.contiguous().view(-1).abs().max() > threshold:
                        # x = x + HL2
                        # print("HL2")
                        hl = HL2
                    if HH2.contiguous().view(-1).abs().max() > threshold:
                        # x = x + HH2    
                        # print("HH2")  
                        hh = HH2
                    # 2
                    # x = self.idwt(x) # IDWT
                    x = self.idwt(LL2, lh, hl, hh)
                    threshold = threshold / 2
                    # threshold = (LL1.contiguous().view(-1).abs().std() + LL1.contiguous().view(-1).abs().mean() + pow(2, int(math.log2(LL1.contiguous().view(-1).abs().max()))))/2 #Self
                    # print("thres: ", threshold)
                    if LH1.contiguous().view(-1).abs().max() > threshold:
                        x = x + LH1
                        # print("LH1")
                    if HL1.contiguous().view(-1).abs().max() > threshold:
                        x = x + HL1
                        # print("HL1")
                    if HH1.contiguous().view(-1).abs().max() > threshold:
                        x = x + HH1   
                        # print("HH1")  
                else: 
                    x = x                               
                return x                      
            elif self.method == 'root':
                ## 4
                # threshold = 7 # Fixed
                if self.sr_ratio == 4:
                    # 4
                    # threshold = LL2.contiguous().view(-1).abs().std() + LL2.contiguous().view(-1).abs().mean() + pow(2, int(math.log2(LL2.contiguous().view(-1).abs().max())))
                    if LH2.contiguous().view(-1).abs().max() > threshold:
                        x = LL2 + LH2
                    else:
                        x = LL2
                    if HL2.contiguous().view(-1).abs().max() > threshold:
                        x = x + HL2
                    if HH2.contiguous().view(-1).abs().max() > threshold:
                        x = x + HH2   
                elif self.sr_ratio == 2:
                    # print("!!!!!!!!")
                    # 4
                    # threshold = LL2.contiguous().view(-1).abs().std() + LL2.contiguous().view(-1).abs().mean() + pow(2, int(math.log2(LL2.contiguous().view(-1).abs().max())))
                    # print('thres: ', threshold)
                    if LH2.contiguous().view(-1).abs().max() > threshold:
                        # x = LL2 + LH2
                        tree['LH2'] = 1
                        lh = LH2
                    # else:
                    #     x = LL2
                    if HL2.contiguous().view(-1).abs().max() > threshold:
                        # x = x + HL2
                        tree['HL2'] = 1
                        hl = HL2
                    if HH2.contiguous().view(-1).abs().max() > threshold:
                        # x = x + HH2 
                        tree['HH2'] = 1
                        hh = HH2
                    # 2
                    # x = self.idwt(x) # IDWT
                    x = self.idwt(LL2, lh, hl, hh)
                    # x = x.half()
                    # threshold = 5 # Fixed
                    threshold = threshold / 2 # EZW
                    # threshold = (LL1.contiguous().view(-1).abs().std() + LL1.contiguous().view(-1).abs().mean() + pow(2, int(math.log2(LL1.contiguous().view(-1).abs().max()))))/2 #Self
                    # print('thres2: ', threshold)
                    if LH1.contiguous().view(-1).abs().max() > threshold and tree['LH2'] == 1:
                        x = x + LH1
                        tree['LH1'] = 1
                    if HL1.contiguous().view(-1).abs().max() > threshold and tree['HL2'] == 1:
                        x = x + HL1
                        tree['HL1'] = 1
                    if HH1.contiguous().view(-1).abs().max() > threshold and tree['HH2'] == 1:
                        x = x + HH1    
                        tree['HH1'] = 1   
                    # print("sr=2:", tree)                                                                          
                else: 
                    x = x                               
                return x
        else:
            return x

class WaveletTree1(nn.Module):
    def __init__(self, in_planes, sr_ratio=1, ratio=8, threshold=1, use_thres = False):
        super(WaveletTree1, self).__init__()
        self.in_planes = in_planes
        self.ratio = ratio
        self.sr_ratio = sr_ratio
        self.threshold = threshold
        self.use_thres = use_thres
        # self.idwt = IDWT_2D(wave='haar')
        self.idwt = inverse_wav_transform(wave = 'haar')
        # self.wavelettree1 = Wavelet2D(in_planes, name="db2")
        # self.wavelettree2 = Wavelet2D(in_planes, name="db2")
        # self.wavelettree3 = Wavelet2D(in_planes, name="db2")
        self.wavelettree1 = Wavelet2D(in_planes, name="haar")
        self.wavelettree2 = Wavelet2D(in_planes, name="haar")
        self.wavelettree3 = Wavelet2D(in_planes, name="haar")


    def forward(self, x):
        # print("sr_ratio: ", self.sr_ratio)
        # print("x: ", x.shape)
        if self.ratio == 8:
            (LL1, LH1, HL1, HH1) = self.wavelettree1(x)
            # print("LL1: ", LL1.shape)
            (LL2, LH2, HL2, HH2) = self.wavelettree2(LL1)
            # print("LL2: ", LL2.shape)
            (LL3, LH3, HL3, HH3) = self.wavelettree3(LL2)
            if self.use_thres == False:
                # std + mean + 小波零树初始化
                threshold =  LL3.view(-1).abs().std() + LL3.view(-1).abs().mean() + pow(2, int(math.log2(LL3.view(-1).abs().max()))) # 小波零树初始化
            else:
                threshold = self.threshold
            ## 8
            if self.sr_ratio == 8:
                x = LL3
            ## 4
            elif self.sr_ratio == 4:
                # 8
                x = LL3
                # 4
                x = self.idwt(x) # IDWT    
            ## 2
            elif self.sr_ratio == 2:
                # 8
                x = LL3
                # 4
                x = self.idwt(x) # IDWT 
                # 2
                x = self.idwt(x) # IDWT 
            else: 
                x = x                               
            return x
        else:
            return x

class WaveTreeHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio, tree_level = 8, use_lifting_scheme = True, model_method = 'hand', use_wavelet_tree = True, use_idwt = True, method = 'root'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.sr_ratio = sr_ratio
        # print('sr_ratio: ', sr_ratio)
        self.use_lifting_scheme = use_lifting_scheme
        self.use_wavelet_tree = use_wavelet_tree
        self.use_idwt = use_idwt
        self.tree_level = tree_level
        self.method = method
        
        if use_lifting_scheme == True and num_heads in [2, 4]:
            self.lifting_by_channel = LiftingScheme_by_channel(dim//2, num_heads = num_heads, 
                                                                k_size=3,
                                                                simple_lifting=True,
                                                                model_importance_method = model_method)
            
        self.filter = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=1, bias = False),
            build_norm_layer(dict(type='BN', requires_grad=False), dim)[1],
            nn.ReLU(inplace=True),
        )

        if use_wavelet_tree == True:
            # self.dwt = DWT_2D(wave='haar')
            # self.idwt = IDWT_2D(wave='haar')

            # self.reduce = nn.Sequential(
            #     nn.Conv2d(dim, dim//4, kernel_size=1, padding=0, stride=1),
            #     build_norm_layer(dict(type='BN', requires_grad=False), dim//4)[1],
            #     nn.ReLU(inplace=True),
            # )

            self.kv_embed_dwt = WaveletTree(dim, sr_ratio, tree_level = tree_level, method = method)
            # if sr_ratio == 8:
            #     self.idwt_by_tree = nn.Sequential(IDWT_2D(wave='haar'), IDWT_2D(wave='haar'), IDWT_2D(wave='haar'))
            #     # self.kv_embed_dwt = WaveletTree(dim, sr_ratio)
            #     # self.kv_embed_dwt = nn.Sequential(
            #     #                     self.reduce,
            #     #                     DWT_2D(wave='haar'),
            #     #                     self.reduce,
            #     #                     DWT_2D(wave='haar'),
            #     #                     self.reduce,
            #     #                     DWT_2D(wave='haar'),
            #     #                     )
            # elif sr_ratio == 4:
            #     self.idwt_by_tree = nn.Sequential(IDWT_2D(wave='haar'), IDWT_2D(wave='haar'))
            #     # self.kv_embed_dwt = WaveletTree(dim, sr_ratio)
            #     # self.kv_embed_dwt = nn.Sequential(
            #     #                     self.reduce,
            #     #                     DWT_2D(wave='haar'),
            #     #                     self.reduce,
            #     #                     DWT_2D(wave='haar'),
            #     #                     )
            # elif sr_ratio == 2:
            #     self.idwt_by_tree = nn.Sequential(IDWT_2D(wave='haar'))
            #     # self.kv_embed_dwt = WaveletTree(dim, sr_ratio)
            #     # self.kv_embed_dwt = nn.Sequential(self.reduce, DWT_2D(wave='haar'))
            # # else:
            # #     self.kv_embed_dwt = WaveletTree(dim, sr_ratio)
            # #     # self.idwt_by_tree = nn.Sequential() 无
            # #     self.kv_embed_dwt = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)

        else:
            self.kv_embed_dwt = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)#这里仿照pvt的spatial reduce，但是又不一样，
                                                                                  #感觉是错的，除非sr_ratio一直为1
                                                                                  #transformer中滑窗的大小，即每个patch的大小为sr_ratio
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2)
        )

        if use_idwt == True and sr_ratio > 1:
            if sr_ratio == 8:
                self.idwt_by_tree = nn.Sequential(IDWT_2D(wave='haar'), IDWT_2D(wave='haar'), IDWT_2D(wave='haar'))
                # self.idwt_by_tree = nn.Sequential(inverse_wav_transform(wave = 'haar'), inverse_wav_transform(wave = 'haar'), inverse_wav_transform(wave = 'haar'))
            elif sr_ratio == 4:
                self.idwt_by_tree = nn.Sequential(IDWT_2D(wave='haar'), IDWT_2D(wave='haar'))
                # self.idwt_by_tree = nn.Sequential(inverse_wav_transform(wave = 'haar'), inverse_wav_transform(wave = 'haar'))
            elif sr_ratio == 2:
                self.idwt_by_tree = nn.Sequential(IDWT_2D(wave='haar'))
                # self.idwt_by_tree = nn.Sequential(inverse_wav_transform(wave = 'haar'))
            self.proj = nn.Linear(dim + dim, dim)
        else:
            self.proj = nn.Linear(dim, dim)

        self.apply(self._init_weights) # 初始化权重

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
        # print("(H, W): ", (H, W))
        B, N, C = x.shape # x.shape: (B, N, C) 
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)# B,num_head,N,c/num_head        
        x = x.view(B, H, W, C).permute(0, 3, 1, 2) # x:(B,C,H,W)

        #######################################################/channel_lifting_scheme_head_att/#############################################
        if self.use_lifting_scheme == True and self.num_heads in [2, 4]:
            # print("num_head: ", self.num_heads)
            x = self.lifting_by_channel(x) # (B,C,H,W)
        #######################################################/channel_lifting_scheme_head_att/#############################################

        #######################################################/wavelet_tree_reduction_block/#############################################
        # if self.use_wavelet_tree == True:
        #     x_dwt = x
        # kv = self.kv_embed_dwt(x).reshape(B, C, -1).permute(0, 2, 1) #(B,N,C)
        kv = self.filter(self.kv_embed_dwt(x)).reshape(B, C, -1).permute(0, 2, 1) #(B,N,C)
        kv = self.kv(kv).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)# KV,B,num_head,N,c/num_head
        k, v = kv[0], kv[1]
        # v = self.filter(v)
        #######################################################/wavelet_tree_reduction_block/#############################################

        #/////////////////////////////////////////////////////ori block///////////////////////////////////////////////////
        # print("q_shape: ", q.shape)
        # # #### Q:lifing_by_channel
        # q_ = q.permute(0, 2, 1, 3)#B, N, num_head, C/num_head
        # q_ = q_.reshape(B, N, C)
        # q_ = q_.reshape(B, H, W, C).permute(0, 3, 1, 2)# B, C, H, W
        # q_ = self.lifting_by_channel(q_)# B, C, H, W
        # q = q_.permute(0, 2, 3, 1).reshape(B, N, C)
        # q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # print("q_shape: ", q_.shape)
        # ####

        # x_dwt = self.dwt(self.reduce(x))#linear c/4
        # x_dwt = self.filter(x_dwt)#conv3*3

        # x_idwt = self.idwt(x_dwt)
        # x_idwt = x_idwt.view(B, -1, x_idwt.size(-2)*x_idwt.size(-1)).transpose(1, 2)

        # kv = self.kv_embed(x_dwt).reshape(B, C, -1).permute(0, 2, 1)        
        # kv = self.kv_embed_dwt(x_dwt).reshape(B, C, -1).permute(0, 2, 1)    

        # kv = self.kv_embed_dwt(x).reshape(B, C, -1).permute(0, 2, 1) #(B,N,C)
        # kv = self.kv(kv).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)# KV,B,num_head,N,c/num_head
        # k, v = kv[0], kv[1]

        # print("v_shape: ", v.shape)
        # #### K,V lifting_by_channel
        # k_ = k.permute(0, 2, 1, 3)#B, N, num_head, C/num_head
        # n = int(k_.shape[1])
        # k_ = k_.reshape(B, n, C) #B, N, C
        # k_ = k_.reshape(B, int(H/2), int(W/2), C).permute(0, 3, 1, 2)# B, C, H/2, W/2
        # k_ = self.lifting_by_channel(k_)# B, C, H/2, W/2
        # k = k_.permute(0, 2, 3, 1).reshape(B, n, C)
        # k = k.reshape(B, n, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        #
        # v_ = v.permute(0, 2, 1, 3)#B, N, num_head, C/num_head
        # n = int(v_.shape[1])
        # h, w = int(n ** 0.5), int(n ** 0.5)
        # # print("n: ", n)
        # # print("h: ", h)
        # v_ = v_.reshape(B, n, C) #B, N, C
        # v_ = v_.reshape(B, h, w, C).permute(0, 3, 1, 2)# B, C, H/2, W/2
        # v_ = self.lifting_by_channel(v_)# B, C, H/2, W/2
        # v = v_.permute(0, 2, 3, 1).reshape(B, n, C)
        # v = v.reshape(B, n, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # ####
        #/////////////////////////////////////////////////////ori block///////////////////////////////////////////////////

        #######################################################/Scaled Dot-Product Att/#############################################
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # B,num_head,N,c/num_head --> B,N,C
        #######################################################/Scaled Dot-Product Att/#############################################

        #######################################################/use_idwt/#############################################
        # print("x.shape: ", x.shape)
        # print("x_idwt.shape: ", x_idwt.shape)
        # if self.use_idwt == True and self.use_wavelet_tree == True and self.sr_ratio > 1:
        if self.use_idwt == True and self.sr_ratio > 1:
            # print("sr_ratio: ", self.sr_ratio)
            v = v.permute(0, 2, 1, 3)#B, N, num_head, C/num_head
            n = int(v.shape[1])
            h, w = int(n ** 0.5), int(n ** 0.5)
            v = v.view(B, n, C)
            v = v.view(B, h, w, C).permute(0, 3, 1, 2)# B, C, H, W
            # print("v.shape: ", v.shape)
            x_idwt = self.idwt_by_tree(v) # 此处需要根据设置的分辨率来进行逆变换
            x_idwt = x_idwt.view(B, -1, x_idwt.size(-2)*x_idwt.size(-1)).transpose(1, 2)
            # print("x_idwt.shape: ", x_idwt.shape)
            # print("x.shape: ", x.shape)
            x = self.proj(torch.cat([x, x_idwt], dim=-1))
            # print("###################################")
        else:
            # print("!!!!!!!!!!!!!!!!!!!!!!!!")
            x = self.proj(x)
        #######################################################/use_idwt/#############################################
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
        # print("###(H, W): ", (H, W))
        # print("###x.shape: ", x.shape)
        x = self.proj(x)
        return x


class Block(nn.Module):#一个encode模块
    def __init__(self, 
        dim, 
        num_heads, 
        mlp_ratio,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        sr_ratio=1, 
        block_type = 'wave',
        tree_level = 8,
        use_lifting_scheme = True,
        model_method = 'hand',
        use_wavelet_tree = True,
        use_idwt = True,
        method = 'root'
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        if block_type == 'std_att':
            self.attn = Attention(dim, num_heads)
        else:
            self.attn = WaveTreeHeadAttention(dim, num_heads, sr_ratio, tree_level = tree_level, use_lifting_scheme = use_lifting_scheme, 
                                              model_method = model_method, use_wavelet_tree = use_wavelet_tree, use_idwt = use_idwt, method = method)
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
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)#下采样作用
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
        x = x.flatten(2).transpose(1, 2)#指定维度后的维度进行合并，即H,W合并为N。（B,N,C）
        x = self.norm(x)
        return x, H, W

class Stem(nn.Module):#CNN的前几层，用作下采样
    def __init__(self, in_channels, stem_hidden_dim, out_channels):
        super().__init__()
        hidden_dim = stem_hidden_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=7, stride=2,#下采样
                      padding=3, bias=False),  # 112x112
            build_norm_layer(dict(type='BN', requires_grad=False), hidden_dim)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                      padding=1, bias=False),  # 112x112
            build_norm_layer(dict(type='BN', requires_grad=False), hidden_dim)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                      padding=1, bias=False),  # 112x112
            build_norm_layer(dict(type='BN', requires_grad=False), hidden_dim)[1],
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
        x = self.conv(x)# x的大小减半
        x = self.proj(x)# x的大小再次减半
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)# (B,N,C)
        x = self.norm(x)
        return x, H, W

class WaveTreeHeadAttention_ViT(nn.Module):
    def __init__(self, 
        stem_width=32, 
        in_chans=3, 
        embed_dims=[64, 128, 320, 448],
        num_heads=[2, 4, 10, 14], 
        mlp_ratios=[8, 8, 4, 4], 
        drop_path_rate=0., 
        norm_layer=nn.LayerNorm, 
        depths=[3, 4, 6, 3],
        sr_ratios=[4, 2, 1, 1], 
        num_stages=4, 
        pretrained=None,
        ##
        tree_level = [8, 8, 4, 1],
        use_lifting_scheme = True,
        model_method = 'hand',
        use_wavelet_tree = True,
        use_idwt = True,
        method = 'root'
    ):
        super().__init__()
        self.depths = depths
        self.num_stages = num_stages
        

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = Stem(in_chans, stem_width, embed_dims[i])# /4
            else:
                patch_embed = DownSamples(embed_dims[i - 1], embed_dims[i])# /2

            block = nn.ModuleList([Block(
                dim = embed_dims[i], 
                num_heads = num_heads[i], 
                mlp_ratio = mlp_ratios[i], 
                drop_path=dpr[cur + j], 
                norm_layer=norm_layer,
                sr_ratio = sr_ratios[i],
                # block_type='wave' if i < 2 else 'std_att')
                block_type='wave',
                tree_level= tree_level[i],
                use_lifting_scheme = use_lifting_scheme,
                model_method = model_method, 
                use_wavelet_tree = use_wavelet_tree, 
                use_idwt = use_idwt,
                method = method)
            for j in range(depths[i])])

            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)# 用于设置属性值，该属性不一定是存在的
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

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
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x) # x.shape: (B, N, C)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x



@BACKBONES.register_module() # backbone登记
class WTHA_ViT_s(WaveTreeHeadAttention_ViT):
    def __init__(self, **kwargs):
        super(WTHA_ViT_s, self).__init__(
            stem_width=32,
            embed_dims=[64, 128, 256, 448],
            num_heads=[2, 4, 8, 14],
            mlp_ratios=[8, 8, 4, 4],
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 5, 6, 3],
            sr_ratios=[8, 4, 2, 1], #PVT
            # sr_ratios=[4, 2, 1, 1], #WaveViT
            # sr_ratios=[1, 1, 1, 1],
            drop_path_rate=0.1, 
            # pretrained=kwargs['pretrained'],
            tree_level=[8, 8, 4, 1],
            use_lifting_scheme = True,
            model_method = 'hand',
            use_wavelet_tree = True,
            use_idwt = True,
            method = 'root')

@BACKBONES.register_module()
class WTHA_ViT_m(WaveTreeHeadAttention_ViT):
    def __init__(self, **kwargs):
        super(WTHA_ViT_m, self).__init__(
            stem_width=64,
            embed_dims=[64, 128, 320, 512],
            num_heads=[2, 4, 10, 16],
            mlp_ratios=[8, 8, 4, 4],
            norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            depths=[3, 5, 10, 3],
            sr_ratios=[8, 4, 2, 1],#之前是[4,2,1,1]
            drop_path_rate=0.1, 
            # pretrained=kwargs['pretrained'],
            tree_level=[8, 8, 4, 1],
            # tree_level=[8, 4, 4, 1],
            use_lifting_scheme = True,
            model_method = 'hand',
            use_wavelet_tree = True,
            use_idwt = True,
            method = 'root')

@BACKBONES.register_module()
class WTHA_ViT_l(WaveTreeHeadAttention_ViT):
    def __init__(self, **kwargs):
        super(WTHA_ViT_l, self).__init__(
            stem_width=64,
            embed_dims=[64, 128, 384, 512],
            num_heads=[2, 4, 12, 16],
            mlp_ratios=[8, 8, 4, 4],
            norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            depths=[3, 6, 14, 3],
            sr_ratios=[8, 4, 2, 1],#之前是[4,2,1,1]
            drop_path_rate=0.1, 
            # pretrained=kwargs['pretrained'],
            tree_level=[8, 8, 4, 1],
            # tree_level=[8, 4, 4, 1],
            use_lifting_scheme = True,
            model_method = 'hand',
            use_wavelet_tree = True,
            use_idwt = True,
            method = 'root')

###
@BACKBONES.register_module() # backbone登记
class WTHA_ViT_s_all(WaveTreeHeadAttention_ViT):
    def __init__(self, **kwargs):
        super(WTHA_ViT_s_all, self).__init__(
            stem_width=32,
            embed_dims=[64, 128, 256, 448],
            num_heads=[2, 4, 8, 14],
            mlp_ratios=[8, 8, 4, 4],
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 5, 6, 3],
            sr_ratios=[8, 4, 2, 1], #PVT
            # sr_ratios=[4, 2, 1, 1], #WaveViT
            # sr_ratios=[1, 1, 1, 1],
            drop_path_rate=0.1, 
            # pretrained=kwargs['pretrained'],
            tree_level=[8, 8, 4, 1],
            use_lifting_scheme = True,
            model_method = 'network',
            use_wavelet_tree = True,
            use_idwt = True,
            method = 'all')
        
@BACKBONES.register_module() # backbone登记
class WTHA_ViT_s_root(WaveTreeHeadAttention_ViT):
    def __init__(self, **kwargs):
        super(WTHA_ViT_s_root, self).__init__(
            stem_width=32,
            embed_dims=[64, 128, 256, 448],
            num_heads=[2, 4, 8, 14],
            mlp_ratios=[8, 8, 4, 4],
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 5, 6, 3],
            sr_ratios=[8, 4, 2, 1], #PVT
            # sr_ratios=[4, 2, 1, 1], #WaveViT
            # sr_ratios=[1, 1, 1, 1],
            drop_path_rate=0.1, 
            # pretrained=kwargs['pretrained'],
            tree_level=[8, 8, 4, 1],
            use_lifting_scheme = True,
            model_method = 'network',
            use_wavelet_tree = True,
            use_idwt = True,
            method = 'root')

@BACKBONES.register_module() # backbone登记
class WTHA_ViT_s_network(WaveTreeHeadAttention_ViT):
    def __init__(self, **kwargs):
        super(WTHA_ViT_s_network, self).__init__(
            stem_width=32,
            embed_dims=[64, 128, 256, 448],
            num_heads=[2, 4, 8, 14],
            mlp_ratios=[8, 8, 4, 4],
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 5, 6, 3],
            sr_ratios=[8, 4, 2, 1], #PVT
            # sr_ratios=[4, 2, 1, 1], #WaveViT
            # sr_ratios=[1, 1, 1, 1],
            drop_path_rate=0.1, 
            # pretrained=kwargs['pretrained'],
            tree_level=[8, 8, 4, 1],
            # tree_level=[8, 4, 4, 1],
            use_lifting_scheme = True,
            model_method = 'network',
            use_wavelet_tree = True,
            use_idwt = True,
            method = 'root')
        
@BACKBONES.register_module() # backbone登记
class WTHA_ViT_s_network2(WaveTreeHeadAttention_ViT):
    def __init__(self, **kwargs):
        super(WTHA_ViT_s_network2, self).__init__(
            stem_width=32,
            embed_dims=[64, 128, 256, 448],
            num_heads=[2, 4, 8, 14],
            mlp_ratios=[8, 8, 4, 4],
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 5, 6, 3],
            sr_ratios=[8, 4, 2, 1], #PVT
            # sr_ratios=[4, 2, 1, 1], #WaveViT
            # sr_ratios=[1, 1, 1, 1],
            drop_path_rate=0.1, 
            # pretrained=kwargs['pretrained'],
            # tree_level=[8, 8, 4, 1],
            tree_level=[8, 4, 4, 1],
            use_lifting_scheme = True,
            model_method = 'hand',
            use_wavelet_tree = True,
            # use_idwt = True,
            use_idwt = False,
            method = 'root')

@BACKBONES.register_module() # backbone登记
class WTHA_ViT_s_network3(WaveTreeHeadAttention_ViT):
    def __init__(self, **kwargs):
        super(WTHA_ViT_s_network3, self).__init__(
            stem_width=32,
            embed_dims=[64, 128, 256, 448],
            num_heads=[2, 4, 8, 14],
            mlp_ratios=[8, 8, 4, 4],
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 5, 6, 3],
            sr_ratios=[8, 4, 2, 1], #PVT
            # sr_ratios=[4, 2, 1, 1], #WaveViT
            # sr_ratios=[1, 1, 1, 1],
            drop_path_rate=0.1, 
            # pretrained=kwargs['pretrained'],
            # tree_level=[8, 8, 4, 1],
            tree_level=[8, 4, 4, 1],
            use_lifting_scheme = True,
            model_method = 'hand',
            use_wavelet_tree = True,
            use_idwt = True,
            method = 'root')

@BACKBONES.register_module() # backbone登记
class WTHA_ViT_s_similarity(WaveTreeHeadAttention_ViT):
    def __init__(self, **kwargs):
        super(WTHA_ViT_s_similarity, self).__init__(
            stem_width=32,
            embed_dims=[64, 128, 256, 448],
            num_heads=[2, 4, 8, 14],
            mlp_ratios=[8, 8, 4, 4],
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 5, 6, 3],
            sr_ratios=[8, 4, 2, 1], #PVT
            # sr_ratios=[4, 2, 1, 1], #WaveViT
            # sr_ratios=[1, 1, 1, 1],
            drop_path_rate=0.1, 
            # pretrained=kwargs['pretrained'],
            tree_level=[8, 8, 4, 1],
            use_lifting_scheme = True,
            model_method = 'similarity',
            use_wavelet_tree = True,
            use_idwt = True,
            method = 'root')

@BACKBONES.register_module() # backbone登记
class WTHA_ViT_s_noidwt(WaveTreeHeadAttention_ViT):
    def __init__(self, **kwargs):
        super(WTHA_ViT_s_noidwt, self).__init__(
            stem_width=32,
            embed_dims=[64, 128, 256, 448],
            num_heads=[2, 4, 8, 14],
            mlp_ratios=[8, 8, 4, 4],
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 5, 6, 3],
            sr_ratios=[8, 4, 2, 1], #PVT
            # sr_ratios=[4, 2, 1, 1], #WaveViT
            # sr_ratios=[1, 1, 1, 1],
            drop_path_rate=0.1, 
            # pretrained=kwargs['pretrained'],
            tree_level=[8, 8, 4, 1],
            use_lifting_scheme = True,
            model_method = 'network',
            use_wavelet_tree = True,
            use_idwt = False,
            method = 'root')

@BACKBONES.register_module() # backbone登记
class WTHA_ViT_s_nolifting_noidwt(WaveTreeHeadAttention_ViT):
    def __init__(self, **kwargs):
        super(WTHA_ViT_s_nolifting_noidwt, self).__init__(
            stem_width=32,
            embed_dims=[64, 128, 256, 448],
            num_heads=[2, 4, 8, 14],
            mlp_ratios=[8, 8, 4, 4],
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 5, 6, 3],
            sr_ratios=[8, 4, 2, 1], #PVT
            # sr_ratios=[4, 2, 1, 1], #WaveViT
            # sr_ratios=[1, 1, 1, 1],
            drop_path_rate=0.1, 
            # pretrained=kwargs['pretrained'],
            tree_level=[8, 8, 4, 1],
            use_lifting_scheme = False,
            model_method = 'network',
            use_wavelet_tree = True,
            use_idwt = False,
            method = 'root')

@BACKBONES.register_module() # backbone登记
class WTHA_ViT_s_nolifting_nowavetree_noidwt(WaveTreeHeadAttention_ViT):
    def __init__(self, **kwargs):
        super(WTHA_ViT_s_nolifting_nowavetree_noidwt, self).__init__(
            stem_width=32,
            embed_dims=[64, 128, 256, 448],
            num_heads=[2, 4, 8, 14],
            mlp_ratios=[8, 8, 4, 4],
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 5, 6, 3],
            sr_ratios=[8, 4, 2, 1], #PVT
            # sr_ratios=[4, 2, 1, 1], #WaveViT
            # sr_ratios=[1, 1, 1, 1],
            drop_path_rate=0.1, 
            # pretrained=kwargs['pretrained'],
            tree_level=[8, 8, 4, 1],
            use_lifting_scheme = False,
            model_method = 'network',
            use_wavelet_tree = False,
            use_idwt = False,
            method = 'root')

@BACKBONES.register_module() # backbone登记
class WTHA_ViT_s_nolifting(WaveTreeHeadAttention_ViT):
    def __init__(self, **kwargs):
        super(WTHA_ViT_s_nolifting, self).__init__(
            stem_width=32,
            embed_dims=[64, 128, 256, 448],
            num_heads=[2, 4, 8, 14],
            mlp_ratios=[8, 8, 4, 4],
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 5, 6, 3],
            sr_ratios=[8, 4, 2, 1], #PVT
            # sr_ratios=[4, 2, 1, 1], #WaveViT
            # sr_ratios=[1, 1, 1, 1],
            drop_path_rate=0.1, 
            # pretrained=kwargs['pretrained'],
            tree_level=[8, 8, 4, 1],
            use_lifting_scheme = False,
            model_method = 'network',
            use_wavelet_tree = True,
            use_idwt = True,
            method = 'root')

@BACKBONES.register_module() # backbone登记
class WTHA_ViT_s_nowavetree_noidwt(WaveTreeHeadAttention_ViT):
    def __init__(self, **kwargs):
        super(WTHA_ViT_s_nowavetree_noidwt, self).__init__(
            stem_width=32,
            embed_dims=[64, 128, 256, 448],
            num_heads=[2, 4, 8, 14],
            mlp_ratios=[8, 8, 4, 4],
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 5, 6, 3],
            sr_ratios=[8, 4, 2, 1], #PVT
            # sr_ratios=[4, 2, 1, 1], #WaveViT
            # sr_ratios=[1, 1, 1, 1],
            drop_path_rate=0.1, 
            # pretrained=kwargs['pretrained'],
            tree_level=[8, 8, 4, 1],
            use_lifting_scheme = True,
            model_method = 'network',
            use_wavelet_tree = False,
            use_idwt = False,
            method = 'root')