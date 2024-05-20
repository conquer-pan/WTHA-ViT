import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from timm.models.layers import trunc_normal_
from mmdet.models.backbones.WTHA_ViT import IDWT_2D
class WaveletTree(nn.Module):
    def __init__(self, in_planes, ratio=8, sr_ratio=1, threshold=1, use_thres = False):
        super(WaveletTree, self).__init__()
        self.in_planes = in_planes
        self.ratio = ratio
        self.sr_ratio = sr_ratio
        self.threshold = threshold
        self.use_thres = use_thres
        self.idwt = IDWT_2D(wave='haar')
        self.wavelettree1 = Wavelet2D(in_planes, name="db2")
        self.wavelettree2 = Wavelet2D(in_planes, name="db2")
        self.wavelettree3 = Wavelet2D(in_planes, name="db2")


    def forward(self, x):
        print("x: ", x.shape)
        if self.ratio == 8:
            (LL1, LH1, HL1, HH1) = self.wavelettree1(x)
            print("LL1: ", LL1.shape)
            (LL2, LH2, HL2, HH2) = self.wavelettree2(LL1)
            print("LL2: ", LL2.shape)
            (LL3, LH3, HL3, HH3) = self.wavelettree3(LL2)
            print(LL3.ravel().abs().mean(), LL3.ravel().abs().max())
            print(LH3.ravel().abs().mean(), LH3.ravel().abs().max())#低频通常要大一些
            print(HL3.ravel().abs().mean(), HL3.ravel().abs().max())
            print(HH3.ravel().abs().mean(), HH3.ravel().abs().max())

            print(LL2.ravel().abs().mean(), LL2.ravel().abs().max())
            print(LH2.ravel().abs().mean(), LH2.ravel().abs().max())#低频通常要大一些
            print(HL2.ravel().abs().mean(), HL2.ravel().abs().max())
            print(HH2.ravel().abs().mean(), HH2.ravel().abs().max())

            print(LL1.ravel().abs().mean(), LL1.ravel().abs().max())
            print(LH1.ravel().abs().mean(), LH1.ravel().abs().max())#低频通常要大一些
            print(HL1.ravel().abs().mean(), HL1.ravel().abs().max())
            print(HH1.ravel().abs().mean(), HH1.ravel().abs().max())
            if self.use_thres == False:
                # std + mean + 小波零树初始化
                threshold =  LL3.ravel().abs().std() + LL3.ravel().abs().mean() + pow(2, int(math.log2(LL3.ravel().abs().max()))) # 小波零树初始化
            else:
                threshold = self.threshold
            ## 8
            if self.sr_ratio == 8:
                print("thres: ", threshold)
                if LH3.ravel().abs().max() > threshold:
                    x = LL3 + LH3
                    print("LH3")
                else:
                    x = LL3
                if HL3.ravel().abs().max() > threshold:
                    x = x + HL3
                    print("HL3")
                if HH3.ravel().abs().max() > threshold:
                    x = x + HH3
                    print("HH3")
                print("x2: ", x.shape)
            ## 4
            elif self.sr_ratio == 4:
                # 8
                if LH3.ravel().abs().max() > threshold:
                    x = LL3 + LH3
                    print("LH3")
                else:
                    x = LL3
                if HL3.ravel().abs().max() > threshold:
                    x = x + HL3
                    print("HL3")
                if HH3.ravel().abs().max() > threshold:
                    x = x + HH3
                    print("HH3")
                # 4
                # x = torch.FloatTensor(x)
                # x = x.half()
                x = self.idwt(x) # IDWT
                threshold = threshold // 2
                print("thres: ", threshold)
                if LH2.ravel().abs().max() > threshold:
                    x = x + LH2
                    print("LH2")
                if HL2.ravel().abs().max() > threshold:
                    x = x + HL2
                    print("HL2")
                if HH2.ravel().abs().max() > threshold:
                    x = x + HH2    
                    print("HH2")     
            ## 2
            elif self.sr_ratio == 2:
                # 8
                print("thres: ", threshold)
                if LH3.ravel().abs().max() > threshold:
                    x = LL3 + LH3
                    print("LH3")
                else:
                    x = LL3
                if HL3.ravel().abs().max() > threshold:
                    x = x + HL3
                    print("HL3")
                if HH3.ravel().abs().max() > threshold:
                    x = x + HH3
                    print("HH3")
                # 4
                # x = torch.FloatTensor(x)
                # x = x.half()
                x = self.idwt(x) # IDWT
                threshold = threshold // 2
                print("thres: ", threshold)
                if LH2.ravel().abs().max() > threshold:
                    x = x + LH2
                    print("LH2")
                if HL2.ravel().abs().max() > threshold:
                    x = x + HL2
                    print("HL2")
                if HH2.ravel().abs().max() > threshold:
                    x = x + HH2   
                    print("HH2")
                # 2
                # x = torch.FloatTensor(x)
                # x = x.half()
                x = self.idwt(x) # IDWT
                threshold = threshold // 2
                print("thres: ", threshold)
                if LH1.ravel().abs().max() > threshold:
                    x = x + LH1
                    print("LH1")
                if HL1.ravel().abs().max() > threshold:
                    x = x + HL1
                    print("HL1")
                if HH1.ravel().abs().max() > threshold:
                    x = x + HH1   
                    print("HH1")                 
            # return (LL1, LH1, HL1, HH1, LH2, HL2, HH2, LH3, HL3, HH3)
            return x
        else:
            return x


# class WaveletHaar(nn.Module):
#     def __init__(self, horizontal):
#         super(WaveletHaar, self).__init__()
#         self.split = Splitting(horizontal)
#         self.norm = math.sqrt(2.0)

#     def forward(self, x):
#         '''Returns the approximation and detail part'''
#         (x_even, x_odd) = self.split(x)

#         # Haar wavelet definition
#         d = (x_odd - x_even) / self.norm
#         c = (x_odd + x_even) / self.norm
#         return (c, d)


# class WaveletHaar2D(nn.Module):
#     def __init__(self):
#         super(WaveletHaar2D, self).__init__()
#         self.horizontal_haar = WaveletHaar(horizontal=True)
#         self.vertical_haar = WaveletHaar(horizontal=False)

#     def forward(self, x):
#         '''Returns (LL, LH, HL, HH)'''
#         (c, d) = self.horizontal_haar(x)
#         (LL, LH) = self.vertical_haar(c)
#         (HL, HH) = self.vertical_haar(d)
#         return (LL, LH, HL, HH)


class Wavelet(nn.Module):
    """This module extract wavelet coefficient defined in pywt
    and create 2D convolution kernels to be able to use GPU"""

    def _coef_h(self, in_planes, coef):
        """Construct the weight matrix for horizontal 2D convolution.
        The weights are repeated on the diagonal"""
        v = []
        for i in range(in_planes):
            l = []
            for j in range(in_planes):
                if i == j:
                    l.append([[c for c in coef]])
                else:
                    l.append([[0.0 for c in coef]])
            v.append(l)
        return v

    def _coef_v(self, in_planes, coef):
        """Construct the weight matrix for vertical 2D convolution.
        The weights are repeated on the diagonal"""
        v = []
        for i in range(in_planes):
            l = []
            for j in range(in_planes):
                if i == j:
                    l.append([[c] for c in coef])
                else:
                    l.append([[0.0] for c in coef])
            v.append(l)
        return v

    def __init__(self, in_planes, horizontal, name="db2"):
        super(Wavelet, self).__init__()

        # Import wavelet coefficients
        import pywt
        wavelet = pywt.Wavelet(name)
        coef_low = wavelet.dec_lo
        coef_high = wavelet.dec_hi
        # Determine the kernel 2D shape
        nb_coeff = len(coef_low)
        if horizontal:
            kernel_size = (1, nb_coeff)
            stride = (1, 2)
            pad = (nb_coeff // 2, nb_coeff - 1 - nb_coeff // 2, 0, 0)
            weights_low = self._coef_h(in_planes, coef_low)
            weights_high = self._coef_h(in_planes, coef_high)
        else:
            kernel_size = (nb_coeff, 1)
            stride = (2, 1)
            pad = (0, 0, nb_coeff // 2, nb_coeff - 1 - nb_coeff // 2)
            weights_low = self._coef_v(in_planes, coef_low)
            weights_high = self._coef_v(in_planes, coef_high)
        # TODO: Debug prints
        # print("")
        # print("Informations: ")
        # print("- kernel_size: ", kernel_size)
        # print("- stride     : ", stride)
        # print("- pad        : ", pad)
        # print("- low        : ", weights_low)
        # print("- high       : ", weights_high)

        # Create the conv2D
        self.conv_high = nn.Conv2d(
            in_planes, in_planes, kernel_size=kernel_size, stride=stride, bias=False)
        self.conv_low = nn.Conv2d(
            in_planes, in_planes, kernel_size=kernel_size, stride=stride, bias=False)
        self.padding = nn.ReflectionPad2d(padding=pad)
        # TODO: Debug prints
        # print("- low        : ", self.conv_low.weight)
        # print("- high       : ", self.conv_high.weight)

        # Replace their weights
        self.conv_high.weight = torch.nn.Parameter(
            data=torch.Tensor(weights_high), requires_grad=False)
        self.conv_low.weight = torch.nn.Parameter(
            data=torch.Tensor(weights_low), requires_grad=False)

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        '''Returns the approximation and detail part'''
        x = self.padding(x)
        return (self.conv_low(x), self.conv_high(x))


class Wavelet2D(nn.Module):
    def __init__(self, in_planes, name="db1"):
        super(Wavelet2D, self).__init__()
        self.horizontal_wavelet = Wavelet(in_planes, horizontal=True, name=name)
        self.vertical_wavelet = Wavelet(in_planes, horizontal=False, name=name)

    def forward(self, x):
        '''Returns (LL, LH, HL, HH)'''
        (c, d) = self.horizontal_wavelet(x)
        (LL, LH) = self.vertical_wavelet(c)
        (HL, HH) = self.vertical_wavelet(d)
        return (LL, LH, HL, HH)



if __name__ == "__main__":
    input = torch.randn(4, 64, 400, 400)
    #m_harr = WaveletLiftingHaar2D()
    m_wavelet = Wavelet2D(64, name="db2")
    wavelettree = WaveletTree(64, sr_ratio=8)
    # print(input)
    # print(m_wavelet(input))
    # (LL, LH, HL, HH) = m_wavelet(input)
    # print(LL.shape)
    # print(max(LL))
    x = wavelettree(input)
    print(x.shape)


