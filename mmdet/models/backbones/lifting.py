import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

# # 计算相关性矩阵
# correlation_matrix = calculate_correlation(feature_maps[0][0])
# # correlation_matrix = cal_cor2(feature_maps[0])
# 定义计算相关性的函数
def calculate_correlation(feature_map):
    # 将特征图展平为二维矩阵
    flat_feature_map = feature_map.view(feature_map.shape[0], -1)
    # 计算协方差矩阵
    cov_matrix = torch.mm(flat_feature_map, flat_feature_map.t())
    # 计算每一对通道之间的Pearson相关系数
    std = torch.sqrt(torch.diag(cov_matrix))
    correlation_matrix = cov_matrix / torch.mm(std.view(-1,1), std.view(1,-1))
    return correlation_matrix

def cal_cor2(features):
    # 将特征图扁平化成形状为 (batch_size, num_channels, height*width) 的张量
    batch_size = features.shape[0]
    num_channels = features.shape[1]
    N = features.shape[2]*features.shape[3]
    features_flat = features.view(batch_size, num_channels, N)

    # 计算特征图的协方差矩阵
    covariance_matrix = torch.zeros(num_channels, num_channels)
    for i in range(num_channels):
        for j in range(num_channels):
        # 计算通道i和通道j之间的协方差
            covariance_matrix[i, j] = torch.mean(features_flat[:, i, :] * features_flat[:, j, :]) - \
                                  torch.mean(features_flat[:, i, :]) * torch.mean(features_flat[:, j, :])

    # 计算特征图的相关性矩阵
    std_dev = torch.sqrt(torch.diag(covariance_matrix))
    correlation_matrix = covariance_matrix / (std_dev.unsqueeze(0) * std_dev.unsqueeze(1))

    # 打印相关性矩阵
    # print(correlation_matrix.numpy())
    # print("size:", correlation_matrix.shape)
    return correlation_matrix

# To change if we do horizontal first inside the LS
HORIZONTAL_FIRST = True

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.in_planes = in_planes
        self.conv1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        # self.sharedMLP = nn.Sequential(
        #     nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), 
        #     nn.ReLU(),
        #     nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.act = nn.Tanh()
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        # if isinstance(m, nn.AdaptiveAvgPool2d):
        #     trunc_normal_(m.weight, std=.02)
        #     if isinstance(m, nn.AdaptiveAvgPool2d) and m.bias is not None:
        #         nn.init.constant_(m.bias, 0.01)
        # if isinstance(m, nn.ReLU):
        #     # trunc_normal_(m.weight, std=.02)
        #     if isinstance(m, nn.ReLU) and m.bias is not None:
        #         nn.init.constant_(m.bias, 0)
        # elif isinstance(m, nn.Tanh):
        #     # trunc_normal_(m.weight, std=.02)
        #     if isinstance(m, nn.Tanh) and m.bias is not None:
        #         nn.init.constant_(m.bias, 0.01)
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # print("self.in_planes: ", self.in_planes)
        # print('self.avg_pool(x)_shape: ', self.avg_pool(x).shape)
        # avgout = self.sharedMLP(self.avg_pool(x))
        # maxout = self.sharedMLP(self.max_pool(x))
        avgout = self.conv2(self.relu(self.conv1(self.avg_pool(x))))
        maxout = self.conv2(self.relu(self.conv1(self.max_pool(x))))
        return self.act(avgout + maxout)

# class ChannelAttention(nn.Module):
#     """ Channel Attention Module """
#     def __init__(self, in_channels, ratio=16):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
#         self.max_pool = nn.AdaptiveMaxPool2d((1,1))
#         self.mlp = nn.Sequential(
#             nn.Linear(in_channels, in_channels//ratio),
#             nn.ReLU(),
#             nn.Linear(in_channels//ratio, in_channels)
#         )
    
#     def forward(self, x):
#         # print("x1_shape: ", x.shape)
#         avg_feat = self.mlp(self.avg_pool(x).flatten(1))
#         max_feat = self.mlp(self.max_pool(x).flatten(1))
#         att_feat = avg_feat + max_feat
#         att_weight = torch.sigmoid(att_feat).unsqueeze(2).unsqueeze(3)
#         return att_weight

class Splitting_by_channel(nn.Module):
    def __init__(self):
        super(Splitting_by_channel, self).__init__()
        # Deciding the stride base on the direction
        self.conv_even = lambda x: x[:, ::2, :, :]
        self.conv_odd = lambda x: x[:, 1::2, :, :]

    def forward(self, x):
        '''Returns the odd and even part'''
        return (self.conv_even(x), self.conv_odd(x))

class LiftingScheme_by_channel(nn.Module):
    def __init__(self, in_planes, num_heads=1, modified=False, splitting=True, k_size=4, simple_lifting=True, model_importance_method = 'hand'):
        super(LiftingScheme_by_channel, self).__init__()
        self.modified = modified
        self.num_heads = num_heads
        kernel_size = (k_size, k_size) 
        pad = (k_size // 2, k_size - 1 - k_size // 2, k_size // 2, k_size - 1 - k_size // 2)
        self.splitting = splitting
        self.model_importance_method = model_importance_method
        self.split = Splitting_by_channel()
        if model_importance_method == 'network':
            self.cam0 = ChannelAttention(int(in_planes * 0.5))
            self.cam1 = ChannelAttention(in_planes)
        # self.cam2 = ChannelAttention(in_planes * 2)

        # Dynamic build sequential network
        modules_P = []
        modules_U = []
        prev_size = 1
        modules_P1 = []
        modules_U1 = []

        # HARD CODED Architecture
        if simple_lifting:            
            modules_P += [
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_planes, in_planes,
                          kernel_size=kernel_size, stride=1),
                nn.Tanh()
            ]
            modules_U += [
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_planes, in_planes,
                          kernel_size=kernel_size, stride=1),
                nn.Tanh()
            ]
            modules_P1 += [
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_planes // 2, in_planes // 2,
                          kernel_size=kernel_size, stride=1),
                nn.Tanh()
            ]
            modules_U1 += [
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_planes // 2, in_planes // 2,
                          kernel_size=kernel_size, stride=1),
                nn.Tanh()
            ]
        else:#论文中的架构
            size_hidden = 2
            
            modules_P += [
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_planes*prev_size, in_planes*size_hidden,
                          kernel_size=kernel_size, stride=1),
                nn.ReLU()
            ]
            modules_U += [
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_planes*prev_size, in_planes*size_hidden,
                          kernel_size=kernel_size, stride=1),
                nn.ReLU()
            ]
            prev_size = size_hidden

            # Final dense
            modules_P += [
                nn.Conv2d(in_planes*prev_size, in_planes,
                          kernel_size=(1, 1), stride=1),
                nn.Tanh()
            ]
            modules_U += [
                nn.Conv2d(in_planes*prev_size, in_planes,
                          kernel_size=(1, 1), stride=1),
                nn.Tanh()
            ]

        self.P = nn.Sequential(*modules_P)
        self.U = nn.Sequential(*modules_U)
        self.P1 = nn.Sequential(*modules_P1)
        self.U1 = nn.Sequential(*modules_U1)
        # self.cam = channelAttention(2048)

    def forward(self, x):
        if self.num_heads == 2:
            if self.splitting:
                (x_even, x_odd) = self.split(x)
            else:
                (x_even, x_odd) = x

            if self.modified:#先更新再预测
                c = x_even + self.U(x_odd)
                d = x_odd - self.P(c)
            else:           #先预测再更新
                d = x_odd - self.P(x_even)
                c = x_even + self.U(d)
            if self.model_importance_method == 'hand':
                ## 4seg_att_first
                shape = c.shape   
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                c_att = torch.ones(shape[0], shape[1], shape[2], shape[3]) * 0.7
                d_att = torch.ones(shape[0], shape[1], shape[2], shape[3]) * 0.3
                att = torch.tanh(torch.cat((c_att, d_att), dim=1))
                att = att.to(device)
                x = torch.cat((c, d), dim=1)
                x = x * att
                channel = int(shape[1])
                c, d = x[:, :channel, :, :],  x[:, channel:2*channel, :, :]
                x = torch.stack((c,d),dim=2).reshape(shape[0], -1, shape[2], shape[3])
                ##
            elif self.model_importance_method == 'network':
                # 4seg_cam_att_first
                shape = c.shape
                x = torch.cat((c, d), dim=1)
                c_att, d_att = self.cam1(c), self.cam1(d)
                att = torch.cat((c_att, d_att), dim=1)
                x = x * att
                channel = int(shape[1])
                c, d = x[:, :channel, :, :],  x[:, channel:2*channel, :, :]
                x = torch.stack((c,d),dim=2).reshape(shape[0], -1, shape[2], shape[3])
            elif self.model_importance_method == 'similarity':               
                # c 计算相关性矩阵
                correlation_matrix = cal_cor2(c)
                # 对每一行进行求和
                c_row_sum = torch.sum(correlation_matrix, dim=1)
                # 将结果变成一维张量
                c_row_sum_1d = c_row_sum.view(-1)
                shape = c_row_sum_1d.shape[0]
                c_att = c_row_sum_1d.reshape((1, shape, 1, 1))
                # d
                # 计算相关性矩阵
                correlation_matrix = cal_cor2(d)
                d_row_sum = torch.sum(correlation_matrix, dim=1)
                d_row_sum_1d = d_row_sum.view(-1)
                shape = d_row_sum_1d.shape[0]
                d_att = d_row_sum_1d.reshape((1, shape, 1, 1))
                att = torch.cat((c_att, d_att), dim=1)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                att = att.to(device)
                x = x * att
                shape = c.shape
                channel = int(shape[1])
                c, d = x[:, :channel, :, :],  x[:, channel:2*channel, :, :]
                x = torch.stack((c,d),dim=2).reshape(shape[0], -1, shape[2], shape[3])                
                ##
            # return x * att
            return x

        elif self.num_heads == 4:
            (x_even, x_odd) = self.split(x)
            if self.modified:#先更新再预测
                c = x_even + self.U(x_odd)
                d = x_odd - self.P(c)
            else:           #先预测再更新
                d = x_odd - self.P(x_even)
                c = x_even + self.U(d)
            ####
            (c_even, c_odd) = self.split(c)
            if self.modified:#先更新再预测
                c1 = c_even + self.U1(c_odd)
                d1 = c_odd - self.P1(c1)
            else:           #先预测再更新
                d1 = c_odd - self.P1(c_even)
                c1 = c_even + self.U1(d1)
            # d
            (d_even, d_odd) = self.split(d)
            if self.modified:#先更新再预测
                c2 = d_even + self.U1(d_odd)
                d2 = d_odd - self.P1(c2)
            else:           #先预测再更新
                d2 = d_odd - self.P1(d_even)
                c2 = d_even + self.U1(d2)  
            # shape = c1.shape
            # f1 = torch.stack((c1,d1),dim=2).reshape(shape[0], -1, shape[2], shape[3])
            # shape = c2.shape
            # f2 = torch.stack((c2,d2),dim=2).reshape(shape[0], -1, shape[2], shape[3]) 
            # shape = f1.shape 
            # x = torch.stack((f1,f2),dim=2).reshape(shape[0], -1, shape[2], shape[3])
            # x = torch.cat((f1, f2), dim=1)
            # x = torch.cat((c1, d1, c2, d2), dim=1)

            if self.model_importance_method == 'hand':
                ## 4seg_att_first
                shape = c1.shape
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                c1_att = torch.ones(shape[0], shape[1], shape[2], shape[3]) * 0.9
                d1_att = torch.ones(shape[0], shape[1], shape[2], shape[3]) * 0.7
                c2_att = torch.ones(shape[0], shape[1], shape[2], shape[3]) * 0.5
                d2_att = torch.ones(shape[0], shape[1], shape[2], shape[3]) * 0.3
                att = torch.tanh(torch.cat((c1_att, d1_att, c2_att, d2_att), dim=1))
                att = att.to(device) 
                x = torch.cat((c1, d1, c2, d2), dim=1)
                x = x * att
                channel = int(shape[1])
                c1, d1, c2, d2 = x[:, :channel, :, :],  x[:, channel:2*channel, :, :], x[:, 2*channel:3*channel, :, :], x[:, 3*channel:4*channel, :, :]    
                f1 = torch.stack((c1,d1),dim=2).reshape(shape[0], -1, shape[2], shape[3])
                f2 = torch.stack((c2,d2),dim=2).reshape(shape[0], -1, shape[2], shape[3]) 
                shape = f1.shape
                x = torch.stack((f1,f2),dim=2).reshape(shape[0], -1, shape[2], shape[3])
                ###

            elif self.model_importance_method == 'network':
                # 4seg_cam_att_first
                shape = c1.shape
                x = torch.cat((c1, d1, c2, d2), dim=1)
                c1_att, d1_att, c2_att, d2_att = self.cam0(c1), self.cam0(d1), self.cam0(c2), self.cam0(d2)
                att = torch.cat((c1_att, d1_att, c2_att, d2_att), dim=1)
                x = x * att
                channel = int(shape[1])
                c1, d1, c2, d2 = x[:, :channel, :, :],  x[:, channel:2*channel, :, :], x[:, 2*channel:3*channel, :, :], x[:, 3*channel:4*channel, :, :]    
                f1 = torch.stack((c1,d1),dim=2).reshape(shape[0], -1, shape[2], shape[3])
                f2 = torch.stack((c2,d2),dim=2).reshape(shape[0], -1, shape[2], shape[3]) 
                shape = f1.shape
                x = torch.stack((f1,f2),dim=2).reshape(shape[0], -1, shape[2], shape[3])
                ##
            # att = torch.cat((c1, d1, c2, d2), dim=1)
            # return x * self.cam(att)
            elif self.model_importance_method == 'similarity':               
                # c1 计算相关性矩阵
                correlation_matrix = cal_cor2(c1)
                row_sum = torch.sum(correlation_matrix, dim=1)
                row_sum_1d = row_sum.view(-1)
                c1_att = row_sum_1d.reshape((1, row_sum_1d.shape[0], 1, 1))
                # d1
                # 计算相关性矩阵
                correlation_matrix = cal_cor2(d1)
                row_sum = torch.sum(correlation_matrix, dim=1)
                row_sum_1d = row_sum.view(-1)
                d1_att = row_sum_1d.reshape((1, row_sum_1d.shape[0], 1, 1))
                # c2
                # 计算相关性矩阵
                correlation_matrix = cal_cor2(c2)
                row_sum = torch.sum(correlation_matrix, dim=1)
                row_sum_1d = row_sum.view(-1)
                c2_att = row_sum_1d.reshape((1, row_sum_1d.shape[0], 1, 1))
                # d2
                # 计算相关性矩阵
                correlation_matrix = cal_cor2(d2)
                row_sum = torch.sum(correlation_matrix, dim=1)
                row_sum_1d = row_sum.view(-1)
                d2_att = row_sum_1d.reshape((1, row_sum_1d.shape[0], 1, 1))

                att = torch.cat((c1_att, d1_att, c2_att, d2_att), dim=1)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                att = att.to(device)
                x = x * att
                shape = c1.shape
                channel = int(shape[1])
                c1, d1, c2, d2 = x[:, :channel, :, :],  x[:, channel:2*channel, :, :], x[:, 2*channel:3*channel, :, :], x[:, 3*channel:4*channel, :, :]    
                f1 = torch.stack((c1,d1),dim=2).reshape(shape[0], -1, shape[2], shape[3])
                f2 = torch.stack((c2,d2),dim=2).reshape(shape[0], -1, shape[2], shape[3]) 
                shape = f1.shape
                x = torch.stack((f1,f2),dim=2).reshape(shape[0], -1, shape[2], shape[3])
            return x
        else:
            return x
        
        ##通道
        # shape = c.shape
        # # print(shape)
        # f = torch.stack((c,d),dim=2).view(shape[0], -1, shape[2], shape[3])
        # # f = x * self.cam(f)
        # return f


class LiftingScheme2D_by_channel(nn.Module):
    def __init__(self, in_planes, num_heads=1, modified=False, kernel_size=4, simple_lifting=True):
        super(LiftingScheme2D_by_channel, self).__init__()
        self.level1_lf = LiftingScheme_by_channel(
            in_planes=in_planes, num_heads=num_heads, modified=modified,
            k_size=kernel_size, simple_lifting=simple_lifting)

    def forward(self, x):
        '''Returns (LL, LH, HL, HH)'''
        # (c, d) = self.level1_lf(x)#c是低频
        f = self.level1_lf(x)
        # return torch.cat((c, d), dim=1)
        # print("c:", c)
        # print("d:", d)
        # shape = c.shape
        # f = torch.stack((c,d),dim=2).reshape(shape[0], -1, shape[2], shape[3])
        # return (c, d)
        # print("f:", f)
        # print(f[0][0].eq(c[0][0]))
        # print(f[0][2].eq(c[0][1]))
        # if (f[0][0] == c[0][0]):
        #     if f[0][1] == d[0][1]:
        #         print("True!!")
        # else:
        #     print("False!!")
        return f

############################################################################
class Splitting(nn.Module):
    def __init__(self, horizontal):
        super(Splitting, self).__init__()
        # Deciding the stride base on the direction
        self.horizontal = horizontal
        if(horizontal):
            self.conv_even = lambda x: x[:, :, :, ::2]
            self.conv_odd = lambda x: x[:, :, :, 1::2]
        else:
            self.conv_even = lambda x: x[:, :, ::2, :]
            self.conv_odd = lambda x: x[:, :, 1::2, :]

    def forward(self, x):
        '''Returns the odd and even part'''
        return (self.conv_even(x), self.conv_odd(x))


class WaveletHaar(nn.Module):
    def __init__(self, horizontal):
        super(WaveletHaar, self).__init__()
        self.split = Splitting(horizontal)
        self.norm = math.sqrt(2.0)

    def forward(self, x):
        '''Returns the approximation and detail part'''
        (x_even, x_odd) = self.split(x)

        # Haar wavelet definition
        d = (x_odd - x_even) / self.norm
        c = (x_odd + x_even) / self.norm
        return (c, d)


class WaveletHaar2D(nn.Module):
    def __init__(self):
        super(WaveletHaar2D, self).__init__()
        self.horizontal_haar = WaveletHaar(horizontal=True)
        self.vertical_haar = WaveletHaar(horizontal=False)

    def forward(self, x):
        '''Returns (LL, LH, HL, HH)'''
        (c, d) = self.horizontal_haar(x)
        (LL, LH) = self.vertical_haar(c)
        (HL, HH) = self.vertical_haar(d)
        return (LL, LH, HL, HH)


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

    def forward(self, x):
        '''Returns the approximation and detail part'''
        x = self.padding(x)
        return (self.conv_low(x), self.conv_high(x))


class Wavelet2D(nn.Module):
    def __init__(self, in_planes, name="db1"):
        super(Wavelet2D, self).__init__()
        self.horizontal_wavelet = Wavelet(
            in_planes, horizontal=True, name=name)
        self.vertical_wavelet = Wavelet(in_planes, horizontal=False, name=name)

    def forward(self, x):
        '''Returns (LL, LH, HL, HH)'''
        (c, d) = self.horizontal_wavelet(x)
        (LL, LH) = self.vertical_wavelet(c)
        (HL, HH) = self.vertical_wavelet(d)
        return (LL, LH, HL, HH)


class LiftingScheme(nn.Module):
    def __init__(self, horizontal, in_planes, modified=True, size=[], splitting=True, k_size=4, simple_lifting=False):
        super(LiftingScheme, self).__init__()
        self.modified = modified
        if horizontal:
            kernel_size = (1, k_size)
            pad = (k_size // 2, k_size - 1 - k_size // 2, 0, 0)
        else:
            kernel_size = (k_size, 1)
            pad = (0, 0, k_size // 2, k_size - 1 - k_size // 2)

        self.splitting = splitting
        self.split = Splitting(horizontal)

        # Dynamic build sequential network
        modules_P = []
        modules_U = []
        prev_size = 1

        # HARD CODED Architecture
        if simple_lifting:            
            modules_P += [
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_planes, in_planes,
                          kernel_size=kernel_size, stride=1),
                nn.Tanh()
            ]
            modules_U += [
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_planes, in_planes,
                          kernel_size=kernel_size, stride=1),
                nn.Tanh()
            ]
        else:#论文中的架构
            size_hidden = 2
            
            modules_P += [
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_planes*prev_size, in_planes*size_hidden,
                          kernel_size=kernel_size, stride=1),
                nn.ReLU()
            ]
            modules_U += [
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_planes*prev_size, in_planes*size_hidden,
                          kernel_size=kernel_size, stride=1),
                nn.ReLU()
            ]
            prev_size = size_hidden

            # Final dense
            modules_P += [
                nn.Conv2d(in_planes*prev_size, in_planes,
                          kernel_size=(1, 1), stride=1),
                nn.Tanh()
            ]
            modules_U += [
                nn.Conv2d(in_planes*prev_size, in_planes,
                          kernel_size=(1, 1), stride=1),
                nn.Tanh()
            ]

        self.P = nn.Sequential(*modules_P)
        self.U = nn.Sequential(*modules_U)

    def forward(self, x):
        if self.splitting:
            (x_even, x_odd) = self.split(x)
        else:
            (x_even, x_odd) = x

        if self.modified:#先更新再预测
            c = x_even + self.U(x_odd)
            d = x_odd - self.P(c)
            return (c, d)
        else:           #先预测再更新
            d = x_odd - self.P(x_even)
            c = x_even + self.U(d)
            return (c, d)


class LiftingScheme2D(nn.Module):
    def __init__(self, in_planes, share_weights, modified=True, size=[2, 1], kernel_size=4, simple_lifting=False):
        super(LiftingScheme2D, self).__init__()
        self.level1_lf = LiftingScheme(
            horizontal=HORIZONTAL_FIRST, in_planes=in_planes, modified=modified,
            size=size, k_size=kernel_size, simple_lifting=simple_lifting)
        self.share_weights = share_weights
        if share_weights:
            self.level2_1_lf = LiftingScheme(
                horizontal=not HORIZONTAL_FIRST,  in_planes=in_planes, modified=modified,
                size=size, k_size=kernel_size, simple_lifting=simple_lifting)
            self.level2_2_lf = self.level2_1_lf  # Double check this
        else:
            self.level2_1_lf = LiftingScheme(
                horizontal=not HORIZONTAL_FIRST,  in_planes=in_planes, modified=modified,
                size=size, k_size=kernel_size, simple_lifting=simple_lifting)
            self.level2_2_lf = LiftingScheme(
                horizontal=not HORIZONTAL_FIRST,  in_planes=in_planes, modified=modified,
                size=size, k_size=kernel_size, simple_lifting=simple_lifting)

    def forward(self, x):
        '''Returns (LL, LH, HL, HH)'''
        (c, d) = self.level1_lf(x)
        (LL, LH) = self.level2_1_lf(c)
        (HL, HH) = self.level2_2_lf(d)
        return (c, d, LL, LH, HL, HH)


if __name__ == "__main__":
    input = torch.randn(1, 6, 10, 10)
    #m_harr = WaveletLiftingHaar2D()
    # m_wavelet = Wavelet2D(1, name="db2")
    # print(input)
    # print(m_wavelet(input))
    # lift = LiftingScheme2D(6, share_weights=False, simple_lifting=False)
    # (c, d, LL, LH, HL, HH) = lift(input)
    # print(LL.shape)
    lift_by_channel = LiftingScheme2D_by_channel(3, share_weights=False, kernel_size=3, simple_lifting=False)
    print(lift_by_channel)
    (c, d) = lift_by_channel(input)
    print(c.shape)
    # TODO: Do more experiments with the code
