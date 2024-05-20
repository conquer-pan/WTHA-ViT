import math
import pywt
import torch
import torch.nn as nn
from torch.autograd import Function
from functools import partial

class IDWT_2D(nn.Module):
    def __init__(self, wave):
        super(IDWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)
        
        w_ll = rec_lo.unsqueeze(0)*rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0)*rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0)*rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0)*rec_hi.unsqueeze(1)

        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)

        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer('filters', filters)
        # self.filters = self.filters.to(dtype=torch.float16)


    def forward(self, LL, LH = None, HL = None, HH = None):
        B, C, H, W = LL.shape
        if LH == None:
            LH = torch.zeros(B, C, H, W)
        if HL == None:
            HL = torch.zeros(B, C, H, W)
        if HH == None:
            HH = torch.zeros(B, C, H, W)
        device = torch.device("cuda")
        LH, HL, HH = LH.to(device), HL.to(device), HH.to(device)
        x = torch.cat([LL, LH, HL, HH], dim = 1)
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        # C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = self.filters.repeat(C, 1, 1, 1)
        # print("filters.shape:", filters.shape)
        # print("x.shape:", x.shape)
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        return x

class DWT_2D(nn.Module):
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1]) #逆序
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0)*dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0)*dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0)*dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0)*dec_hi.unsqueeze(1)

        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))#该方法的作用是定义一组参数，该组参数的特别之处在于：
                                                                    #模型训练时不会更新（即调用 optimizer.step() 后该组参数不会变化，只可人为地改变它们的值），
                                                                    #但是保存模型时，该组参数又作为模型参数不可或缺的一部分被保存。
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

        # self.w_ll = self.w_ll.to(dtype=torch.float16)
        # self.w_lh = self.w_lh.to(dtype=torch.float16)
        # self.w_hl = self.w_hl.to(dtype=torch.float16)
        # self.w_hh = self.w_hh.to(dtype=torch.float16)

    def forward(self, x):
        dim = x.shape[1] #通道C
        x_ll = torch.nn.functional.conv2d(x, self.w_ll.expand(dim, -1, -1, -1), stride = 2, groups = dim)#torch.nn.functional.conv2d(input,filters,bias,stride,padding,dilation,groups)
        x_lh = torch.nn.functional.conv2d(x, self.w_lh.expand(dim, -1, -1, -1), stride = 2, groups = dim)#filters代表卷积核的大小(out_channels，in_channe/groups，H，W)，是一个四维tensor
        x_hl = torch.nn.functional.conv2d(x, self.w_hl.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x_hh = torch.nn.functional.conv2d(x, self.w_hh.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        return x_ll, x_lh, x_hl, x_hh
        
def start():
    x = torch.Tensor([[[[1,2,3,4],
                        [4,5,3,8],
                        [6,7,1,2],
                        [5,9,6,3]],
                        [[1,2,3,4],
                        [5,6,7,8],
                        [9,8,7,6],
                        [5,4,3,2]]]])
    print(x)
    print(x.shape)
    haar_wav = DWT_2D(wave='rbio1.1')
    LL, LH, HL, HH = haar_wav(x)
    print(LL.shape)
    # print(LL, LH, HL, HH)

    haar_idwt = IDWT_2D(wave='rbio1.1')
    y = haar_idwt(LL, None, None, None)
    # y = haar_idwt(LL, LH, HL, HH)
    print(y)

#start()