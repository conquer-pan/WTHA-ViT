import math
import torch
import torch.nn as nn
import numpy as np
import cv2 as cv
import torchvision.transforms as transforms
import pywt
import matplotlib.image as mpig
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# from mmdet.models.backbones.WTHA_ViT import Wavelet2D,WaveletTree
from mmdet.models.backbones.WaveletTransform_filters import DWT_2D

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
        # device = torch.device("cuda")
        # LH, HL, HH = LH.to(device), HL.to(device), HH.to(device)
        x = torch.cat([LL, LH, HL, HH], dim = 1)
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        # C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = self.filters.repeat(C, 1, 1, 1)
        # print("filters.shape:", filters.shape)
        # print("x.shape:", x.shape)
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        return x

class inverse_wav_transform(nn.Module):
    def __init__(self, wave="haar"):
        super(inverse_wav_transform, self).__init__()
        self.wave = wave

    def forward(self, x ,LH, HL, HH):
    # def forward(self, x):
        # print("X.shape: ", x.shape)
        x = x.cpu().detach().numpy() 
        w = pywt.Wavelet(self.wave)
        x = pywt.idwt2((x, (LH, HL, HH)), wavelet = w)   
        # x = pywt.idwt2((x, (None, None, None)), self.wave)
        x = torch.from_numpy(x)
        # device = torch.device("cuda:0")
        # x = x.to(device)
        return x

class wav_transform(nn.Module):
    def __init__(self, wave="haar"):
        super(wav_transform, self).__init__()
        self.wave = wave

    def forward(self, x):
        x = x.cpu().detach().numpy() 
        LL, (LH, HL, HH) = pywt.dwt2(x, 'haar')
        LL = torch.from_numpy(LL)
        LH = torch.from_numpy(LH)
        HL = torch.from_numpy(HL)
        HH = torch.from_numpy(HH)
        # device = torch.device("cuda:0")
        # LL = LL.to(device)
        # LH = LH.to(device)
        # HL = HL.to(device)
        # HH = HH.to(device)
        return LL, LH, HL, HH

# 图像缩放
def resize_pic():

    x = cv.imread('/home/omeneis/pjh/dataset/P1042/P1042.png')

    #plt.imshow(x)

    x=transforms.ToTensor()(x)
    x=torch.unsqueeze(x,0)

    # 进行缩放
    x=transforms.Resize(size=(1024,1024))(x)
    x_ = torch.permute(x[0,:,:,:],dims=[1,2,0])
    x_ = np.array(x_)*255
    cv.imwrite('/home/omeneis/pjh/dataset/P1042/P1042_1024.png',x_)

# dwt_idwt
def dwt_idwt():
    # 输入图像检验代码是否正常运行
    # haar_wav = Wavelet2D(3,name="haar")
    # haar_wav = wav_transform(wave="haar")
    haar_wav = DWT_2D(wave="haar")

    # x = cv.imread('/home/omeneis/hw/dataset/lena.jpg')
    x = cv.imread('/home/omeneis/pjh/P0991.png')

    #plt.imshow(x)

    x=transforms.ToTensor()(x)
    x=torch.unsqueeze(x,0)

    #print(x)
    #print(x.shape)

    LL ,LH ,HL ,HH = haar_wav(x)
    # LL2 ,LH2 ,HL2 ,HH2 = haar_wav(LL)
    # LL3 ,LH3 ,HL3 ,HH3 = haar_wav(LL2)

    x_LL_image = torch.permute(LL[0,:,:,:],dims=[1,2,0])
    x_LL_array = np.array(x_LL_image)*255
    cv.imwrite('/home/omeneis/pjh/dataset/lena_LL.jpg',x_LL_array)

    # x_LL_image = torch.permute(LL2[0,:,:,:],dims=[1,2,0])
    # x_LL_array = np.array(x_LL_image)*255
    # cv.imwrite('/home/omeneis/pjh/dataset/lena_LL2.jpg',x_LL_array)

    # x_LL_image = torch.permute(LL3[0,:,:,:],dims=[1,2,0])
    # x_LL_array = np.array(x_LL_image)*255
    # cv.imwrite('/home/omeneis/pjh/dataset/lena_LL3.jpg',x_LL_array)

##
    # haar_idwt = inverse_wav_transform(wave="haar")
    # x_idwt = haar_idwt(LL3, LH3, HL3, HH3)
    # x_idwt_image = torch.permute(x_idwt[0,:,:,:],dims=[1,2,0])
    # x_idwt_image = x_idwt_image.cpu().detach().numpy()
    # x_idwt_array = np.array(x_idwt_image)*255
    # cv.imwrite('/home/omeneis/pjh/dataset/lena_idwt2.jpg',x_idwt_array)

    # x_idwt = haar_idwt(x_idwt, LH2, HL2, HH2)
    # x_idwt_image = torch.permute(x_idwt[0,:,:,:],dims=[1,2,0])
    # x_idwt_image = x_idwt_image.cpu().detach().numpy()
    # x_idwt_array = np.array(x_idwt_image)*255
    # cv.imwrite('/home/omeneis/pjh/dataset/lena_idwt1.jpg',x_idwt_array)

    x_idwt = haar_idwt(x_idwt, LH, HL, HH)
    x_idwt_image = torch.permute(x_idwt[0,:,:,:],dims=[1,2,0])
    x_idwt_image = x_idwt_image.cpu().detach().numpy()
    x_idwt_array = np.array(x_idwt_image)*255
    cv.imwrite('/home/omeneis/pjh/dataset/lena_idwt0.jpg',x_idwt_array)

def down_up_sample():
    x = cv.imread('/home/omeneis/pjh/dataset/P1042/P1042_1024.png')
    # x = cv.imread('/home/omeneis/pjh/P0991.png')
    #plt.imshow(x)

    x=transforms.ToTensor()(x)
    x=torch.unsqueeze(x,0)

    average_pool = nn.AvgPool2d(2,stride=2)
    pool_out = average_pool(x)

    x_down_image = torch.permute(pool_out[0,:,:,:],dims=[1,2,0])
    x_down_array = np.array(x_down_image)*255
    cv.imwrite('/home/omeneis/pjh/dataset/P1042/downsample.png',x_down_array)

    # average_unpool = nn.UpsamplingNearest2d(scale_factor=2)
    average_unpool = nn.UpsamplingBilinear2d(scale_factor=2)
    unpool_out = average_unpool(pool_out)
    
    x_up_image = torch.permute(unpool_out[0,:,:,:],dims=[1,2,0])
    x_up_array = np.array(x_up_image)*255
    cv.imwrite('/home/omeneis/pjh/dataset/P1042/up.png',x_up_array)

def wt_iwt():
    # x = cv.imread('/home/omeneis/hw/dataset/lena.jpg')
    # x = cv.imread('/home/omeneis/pjh/P0991.png')
    # x = cv.imread('/home/omeneis/pjh/P0128.png')
    x = cv.imread('/home/omeneis/pjh/dataset/P1042/P1042_1024.png')

    #plt.imshow(x)

    x=transforms.ToTensor()(x)
    x=torch.unsqueeze(x,0)

    wavelet_filter = DWT_2D(wave="haar")
    LL, LH, HL, HH = wavelet_filter(x)
    LL2, LH2, HL2, HH2 = wavelet_filter(LL)
    LL3, LH3, HL3, HH3 = wavelet_filter(LL2)
#
    image = torch.permute(LL[0,:,:,:],dims=[1,2,0])
    image = image.cpu().detach().numpy()
    array = np.array(image)*255
    cv.imwrite('/home/omeneis/pjh/dataset/P1042/LL.jpg',array)
    #
    image = torch.permute(LH[0,:,:,:],dims=[1,2,0])
    image = image.cpu().detach().numpy()
    array = np.array(image)*255
    cv.imwrite('/home/omeneis/pjh/dataset/P1042/LH.jpg',array)
    #
    image = torch.permute(HL[0,:,:,:],dims=[1,2,0])
    image = image.cpu().detach().numpy()
    array = np.array(image)*255
    cv.imwrite('/home/omeneis/pjh/dataset/P1042/HL.jpg',array)
    #
    image = torch.permute(HH[0,:,:,:],dims=[1,2,0])
    image = image.cpu().detach().numpy()
    array = np.array(image)*255
    cv.imwrite('/home/omeneis/pjh/dataset/P1042/HH.jpg',array)
    #
    image = torch.permute(LL2[0,:,:,:],dims=[1,2,0])
    image = image.cpu().detach().numpy()
    array = np.array(image)*255
    cv.imwrite('/home/omeneis/pjh/dataset/P1042/LL2.jpg',array)
    #
    image = torch.permute(LH2[0,:,:,:],dims=[1,2,0])
    image = image.cpu().detach().numpy()
    array = np.array(image)*255
    cv.imwrite('/home/omeneis/pjh/dataset/P1042/LH2.jpg',array)
    #
    image = torch.permute(HL2[0,:,:,:],dims=[1,2,0])
    image = image.cpu().detach().numpy()
    array = np.array(image)*255
    cv.imwrite('/home/omeneis/pjh/dataset/P1042/HL2.jpg',array)
    #
    image = torch.permute(HH2[0,:,:,:],dims=[1,2,0])
    image = image.cpu().detach().numpy()
    array = np.array(image)*255
    cv.imwrite('/home/omeneis/pjh/dataset/P1042/HH2.jpg',array)
    #
    image = torch.permute(LL3[0,:,:,:],dims=[1,2,0])
    image = image.cpu().detach().numpy()
    array = np.array(image)*255
    cv.imwrite('/home/omeneis/pjh/dataset/P1042/LL3.jpg',array)
    #
    image = torch.permute(LH3[0,:,:,:],dims=[1,2,0])
    image = image.cpu().detach().numpy()
    array = np.array(image)*255
    cv.imwrite('/home/omeneis/pjh/dataset/P1042/LH3.jpg',array)
    #
    image = torch.permute(HL3[0,:,:,:],dims=[1,2,0])
    image = image.cpu().detach().numpy()
    array = np.array(image)*255
    cv.imwrite('/home/omeneis/pjh/dataset/P1042/HL3.jpg',array)
    #
    image = torch.permute(HH3[0,:,:,:],dims=[1,2,0])
    image = image.cpu().detach().numpy()
    array = np.array(image)*255
    cv.imwrite('/home/omeneis/pjh/dataset/P1042/HH3.jpg',array)
#
    haar_idwt = IDWT_2D(wave="haar")
    idwt4 = haar_idwt(LL3, LH3, HL3, None)
    idwt2 = haar_idwt(idwt4, LH2, HL2, None)
    idwt = haar_idwt(idwt2, LH, HL, None)
    ##
    idwt_LL_only = haar_idwt(LL, None, None, None)
    image = torch.permute(idwt_LL_only[0,:,:,:],dims=[1,2,0])
    image = image.cpu().detach().numpy()
    array = np.array(image)*255
    cv.imwrite('/home/omeneis/pjh/dataset/P1042/idwt_LL_only.jpg', array)
    ##
    #
    image = torch.permute(idwt4[0,:,:,:],dims=[1,2,0])
    image = image.cpu().detach().numpy()
    array = np.array(image)*255
    cv.imwrite('/home/omeneis/pjh/dataset/P1042/lena_iwt4.jpg', array)
    #
    image = torch.permute(idwt2[0,:,:,:],dims=[1,2,0])
    image = image.cpu().detach().numpy()
    array = np.array(image)*255
    cv.imwrite('/home/omeneis/pjh/dataset/P1042/lena_iwt2.jpg', array)
    #
    image = torch.permute(idwt[0,:,:,:],dims=[1,2,0])
    image = image.cpu().detach().numpy()
    array = np.array(image)*255
    cv.imwrite('/home/omeneis/pjh/dataset/P1042/lena_iwt.jpg', array)


    # wtree = WaveletTree(in_planes=3,sr_ratio=2)

    # x_wt = wtree(x)
    # x_wt_image = torch.permute(x_wt[0,:,:,:],dims=[1,2,0])
    # # x_wt_image = x_wt[0,:,:,:].permute(1,2,0)
    # x_wt_image = x_wt_image.cpu().detach().numpy()
    # x_wt_array = np.array(x_wt_image)*255
    # cv.imwrite('/home/omeneis/pjh/dataset/lena_wt2.jpg',x_wt_array)

    # haar_idwt = inverse_wav_transform(wave="haar")
    # # average_unpool = nn.UpsamplingNearest2d(scale_factor=2)
    # # x_iwt = average_unpool(x_wt)
    # x_iwt = haar_idwt(x_wt, None, None, None)     # sr_ratio=2，进行以此idwt恢复原分辨率
    # # x_iwt = haar_idwt(x_iwt)    # sr_ratio=4，进行两次idwt
    # # x_iwt = haar_idwt(x_iwt)    # sr_ratio=8,进行三次idwt
    # x_iwt_image = torch.permute(x_iwt[0,:,:,:],dims=[1,2,0])
    # x_iwt_image = x_iwt_image.cpu().detach().numpy()
    # x_iwt_array = np.array(x_iwt_image)*255
    # cv.imwrite('/home/omeneis/pjh/dataset/lena_iwt2.jpg',x_iwt_array)

def test():
    # x = torch.Tensor([[[[1,2,3,4,5,6,7,8],
    #                 [4,5,3,8,3,8,5,9],
    #                 [6,7,1,2,1,4,6,3],
    #                 [5,9,6,3,3,5,9,3]]]])
    x = torch.Tensor([[[[1,2,3,4],
                    [4,5,3,8],
                    [6,7,1,2],
                    [5,9,6,3]]]])
    print(x.shape)
    haar_wav = Wavelet2D(1, name="haar")
    LL, LH, HL, HH = haar_wav(x)
    print(LL)
    haar_idwt = inverse_wav_transform(wave="haar")
    x_idwt = haar_idwt(LL, LH, HL, HH)
    print(x_idwt)

    #
    haar_wav2 = DWT_2D(wave="haar")
    LL, LH, HL, HH = haar_wav2(x)
    print("LL_2: ", LL)
    x_idwt2 = haar_idwt(LL, LH, HL, HH)
    print("x_idwt2: ", x_idwt2)


wt_iwt()
# dwt_idwt()
# test()
# resize_pic()
# down_up_sample()