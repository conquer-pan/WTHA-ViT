#coding: utf-8
import cv2
import mmcv
import numpy as np
import os
import torch

from mmdet.apis import inference_detector, init_detector, show_result_pyplot

def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    heatmap = feature_map[:,0,:,:]*0
    for c in range(feature_map.shape[1]):
        heatmap+=feature_map[:,c,:,:]
    heatmap = heatmap.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap

def draw_feature_map(model, img_path, save_dir):
    '''
    :param model: 加载了参数的模型
    :param img_path: 测试图像的文件路径
    :param save_dir: 保存生成图像的文件夹
    :return:
    '''
    img = mmcv.imread(img_path)
    img_name = img_path.split('/')[-1]
    modeltype = str(type(model)).split('.')[-1].split('\'')[0]
    model.eval()
    model.draw_heatmap = True
    featuremaps,result = inference_detector(model, img) #这里需要改model，让其在forward的最后return特征图。我这里return的是一个Tensor的tuple，每个Tensor对应一个level上输出的特征图。
    i=0
    for featuremap in featuremaps:
        heatmap = featuremap_2_heatmap(featuremap)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
        heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
        superimposed_img = heatmap * 0.7 + img*0.4  # 这里的0.4是热力图强度因子
        # cv2.imshow("1",superimposed_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite(os.path.join(save_dir,'featuremap_'+str(i)+'_'+str(img_name)), superimposed_img)  # 将图像保存到硬盘
        i=i+1
    show_result_pyplot(model, img, result, score_thr=0.05)


from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image folder')
    parser.add_argument('save_dir', help='Dir to save heatmap image')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # draw_feature_map(model,args.img,args.save_dir)
    imgs_list = os.listdir(args.img)
    for i in imgs_list:
        draw_feature_map(model,os.path.join(args.img, i),args.save_dir)

if __name__ == '__main__':
    main()
