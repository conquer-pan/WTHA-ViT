_base_ = 'retinanet_pvtv2-b0_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        embed_dims=64,
        num_layers=[3, 4, 6, 3],
        init_cfg=dict(checkpoint='/home/omeneis/pjh/mmdetection-2.26.0/pvtv2_b2.pth')),
    neck=dict(in_channels=[64, 128, 320, 512]))
