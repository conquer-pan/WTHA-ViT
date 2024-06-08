_base_ = './retinanet_pcpvt_s_fpn_1x_coco_pvt_setting.py'

model = dict(
    pretrained='/home/omeneis/pjh/mmdetection-2.26.0/work_COCO/retinanet/epoch_12.pth',
    backbone=dict(
        type='alt_gvt_large',
        init_cfg=dict(type='Pretrained', checkpoint='/home/omeneis/pjh/mmdetection-2.26.0/work_COCO/retinanet/epoch_12.pth'),
        style='pytorch'),
    neck=dict(
        in_channels=[128, 256, 512, 1024],
        out_channels=256,))
load_from = "/home/omeneis/pjh/mmdetection-2.26.0/work_COCO/retinanet/epoch_12.pth"

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    )
# auto_scale_lr = dict(base_batch_size=2)
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric='bbox')