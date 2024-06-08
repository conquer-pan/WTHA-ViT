_base_ = './retinanet_pcpvt_s_fpn_1x_coco_pvt_setting.py'

model = dict(
    # pretrained='/home/omeneis/pjh/mmdetection-2.26.0/pretrain/alt_gvt_small.pth',
    backbone=dict(
        _delete_=True,
        type='alt_gvt_small',
        init_cfg=dict(type='Pretrained', checkpoint='/home/omeneis/pjh/mmdetection-2.26.0/pretrain/alt_gvt_small.pth'),
        style='pytorch'),
    neck=dict(
        in_channels=[64, 128, 256, 512],
        out_channels=256,))

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    )
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
coptimizer = dict(_delete_=True, type='AdamW', lr=0.00005, weight_decay=0.0001)
load_from = "/home/omeneis/pjh/mmdetection-2.26.0/work_COCO/retinanet/epoch_12.pth"
# auto_scale_lr = dict(base_batch_size=4)
checkpoint_config = dict(interval=2)
evaluation = dict(interval=2, metric='bbox')