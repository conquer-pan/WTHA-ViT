_base_ = './cascade_rcnn_twins_fpn_1x_coco.py'
# data = dict(
#     samples_per_gpu=16,
#     workers_per_gpu=8,
#     )
model = dict(
    pretrained='/home/omeneis/pjh/mmdetection-2.26.0/pretrain/pcpvt_small.pth',
    backbone=dict(
        type='alt_gvt_small',
        init_cfg=dict(type='Pretrained', checkpoint='/home/omeneis/pjh/mmdetection-2.26.0/pretrain/alt_gvt_small.pth'),
        style='pytorch'),
    neck=dict(
        in_channels=[64, 128, 256, 512],
        out_channels=256,))
# load_from = "/home/omeneis/SDA/data/work_DOTA1-1024/afno_t/epoch_12.pth"
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    )
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric='bbox')

load_from = "/home/omeneis/pjh/mmdetection-2.26.0/work_COCO/cascade/epoch_12.pth"