_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    type='RetinaNet',
    backbone=dict(
        _delete_=True,
        type='PyramidVisionTransformer',
        num_layers=[2, 2, 2, 2],
        init_cfg=dict(checkpoint='/home/omeneis/pjh/mmdetection-2.23.0/pvt_tiny.pth')),
    neck=dict(in_channels=[64, 128, 320, 512]))
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    )
load_from = "/home/omeneis/pjh/mmdetection-2.26.0/work_COCO/retinanet/epoch_12.pth"
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric='bbox')