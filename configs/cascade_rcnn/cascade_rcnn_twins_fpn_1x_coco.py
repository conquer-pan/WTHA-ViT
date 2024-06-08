_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# data = dict(
#     samples_per_gpu=16,
#     workers_per_gpu=8,
#     )
model = dict(
    pretrained='/home/omeneis/pjh/mmdetection-2.26.0/pretrain/pcpvt_small.pth',
    backbone=dict(
        _delete_=True,
        type='pcpvt_small',
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],))
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)
load_from = "/home/omeneis/pjh/mmdetection-2.26.0/work_COCO/cascade/epoch_12.pth"