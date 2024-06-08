_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# _base_ = './cascade_rcnn_r50_fpn_1x_coco.py'
model = dict(
    pretrained='/home/omeneis/pjh/mmdetection-2.26.0/work_COCO/cascade/epoch_12.pth',
    backbone=dict(
        _delete_=True,
        type='wavevit_s',
        init_cfg=dict(type='Pretrained', checkpoint='/home/omeneis/pjh/mmdetection-2.26.0/work_COCO/cascade/epoch_12.pth'),
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 448],))

fp16 = dict(loss_scale=512.)
find_unused_parameters = True
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    )
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric='bbox')
load_from = "/home/omeneis/pjh/mmdetection-2.26.0/work_COCO/cascade/epoch_12.pth"
