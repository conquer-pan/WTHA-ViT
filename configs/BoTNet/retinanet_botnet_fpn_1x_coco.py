_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
P='/home/omeneis/pjh/mmdetection-2.26.0/work_COCO/retinanet/epoch_12.pth'
model = dict(
    pretrained='/home/omeneis/pjh/mmdetection-2.26.0/work_COCO/retinanet/epoch_12.pth',
    backbone=dict(
        _delete_=True,
        type='BoTNet',
        # type='WTHA_ViT_s',
        init_cfg=dict(type='Pretrained', checkpoint=P),
        style='pytorch'),)

fp16 = dict(loss_scale=512.)
find_unused_parameters = True
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,)

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)
# optimizer = dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0001)
load_from = "/home/omeneis/pjh/mmdetection-2.26.0/work_COCO/retinanet/epoch_12.pth"
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    )
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric='bbox')



