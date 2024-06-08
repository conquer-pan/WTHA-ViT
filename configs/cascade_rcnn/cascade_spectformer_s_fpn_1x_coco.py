_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
# _base_ = './cascade_rcnn_r50_fpn_1x_coco.py'
# optimizer
model = dict(
    pretrained='/home/omeneis/pjh/mmdetection-2.26.0/work_COCO/cascade/epoch_12.pth',
    backbone=dict(
        type='spectformer_s',
        style='pytorch'),
    neck=dict(
        # type='FPN',
        in_channels=[64, 128, 320, 448],
        # out_channels=256,
        # start_level=1,
        # add_extra_convs='on_output',
        # num_outs=5
        ))

load_from = "/home/omeneis/pjh/mmdetection-2.26.0/work_COCO/cascade/epoch_12.pth"

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)
# optimizer = dict(_delete_=True, type='AdamW', lr=0.00005, weight_decay=0.0001)
# optimizer = dict(_delete_=True, type='AdamW', lr=0.00001, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
# runner = dict(type='EpochBasedRunner', max_epochs=12)
fp16 = dict(loss_scale=512.)
find_unused_parameters = True
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    )
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric='bbox')
