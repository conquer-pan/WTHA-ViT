_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
Pretrained='/home/omeneis/pjh/mmdetection-2.26.0/work_COCO/retinanet/epoch_12.pth'
# optimizer
model = dict(
    pretrained='/home/omeneis/pjh/mmdetection-2.26.0/work_COCO/retinanet/epoch_12.pth',
    backbone=dict(
        _delete_=True,
        type='WTHA_ViT_s_network2',
        # type='WTHA_ViT_s',
        init_cfg=dict(type='Pretrained', checkpoint=Pretrained),
        style='pytorch'),
    neck=dict(
        type='FPN',
        # in_channels=[64, 128, 320, 448],
        in_channels = [64, 128, 256, 448],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5))

load_from = '/home/omeneis/pjh/mmdetection-2.26.0/work_COCO/retinanet/epoch_12.pth' #coco预
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)
# lr_config = dict(_delete_=True,
#     policy='CosineAnnealing',
#     warmup='linear',
#     warmup_iters=1000,
#     warmup_ratio=1.0 / 10,
#     min_lr_ratio=1e-7)
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=1000,
#     warmup_ratio=0.001,
#     step=[12, 22, 32])
# optimizer = dict(_delete_=True, type='AdamW', lr=0.000001, weight_decay=0.0001)
# optimizer = dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# runner = dict(type='EpochBasedRunner', max_epochs=12)
fp16 = dict(loss_scale=512.)
find_unused_parameters = True
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,)
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric='bbox')