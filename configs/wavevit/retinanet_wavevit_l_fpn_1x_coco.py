_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
# optimizer
model = dict(
    pretrained='/home/omeneis/pjh/mmdetection-2.26.0/work_COCO/retinanet/epoch_12.pth',
    backbone=dict(
        type='wavevit_l',
        init_cfg=dict(type='Pretrained', checkpoint='/home/omeneis/pjh/mmdetection-2.26.0/work_COCO/retinanet/epoch_12.pth'),
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 512],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5))

load_from = "/home/omeneis/pjh/mmdetection-2.26.0/work_COCO/retinanet/epoch_12.pth"
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)
# optimizer = dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
# optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=2000,  # 修改这里 Epoch [1][500/xxxx]之前的学习率的意思
#     warmup_ratio=0.001,
#     step=[8, 11])
# runner = dict(type='EpochBasedRunner', max_epochs=12)
fp16 = dict(loss_scale=512.)
find_unused_parameters = True
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    )
# auto_scale_lr = dict(base_batch_size=1)
checkpoint_config = dict(interval=2)
evaluation = dict(interval=2, metric='bbox')