# We follow the original implementation which
# adopts the Caffe pre-trained backbone.
_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    type='AutoAssign',
    pretrained='/home/omeneis/pjh/mmdetection-2.26.0/work_COCO/epoch_12.pth',
    backbone=dict(
        _delete_=True,
        # type='WTHA_ViT_s',
        type='WTHA_ViT_s_network3',
        init_cfg=dict(type='Pretrained', checkpoint='/home/omeneis/pjh/mmdetection-2.26.0/work_COCO/epoch_12.pth'),
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels = [64, 128, 256, 448],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=5,
        relu_before_extra_convs=True,
        init_cfg=dict(type='Caffe2Xavier', layer='Conv2d')),
    bbox_head=dict(
        type='AutoAssignHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_bbox=dict(type='GIoULoss', loss_weight=5.0)),
    train_cfg=None,
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))
img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='Resize', img_scale=(832, 832), keep_ratio=False),
    # dict(type='Resize', img_scale=(1024,1024), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(1333, 800),
        img_scale=(832, 832),
        # img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
# # optimizer
# optimizer = dict(lr=0.01, paramwise_cfg=dict(norm_decay_mult=0.))
# # learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=1000,
#     warmup_ratio=1.0 / 1000,
#     step=[8, 11])
# total_epochs = 12

# load_from = "/home/omeneis/pjh/mmdetection-2.26.0/work_COCO/wtha_s/epoch_12.pth"

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
load_from = "/home/omeneis/pjh/mmdetection-2.26.0/work_COCO/epoch_12.pth"

# runner = dict(type='EpochBasedRunner', max_epochs=12)
fp16 = dict(loss_scale=512.)
find_unused_parameters = True
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric='bbox')