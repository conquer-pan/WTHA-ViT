# _base_ = [
#     '../_base_/datasets/coco_detection.py',
#     '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
# ]
_base_ = './atss_r50_fpn_1x_coco.py'
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
model = dict(
    type='ATSS',
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        # out_indices=(1, 2, 3),
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    # bbox_head=dict(
    #     type='ATSSHead',
    #     num_classes=15,
    #     in_channels=256,
    #     stacked_convs=4,
    #     feat_channels=256,
    #     anchor_generator=dict(
    #         type='AnchorGenerator',
    #         ratios=[1.0],
    #         octave_base_scale=8,
    #         scales_per_octave=1,
    #         strides=[8, 16, 32, 64, 128]),
    #     bbox_coder=dict(
    #         type='DeltaXYWHBBoxCoder',
    #         target_means=[.0, .0, .0, .0],
    #         target_stds=[0.1, 0.1, 0.2, 0.2]),
    #     loss_cls=dict(
    #         type='FocalLoss',
    #         use_sigmoid=True,
    #         gamma=2.0,
    #         alpha=0.25,
    #         loss_weight=1.0),
    #     loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
    #     loss_centerness=dict(
    #         type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # # training and testing settings
    # train_cfg=dict(
    #     assigner=dict(type='ATSSAssigner', topk=9),
    #     allowed_border=-1,
    #     pos_weight=-1,
    #     debug=False),
    # test_cfg=dict(
    #     nms_pre=1000,
    #     min_bbox_size=0,
    #     score_thr=0.05,
    #     nms=dict(type='nms', iou_threshold=0.6),
    #     max_per_img=100)
        )
# optimizer
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    )
load_from = "/home/omeneis/pjh/mmdetection-2.26.0/work_COCO/atss/epoch_12.pth"
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric='bbox')
