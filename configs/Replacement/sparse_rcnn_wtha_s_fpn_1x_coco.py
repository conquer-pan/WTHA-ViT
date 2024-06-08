_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
num_stages = 6
num_proposals = 100
model = dict(
    type='SparseRCNN',
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
        start_level=0,
        add_extra_convs='on_input',
        num_outs=4),
    rpn_head=dict(
        type='EmbeddingRPNHead',
        num_proposals=num_proposals,
        proposal_feature_channel=256),
    roi_head=dict(
        type='SparseRoIHead',
        num_stages=num_stages,
        stage_loss_weights=[1] * num_stages,
        proposal_feature_channel=256,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='DIIHead',
                num_classes=1,
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_reg_fcs=3,
                feedforward_channels=2048,
                in_channels=256,
                dropout=0.0,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                dynamic_conv_cfg=dict(
                    type='DynamicConv',
                    in_channels=256,
                    feat_channels=64,
                    out_channels=256,
                    input_feat_shape=7,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')),
                loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    clip_border=False,
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.5, 0.5, 1., 1.])) for _ in range(num_stages)
        ]),
    # training and testing settings
    train_cfg=dict(
        rpn=None,
        rcnn=[
            dict(
                assigner=dict(
                    type='HungarianAssigner',
                    cls_cost=dict(type='FocalLossCost', weight=2.0),
                    reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                    iou_cost=dict(type='IoUCost', iou_mode='giou',
                                  weight=2.0)),
                sampler=dict(type='PseudoSampler'),
                pos_weight=1) for _ in range(num_stages)
        ]),
    test_cfg=dict(rpn=None, rcnn=dict(max_per_img=num_proposals)))

# optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)
# optimizer = dict(type='SGD', lr=0.00005, momentum=0.9, weight_decay=0.0001)
load_from = "/home/omeneis/pjh/mmdetection-2.26.0/work_COCO/epoch_12.pth"

fp16 = dict(loss_scale=512.)
find_unused_parameters = True
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric='bbox')
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    )