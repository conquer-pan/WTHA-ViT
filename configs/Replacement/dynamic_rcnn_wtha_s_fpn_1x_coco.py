_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
model = dict(
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
    ),
    roi_head=dict(
        type='DynamicRoIHead',
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    train_cfg=dict(
        rpn_proposal=dict(nms=dict(iou_threshold=0.85)),
        rcnn=dict(
            dynamic_rcnn=dict(
                iou_topk=75,
                beta_topk=10,
                update_iter_interval=100,
                initial_iou=0.4,
                initial_beta=1.0))),
    test_cfg=dict(rpn=dict(nms=dict(iou_threshold=0.85))))
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)
load_from = "/home/omeneis/pjh/mmdetection-2.26.0/work_COCO/epoch_12.pth"
fp16 = dict(loss_scale=512.)
find_unused_parameters = True
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric='bbox')
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    )