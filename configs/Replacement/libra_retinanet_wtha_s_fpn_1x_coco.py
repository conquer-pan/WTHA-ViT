_base_ = '../retinanet/retinanet_r50_fpn_1x_coco.py'
# model settings
model = dict(
    pretrained='/home/omeneis/pjh/mmdetection-2.26.0/work_COCO/epoch_12.pth',
    backbone=dict(
        _delete_=True,
        # type='WTHA_ViT_s',
        type='WTHA_ViT_s_network3',
        init_cfg=dict(type='Pretrained', checkpoint='/home/omeneis/pjh/mmdetection-2.26.0/work_COCO/epoch_12.pth'),
        style='pytorch'),
    neck=[
        dict(
            type='FPN',
            in_channels=[64, 128, 256, 448],
            out_channels=256,
            start_level=1,
            add_extra_convs='on_input',
            num_outs=5),
        dict(
            type='BFP',
            in_channels=256,
            num_levels=5,
            refine_level=1,
            refine_type='non_local')
    ],
    bbox_head=dict(
        loss_bbox=dict(
            _delete_=True,
            type='BalancedL1Loss',
            alpha=0.5,
            gamma=1.5,
            beta=0.11,
            loss_weight=1.0)))

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