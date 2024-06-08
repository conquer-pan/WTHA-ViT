# _base_ = [
#     '../_base_/models/retinanet_r50_fpn.py',
#     '../_base_/datasets/coco_detection.py',
#     '../_base_/schedules/schedule_1x.py',
#     '../_base_/default_runtime.py'
# ]
_base_ = './cascade_rcnn_r50_fpn_1x_coco.py'
pretrained = '/home/omeneis/pjh/mmdetection-2.26.0/work_COCO/cascade/epoch_12.pth'
model = dict(
    backbone=dict(
        _delete_=True,
        type='UniFormer',
        embed_dim=[64, 128, 320, 512],
        layers=[3, 4, 8, 3],
        head_dim=64,
        drop_path_rate=0.2,
        use_checkpoint=False,
        checkpoint_num=[0, 0, 8, 0],
        windows=False,
        hybrid=True,
        window_size=14,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)
    ),
    neck=dict(in_channels=[64, 128, 320, 512]))

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    grad_clip=None,
    type='Fp16OptimizerHook',
    coalesce=True,
    bucket_size_mb=-1,
    loss_scale='dynamic',
    distributed=True)
# optimizer_config = dict(
#     type="DistOptimizerHook",
#     update_interval=1,
#     grad_clip=None,
#     coalesce=True,
#     bucket_size_mb=-1,
#     use_fp16=True,
# )

load_from = "/home/omeneis/pjh/mmdetection-2.26.0/work_COCO/cascade/epoch_12.pth"
# optimizer
# optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)
# optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)
# optimizer = dict(_delete_=True, type='AdamW', lr=0.00001, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
# runner = dict(type='EpochBasedRunner', max_epochs=12)
# fp16 = dict(loss_scale=512.)
find_unused_parameters = True
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    )
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric='bbox')
