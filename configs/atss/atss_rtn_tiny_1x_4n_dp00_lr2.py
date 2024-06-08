_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
# _base_ = './atss_r50_fpn_1x_coco.py'
pretrained = '/home/omeneis/pjh/DAT-Detection-main/rtn_dat_t_1x.pth'

model = dict(
    backbone=dict(
        type='DAT',
        dim_stem=64,
        dims=[64, 128, 256, 512],
        depths=[2, 4, 18, 2],
        stage_spec=[
            ["N", "D"], 
            ["N", "D", "N", "D"], 
            ["N", "D", "N", "D", "N", "D", "N", "D", "N", "D", "N", "D", "N", "D", "N", "D", "N", "D"], 
            ["D", "D"]],
        heads=[2, 4, 8, 16],
        groups=[1, 2, 4, 8],
        use_pes=[True, True, True, True],
        strides=[8, 4, 2, 1],
        offset_range_factor=[-1, -1, -1, -1],
        use_dwc_mlps=[True, True, True, True],
        use_lpus=[True, True, True, True],
        use_conv_patches=True,
        ksizes=[9, 7, 5, 3],
        nat_ksizes=[7, 7, 7, 7],
        drop_path_rate=0.0,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)
    ),
    neck=dict(in_channels=[64, 128, 256, 512])
)

load_from = '/home/omeneis/pjh/mmdetection-2.26.0/work_COCO/atss/epoch_12.pth'
lr = 0.0001
data = dict(samples_per_gpu=4, workers_per_gpu=4)

optimizer = dict(_delete_=True, type='AdamW', lr=lr, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(norm_decay_mult=0.,
                     custom_keys={'absolute_pos_embed': dict(decay_mult=0.), 
                                  'relative_position_bias_table': dict(decay_mult=0.),
                                  'rpe_table': dict(decay_mult=0.),
                                  'norm': dict(decay_mult=0.)
                                 }
                                   )
                )
# fp16 = None
optimizer_config = dict(
    grad_clip=None,
    type='Fp16OptimizerHook',
    coalesce=True,
    bucket_size_mb=-1,
    loss_scale='dynamic',
    distributed=True)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
# runner = dict(type='EpochBasedRunner', max_epochs=12)
# log_config = dict(interval=50)

# fp16 = dict(loss_scale=512.)
find_unused_parameters = True
checkpoint_config = dict(interval=2)
evaluation = dict(interval=2, metric='bbox')
