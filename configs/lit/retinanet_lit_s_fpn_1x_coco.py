_base_ = [
    # '../_base_/models/retinanet_fpn_lit_s.py',
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    # pretrained='/home/omeneis/pjh/LIT-main/detection/retina_lit_s.pth',
    backbone=dict(
        _delete_=True,
        type='LIT',
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=True,
        init_cfg=dict(type='Pretrained', checkpoint='/home/omeneis/pjh/LIT-main/detection/retina_lit_s.pth')),
    # backbone=dict(
    #     embed_dim=96,
    #     depths=[2, 2, 6, 2],
    #     num_heads=[3, 6, 12, 24],
    #     window_size=7,
    #     ape=False,
    #     drop_path_rate=0.2,
    #     patch_norm=True,
    #     use_checkpoint=True
    # ),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5)
)

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)})
                 )

# do not use mmdet version fp16
# fp16 = None
# optimizer_config = dict(
#     type="DistOptimizerHook",
#     update_interval=1,
#     grad_clip=None,
#     coalesce=True,
#     bucket_size_mb=-1,
#     use_fp16=True,
# )

load_from = "/home/omeneis/pjh/mmdetection-2.26.0/work_COCO/retinanet/epoch_12.pth"
# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(grad_clip=None)
# optimizer_config = dict(
#     grad_clip=None,
#     type='Fp16OptimizerHook',
#     coalesce=True,
#     bucket_size_mb=-1,
#     loss_scale='dynamic',
#     distributed=True)

# fp16 = dict(loss_scale=512.)
find_unused_parameters = True
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    )
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric='bbox')