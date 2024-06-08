_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
model = dict(
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
    neck=dict(in_channels=[96, 192, 384, 768]))
    # neck=dict(in_channels=[192, 384, 768], start_level=0, num_outs=5))

# optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001)
# optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    )
load_from = '/home/omeneis/pjh/mmdetection-2.26.0/work_COCO/retinanet/epoch_12.pth' #coco预
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric='bbox')

