# dataset settings
# dataset_type = 'CocoDataset'
# data_root = 'data/coco/'
dataset_type = 'CocoDataset'
# data_root = '/home/omeneis/pjh/dataset/other_dataset/coco2017/'
data_root = '/home/omeneis/pjh/dataset/dota_cut/dota1-split-1024/'
# data_root = '/home/spl0/HDD-2T/pjh/pjh/dataset/coco2017/'
# data_root = '/home/omeneis/pjh/dataset/other_dataset/HRSID/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='Resize', img_scale=(1024,1024), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(1333, 800),
        img_scale=(1024,1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/instances_train2017.json',
        # img_prefix=data_root + 'train2017/',
        ann_file=[data_root + 'train1024/DOTA_train1024_remove.json', '/home/omeneis/pjh/dataset/dota_cut/dota1-split-1024_gap400/train1024_multi/DOTA1_train1024_multi_gap400_remove.json'],
        img_prefix=[data_root + 'train1024/images', '/home/omeneis/pjh/dataset/dota_cut/dota1-split-1024_gap400/train1024_multi/images'],
        # ann_file=data_root + 'annotations/train2017.json',#HRSID
        # img_prefix=data_root + 'images',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/instances_val2017.json',
        # img_prefix=data_root + 'val2017/',
        ann_file=data_root + 'val1024/DOTA_val1024_remove.json',
        img_prefix=data_root + 'val1024/images',
        # ann_file=data_root + 'annotations/test2017.json',#HRSID
        # img_prefix=data_root + 'images',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/instances_val2017.json',
        # img_prefix=data_root + 'val2017/',
        ann_file=data_root + 'val1024/DOTA_val1024_remove.json',
        img_prefix=data_root + 'val1024/images',
        # ann_file=data_root + 'annotations/test2017.json',
        # img_prefix=data_root + 'images',
        pipeline=test_pipeline))
evaluation = dict(interval=2, metric=['bbox', 'segm'])
