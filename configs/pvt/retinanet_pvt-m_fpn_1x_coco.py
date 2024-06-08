_base_ = 'retinanet_pvt-t_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        num_layers=[3, 4, 18, 3],
        init_cfg=dict(checkpoint='https://github.com/whai362/PVT/'
                      'releases/download/v2/pvt_medium.pth')))
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)
# optimizer = dict(_delete_=True, type='AdamW', lr=0.00005, weight_decay=0.0001)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    )
load_from = "/home/omeneis/pjh/mmdetection-2.26.0/work_COCO/retinanet/epoch_12.pth"
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric='bbox')