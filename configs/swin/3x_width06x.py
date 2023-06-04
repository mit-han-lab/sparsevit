_base_ = './mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py'
pretrained = 'checkpoint/swint_in1k_06x.pth'

model = dict(
    backbone=dict(
        embed_dims=60,
        drop_path_rate=0.1,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)
    ),
    neck=dict(in_channels=[60, 120, 240, 480])
)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
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
    val=dict(
        pipeline=test_pipeline),
    test=dict(
        pipeline=test_pipeline))