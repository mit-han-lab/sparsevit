_base_ = './mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py'

load_from = 'checkpoint/mask_rcnn_swin-t-p4-w7_fpn-3x_coco.pth'

lr_config = dict(warmup_iters=1000, step=[60, 66])
runner = dict(max_epochs=72)
checkpoint_config = dict(interval=9, max_keep_ckpts=10)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(576, 576),
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