_base_ = '../swin/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

model = dict(
    type='MaskRCNN',
    backbone=dict(
        type='SparseViT',
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
        with_cp=False,
        convert_weights=True,
        #init_cfg=dict(type='Pretrained', checkpoint=pretrained)
        ),
    neck=dict(in_channels=[96, 192, 384, 768]))

#load_from = 'work_dirs/mask_rcnn_long_schedule/epoch_9.pth'
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
lr_config = dict(warmup_iters=10, step=[2])
runner = dict(max_epochs=3)
checkpoint_config = dict(interval=1, max_keep_ckpts=1)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(672, 672),
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

random_sample = False

pruning_ratios=[[0.6, 0.6], [0., 0.], [0.2, 0.2, 0.4, 0.4, 0.5, 0.5], [0., 0.]]
