_base_ = './mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py'
pretrained = 'checkpoint/swint_in1k_09x.pth'

model = dict(
    backbone=dict(
        embed_dims=84,
        drop_path_rate=0.15,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)
    ),
    neck=dict(in_channels=[84, 168, 336, 672])
)