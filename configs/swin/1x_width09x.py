_base_ = './mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-1x_coco.py'

model = dict(
    backbone=dict(
        embed_dims=84,
        drop_path_rate=0.15,
    ),
    neck=dict(in_channels=[84, 168, 336, 672])
)