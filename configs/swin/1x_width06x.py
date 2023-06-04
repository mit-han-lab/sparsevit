_base_ = './mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-1x_coco.py'

pretrained = 'checkpoint/swint_in1k_06x.pth'

model = dict(
    backbone=dict(
        embed_dims=60,
        drop_path_rate=0.1,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)
    ),
    neck=dict(in_channels=[60, 120, 240, 480])
)