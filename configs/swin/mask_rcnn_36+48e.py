_base_ = './mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py'

load_from = 'work_dirs/mask_rcnn_long_schedule/epoch_36.pth'
optimizer = dict(lr=0.00001)
lr_config = dict(warmup_iters=10, step=[8])
runner = dict(max_epochs=12)
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
