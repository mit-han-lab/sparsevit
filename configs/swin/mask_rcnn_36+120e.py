_base_ = './mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py'

load_from = 'work_dirs/mask_rcnn_long_schedule/epoch_54.pth'

lr_config = dict(warmup_iters=1, step=[36, 56])
runner = dict(max_epochs=66)
checkpoint_config = dict(interval=9, max_keep_ckpts=10)

