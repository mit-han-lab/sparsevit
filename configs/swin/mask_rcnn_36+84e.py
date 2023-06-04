_base_ = './mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py'

load_from = 'work_dirs/mask_rcnn_36+120e/epoch_9.pth'
optimizer = dict(lr=0.00001)
lr_config = dict(warmup_iters=10, step=[14])
runner = dict(max_epochs=21)
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
