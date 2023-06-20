_base_ = '../swin/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py'

lr_config = dict(warmup_iters=1000, step=[27, 33])
runner = dict(max_epochs=36)

pruning_ratios=[[0.7, 0.7], [0., 0.], [0.2, 0.2, 0.4, 0.4, 0.5, 0.5], [0., 0.]]

load_from = 'work_dirs/mask_rcnn_sparsevit_saa/exp/epoch_12.pth'