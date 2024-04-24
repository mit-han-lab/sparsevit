_base_ = './mask_rcnn_sparsevit.py'

lr_config = dict(warmup_iters=1000, step=[27, 33])
runner = dict(max_epochs=36)

pruning_ratios=[[0.3, 0.3], [0., 0.], [0.1, 0.1, 0.2, 0.2, 0.2, 0.2], [0., 0.]]

load_from = 'work_dirs/mask_rcnn_sparsevit_saa/exp/epoch_12.pth'
