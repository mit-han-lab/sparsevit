# SparseViT: Revisiting Activation Sparsity for Efficient High-Resolution Vision Transformer

### [website](https://sparsevit.mit.edu/) | [paper](https://arxiv.org/abs/2303.17605)

## Prerequisite

Our code is based on mmdetection 2.28.2
1. mmcv >= 1.3.17, and <= 1.8.0
2. torchpack
   

## Training Pipeline

### Swin Pre-train

```
bash tools/dist_train.sh configs/swin/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py 8
```

or download the checkpoint from [model](https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_20210908_165006-90a4008c.pth) in https://github.com/open-mmlab/mmdetection/tree/main/configs/swin.

### Sparsity-Aware Adaption



```
bash tools/dist_train.sh configs/sparsevit/mask_rcnn_sparsevit_saa.py 8
```


### Latency-Constrained Evolutionary Search

```
torchpack dist-run -np 8 python tools/search.py configs/sparsevit/mask_rcnn_sparsevit.py [checkpoint_path] --max [max_latency] --min [min_latency]
```

### Finetune

Finetune the SAA model with optimal sparsity configuration.

For example, the best sparsity configuration under 42ms is 

backbone.stages.0 : 0.3  
backbone.stages.1 :  0     
backbone.stages.2_1 : 0.1     
backbone.stages.2_2 : 0.2     
backbone.stages.2_3 : 0.2     
backbone.stages.3 :  0 

```
bash tools/dist_train.sh configs/sparsevit/sparsevit_cfg1_42ms.py 8
```

## Latency Measure

```
python tools/measure_latency.py [config] --img_size [img_size]
```

For example,
```
python tools/measure_latency.py configs/sparsevit/mask_rcnn_sparsevit.py --img_size 672
```


## Results


| sparsity configuration | resolution | latency  | bbox mAP | mask mAP | model |
| ------------- | ---------- | -------- | -------- | -------- | ------| 
| -- | 672x672 | 47.8 | 42.6 | 38.8 | |
|[config](configs/sparsevit/sparsevit_cfg1_42ms.py) | 672x672 | 41.3 | 42.4 | 38.5 | |
|[config](configs/sparsevit/sparsevit_cfg2_35ms.py) | 672x672 | 34.2 | 41.6 | 37.7 | |
|[config](configs/sparsevit/sparsevit_cfg3_33ms.py)  | 672x672 | 32.9 | 41.3 | 37.4 | |


