# SparseViT: Revisiting Activation Sparsity for Efficient High-Resolution Vision Transformer

### [website](https://sparsevit.mit.edu/) | [paper](https://arxiv.org/abs/2303.17605)

![Alt text](resources/image.png)

## Abstract

High-resolution images enable neural networks to learn
richer visual representations. However, this improved performance
comes at the cost of growing computational complexity,
hindering their usage in latency-sensitive applications.
As not all pixels are equal, skipping computations
for less-important regions offers a simple and effective measure
to reduce the computation. This, however, is hard to
be translated into actual speedup for CNNs since it breaks
the regularity of the dense convolution workload. In this
paper, we introduce *SparseViT* that revisits activation sparsity
for recent window-based vision transformers (ViTs). As
window attentions are naturally batched over blocks, actual
speedup with window activation pruning becomes possible:
i.e., ∼50% latency reduction with 60% sparsity. Different
layers should be assigned with different pruning ratios due to
their diverse sensitivities and computational costs. We introduce
sparsity-aware adaptation and apply the evolutionary
search to efficiently find the optimal layerwise sparsity configuration
within the vast search space.  *SparseViT* achieves
speedups of 1.5×, 1.4×, and 1.3× compared to its dense
counterpart in monocular 3D object detection, 2D instance
segmentation, and 2D semantic segmentation, respectively,
with negligible to no loss of accuracy.


## Prerequisite

Our code is based on mmdetection 2.28.2
1. mmcv >= 1.3.17, and <= 1.8.0
2. torchpack
3. OpenMPI = 4.0.4 and mpi4py = 3.0.3 (Needed for torchpack)

   

## Training Pipeline

### Swin Pre-train

```
bash tools/dist_train.sh configs/swin/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py 8
```

or download the checkpoint from [model](https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_20210908_165006-90a4008c.pth) in https://github.com/open-mmlab/mmdetection/tree/main/configs/swin.

### Sparsity-Aware Adaption

In Sparsity-Aware Adaption, we randomly sample layerwise sparsity at each iteration.

```
bash tools/dist_train.sh configs/sparsevit/mask_rcnn_sparsevit_saa.py 8
```


### Latency-Constrained Evolutionary Search

We use evolutionary search based on Sparsity-Aware Adaption model to find the optimal sparsity configuration whose lantency is between `max_latency` and `min_latency`.

```
torchpack dist-run -np 8 python tools/search.py configs/sparsevit/mask_rcnn_sparsevit.py [checkpoint_path] --max [max_latency] --min [min_latency]
```

For example, the dense model with `672x672` input resolution has about `47.8ms` latency. We want to find the optimal sparsity configuration with latency under 42ms.

```
torchpack dist-run -np 8 python tools/search.py configs/sparsevit/mask_rcnn_sparsevit.py mask_rcnn_sparsevit_saa.pth --max 42 --min 37
```

The search log will be stored in `work_dirs/search_max42_min37.txt`. 

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

We measure our model's latency using input image batch of 4.

```
python tools/measure_latency.py [config] --img_size [img_size]
```

For example,
```
python tools/measure_latency.py configs/sparsevit/sparsevit_cfg1_42ms.py --img_size 672
```


## Results

We report our latency on NVIDIA RTX A6000 GPU.

The pre-trained SAA(Sparsity-Aware Adaption) model is [here](https://drive.google.com/file/d/1a_AeW0SH9u5aC4htnTpk_Svsqgy5V9h4/view?usp=sharing).

| sparsity configuration | resolution | latency  | bbox mAP | mask mAP | model |
| ------------- | ---------- | -------- | -------- | -------- | ------| 
| -- | 672x672 | 47.8 | 42.6 | 38.8 | |
|[config](configs/sparsevit/sparsevit_cfg1_42ms.py) | 672x672 | 41.3 | 42.4 | 38.5 | [link](https://drive.google.com/file/d/1tIs-r9IgepnATeidmQq34QqCH-Wp5OgY/view?usp=sharing) |
|[config](configs/sparsevit/sparsevit_cfg2_35ms.py) | 672x672 | 34.2 | 41.6 | 37.7 | [link](https://drive.google.com/file/d/1Md_IXHuwQY1B21XulNPAGaN2YZhOqjeu/view?usp=sharing) |
|[config](configs/sparsevit/sparsevit_cfg3_33ms.py)  | 672x672 | 32.9 | 41.3 | 37.4 | [link](https://drive.google.com/file/d/1ruJ3SbFwJMaV4NAFY-U4nUzS8gTXZiV-/view?usp=drive_link) |


