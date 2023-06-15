# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings
import random
import numpy as np
import types

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import (build_ddp, build_dp, compat_cfg, get_device,
                         replace_cfg_vals, rfnext_init_model,
                         setup_multi_processes, update_data_root)
from mmdet.models.backbones.sparsevit import SwinBlockSequence

def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()

def random_sample(model):
    configs = {}
    for name, module in model.named_modules():
        if isinstance(module, SwinBlockSequence):
            configs[name] = []
            pre_ratio = 0
            for i in range(module.depth // 2):
                ratio = random.randint(pre_ratio, 8)
                configs[name].append(ratio / 10)
                configs[name].append(ratio / 10)
                pre_ratio = ratio
            module.pruning_ratios = configs[name]
    return configs

def select(model, configs):
    for name, module in model.named_modules():
        if isinstance(module, SwinBlockSequence):
            module.pruning_ratios = configs[int(name[16])]
            print(name, configs[int(name[16])])

def monkey_patch(model):
    def forward(self, *args, **kwargs):
        random_sample(self)
        return type(self).forward(self, *args, **kwargs)

    model.forward = types.MethodType(forward, model)


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--img_size', type=int)
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    elif 'init_cfg' in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    # init rfnext if 'RFSearchHook' is defined in cfg
    rfnext_init_model(model, cfg=cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is None and cfg.get('device', None) == 'npu':
        fp16_cfg = dict(loss_scale='dynamic')
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    
    if cfg.get('random_sample', False):
        monkey_patch(model)
    if cfg.get('pruning_ratios', None):
        print(cfg.pruning_ratios)
        select(model, cfg.pruning_ratios)

    random.seed(0)
    @torch.inference_mode()
    def measure(model, img_size, num_repeats=500, num_warmup=500):
        model.cuda()
        model.eval()
        
        backbone = model.backbone
        inputs = torch.randn(4, 3, img_size, img_size).cuda()

        latencies = []
        for k in range(num_repeats + num_warmup):
            start = cuda_time()
            backbone(inputs)
            if k >= num_warmup:
                latencies.append((cuda_time() - start) * 1000)

        #latencies = itertools.chain(dist.allgather(latencies))
        latencies = sorted(latencies)

        drop = int(len(latencies) * 0.25)
        return np.mean(latencies[drop:-drop])
    img_size = args.img_size
    #for i in range(11):
    #    img_size = 480 + i*32
    print(img_size, measure(model, img_size))


if __name__ == '__main__':
    main()
