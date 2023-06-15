import argparse
import copy
import hashlib
import itertools
import json
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import tqdm
from mmcv import Config
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model, get_dist_info
from torch import nn
from torchpack import distributed as dist
from torchpack.utils.config import configs

from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.models.backbones.sparsevit import SwinBlockSequence
from mmdet.apis import multi_gpu_test


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, "w") as fnull:
        with redirect_stderr(fnull), redirect_stdout(open(devnull, "w")):
            yield


def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def num2hrb(num: float, suffix="", base=1000) -> str:
    """Convert big floating number to human readable string."""
    for unit in ["", "K", "M", "G", "T"]:
        if abs(num) < base:
            return f"{num:3.2f}{unit}{suffix}"
        num /= base
    return f"{num:.2f}{suffix}"


def stats(vals: List[float]) -> Dict[str, float]:
    """Compute min, max, avg, std of vals."""
    STATS = {"min": np.min, "max": np.max, "avg": np.mean, "std": np.std}
    return {name: fn(vals) for name, fn in STATS.items()} if vals else {}


class BaseSearcher(ABC):
    iter_num: int
    num_satisfied: int
    candidate: Dict[str, Any]
    best: Dict[str, Any]
    samples: Dict[str, Any]
    history: Dict[str, Any]

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        for key, val in self.default_state_dict.items():
            setattr(self, key, copy.deepcopy(val))

    def search(
        self,
        model: nn.Module,
        num_iters: int,
        filter_func,
        score_func: Callable[[nn.Module], float],
        save_func: Optional[Callable[[Dict[str, Any]], None]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        # initialize search
        self.model = model
        # construct constraints checker function from constraints dict
        self.filter_func = filter_func
        self.score_func = score_func
        self.save_func = save_func

        if verbose:
            pbar = tqdm.trange(
                self.iter_num,
                num_iters,
                initial=self.iter_num,
                total=num_iters,
                position=0,
                leave=True,
            )

        # run initial step and sanity checks before search step
        self.before_search()

        for self.iter_num in range(self.iter_num, num_iters):
            self.before_step()
            self.run_step()
            self.after_step()

            if verbose:
                info = {
                    "num_satisfied": self.num_satisfied,
                    "metric": self.candidate["metric"],
                    "constraints": self.candidate["constraints"],
                    "best_subnet_metric": self.best["metric"],
                    "best_subnet_constraints": self.best["constraints"],
                }

                # display the full stats only once a while
                if len(self.history["metric"]) == 100:
                    info["metric/stats"] = stats(self.history["metric"])
                    info["constraints/stats"] = {
                        name: stats(vals)
                        for name, vals in self.history["constraints"].items()
                    }
                    self.history["metric"].clear()
                    self.history["constraints"].clear()

                def _recursive_format(obj, fmt):
                    if isinstance(obj, float):
                        return num2hrb(obj)
                    if isinstance(obj, dict):
                        return {k: _recursive_format(v, fmt) for k, v in obj.items()}
                    return obj

                pbar.update()
                info = _recursive_format(info, fmt="{:.4g}")
                print("".join(f"\n+ [{k}] = {v}" for k, v in info.items()) + "\n")

            if self.early_stop():
                break

        if verbose:
            pbar.close()
            print(
                f'[best_subnet_metric] = {info["best_subnet_metric"]} \
                    [best_subnet_constraints] = {info["best_subnet_constraints"]}'
            )

        return self.best

    def _sample(self) -> Dict[str, Any]:
        return {
            "config": sample(self.model),
        }

    @abstractmethod
    def sample(self) -> Dict[str, Any]:
        """Sample and select new sub-net configuration and return configuration."""
        raise NotImplementedError

    def before_search(self) -> None:
        pass

    def before_step(self) -> None:
        pass

    def run_step(self) -> None:
        # sample and select the candidate
        self.candidate = self.sample()

        # obtain the active config and hparams
        config = self.candidate["config"]

        # serialize and hash the candidate
        buffer = json.dumps({"config": config}, sort_keys=True)
        ckey = hashlib.sha256(buffer.encode()).hexdigest()

        if ckey not in self.samples:
            # check constraints
            (
                self.candidate["is_satisfied"],
                self.candidate["constraints"],
            ) = self.filter_func(self.model)

            # evaluate the metric
            if self.candidate["is_satisfied"]:
                self.candidate["metric"] = float(self.score_func(self.model))
            else:
                self.candidate["metric"] = -float("inf")

            self.samples[ckey] = copy.deepcopy(self.candidate)
        else:
            self.candidate = copy.deepcopy(self.samples[ckey])
        rank, _ = get_dist_info()
        if rank == 0:
            with open('evolve_search.txt', 'a+') as f:
                f.write(str(self.iter_num) + ' ' + str(self.candidate["metric"]))
                f.write('\n')

        self.num_satisfied += int(self.candidate["is_satisfied"])
        self.candidate["is_best"] = (
            self.candidate["metric"] >= self.best["metric"] or self.iter_num == 0
        )

        # update the stats if satisfied
        if self.candidate["is_satisfied"]:
            self.history["metric"].append(self.candidate["metric"])
            for name, val in self.candidate["constraints"].items():
                self.history["constraints"][name].append(val)

        # update the best if necessary
        if self.candidate["is_best"]:
            self.best = copy.deepcopy(self.candidate)
            best_res = self.candidate['metric']
            best_latency = self.candidate["constraints"]
            best_cfg = self.candidate["config"]
            if dist.is_master():
                with open(self.search_log, 'a+') as f:
                    f.write(str(self.iter_num) + ' ' + str(best_res) + ' ' + str(best_latency))
                    f.write('\n')
                    for k,v in best_cfg.items():
                        f.write(k+' '+str(v)+' ')
                    f.write('\n')

        # save the state dict
        if self.save_func is not None:
            self.save_func(self.state_dict())

    def after_step(self) -> None:
        pass

    def early_stop(self) -> bool:
        return False

    @property
    def default_state_dict(self) -> Dict[str, Any]:
        return {
            "iter_num": 0,
            "num_satisfied": 0,
            "candidate": {},
            "best": {"metric": -float("inf"), "constraints": None},
            "samples": {},
            "history": {"metric": [], "constraints": defaultdict(list)},
        }

    def state_dict(self) -> Dict[str, Any]:
        return {key: getattr(self, key) for key in self.default_state_dict}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        for key in self.default_state_dict:
            setattr(self, key, state_dict[key])


class RandomSearcher(BaseSearcher):
    def sample(self) -> Dict[str, Any]:
        return self._sample()


class EvolveSearcher(BaseSearcher):
    population: List[Dict[str, Any]]
    candidates: List[Dict[str, Any]]

    def __init__(
        self,
        population_size: int = 100,
        candidate_size: int = 25,
        mutation_prob: float = 0.1,
        search_log: str = '',
    ) -> None:
        super().__init__()
        self.population_size = population_size
        self.candidate_size = candidate_size
        self.mutation_prob = mutation_prob
        self.search_log = search_log

    def sort_cfg(self, input: Dict[str, Any]) -> Dict[str, Any]:
        for var in input:
            stage_name = None
            p_ratios = []
            for name in input[var]:
                if name[-2] == '_':
                    stage_name = name[:-2]
                    p_ratios.append(input[var][name])
            p_ratios.sort()
            input[var][stage_name+'_1'] = p_ratios[0]
            input[var][stage_name+'_2'] = p_ratios[1]
            input[var][stage_name+'_3'] = p_ratios[2]
        return input

    def _mutate(self, input: Dict[str, Any]) -> Dict[str, Any]:
        output = self._sample()
        # only considers independent hparams
        output["config"] = config(self.model)
        for var in output:  # pylint: disable=C0206
            for name in output[var]:
                if random.random() > self.mutation_prob:
                    output[var][name] = input[var][name]
        output = self.sort_cfg(output)
        # returns reduced, independent config
        return output

    def _crossover(self, inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        output = {"config": {}}
        # only considers independent hparams
        for name in config(self.model):
            output["config"][name] = random.choice(inputs)["config"][name]
        output = self.sort_cfg(output)
        # returns reduced, independent config
        return output

    def sample(self) -> Dict[str, Any]:
        if not self.candidates:
            return self._sample()
        if len(self.population) < self.population_size // 2:
            output = self._mutate(random.choice(self.candidates))
        else:
            output = self._crossover(random.sample(self.candidates, 2))
        # returns full config
        select(self.model, output["config"])
        output["config"] = config(self.model)
        return output

    def before_step(self) -> None:
        if len(self.population) >= self.population_size:
            self.candidates = sorted(
                self.population,
                key=lambda x: x["metric"],
                reverse=True,
            )[: self.candidate_size]
            self.population = []

    def after_step(self) -> None:
        if self.candidate["is_satisfied"]:
            self.population.append(copy.deepcopy(self.candidate))

    @property
    def default_state_dict(self) -> Dict[str, Any]:
        return {
            **super().default_state_dict,
            "population": [],
            "candidates": [],
        }


def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj


def select(model: nn.Module, configs: Dict[str, Any]) -> None:
    for name, module in model.named_modules():
        if isinstance(module, SwinBlockSequence):
            if module.depth == 2:
                module.pruning_ratios = [configs[name], configs[name]]
            elif module.depth == 6:
                p_ratios = [configs[name+'_1'], configs[name+'_2'], configs[name+'_3']]
                assert p_ratios[0] <= p_ratios[1] and p_ratios[1] <= p_ratios[2]
                module.pruning_ratios = []
                for i in range(3):
                    module.pruning_ratios.append(p_ratios[i])
                    module.pruning_ratios.append(p_ratios[i])

def sample(model: nn.Module, sample_func: Optional[Callable] = None) -> Dict[str, Any]:
    if sample_func is None:
        sample_func = random.choice

    # TODO: move this choices to the module
    choices = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    configs = {}
    for name, module in model.named_modules():
        if isinstance(module, SwinBlockSequence):
            if module.depth == 2:
                configs[name] = sample_func(choices)
            elif module.depth == 6:
                p_ratios = [sample_func(choices) for i in range(3)]
                p_ratios.sort()
                configs[name + '_1'] = p_ratios[0]
                configs[name + '_2'] = p_ratios[1]
                configs[name + '_3'] = p_ratios[2]
            else:
                raise Exception('wrong stage depth')
    select(model, configs)
    return configs


def config(model: nn.Module) -> Dict[str, Any]:
    configs = {}
    for name, module in model.named_modules():
        if isinstance(module, SwinBlockSequence):
            if module.depth == 2:
                assert len(module.pruning_ratios) == 2 and module.pruning_ratios[0] == module.pruning_ratios[1]
                configs[name] = module.pruning_ratios[0]
            elif module.depth == 6:
                assert len(module.pruning_ratios) == 6
                configs[name+'_1'] = module.pruning_ratios[0]
                configs[name+'_2'] = module.pruning_ratios[2]
                configs[name+'_3'] = module.pruning_ratios[4]
    return configs


def main():
    dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--max", type=float)
    parser.add_argument("--min", type=float)
    parser.add_argument("--img_size", type=int, default=[672, 672])
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    #configs.load(args.config, recursive=True)
    #cfg = Config(recursive_eval(configs), filename=args.config)
    #print(cfg.pretty_text)

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())

    cfg.data.test.test_mode = True
    samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=True,
        shuffle=False,
    )

    # build the model and load checkpoint
    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    model = model.cuda()
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = fuse_conv_bn(model)

    #fp16_cfg = cfg.get("fp16", None)
    #if fp16_cfg is not None:
    #    wrap_fp16_model(model)

    random.seed(0)

    @torch.inference_mode()
    def measure(model, num_repeats=300, num_warmup=200, img_size=args.img_size):
        model.eval()

        backbone = model.backbone
        inputs = torch.randn(4, 3, img_size[0], img_size[1]).cuda()

        latencies = []
        for k in range(num_repeats + num_warmup):
            start = cuda_time()
            backbone(inputs)
            if k >= num_warmup:
                latencies.append((cuda_time() - start) * 1000)

        latencies = itertools.chain(dist.allgather(latencies))
        latencies = sorted(latencies)

        drop = int(len(latencies) * 0.25)
        return np.mean(latencies[drop:-drop])

    def evaluate(model: nn.Module) -> float:
        model = MMDistributedDataParallel(
            model,
            device_ids=[dist.local_rank()],
            broadcast_buffers=False,
        )
        with redirect_stdout(open(devnull, "w")):
            outputs = multi_gpu_test(model, data_loader)
            if dist.is_master():
                metrics = dataset.evaluate(outputs, metric=["bbox", "segm"])
                metric = metrics["bbox_mAP"] * 100
            else:
                metric = None
        return dist.broadcast(metric)

    def filter_func(model):
        latency = measure(model)
        return args.min <= latency <= args.max, {"latency": latency}

    def save_func(state_dict):
        print(state_dict["best"])

    #for sample_func in [min, lambda *_: 0.5]:
    #    print(sample_func, sample(model, sample_func))
    #    print("Latency: {:.2f} ms".format(measure(model)))
    #    print("Metric: {:.4f}".format(evaluate(model)))

    search_log = 'work_dirs/search'+'_max'+str(args.max)+'_min'+str(args.min)+'.txt'
    searcher = EvolveSearcher(search_log=search_log,)
    searcher.search(
        model,
        num_iters=10000,
        filter_func=lambda model: filter_func(model),
        score_func=lambda model: evaluate(model),
        save_func=save_func,

    )


if __name__ == "__main__":
    main()