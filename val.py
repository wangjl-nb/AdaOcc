import argparse
import importlib
import os
import os.path as osp

import torch
from mmengine.config import Config


def _force_offline_sweeps(dataloader_cfg):
    if not dataloader_cfg:
        return
    dataset_cfg = dataloader_cfg.get('dataset', None)
    if not dataset_cfg:
        return
    pipeline = dataset_cfg.get('pipeline', None)
    if not pipeline:
        return

    for transform in pipeline:
        if not isinstance(transform, dict):
            continue
        if transform.get('type') == 'LoadMultiViewImageFromMultiSweeps':
            transform['force_offline'] = True


def main():
    parser = argparse.ArgumentParser(description='Validate a detector')
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    from mmdet3d.utils import register_all_modules
    register_all_modules(init_default_scope=True)

    cfg = Config.fromfile(args.config)
    importlib.import_module('models')
    importlib.import_module('loaders')
    from models.runtime import AdaOccRunner

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = str(args.world_size)

    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    cfg.launcher = 'pytorch' if world_size > 1 else 'none'
    if world_size == 1:
        # Single-GPU eval must load historical sweeps as images; online mode
        # only appends metadata and can break temporal models.
        _force_offline_sweeps(cfg.get('val_dataloader'))
        _force_offline_sweeps(cfg.get('test_dataloader'))
    if cfg.get('val_dataloader') is not None:
        cfg.val_dataloader.batch_size = args.batch_size

    cfg.load_from = args.weights
    cfg.resume = False

    if not cfg.get('work_dir'):
        cfg.work_dir = osp.dirname(args.weights)

    cfg.train_dataloader = None
    cfg.train_cfg = None
    cfg.optim_wrapper = None
    cfg.param_scheduler = None

    runner = AdaOccRunner.from_cfg(cfg)
    runner.val()


if __name__ == '__main__':
    main()
