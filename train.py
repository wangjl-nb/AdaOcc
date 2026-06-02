import argparse
import importlib
import os
import os.path as osp
import time

import torch
from mmengine.config import Config, DictAction

import utils


def _split_run_timestamp(run_timestamp: str):
    if '/' in run_timestamp:
        date_part, time_part = run_timestamp.split('/', 1)
        return date_part, time_part
    if '_' in run_timestamp:
        date_part, time_part = run_timestamp.split('_', 1)
        return date_part, time_part
    return run_timestamp, ''


def _normalize_run_label(label: str):
    if not label:
        return ''
    safe = []
    last_was_sep = False
    for ch in label.strip():
        if ch.isalnum() or ch in ('-', '_'):
            safe.append(ch)
            last_was_sep = False
        else:
            if not last_was_sep:
                safe.append('_')
                last_was_sep = True
    return ''.join(safe).strip('_')


def main():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', required=True)
    parser.add_argument('--override', nargs='+', action=DictAction)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    args = parser.parse_args()

    from mmdet3d.utils import register_all_modules
    register_all_modules(init_default_scope=True)

    cfg = Config.fromfile(args.config)
    override_keys = set(args.override.keys()) if args.override is not None else set()
    if args.override is not None:
        cfg.merge_from_dict(args.override)
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

    def _scale_dataloader_batch_size(loader_cfg, world_size):
        if loader_cfg is None or world_size <= 1:
            return
        if isinstance(loader_cfg, (list, tuple)):
            for item in loader_cfg:
                _scale_dataloader_batch_size(item, world_size)
            return
        if isinstance(loader_cfg, dict) and 'batch_size' in loader_cfg:
            loader_cfg['batch_size'] = max(1, loader_cfg['batch_size'] // world_size)

    run_timestamp = os.environ.get('ADAOCC_RUN_TIMESTAMP')
    if not run_timestamp:
        run_timestamp = time.strftime('%Y-%m-%d/%H-%M-%S', time.localtime())
    run_label = _normalize_run_label(os.environ.get('ADAOCC_RUN_LABEL', ''))

    repo_root = os.environ.get('ADAOCC_REPO_ROOT') or osp.dirname(osp.abspath(__file__))
    repo_root = osp.abspath(osp.expanduser(repo_root))

    resume_from = cfg.get('resume_from', None)
    if resume_from:
        if not osp.isfile(resume_from):
            raise FileNotFoundError(resume_from)
        cfg.work_dir = osp.dirname(resume_from)
        cfg.load_from = resume_from
        cfg.resume = True
    else:
        config_name = osp.splitext(osp.split(args.config)[-1])[0]
        date_part, time_part = _split_run_timestamp(run_timestamp)
        run_root = f'{date_part}_{config_name}'
        output_root = cfg.get('output_root', None) or os.environ.get('ADAOCC_OUTPUT_ROOT') or osp.join(repo_root, 'outputs')
        output_root = osp.expanduser(str(output_root))
        if not osp.isabs(output_root):
            output_root = osp.join(repo_root, output_root)
        output_subdir = cfg.get('output_subdir', None)
        if output_subdir:
            cfg.work_dir = osp.join(output_root, cfg.model['type'], str(output_subdir), run_root)
        else:
            cfg.work_dir = osp.join(output_root, cfg.model['type'], run_root)
        if time_part:
            if run_label:
                time_part = f'{time_part}_{run_label}'
            cfg.work_dir = osp.join(cfg.work_dir, time_part)
        cfg.resume = False

    cfg.launcher = 'pytorch' if world_size > 1 else 'none'

    # Respect an explicit CLI override for train_dataloader.batch_size.
    # Otherwise, keep the legacy behavior that derives per-rank batch size
    # from the top-level global batch_size field.
    if (cfg.get('batch_size') is not None
            and cfg.get('train_dataloader') is not None
            and 'train_dataloader.batch_size' not in override_keys):
        cfg.train_dataloader.batch_size = max(1, cfg.batch_size // world_size)

    # Treat val/test batch_size as global and scale by world_size.
    _scale_dataloader_batch_size(cfg.get('val_dataloader'), world_size)
    _scale_dataloader_batch_size(cfg.get('test_dataloader'), world_size)

    if 'randomness' not in cfg:
        cfg.randomness = dict(seed=0, deterministic=True)

    runner = AdaOccRunner.from_cfg(cfg)

    # Eagerly build the optimizer wrapper before entering Runner.train().
    # On the current H20 / torch2.2 stack, the lazy path can stall or crash
    # before the first train iter for this EfficientNet-native-FPN setup,
    # while the explicit build path runs stably.
    runner.optim_wrapper = runner.build_optim_wrapper(runner.optim_wrapper)

    if runner.rank == 0:
        utils.backup_code(cfg.work_dir)
    runner.train()


if __name__ == '__main__':
    main()
