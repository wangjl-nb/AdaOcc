"""One-epoch smoke config for the public OccScanNet-mini reproduction.

This reuses the normal mini PKLs/data and only shortens training to one epoch.
Use it after preparing data and weights to verify train + checkpoint + val before
launching the 200-epoch reference run.
"""

_base_ = ["radio_occscannet_mini.py"]

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=1, val_interval=1)
default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=1, save_last=True),
)
