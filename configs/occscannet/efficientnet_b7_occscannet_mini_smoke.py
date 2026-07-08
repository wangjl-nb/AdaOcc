"""One-iteration smoke config for the EfficientNet-B7 OccScanNet-mini experiment."""

_base_ = ["efficientnet_b7_occscannet_mini.py"]

global_batch_size = 1
batch_size = 1

train_cfg = dict(_delete_=True, type="IterBasedTrainLoop", max_iters=1, val_interval=1000000)
val_cfg = None
val_dataloader = None
val_evaluator = None

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(_delete_=True, type="DefaultSampler", shuffle=False),
)

default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=1, save_last=False),
)
