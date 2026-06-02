import math
from typing import Iterator, Optional, Sized

import torch
from torch.utils.data import Sampler

from mmengine.dist import get_dist_info, sync_random_seed
from mmengine.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class BatchAlignedDefaultSampler(Sampler):
    """DefaultSampler variant that pads each distributed rank to full local batches.

    This keeps all ranks on the same number of iterations while avoiding a
    singleton tail batch on distributed runs. In single-GPU mode it falls back
    to the standard DefaultSampler behavior.
    """

    def __init__(
        self,
        dataset: Sized,
        shuffle: bool = True,
        seed: Optional[int] = None,
        round_up: bool = True,
        batch_size: int = 1,
        batch_size_scale: str = "local",
    ) -> None:
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.shuffle = bool(shuffle)
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.epoch = 0
        self.round_up = bool(round_up)
        self.batch_size = int(batch_size)
        self.batch_size_scale = str(batch_size_scale)

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.batch_size_scale not in ("local", "global"):
            raise ValueError(
                f"batch_size_scale must be 'local' or 'global', got {self.batch_size_scale}"
            )

        local_batch_size = self.batch_size
        if self.batch_size_scale == "global":
            local_batch_size = max(1, self.batch_size // max(1, self.world_size))
        self.local_batch_size = int(local_batch_size)

        if self.round_up:
            base_num_samples = math.ceil(len(self.dataset) / self.world_size)
            if self.world_size > 1:
                self.num_samples = (
                    math.ceil(base_num_samples / self.local_batch_size) * self.local_batch_size
                )
            else:
                self.num_samples = base_num_samples
            self.total_size = self.num_samples * self.world_size
        else:
            self.num_samples = math.ceil((len(self.dataset) - rank) / self.world_size)
            self.total_size = len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        if self.round_up:
            indices = (indices * int(self.total_size / len(indices) + 1))[: self.total_size]

        indices = indices[self.rank : self.total_size : self.world_size]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
