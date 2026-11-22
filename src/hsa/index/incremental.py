import logging

import torch

from hsa.index.ward import Buckets, build_buckets

logger = logging.getLogger(__name__)


class WardIndex:
    # Incremental torch-side Ward bucket index: keys arrive cluster-block by
    # cluster-block (page close = one build_buckets call per block). Torch port
    # of the serving-path incremental index for the NPU eval path; the MLX
    # implementation remains the serving reference.

    def __init__(
        self,
        index_dim: int,
        pool_size: int,
        bucket_size: int,
        bucket_cap: int,
        cluster_block_size: int,
        device: str = "cpu",
    ):
        if index_dim < 1:
            raise ValueError("index_dim must be positive")
        if cluster_block_size % bucket_size or bucket_size % pool_size:
            raise ValueError(
                "cluster_block_size must divide by bucket_size, bucket_size by pool_size"
            )
        self.index_dim = index_dim
        self.pool_size = pool_size
        self.bucket_size = bucket_size
        self.bucket_cap = bucket_cap
        self.cluster_block_size = cluster_block_size
        self.device = device
        self._centroids: list[torch.Tensor] = []
        self._members: list[torch.Tensor] = []
        self._sizes: list[torch.Tensor] = []
        self.total_keys = 0
        self.pages_closed = 0

    def append(self, block_keys: torch.Tensor) -> Buckets:
        # One page close: block_keys [cluster_block_size, index_dim], unit rows.
        if block_keys.shape != (self.cluster_block_size, self.index_dim):
            raise ValueError(
                f"expected ({self.cluster_block_size}, {self.index_dim}), got {tuple(block_keys.shape)}"
            )
        if block_keys.device.type != torch.device(self.device).type:
            block_keys = block_keys.to(self.device)
        buckets = build_buckets(
            block_keys, self.pool_size, self.bucket_size, self.bucket_cap, self.cluster_block_size
        )
        page = self.pages_closed
        members = buckets.bucket_members
        members = torch.where(members >= 0, members + page * self.cluster_block_size, members)
        self._centroids.append(buckets.bucket_centroids)
        self._members.append(members)
        self._sizes.append(buckets.bucket_sizes)
        self.total_keys += self.cluster_block_size
        self.pages_closed += 1
        logger.debug(
            "page close %d: %d buckets, total %d keys",
            self.pages_closed,
            buckets.bucket_centroids.shape[0],
            self.total_keys,
        )
        return buckets

    def snapshot(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Concatenated (centroids, members, sizes) over all closed pages.
        if not self._centroids:
            z = torch.zeros((0, self.index_dim), device=self.device)
            return (
                z,
                torch.zeros((0, self.bucket_cap), dtype=torch.long, device=self.device),
                torch.zeros((0,), dtype=torch.long, device=self.device),
            )
        return torch.cat(self._centroids), torch.cat(self._members), torch.cat(self._sizes)
