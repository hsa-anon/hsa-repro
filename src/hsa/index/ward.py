import logging
import math
from typing import NamedTuple

import torch

logger = logging.getLogger(__name__)

NORM_TOLERANCE = {torch.float32: 1e-4, torch.bfloat16: 2e-2}


class Buckets(NamedTuple):
    # Normalized key sum of each bucket [m, index_dim] fp32
    bucket_centroids: torch.Tensor
    # Positions of each bucket's keys [m, bucket_cap] int64, -1 padded
    bucket_members: torch.Tensor
    # Keys in each bucket [m] int64
    bucket_sizes: torch.Tensor
    # Buckets a cluster block ends with
    block_bucket_count: int
    # Keys in a cluster block
    cluster_block_size: int


def check_unit_rows(name: str, vectors: torch.Tensor) -> None:
    if vectors.ndim != 2:
        raise ValueError(f"{name} must be [N, index_dim] - got shape {tuple(vectors.shape)}")
    if vectors.dtype not in NORM_TOLERANCE:
        raise ValueError(f"{name} must be float32 or bfloat16 - got {vectors.dtype}")
    if not bool(torch.isfinite(vectors).all()):
        raise ValueError(f"{name} holds a non-finite value")
    if vectors.numel():
        off = float((vectors.float().norm(dim=1) - 1).abs().max().item())
        if off > NORM_TOLERANCE[vectors.dtype]:
            raise ValueError(f"{name} must be unit rows - a row is {off:.3g} off unit length")


def check_sizes(pool_size: int, bucket_size: int, bucket_cap: int, cluster_block_size: int) -> None:
    if min(pool_size, bucket_size, bucket_cap, cluster_block_size) < 1:
        raise ValueError(
            "pool_size, bucket_size, bucket_cap and cluster_block_size must be positive"
        )
    if bucket_size % pool_size or cluster_block_size % bucket_size:
        raise ValueError(
            f"pool_size {pool_size} must divide bucket_size {bucket_size}, which must divide "
            f"cluster_block_size {cluster_block_size}"
        )
    if bucket_cap < 2 * bucket_size:
        raise ValueError(
            f"bucket_cap must be at least twice bucket_size {bucket_size} - got {bucket_cap}"
        )


def pool_keys(index_keys: torch.Tensor, pool_size: int) -> torch.Tensor:
    # index_keys [G, T, index_dim] -> [G, T / pool_size, index_dim]
    blocks, key_count, _ = index_keys.shape
    if key_count % pool_size:
        raise ValueError(f"key_count {key_count} must be a multiple of pool_size {pool_size}")
    return index_keys.reshape(blocks, key_count // pool_size, pool_size, index_keys.shape[2]).sum(
        dim=2
    )


def _nonzero_flat(mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    flat = mask.reshape(-1)
    idx = torch.nonzero(flat, as_tuple=False).reshape(-1)
    return idx, int(idx.numel())


def ward_rounds(
    pools: torch.Tensor,
    pool_sizes: torch.Tensor,
    bucket_count: int,
    bucket_cap: int,
    max_rounds: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Merge each block's pools down to bucket_count buckets by mutual best-pair
    # Ward cost, capacity-bounded. Torch port of the serving-path MLX
    # implementation is the reference). The two are line-identical algorithms,
    # but gram matmuls differ across backends (torch CPU vs mlx Metal measured
    # 8e-4 max cost deviation), so near-tied merge decisions can flip and the
    # final bucketing is only equivalent in Ward quality, not bit-identical.
    # tests/test_index_ward.py asserts exact equality on margin-decisive data
    # and quality equivalence on random data.
    # pools [G, P, dim], pool_sizes [G, P]; returns labels [G, P] int64, reached [G] bool.
    blocks, n_pools, _ = pools.shape
    key_sums = pools.float()
    sizes = pool_sizes.float()
    alive = torch.ones((blocks, n_pools), dtype=torch.bool, device=pools.device)
    parent = torch.arange(n_pools, device=pools.device).expand(blocks, n_pools).contiguous()
    index = torch.arange(n_pools, device=pools.device).expand(blocks, n_pools).contiguous()
    diagonal = torch.eye(n_pools, dtype=torch.bool, device=pools.device)
    merges_done = 0
    for _ in range(max_rounds):
        merges_left = alive.sum(dim=1) - bucket_count
        if not bool((merges_left > 0).any()):
            break
        gram = torch.bmm(key_sums, key_sums.transpose(1, 2))
        squares = gram.diagonal(dim1=1, dim2=2)
        merged = (squares[:, :, None] + squares[:, None, :] + 2 * gram).clamp_min(0).sqrt()
        align = squares.clamp_min(0).sqrt()
        merge_costs = align[:, :, None] + align[:, None, :] - merged
        allowed = alive[:, :, None] & alive[:, None, :] & ~diagonal
        allowed = allowed & (sizes[:, :, None] + sizes[:, None, :] <= bucket_cap)
        masked = torch.where(allowed, merge_costs, torch.inf)
        best_costs = masked.min(dim=2).values
        best = masked.argmin(dim=2)
        mutual = (best_costs < torch.inf) & (best.gather(1, best) == index) & (index < best)
        rank = torch.where(mutual, best_costs, torch.inf).argsort(dim=1).argsort(dim=1)
        mutual = mutual & (rank < merges_left.clamp_min(0)[:, None])
        flat_idx, count = _nonzero_flat(mutual)
        if count == 0:
            break
        block_ids = flat_idx // n_pools
        kept = flat_idx % n_pools
        absorbed = best[block_ids, kept]
        # Mutual pairs are disjoint by construction, so one scatter pass per round.
        recv = torch.full((blocks * n_pools,), -1, dtype=torch.int64, device=pools.device)
        recv[block_ids * n_pools + kept] = absorbed
        recv = recv.view(blocks, n_pools)
        has_recv = recv >= 0
        safe_r = recv.clamp_min(0)
        # gather needs the index expanded per column: a [G, P, 1] index would
        # broadcast the gathered vector's first element across all index_dim
        # columns instead of adding the whole absorbed pool vector.
        gathered = torch.gather(key_sums, 1, safe_r[:, :, None].expand(-1, -1, key_sums.shape[2]))
        key_sums = key_sums + torch.where(
            has_recv[:, :, None], gathered, torch.zeros((), device=pools.device)
        )
        sizes = sizes + torch.where(
            has_recv, torch.gather(sizes, 1, safe_r), torch.zeros((), device=pools.device)
        )
        gone = torch.full((blocks * n_pools,), -1, dtype=torch.int64, device=pools.device)
        gone[block_ids * n_pools + absorbed] = kept
        alive = alive & ~(gone >= 0).view(blocks, n_pools)
        parent = torch.where((gone >= 0).view(blocks, n_pools), gone.view(blocks, n_pools), parent)
        merges_done += count
    for _ in range(math.ceil(math.log2(max(n_pools, 2))) + 1):
        parent = torch.gather(parent, 1, parent)
    number = alive.long().cumsum(dim=1) - 1
    labels = torch.gather(number, 1, parent)
    reached = alive.sum(dim=1) <= bucket_count
    logger.debug("ward_rounds done: %d merges over %d blocks", merges_done, blocks)
    return labels, reached


def build_buckets(
    index_keys: torch.Tensor,
    pool_size: int,
    bucket_size: int,
    bucket_cap: int,
    cluster_block_size: int,
    cluster_algorithm: str = "ward",
) -> Buckets:
    check_unit_rows("index_keys", index_keys)
    check_sizes(pool_size, bucket_size, bucket_cap, cluster_block_size)
    if cluster_algorithm not in ("ward", "position"):
        raise ValueError(
            f"cluster_algorithm must be 'ward' or 'position' - got {cluster_algorithm!r}"
        )
    keys = index_keys.float()
    key_count, index_dim = keys.shape
    device = keys.device
    block_count = key_count // cluster_block_size
    block_bucket_count = cluster_block_size // bucket_size
    blocks = keys[: block_count * cluster_block_size].reshape(
        block_count, cluster_block_size, index_dim
    )
    pools = pool_keys(blocks, pool_size)
    pool_sizes = torch.full(pools.shape[:2], float(pool_size), device=device)
    pools_per_bucket = bucket_size // pool_size
    if cluster_algorithm == "position":
        labels = (
            (torch.arange(pools.shape[1], device=device) // pools_per_bucket)
            .expand(block_count, -1)
            .contiguous()
        )
    else:
        labels, reached = ward_rounds(pools, pool_sizes, block_bucket_count, bucket_cap)
        if not bool(reached.all()):
            short = int((~reached).sum().item())
            raise ValueError(
                f"{short} blocks stopped short of {block_bucket_count} buckets under the cap"
            )
    key_labels = labels.repeat_interleave(pool_size, dim=1)
    bucket_ids = torch.arange(block_bucket_count, device=device)
    onehot = (key_labels[:, :, None] == bucket_ids[None, None, :]).float()
    # [G, buckets, dim] - one matmul per block, batched
    key_sums = torch.bmm(onehot.transpose(1, 2), blocks)
    bucket_centroids = key_sums.reshape(-1, index_dim)
    bucket_centroids = bucket_centroids / bucket_centroids.norm(dim=1, keepdim=True).clamp_min(
        1e-30
    )
    onehot_pos = key_labels[:, :, None] == bucket_ids[None, None, :]
    members = torch.where(
        onehot_pos.any(dim=1),
        torch.zeros((), dtype=torch.long, device=device),
        torch.full((), -1, dtype=torch.long, device=device),
    )
    # Members vectorized: per block, keys stably sorted by label; rank within
    # each label group = position minus the group's first position (segment
    # start via cummax over the group-boundary marker).
    if block_count:
        order = key_labels.argsort(dim=1, stable=True)
        sorted_lab = torch.gather(key_labels, 1, order)
        new_group = torch.ones_like(sorted_lab, dtype=torch.bool)
        new_group[:, 1:] = sorted_lab[:, 1:] != sorted_lab[:, :-1]
        positions = torch.arange(cluster_block_size, device=device).expand(block_count, -1)
        seg_start = torch.cummax(
            torch.where(new_group, positions, torch.zeros_like(positions)), dim=1
        ).values
        pos_in_group = positions - seg_start
        valid = pos_in_group < bucket_cap
        flat = torch.full(
            (block_count, block_bucket_count * bucket_cap), -1, dtype=torch.long, device=device
        )
        # global slot = (block * block_bucket_count + label) * bucket_cap + rank
        v_block = torch.arange(block_count, device=device).repeat_interleave(cluster_block_size)
        v_label = sorted_lab.reshape(-1)
        v_rank = pos_in_group.reshape(-1)
        v_mask = valid.reshape(-1)
        v_idx = (
            (v_block * block_bucket_count + v_label) * bucket_cap + v_rank.clamp_max(bucket_cap - 1)
        )[v_mask]
        # local per-block position; the global block offset is added once below
        v_val = order.reshape(-1)[v_mask]
        flat.view(-1).scatter_(0, v_idx, v_val)
        members = flat.view(block_count, block_bucket_count, bucket_cap)
        members = torch.where(
            members >= 0,
            members + torch.arange(block_count, device=device)[:, None, None] * cluster_block_size,
            members,
        )
    else:
        members = torch.zeros((0, block_bucket_count, bucket_cap), dtype=torch.long, device=device)
    members = members.reshape(-1, bucket_cap)
    bucket_sizes = (members >= 0).sum(dim=1)
    logger.debug(
        "build_buckets: %d blocks, %d buckets each, %d keys",
        block_count,
        block_bucket_count,
        key_count,
    )
    return Buckets(bucket_centroids, members, bucket_sizes, block_bucket_count, cluster_block_size)
