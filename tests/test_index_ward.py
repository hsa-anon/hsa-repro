import pytest
import torch

from hsa.index import Buckets, WardIndex, build_buckets, pool_keys, ward_rounds


def _unit_keys(n, dim, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, dim, generator=g)
    return (x / x.norm(dim=1, keepdim=True)).float()


def _make_configs():
    # (pool_size, bucket_size, bucket_cap, cluster_block_size)
    return [
        (2, 4, 8, 16),
        (4, 8, 24, 32),
        (2, 8, 16, 32),
    ]


def test_pool_keys_shapes_and_sum():
    x = _unit_keys(8, 4).reshape(1, 8, 4)
    p = pool_keys(x, 4)
    assert p.shape == (1, 2, 4)
    assert torch.allclose(p[0, 0], x[0, :4].sum(dim=0))


def test_build_buckets_partition_and_sizes():
    keys = _unit_keys(32, 8, seed=1)
    b = build_buckets(keys, pool_size=2, bucket_size=4, bucket_cap=8, cluster_block_size=32)
    assert isinstance(b, Buckets)
    assert b.block_bucket_count == 32 // 4
    assert b.bucket_members.shape == (b.block_bucket_count * (32 // 32), 8)
    total = b.bucket_sizes.sum().item()
    assert total == 32
    # every key appears exactly once, positions valid
    members = b.bucket_members[b.bucket_members >= 0]
    assert members.unique().numel() == 32
    # centroids are unit rows except for empty buckets (zero row, matching
    # the reference's 0/max(norm,1e-30) normalisation)
    nonempty = b.bucket_sizes > 0
    off = (b.bucket_centroids[nonempty].norm(dim=1) - 1).abs().max().item()
    assert off < 1e-4


def test_ward_rounds_capacity_bound():
    keys = _unit_keys(32, 8, seed=2)
    blocks = keys.reshape(1, 32, 8)
    pools = pool_keys(blocks, 2)
    sizes = torch.full((1, pools.shape[1]), 2.0)
    labels, reached = ward_rounds(pools, sizes, bucket_count=4, bucket_cap=16)
    assert reached.all()
    counts = torch.bincount(labels[0], minlength=4)
    assert counts.max().item() * 2 <= 16


def test_rejects_bad_inputs():
    keys = _unit_keys(16, 4)
    with pytest.raises(ValueError):
        build_buckets(keys * 2.0, 2, 4, 8, 16)  # not unit rows
    with pytest.raises(ValueError):
        build_buckets(keys, 3, 4, 8, 16)  # pool_size does not divide bucket_size
    with pytest.raises(ValueError):
        build_buckets(keys, 2, 4, 6, 16)  # bucket_cap < 2 * bucket_size


@pytest.mark.parametrize("pool,bucket,cap,cluster", _make_configs())
def test_cross_check_against_mlx_reference(pool, bucket, cap, cluster):
    # Anti-drift guard against algorithmic changes: on margin-decisive data
    # (tight clusters far apart, so every merge decision has a cost margin far
    # above the ~8e-4 cross-backend matmul deviation) the torch port must
    # produce identical bucketings to the MLX reference.
    ref_mod = pytest.importorskip("reference_mlx")
    mx = pytest.importorskip("mlx.core")

    keys = _separated_keys(cluster * 3, cluster // bucket, bucket, 16, seed=42)
    ref = ref_mod.build_buckets(mx.array(keys.numpy()), pool, bucket, cap, cluster)
    got = build_buckets(keys, pool, bucket, cap, cluster)
    assert torch.allclose(
        got.bucket_centroids, torch.from_numpy(np_array(ref.bucket_centroids)), atol=1e-4
    )
    assert torch.equal(got.bucket_members, torch.from_numpy(np_array(ref.bucket_members)).long())
    assert torch.equal(got.bucket_sizes, torch.from_numpy(np_array(ref.bucket_sizes)).long())


def _separated_keys(key_count, cluster_count, keys_per_cluster, dim, seed=0):
    # Tight clusters around far-apart centers: intra-cluster merge costs sit
    # orders of magnitude below inter-cluster costs, so merge decisions are
    # tie-free and both backends must agree exactly.
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(cluster_count, dim, generator=g)
    centers = centers / centers.norm(dim=1, keepdim=True)
    noise = 0.05 * torch.randn(key_count, dim, generator=g)
    cluster_of_key = (
        torch.arange(key_count) % (keys_per_cluster * cluster_count)
    ) // keys_per_cluster
    keys = centers[cluster_of_key] + noise
    return (keys / keys.norm(dim=1, keepdim=True)).float()


@pytest.mark.parametrize("pool,bucket,cap,cluster", _make_configs())
def test_cross_check_random_quality_equivalence(pool, bucket, cap, cluster):
    # On random data the two backends agree on every decisive merge but can
    # flip near-tied ones (cost deviation ~8e-4 across matmul backends), so
    # the bucketing is not bit-identical. Assert the weaker contract: both
    # reach the bucket count, both partition validly under the cap, and the
    # total merged key-sum norm (the Ward objective this greedy maximizes)
    # matches to within the tie band.
    ref_mod = pytest.importorskip("reference_mlx")
    mx = pytest.importorskip("mlx.core")

    keys = _unit_keys(cluster * 3, 16, seed=42)
    ref = ref_mod.build_buckets(mx.array(keys.numpy()), pool, bucket, cap, cluster)
    got = build_buckets(keys, pool, bucket, cap, cluster)
    assert int(np_array(ref.bucket_sizes).sum()) == cluster * 3
    assert got.bucket_sizes.sum().item() == cluster * 3
    assert got.bucket_sizes.max().item() <= cap
    assert (np_array(ref.bucket_sizes) <= cap).all()
    for members, sizes in (
        (got.bucket_members, got.bucket_sizes),
        (
            torch.from_numpy(np_array(ref.bucket_members)).long(),
            torch.from_numpy(np_array(ref.bucket_sizes)).long(),
        ),
    ):
        flat = members[members >= 0]
        assert flat.unique().numel() == flat.numel() == cluster * 3
    objective_torch = _merged_norm_sum(keys, got.bucket_members)
    objective_ref = _merged_norm_sum(keys, torch.from_numpy(np_array(ref.bucket_members)).long())
    assert abs(objective_torch - objective_ref) <= 5e-3 * max(objective_torch, objective_ref)


def _merged_norm_sum(keys, members):
    sums = torch.zeros((members.shape[0], keys.shape[1]))
    rows, cols = torch.nonzero(members >= 0, as_tuple=True)
    sums.index_add_(0, rows, keys[members[rows, cols]])
    return float(sums.norm(dim=1).sum())


def np_array(mx_array):
    import numpy as np

    return np.asarray(mx_array)


def test_incremental_pages_and_snapshot():
    keys = _unit_keys(64, 8, seed=7)
    idx = WardIndex(index_dim=8, pool_size=2, bucket_size=4, bucket_cap=8, cluster_block_size=16)
    for i in range(0, 64, 16):
        idx.append(keys[i : i + 16])
    assert idx.total_keys == 64
    assert idx.pages_closed == 4
    c, m, s = idx.snapshot()
    single = build_buckets(keys, 2, 4, 8, 16)
    # per-page bucketing is block-local, so concatenation must equal the
    # single-shot build over the same grid
    assert torch.allclose(c, single.bucket_centroids)
    assert torch.equal(m, single.bucket_members)
    assert torch.equal(s, single.bucket_sizes)


def test_incremental_rejects_bad_shape():
    idx = WardIndex(index_dim=8, pool_size=2, bucket_size=4, bucket_cap=8, cluster_block_size=16)
    with pytest.raises(ValueError):
        idx.append(torch.zeros((15, 8)))
