import logging
from types import SimpleNamespace

import pytest
import torch

from hsa.train.kimi_tier import (
    KimiTierCache,
    attach_tier_weights,
    disable_tier_mode,
    enable_tier_mode,
    mla_layers,
    trainable_parameters,
)

logging.basicConfig(level=logging.WARNING)


class TinyRMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dt)


class TinyMLAAttention(torch.nn.Module):
    # Mirrors KimiMLAAttention structure: no q_lora, no q_norm, no RoPE
    # application, compressed kv_a projection expanded per head by kv_b_proj.
    def __init__(self, hidden, heads, kv_lora, nope, rope, v_head):
        super().__init__()
        self.layer_idx = 0
        self.num_heads = heads
        self.hidden_size = hidden
        self.kv_lora_rank = kv_lora
        self.qk_rope_head_dim = rope
        self.qk_nope_head_dim = nope
        self.v_head_dim = v_head
        self.q_head_dim = nope + rope
        self.scaling = self.q_head_dim**-0.5
        self.q_proj = torch.nn.Linear(hidden, heads * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = torch.nn.Linear(hidden, kv_lora + rope, bias=False)
        self.kv_a_layernorm = TinyRMSNorm(kv_lora)
        self.kv_b_proj = torch.nn.Linear(kv_lora, heads * (nope + v_head), bias=False)
        self.o_proj = torch.nn.Linear(heads * v_head, hidden, bias=False)

    def forward(self, hidden_states, attention_mask=None, past_key_values=None, **kwargs):
        b, t = hidden_states.shape[:2]
        q = self.q_proj(hidden_states).view(b, t, self.num_heads, self.q_head_dim).transpose(1, 2)
        compressed = self.kv_a_proj_with_mqa(hidden_states)
        k_pass = self.kv_a_layernorm(compressed[..., : self.kv_lora_rank])
        kv = self.kv_b_proj(k_pass).view(b, t, self.num_heads, -1).transpose(1, 2)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_rot = compressed[..., self.kv_lora_rank :].view(b, 1, t, self.qk_rope_head_dim)
        k_rot = k_rot.expand(b, self.num_heads, t, self.qk_rope_head_dim)
        key = torch.cat([k_nope, k_rot], dim=-1)
        if past_key_values is not None:
            key, v = past_key_values.update(key, v, self.layer_idx)
        scores = q @ key.transpose(-1, -2) * self.scaling
        total = key.shape[2]
        j = torch.arange(total, device=q.device)
        i = torch.arange(t, device=q.device) + (total - t)
        mask = j[None, :] > i[:, None]
        scores = scores.masked_fill(mask[None, None], float("-inf"))
        out = torch.softmax(scores, dim=-1) @ v
        return self.o_proj(out.transpose(1, 2).reshape(b, t, self.num_heads * self.v_head_dim))


class TinyLayer(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        if attn is not None:
            self.self_attn = attn


class TinyKimiModel(torch.nn.Module):
    # depth 4, MLA layers at 1 and 3, KDA layers elsewhere; the config lists
    # an out-of-range full-attn index (27) like the shipped Kimi config does.
    def __init__(self):
        super().__init__()
        attn_kwargs = {"hidden": 32, "heads": 2, "kv_lora": 16, "nope": 8, "rope": 4, "v_head": 8}
        full = [1, 3, 27]
        self.config = SimpleNamespace(
            num_hidden_layers=4,
            linear_attn_config=SimpleNamespace(full_attn_layers=full),
        )
        mla = {i: TinyMLAAttention(**attn_kwargs) for i in full if i < 4}
        inner = torch.nn.Module()
        inner.layers = torch.nn.ModuleList([TinyLayer(mla.get(i)) for i in range(4)])
        self.model = inner


def test_mla_layers_drops_out_of_range_index():
    model = TinyKimiModel()
    picked = mla_layers(model)
    ids = [
        i
        for i, layer in enumerate(model.model.layers)
        if any(picked_layer is layer for picked_layer in picked)
    ]
    assert ids == [1, 3]


def test_attach_shapes_and_trainable():
    model = TinyKimiModel()
    attach_tier_weights(model, d_latent=12)
    params = trainable_parameters(model)
    assert len(params) == 4
    attn = model.model.layers[1].self_attn
    assert attn.w_q_lat.shape == (2 * 12, 12)  # heads * q_head_dim
    assert attn.w_k_lat.shape == (16 + 4, 12)  # kv_lora + qk_rope


def test_decode_exact_topk_matches_dense():
    # Pivot A step-2 control: with the retrieval cap disabled (topk=None)
    # tiered decode over the compressed cache must reproduce dense attention.
    torch.manual_seed(3)
    model = TinyKimiModel()
    attach_tier_weights(model, d_latent=12)
    x = torch.randn(1, 24, 32)
    attn = model.model.layers[1].self_attn
    dense = attn(x)
    enable_tier_mode(model, window=5, sink=2, topk=None)
    attn._tier_decode = True
    outs = []
    for start in (0, 7, 14):  # chunked prefill covers positions 0..20
        outs.append(attn(x[:, start : start + 7])[0])
    for step in range(21, 24):  # tiered decode for the tail positions
        outs.append(attn(x[:, step : step + 1])[0])
    tiered = torch.cat(outs, dim=1)
    assert torch.allclose(dense, tiered, atol=1e-4)


def test_decode_topk_subset_runs_and_finite():
    model = TinyKimiModel()
    attach_tier_weights(model, d_latent=12)
    enable_tier_mode(model, window=4, sink=1, topk=6)
    attn = model.model.layers[3].self_attn
    attn._tier_decode = True
    x = torch.randn(1, 10, 32)
    for start in range(0, 10, 3):
        out = attn(x[:, start : start + 3])
        assert torch.isfinite(out[0]).all()


def test_disable_restores_dense():
    model = TinyKimiModel()
    attach_tier_weights(model, d_latent=12)
    enable_tier_mode(model, window=4)
    disable_tier_mode(model)
    attn = model.model.layers[1].self_attn
    x = torch.randn(1, 6, 32)
    out = attn(x)
    assert torch.isfinite(out).all()
    assert attn._tier_cache is None


@pytest.mark.parametrize(
    "total,window,sink,topk", [(20, 5, 2, 8), (20, 0, 0, 4), (20, 19, 1, None)]
)
def test_cache_decode_selection_bounds(total, window, sink, topk):
    torch.manual_seed(5)
    model = TinyKimiModel()
    attn = model.model.layers[1].self_attn
    attach_tier_weights(model, d_latent=8)
    cache = KimiTierCache(attn)
    cache.append(torch.randn(1, total, 20))
    q = torch.randn(attn.num_heads, attn.q_head_dim)
    out = cache.decode(q, attn.w_q_lat, attn.w_k_lat, topk=topk, window=window, sink=sink)
    assert out.shape == (1, 1, attn.num_heads * attn.v_head_dim)
    assert torch.isfinite(out).all()
