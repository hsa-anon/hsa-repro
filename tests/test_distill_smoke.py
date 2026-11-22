import copy

import numpy as np
import torch

from hsa.train.distill import distill_loop
from hsa.train.inject import attach_latent_proj, three_tier_attention_with

torch.manual_seed(7)


class TinyAttention(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.wq = torch.nn.Linear(d, d, bias=False)
        self.wk = torch.nn.Linear(d, d, bias=False)
        self.wv = torch.nn.Linear(d, d, bias=False)
        self.wo = torch.nn.Linear(d, d, bias=False)

    def project(self, x):
        return self.wq(x), self.wk(x), self.wv(x)

    def dense(self, q, k, v):
        seq = q.shape[0]
        scores = q @ k.transpose(0, 1) / q.shape[-1] ** 0.5
        mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool), 1)
        scores = scores.masked_fill(mask, float("-inf"))
        return torch.softmax(scores, dim=-1) @ v


class TinyLayer(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = TinyAttention(d)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d, 2 * d), torch.nn.GELU(), torch.nn.Linear(2 * d, d)
        )

    def forward(self, x, tier):
        q, k, v = self.attn.project(x)
        a = tier(q, k, v)
        return x + self.mlp(x + a)


class TinyModel(torch.nn.Module):
    def __init__(self, d, n_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList([TinyLayer(d) for _ in range(n_layers)])

    def forward(self, x, window, block, d_lat, use_tier):
        attn_outs = []
        for layer in self.layers:
            q, k, v = layer.attn.project(x)

            def tier(q=q, k=k, v=v, layer=layer):
                if use_tier:
                    return three_tier_attention_with(
                        layer, q[:, None], k[:, None], v[:, None], window, block
                    )[:, 0]
                return layer.attn.dense(q, k, v)

            x = layer(x, tier)
            attn_outs.append(x)
        return x, attn_outs


def test_distill_smoke():
    d, n_layers, seq, window, block, d_lat = 32, 2, 32, 4, 8, 16
    model = TinyModel(d, n_layers)
    for p in model.parameters():
        p.requires_grad_(False)
    attach_latent_proj(model.layers[0], d_lat)
    attach_latent_proj(model.layers[1], d_lat)
    teacher = copy.deepcopy(model)
    for p in teacher.parameters():
        p.requires_grad_(False)
    rng = np.random.default_rng(41)
    batch = torch.tensor(rng.standard_normal((seq, d)), dtype=torch.float32)

    def student_forward():
        return model(batch, window, block, d_lat, use_tier=True)

    def teacher_forward():
        with torch.no_grad():
            return teacher(batch, window, block, d_lat, use_tier=False)

    trainable = [p for layer in model.layers for p in layer.latent_proj.parameters()]
    history = distill_loop(student_forward, teacher_forward, trainable, steps=50, lr=1e-2)
    assert history[-1] < 0.6 * history[0], (
        f"loss must decrease: {history[0]:.4f} -> {history[-1]:.4f}"
    )
    for layer in model.layers:
        for name, p in layer.named_parameters():
            if "latent_proj" not in name:
                assert p.grad is None or p.grad.abs().sum() == 0, (
                    f"frozen param {name} got gradient"
                )
    for layer in model.layers:
        assert layer.latent_proj.w_k.grad is not None and layer.latent_proj.w_k.grad.abs().sum() > 0


def test_distill_handles_partial_block_seq():
    d, n_layers, seq, window, block, d_lat = 32, 1, 30, 4, 8, 16
    model = TinyModel(d, n_layers)
    for p in model.parameters():
        p.requires_grad_(False)
    attach_latent_proj(model.layers[0], d_lat)
    rng = np.random.default_rng(42)
    batch = torch.tensor(rng.standard_normal((seq, d)), dtype=torch.float32)
    out, _ = model(batch, window, block, d_lat, use_tier=True)
    assert out.shape == (seq, d)
    assert torch.isfinite(out).all()
