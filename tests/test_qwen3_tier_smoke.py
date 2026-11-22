import copy

import pytest
import torch

transformers = pytest.importorskip("transformers")

from hsa.train.qwen3_tier import (
    attach_tier_weights,
    disable_tier_mode,
    enable_tier_mode,
    trainable_parameters,
)


def tiny_qwen3():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
        rope_theta=10000.0,
        _attn_implementation="eager",
    )
    torch.manual_seed(6)
    return Qwen3ForCausalLM(cfg)


def test_qwen3_tier_distill_smoke():
    d_lat, window, block, seq, steps = 32, 8, 8, 32, 25
    model = tiny_qwen3()
    for p in model.parameters():
        p.requires_grad_(False)
    teacher = copy.deepcopy(model)
    for p in teacher.parameters():
        p.requires_grad_(False)
    attach_tier_weights(model, d_lat)
    enable_tier_mode(model, window, block)
    input_ids = torch.randint(0, 60, (1, seq))
    attn = torch.ones_like(input_ids)
    opt = torch.optim.Adam(trainable_parameters(model), lr=1e-2)
    history = []
    for _ in range(steps):
        with torch.no_grad():
            t_hs = teacher(input_ids, attention_mask=attn, output_hidden_states=True).hidden_states
        s_hs = model(input_ids, attention_mask=attn, output_hidden_states=True).hidden_states
        # per-layer hidden MSE (skip embeddings) - denser signal than final-only
        loss = sum(torch.nn.functional.mse_loss(s, t) for s, t in zip(s_hs[1:], t_hs[1:])) / (
            len(s_hs) - 1
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(float(loss))
    # mechanism smoke only: random-weight tiny model has no compressible
    # attention structure; convergence is validated on pretrained weights (Task 4)
    assert history[-1] < 0.85 * history[0], (
        f"loss must decrease: {history[0]:.6f} -> {history[-1]:.6f}"
    )
    for layer in model.model.layers:
        for name, p in layer.self_attn.named_parameters():
            if name not in ("w_q_lat", "w_k_lat"):
                assert p.grad is None or p.grad.abs().sum() == 0, f"frozen {name} got gradient"
    disable_tier_mode(model)
    with torch.no_grad():
        dense_out = model(input_ids, attention_mask=attn).logits
    assert dense_out.shape == (1, seq, 64)
