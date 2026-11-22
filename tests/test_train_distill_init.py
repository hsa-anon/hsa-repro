import torch

from hsa.train.train_distill import load_init_ckpt


class _StubAttn(torch.nn.Module):
    def __init__(self, hq_d, hkv_d, lat):
        super().__init__()
        self.w_q_lat = torch.nn.Parameter(torch.zeros(hq_d, lat))
        self.w_k_lat = torch.nn.Parameter(torch.zeros(hkv_d, lat))


class _StubLayer(torch.nn.Module):
    def __init__(self, has_attn, hq_d=8, hkv_d=4, lat=4):
        super().__init__()
        if has_attn:
            self.self_attn = _StubAttn(hq_d, hkv_d, lat)


class _StubModel:
    def __init__(self, n_layers=4):
        # full-attention layers at indices 1 and 3, mirroring the mod-4 layout
        self.layers = torch.nn.ModuleList([_StubLayer(i in (1, 3)) for i in range(n_layers)])


def test_load_init_ckpt_restores_weights(tmp_path):
    model = _StubModel()
    src = _StubModel()
    with torch.no_grad():
        src.layers[1].self_attn.w_q_lat.fill_(1.5)
        src.layers[1].self_attn.w_k_lat.fill_(-2.0)
        src.layers[3].self_attn.w_q_lat.fill_(3.0)
        src.layers[3].self_attn.w_k_lat.fill_(4.0)
    ck = {
        "tier_weights": {
            f"layer{n}.{k}": getattr(l.self_attn, k).detach().cpu()
            for n, l in enumerate(model.layers)
            if hasattr(l, "self_attn")
            for k in ("w_q_lat", "w_k_lat")
        }
    }
    # overwrite with src values under the global-index keying
    ck["tier_weights"]["layer1.w_q_lat"] = src.layers[1].self_attn.w_q_lat.detach().cpu()
    ck["tier_weights"]["layer1.w_k_lat"] = src.layers[1].self_attn.w_k_lat.detach().cpu()
    ck["tier_weights"]["layer3.w_q_lat"] = src.layers[3].self_attn.w_q_lat.detach().cpu()
    ck["tier_weights"]["layer3.w_k_lat"] = src.layers[3].self_attn.w_k_lat.detach().cpu()
    path = tmp_path / "init.pt"
    torch.save(ck, path)

    loaded = load_init_ckpt(model, str(path), device="cpu")
    assert loaded == 2
    assert torch.equal(model.layers[1].self_attn.w_q_lat.data, src.layers[1].self_attn.w_q_lat.data)
    assert torch.equal(model.layers[3].self_attn.w_k_lat.data, src.layers[3].self_attn.w_k_lat.data)
    # non-attention layers untouched (they have no self_attn at all)


def test_load_init_ckpt_missing_layer_raises(tmp_path):
    model = _StubModel()
    ck = {
        "tier_weights": {"layer1.w_q_lat": torch.zeros(8, 4), "layer1.w_k_lat": torch.zeros(4, 4)}
    }
    path = tmp_path / "partial.pt"
    torch.save(ck, path)
    try:
        load_init_ckpt(model, str(path), device="cpu")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "layer 3" in str(e)
