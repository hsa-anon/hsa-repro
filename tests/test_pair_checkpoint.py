import torch

from hsa.train.train_distill import pair_checkpointed_forward


class _TinyCfg:
    def __init__(self, n):
        self.layer_types = ["linear_attention" if i % 4 else "full_attention" for i in range(n)]
        self.num_hidden_layers = n
        self._attn_implementation = "sdpa"


class _TinyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(8, 8)

    def forward(
        self, h, position_embeddings, attention_mask, position_ids, past_key_values, use_cache
    ):
        return self.lin(torch.tanh(h)) + position_embeddings


class _TinyModel:
    # Mimics the Qwen3_5 text stack surface pair_checkpointed_forward uses.
    def __init__(self, n=8):
        self.config = _TinyCfg(n)
        self.embed_tokens = torch.nn.Embedding(20, 8)
        self.layers = torch.nn.ModuleList(_TinyLayer() for _ in range(n))
        self.norm = torch.nn.RMSNorm(8)

    def rotary_emb(self, h, position_ids):
        return h.detach() * 0.01

    def train(self, mode=True):
        return self

    def parameters(self):
        yield from self.embed_tokens.parameters()
        for layer in self.layers:
            yield from layer.parameters()
        yield from self.norm.parameters()


class _Wrapper:
    def __init__(self):
        self.model = _TinyModel()

    def train(self, mode=True):
        self.model.train(mode)
        return self

    def eval(self):
        return self.train(False)


def _plain_forward(model, ids):
    m = model.model
    h = m.embed_tokens(ids)
    pe = h.detach() * 0.01
    for layer in m.layers:
        h = layer(
            h,
            position_embeddings=pe,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            use_cache=False,
        )
    return m.norm(h)


def test_pair_checkpoint_forward_matches_plain():
    torch.manual_seed(7)
    model = _Wrapper()
    model.eval()
    ids = torch.randint(0, 20, (1, 12))
    ref = _plain_forward(model, ids)
    out = pair_checkpointed_forward(model, ids)
    torch.testing.assert_close(out, ref)


def test_pair_checkpoint_grads_flow():
    torch.manual_seed(8)
    model = _Wrapper()
    model.train()
    ids = torch.randint(0, 20, (1, 12))
    out = pair_checkpointed_forward(model, ids)
    out.sum().backward()
    grads = [p.grad for p in model.model.parameters() if p.grad is not None]
    assert grads, "no gradients reached backbone parameters"
    assert all(torch.isfinite(g).all() for g in grads)
