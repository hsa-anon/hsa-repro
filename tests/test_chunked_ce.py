import torch

from hsa.train.train_distill import chunked_ce_grad


def test_chunked_ce_grad_matches_full():
    # Chunked lm_head CE must match the full-sequence CE value and produce
    # the same gradient w.r.t. the hidden states - the 32k stage cannot
    # afford materializing full-sequence fp32 logits. fp32 throughout, the
    # precision the training loop actually uses.
    torch.manual_seed(3)
    B, T, H, V = 1, 12, 8, 50
    lm_head = torch.nn.Linear(H, V, bias=False)
    hidden = torch.randn(B, T, H)
    labels = torch.randint(0, V, (B, T))

    # full reference
    h_ref = hidden.clone().requires_grad_(True)
    logits = lm_head(h_ref).float()
    ce_full = torch.nn.functional.cross_entropy(logits.reshape(-1, V), labels.reshape(-1))
    ce_full.backward()
    ref_grad = h_ref.grad.clone()

    # chunked (leaf + per-chunk backward)
    h_chunk = hidden.clone().requires_grad_(True)
    ce_sum_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    ce_val, grad = chunked_ce_grad(lm_head, h_chunk, labels, chunk=5, ce_sum_fn=ce_sum_fn)

    torch.testing.assert_close(torch.tensor(ce_val), ce_full.detach())
    assert grad is not None
    torch.testing.assert_close(grad, ref_grad)
