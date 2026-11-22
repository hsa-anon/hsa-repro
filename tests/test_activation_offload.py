import torch

from hsa.train.train_distill import activation_offload_hooks


def test_offload_hooks_preserve_grads():
    # Activation offload relocates storage only: gradients must be identical
    # with and without the saved_tensors_hooks (tiny threshold forces packing
    # of every non-leaf saved tensor).
    torch.manual_seed(5)
    l1 = torch.nn.Linear(16, 32)
    l2 = torch.nn.Linear(32, 32)
    l3 = torch.nn.Linear(32, 8)
    x = torch.randn(4, 16)

    def run(with_hooks):
        for p in (*l1.parameters(), *l2.parameters(), *l3.parameters()):
            p.grad = None
        ctx = (
            activation_offload_hooks(threshold_bytes=64) if with_hooks else contextlib.nullcontext()
        )
        with ctx:
            h = torch.tanh(l1(x))
            h = torch.tanh(l2(h))
            out = l3(h)
            out.sum().backward()
        return [p.grad.clone() for p in (*l1.parameters(), *l2.parameters(), *l3.parameters())]

    import contextlib

    g_ref = run(False)
    g_off = run(True)
    for a, b in zip(g_ref, g_off):
        torch.testing.assert_close(a, b)
