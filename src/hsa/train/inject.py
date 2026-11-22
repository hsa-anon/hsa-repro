import logging

import torch

from hsa.torch_tier import three_tier_attention

logger = logging.getLogger(__name__)


class LatentProj(torch.nn.Module):
    # Shared (across heads) key projection d -> d_latent, initialized as the
    # truncation identity so step-0 behavior equals the M1 default latent.
    def __init__(self, d: int, d_latent: int):
        super().__init__()
        w = torch.zeros(d, d_latent)
        n = min(d, d_latent)
        w[:n, :n] = torch.eye(n)
        self.w_k = torch.nn.Parameter(w)


def attach_latent_proj(layer: torch.nn.Module, d_latent: int) -> None:
    d = layer.attn.wk.out_features
    layer.latent_proj = LatentProj(d, d_latent)
    logger.info("attached latent proj d=%d d_latent=%d", d, d_latent)


def three_tier_attention_with(layer: torch.nn.Module, q, k, v, window: int, block_size: int):
    return three_tier_attention(
        q, k, v, window, block_size, layer.latent_proj.w_k.shape[-1], w_k=layer.latent_proj.w_k
    )[0]
