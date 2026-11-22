import logging

import torch

logger = logging.getLogger(__name__)


def distill_step(student_forward, teacher_forward, trainable, optimizer) -> dict:
    """One distillation step: student (three-tier) vs frozen dense teacher.

    student_forward/teacher_forward return (final_hidden, attn_outs list).
    Loss = sum of per-checkpoint MSEs (attention-layer outputs) + final
    hidden MSE. Only `trainable` params carry gradients.
    """
    t_out, t_attn = teacher_forward()
    s_out, s_attn = student_forward()
    loss = torch.zeros((), dtype=s_out.dtype)
    for a, b in zip(s_attn, t_attn):
        loss = loss + torch.nn.functional.mse_loss(a, b)
    loss = loss + torch.nn.functional.mse_loss(s_out, t_out)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    logger.debug("distill step loss=%.6f", float(loss))
    return {"loss": float(loss)}


def distill_loop(student_forward, teacher_forward, trainable, steps: int, lr: float) -> list:
    optimizer = torch.optim.Adam(trainable, lr=lr)
    history = []
    for _ in range(steps):
        result = distill_step(student_forward, teacher_forward, trainable, optimizer)
        history.append(result["loss"])
    logger.info("distill loop steps=%d loss %.6f -> %.6f", steps, history[0], history[-1])
    return history
