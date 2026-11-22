import json
import logging
import os
import time

import httpx

from hsa.bench.prefill_curve import _stream_answer
from hsa.bench.prompts import build_prompt
from hsa.bench.ruler_baseline import floor_report

logger = logging.getLogger(__name__)


def run(
    base_url: str,
    model: str,
    api_key: str,
    length_tokens: int,
    repeats: int,
    out_path: str,
    seed: int = 42,
) -> dict:
    scores = {}
    client = httpx.Client(timeout=7200.0)
    try:
        for r in range(repeats):
            for depth in (0.25, 0.5, 1.0):
                prompt, expected = build_prompt(length_tokens, seed=seed + r, needle_depth=depth)
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 32,
                    "temperature": 0.0,
                    "stream": True,
                }
                headers = {"Authorization": f"Bearer {api_key}"}
                answer, wall = None, None
                for attempt in range(3):
                    try:
                        answer, wall = _stream_answer(client, base_url, payload, headers)
                        break
                    except httpx.HTTPStatusError as exc:
                        wait = 60.0 * (attempt + 1)
                        logger.warning(
                            "HTTP %s on len=%d rep=%d depth=%.2f, retry %d in %.0fs",
                            exc.response.status_code,
                            length_tokens,
                            r,
                            depth,
                            attempt + 1,
                            wait,
                        )
                        time.sleep(wait)
                if answer is None:
                    logger.error(
                        "len=%d rep=%d depth=%.2f: all retries failed, marking miss",
                        length_tokens,
                        r,
                        depth,
                    )
                    answer, wall = "", 0.0
                key = f"needle_{depth}"
                scores.setdefault(key, []).append(1.0 if expected in answer else 0.0)
                logger.info(
                    "len=%d rep=%d depth=%.2f wall=%.1fs hit=%s",
                    length_tokens,
                    r,
                    depth,
                    wall,
                    expected in answer,
                )
        mean_scores = {k: sum(v) / len(v) for k, v in scores.items()}
    finally:
        client.close()
    result = {
        "length_tokens": length_tokens,
        "repeats": repeats,
        "scores": mean_scores,
        "floor": floor_report(mean_scores),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    logger.info("wrote %s", out_path)
    return result
