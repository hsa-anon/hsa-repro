import logging
import time

import httpx

from hsa.bench.logparse import parse_memcheck
from hsa.bench.prompts import build_prompt

logger = logging.getLogger(__name__)


def compute_windows(rows) -> list[dict]:
    windows = []
    for i in range(1, len(rows)):
        _, p0, _, _, _, _phys0 = rows[i - 1]
        _, p1, a1, pk1, c1, ph1 = rows[i]
        if p1 > p0:
            windows.append(
                {
                    "from": p0,
                    "to": p1,
                    "tokens": p1 - p0,
                    "phys_gb_end": ph1,
                    "active_gb_end": a1,
                    "peak_gb_end": pk1,
                    "cache_gb_end": c1,
                }
            )
    return windows


def _stream_answer(client, base_url, payload, headers) -> tuple[str, float]:
    start = time.monotonic()
    parts = []
    with client.stream(
        "POST", f"{base_url}/chat/completions", json=payload, headers=headers
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if data == "[DONE]":
                break
            piece = ""
            try:
                import json

                piece = (
                    json.loads(data).get("choices", [{}])[0].get("delta", {}).get("content") or ""
                )
            except (AttributeError, IndexError, KeyError, TypeError, ValueError):
                piece = ""
            if piece:
                parts.append(piece)
    return "".join(parts), time.monotonic() - start


def run_length(
    base_url: str,
    model: str,
    api_key: str,
    length_tokens: int,
    serve_log: str,
    seed: int = 42,
    max_tokens: int = 8,
    client: httpx.Client | None = None,
) -> dict:
    prompt, expected = build_prompt(length_tokens, seed=seed, needle_depth=1.0)
    rows_before = len(parse_memcheck(serve_log))
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    own_client = client is None
    client = client or httpx.Client(timeout=7200.0)
    try:
        answer, wall = _stream_answer(client, base_url, payload, headers)
    finally:
        if own_client:
            client.close()
    rows_after = parse_memcheck(serve_log)
    new_rows = rows_after[rows_before:]
    windows = compute_windows(new_rows)
    hit = expected in answer
    result = {
        "length_tokens": length_tokens,
        "wall_s": wall,
        "end_to_end": True,
        "needle_hit": hit,
        "windows": windows,
        "max_phys_gb": max((r[5] for r in new_rows), default=0.0),
        "max_active_gb": max((r[2] for r in new_rows), default=0.0),
    }
    logger.info(
        "run_length len=%d wall=%.1fs hit=%s windows=%d max_phys=%.2fGB",
        length_tokens,
        wall,
        hit,
        len(windows),
        result["max_phys_gb"],
    )
    return result
