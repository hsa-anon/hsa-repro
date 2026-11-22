import torch

from hsa.eval.ruler_tier import chunked_generate


def test_chunked_generate_matches_generate():
    # Chunked prefill must reproduce direct generate token-for-token under
    # greedy decoding; this pins position_ids and past_key_values handling.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)
    model.eval()
    text = "The sun rises in the east and sets in the west. " * 40
    ids = tok(text, return_tensors="pt").input_ids
    assert ids.shape[1] > 400
    direct = model.generate(ids, max_new_tokens=24, do_sample=False, pad_token_id=tok.eos_token_id)
    chunked = chunked_generate(model, ids, 24, 200, tok.eos_token_id)
    assert torch.equal(direct, chunked), (
        f"mismatch: direct={direct[0, -8:].tolist()} chunked={chunked[0, -8:].tolist()}"
    )


def test_chunked_generate_chunk_larger_than_prompt():
    # One-shot chunk (chunk >= T) must also match direct generate.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)
    model.eval()
    text = "Grass is green in the summer. " * 30
    ids = tok(text, return_tensors="pt").input_ids
    direct = model.generate(ids, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)
    chunked = chunked_generate(model, ids, 8, ids.shape[1] + 10, tok.eos_token_id)
    assert torch.equal(direct, chunked)
