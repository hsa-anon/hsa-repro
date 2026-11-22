from hsa.bench.prompts import build_prompt


def test_build_prompt_returns_needle_and_value():
    prompt, value = build_prompt(4096, seed=42, needle_depth=1.0)
    assert value in prompt
    assert "special magic number" in prompt
    assert prompt.count("grasveld-") >= 1
