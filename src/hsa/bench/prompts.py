import random

HAYSTACK_WORDS = (
    "The sun rises over the quiet valley and the farmers begin another day of work "
    "tending fields that have fed their families for generations. "
)
NEEDLE_TEMPLATE = "One of the special magic numbers for {key_word} is: {value}. "
QUESTION_TEMPLATE = (
    "What is the special magic number for {key_word} mentioned in the provided text?"
)


def build_prompt(target_tokens: int, seed: int, needle_depth: float = 0.6) -> tuple[str, str]:
    rng = random.Random(seed)
    key_word = f"grasveld-{rng.randint(1000, 9999)}"
    value = f"{rng.randint(100000, 999999)}-{rng.randint(100, 999)}"
    needle = NEEDLE_TEMPLATE.format(key_word=key_word, value=value)
    filler_words = HAYSTACK_WORDS.split()
    words = []
    needle_words = needle.split()
    # ~1.3 tokens per English word; inject the needle at needle_depth of the
    # context (1.0 = end of context, right before the question)
    needle_at = min(int(target_tokens / 1.3 * needle_depth), int(target_tokens / 1.3) - 8)
    while len(words) < target_tokens // 1.3:
        if abs(len(words) - needle_at) < len(needle_words) // 2:
            words.extend(needle_words)
        else:
            words.append(rng.choice(filler_words))
    context = " ".join(words)
    prompt = context + "\n\n" + QUESTION_TEMPLATE.format(key_word=key_word)
    return prompt, value
