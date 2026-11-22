import argparse
import json
import logging

from hsa.bench.runner import run
from hsa.logging_setup import setup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    setup("INFO")
    result = run(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        length_tokens=args.length,
        repeats=args.repeats,
        out_path=args.out,
        seed=args.seed,
    )
    logging.getLogger(__name__).info("floor verdict: %s", json.dumps(result["floor"]))


if __name__ == "__main__":
    main()
