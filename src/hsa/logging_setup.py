import logging


def setup(level: str = "INFO") -> None:
    resolved = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=resolved,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.getLogger().setLevel(resolved)
