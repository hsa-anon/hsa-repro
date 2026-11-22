import logging
import os
import re

logger = logging.getLogger(__name__)

_MEMCHECK_RE = re.compile(
    r"^(\S+ \S+) - \S+ - \S+ - \[-\] - \[memcheck:chunk\] rid=\S+ "
    r"processed=(\d+)/\d+ ctx=\d+ active=([\d.]+)GB peak=([\d.]+)GB "
    r"cache=([\d.]+)GB phys=([\d.]+)GB"
)


def parse_memcheck(path: str) -> list[tuple[str, int, float, float, float, float]]:
    if not os.path.exists(path):
        logger.warning("memcheck source missing: %s", path)
        return []
    rows = []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            match = _MEMCHECK_RE.match(line.strip())
            if not match:
                continue
            ts, processed, active, peak, cache, phys = match.groups()
            rows.append((ts, int(processed), float(active), float(peak), float(cache), float(phys)))
    logger.info("parsed %d memcheck rows from %s", len(rows), path)
    return rows
