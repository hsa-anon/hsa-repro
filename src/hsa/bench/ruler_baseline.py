import logging

logger = logging.getLogger(__name__)

RETRIEVAL_CATEGORIES = ("needle_0.25", "needle_0.5", "needle_1.0")
FLOOR = 0.99


def zone_result(scores: dict) -> dict:
    retrieval = {k: v for k, v in scores.items() if k in RETRIEVAL_CATEGORIES}
    other = {k: v for k, v in scores.items() if k not in RETRIEVAL_CATEGORIES}
    return {"retrieval": retrieval, "report_only": other}


def floor_report(scores: dict) -> dict:
    zoned = zone_result(scores)
    verdict = {}
    for name, score in zoned["retrieval"].items():
        verdict[name] = {"score": score, "floor": FLOOR, "pass": score >= FLOOR}
    overall = all(v["pass"] for v in verdict.values()) if verdict else False
    return {"per_category": verdict, "all_retrieval_pass": overall}
