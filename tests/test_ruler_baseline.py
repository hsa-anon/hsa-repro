from hsa.bench.ruler_baseline import floor_report, zone_result


def test_zone_result_splits_retrieval_from_report_only():
    zoned = zone_result({"needle_1.0": 1.0, "squad_f1": 0.62})
    assert zoned["retrieval"] == {"needle_1.0": 1.0}
    assert zoned["report_only"] == {"squad_f1": 0.62}


def test_floor_report_flags_failing_category():
    report = floor_report({"needle_0.25": 1.0, "needle_1.0": 0.98})
    assert report["per_category"]["needle_1.0"]["pass"] is False
    assert report["all_retrieval_pass"] is False


def test_floor_report_all_pass():
    report = floor_report({"needle_0.25": 1.0, "needle_0.5": 1.0, "needle_1.0": 1.0})
    assert report["all_retrieval_pass"] is True
