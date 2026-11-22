from hsa.bench.logparse import parse_memcheck


def test_parse_memcheck_three_lines(tmp_path):
    log = tmp_path / "serve.log"
    log.write_text(
        "2026-11-08 07:11:41,576 - inference.scheduler.sched_batch - INFO - [-] - "
        "[memcheck:chunk] rid=abc processed=73728/1095638 ctx=1095638 active=52.13GB "
        "peak=55.23GB cache=6.75GB phys=60.83GB\n"
        "2026-11-08 07:12:35,268 - inference.scheduler.sched_batch - INFO - [-] - "
        "[memcheck:chunk] rid=abc processed=75776/1095638 ctx=1095638 active=52.18GB "
        "peak=55.30GB cache=6.80GB phys=60.90GB\n"
        "other log line\n"
        "2026-11-08 07:13:37,126 - inference.scheduler.sched_batch - INFO - [-] - "
        "[memcheck:chunk] rid=abc processed=77824/1095638 ctx=1095638 active=52.20GB "
        "peak=55.40GB cache=6.85GB phys=61.00GB\n"
    )
    rows = parse_memcheck(str(log))
    assert len(rows) == 3
    ts, processed, active, peak, cache, phys = rows[0]
    assert ts == "2026-11-08 07:11:41,576"
    assert processed == 73728
    assert active == 52.13
    assert peak == 55.23
    assert cache == 6.75
    assert phys == 60.83
    assert rows[2][1] == 77824


def test_parse_memcheck_missing_file_returns_empty():
    assert parse_memcheck("/nonexistent/log.log") == []
