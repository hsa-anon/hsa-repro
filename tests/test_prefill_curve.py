from hsa.bench.prefill_curve import compute_windows


def test_compute_windows_skips_reset_and_carries_memory():
    rows = [
        ("t0", 2048, 50.0, 55.0, 1.0, 50.5),
        ("t1", 4096, 51.0, 56.0, 1.5, 55.0),
        ("t2", 4096, 51.0, 56.0, 0.0, 53.0),
        ("t3", 6144, 52.0, 57.0, 2.0, 58.0),
    ]
    windows = compute_windows(rows)
    assert len(windows) == 2
    assert windows[0]["tokens"] == 2048
    assert windows[0]["phys_gb_end"] == 55.0
    assert windows[1]["tokens"] == 2048
    assert windows[1]["phys_gb_end"] == 58.0
