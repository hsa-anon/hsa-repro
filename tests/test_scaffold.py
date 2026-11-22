import logging


def test_setup_configures_root_logger():
    from hsa.logging_setup import setup

    setup("INFO")
    assert logging.getLogger().level == logging.INFO
