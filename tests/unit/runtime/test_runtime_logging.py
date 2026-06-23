from __future__ import annotations

import logging
from datetime import datetime

from redd.core.utils.logging_utils import setup_logging
from redd.orchestration.runtime import setup_runtime_logging


def _close_root_handlers() -> None:
    logger = logging.getLogger()
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()


def test_setup_logging_writes_timestamped_log_under_experiment_date(tmp_path) -> None:
    try:
        log_path = setup_logging(
            exp="demo.experiment",
            log_dir=tmp_path,
            console_log_level=logging.CRITICAL,
            timestamp=datetime(2026, 6, 21, 12, 34, 56),
        )
        logging.info("hello from runtime logging")
        for handler in logging.getLogger().handlers:
            handler.flush()

        assert log_path is not None
        assert log_path == (
            tmp_path
            / "runs"
            / "demo.experiment"
            / "2026-06-21"
            / "12-34-56.log"
        )
        assert "hello from runtime logging" in log_path.read_text(encoding="utf-8")
    finally:
        _close_root_handlers()


def test_setup_logging_uses_collision_suffix_for_same_second(tmp_path) -> None:
    try:
        timestamp = datetime(2026, 6, 21, 12, 34, 56)
        first_log_path = setup_logging(
            exp="demo.experiment",
            log_dir=tmp_path,
            console_log_level=logging.CRITICAL,
            timestamp=timestamp,
        )
        second_log_path = setup_logging(
            exp="demo.experiment",
            log_dir=tmp_path,
            console_log_level=logging.CRITICAL,
            timestamp=timestamp,
        )

        assert first_log_path is not None
        assert second_log_path is not None
        assert first_log_path.name == "12-34-56.log"
        assert second_log_path.name == "12-34-56-02.log"
    finally:
        _close_root_handlers()


def test_setup_runtime_logging_returns_actual_log_path(tmp_path) -> None:
    try:
        log_path = setup_runtime_logging(
            {
                "log_dir": tmp_path,
                "console_log_level": "CRITICAL",
            },
            "demo.experiment",
        )

        assert log_path.relative_to(tmp_path).parts[:2] == ("runs", "demo.experiment")
        assert log_path.name.endswith(".log")
        assert log_path.exists()
    finally:
        _close_root_handlers()
