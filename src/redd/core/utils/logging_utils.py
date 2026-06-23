import logging
import re
from datetime import datetime
from pathlib import Path

_LOG_COMPONENT_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


def get_log_level(level_str):
    """
    Convert string log level to logging level constant.

    Args:
        level_str (str): String representation of log level

    Returns:
        int: Logging level constant
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    return level_map.get(level_str.upper(), logging.WARNING)


class NoHTTPRequestFilter(logging.Filter):
    def filter(self, record):
        return "HTTP Request" not in record.getMessage()


def _safe_log_component(value):
    slug = _LOG_COMPONENT_PATTERN.sub("-", str(value).strip()).strip(".-")
    return slug or "run"


def _create_log_path(log_dir, exp, timestamp):
    base_dir = Path(log_dir) / "runs" / _safe_log_component(exp) / timestamp.strftime("%Y-%m-%d")
    time_slug = timestamp.strftime("%H-%M-%S")
    for index in range(1, 1000):
        suffix = "" if index == 1 else f"-{index:02d}"
        log_path = base_dir / f"{time_slug}{suffix}.log"
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
            log_path.touch(exist_ok=False)
            return log_path
        except FileExistsError:
            continue
    raise RuntimeError(f"Unable to create a unique log file under {base_dir}")


def setup_logging(exp=None, log_dir="outputs/logs", console_log_level=logging.WARNING, timestamp=None):
    """
    Set up logging configuration with different levels for console and file outputs.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler with configurable level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_log_level)  # Configurable console log level
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler for this run.
    if exp:
        timestamp = timestamp or datetime.now()
        log_filepath = _create_log_path(log_dir, exp, timestamp)
        file_handler = logging.FileHandler(log_filepath, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.INFO)  # Log everything to the file
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logging.info(f"Logs will be saved to: {log_filepath}")
    else:
        log_filepath = None
        logging.info("File logging is disabled")

    # for handler in logger.handlers:
    #     handler.addFilter(NoHTTPRequestFilter())

    # Test logging
    logging.info("Logging initialized")
    return log_filepath


if __name__ == "__main__":
    log_file = setup_logging()
    logging.info("This is an info message.")
    logging.debug("This is a debug message.")
    logging.warning("This is a warning message.")
    logging.error("This is an error message.")
    logging.critical("This is a critical message.")
