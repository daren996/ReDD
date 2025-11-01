import os
import logging
from datetime import datetime


class NoHTTPRequestFilter(logging.Filter):
    def filter(self, record):
        return "HTTP Request" not in record.getMessage()
    

def setup_logging(exp=None, log_dir="logs"):
    """
    Set up logging configuration with different levels for console and file outputs.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler with level WARNING and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only log WARNING and above to the console
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler with all levels (DEBUG and above)
    if exp:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_filename = f"{exp}-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
        log_filepath = os.path.join(log_dir, log_filename)
        file_handler = logging.FileHandler(log_filepath, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.INFO)  # Log everything to the file
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logging.info(f"Logs will be saved to: {log_filepath}")
    else:
        logging.info("File logging is disabled")

    # for handler in logger.handlers:
    #     handler.addFilter(NoHTTPRequestFilter())

    # Test logging
    logging.info("Logging initialized")


if __name__ == "__main__":
    log_file = setup_logging()
    logging.info("This is an info message.")
    logging.debug("This is a debug message.")
    logging.warning("This is a warning message.")
    logging.error("This is an error message.")
    logging.critical("This is a critical message.")
