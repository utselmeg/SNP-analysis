
import logging

logger = logging.getLogger()

def setup_logger(run_path):
    log_format = "%(asctime)s - %(levelname)s - %(message)s"

    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(run_path / "output.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
