import logging
import os
import sys
from logging.handlers import RotatingFileHandler

def setup_logger(name="TokiSTT", log_file=None, level=logging.DEBUG):
    """
    Sets up a centralized logger with console and file handlers.
    """
    if log_file is None:
        # Anchor to the project root so the app works from any working directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_file = os.path.join(base_dir, "debug.log")

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Console Handler: keep the console readable (INFO and up)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler: keep full detail (DEBUG and up) for post-mortem analysis
    try:
        file_handler = RotatingFileHandler(
            log_file, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Failed to setup file logging: {e}")

    return logger

# Global logger instance
logger = setup_logger()
