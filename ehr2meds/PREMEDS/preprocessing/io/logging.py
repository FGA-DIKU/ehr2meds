import logging
import os
from os.path import join


def setup_logging(log_dir: str, log_level: str = "INFO") -> None:
    """
    Sets up logging to a file in the specified directory.

    Args:
        log_dir: Directory to store log files
        log_level: Logging level (default: INFO)
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Set up logging configuration
    logging.basicConfig(
        filename=join(log_dir, "normalise.log"),
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
