import logging
import os
from os.path import join


def setup_logging(log_dir: str, log_level: str = "INFO", name: str = "std.log") -> None:
    """
    Sets up logging to a file in the specified directory.

    Args:
        log_dir: Directory to store log files
        log_level: Logging level (default: INFO)
    """
    if log_dir is None:
        log_dir = "./outputs/logs"
    if log_level is None:
        log_level = "INFO"
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Set up logging configuration
    logging.basicConfig(
        filename=join(log_dir, name),
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
