import logging
import os
from pathlib import Path


def setup_logger(name: str, log_file: str = None, level=logging.INFO):
    """
    Set up a logger with both console and file handlers.
    
    Args:
        name: Logger name (usually __name__)
        log_file: Name of the log file (e.g., 'runner.log')
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    # Path(__file__) is src/logging/logging_config.py
    # .parent is src/logging
    # .parent.parent is src
    # .parent.parent.parent is project_root
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file is specified)
    if log_file:
        log_path = log_dir / log_file
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    return logger
