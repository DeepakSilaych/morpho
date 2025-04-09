"""Logging utilities for the morpho tokenizer library."""

import logging
import sys
from typing import Optional


def get_logger(name: str, 
               level: int = logging.INFO, 
               file_path: Optional[str] = None,
               format_string: Optional[str] = None) -> logging.Logger:
    """Get a logger with the specified name and level.
    
    Args:
        name: Logger name
        level: Logging level
        file_path: Optional path to log file
        format_string: Optional format string for log messages
        
    Returns:
        Logger object
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if file_path is provided
    if file_path:
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def set_global_logging_level(level: int = logging.INFO):
    """Set global logging level.
    
    Args:
        level: Logging level
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


class TqdmLoggingHandler(logging.Handler):
    """Logging handler for tqdm progress bars."""
    
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
    
    def emit(self, record):
        """Emit a log record."""
        try:
            msg = self.format(record)
            from tqdm import tqdm
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record) 