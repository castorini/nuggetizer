from __future__ import annotations

import logging


def setup_logging(log_level: int, *, quiet: bool = False) -> logging.Logger:
    """Configure logging based on verbosity level and return the module logger."""
    if quiet:
        logging_level = logging.CRITICAL
    elif log_level >= 2:
        logging_level = logging.DEBUG
    elif log_level >= 1:
        logging_level = logging.INFO
    else:
        logging_level = logging.WARNING

    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)
