"""Application entry point."""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.logging_config import setup_logging
from src.ui.app import main

setup_logging()
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting resume-analyzer")
    main()
