"""Logging configuration for resume-analyzer.

Configures root logger with console + file handlers.
Call setup_logging() once at application startup.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_dir: Optional[str] = None,
    log_file: str = "app.log",
) -> None:
    """Configure the root logger with console and optional file handlers.

    Args:
        level: Logging level (default INFO).
        log_dir: Directory for log files. Defaults to PROJECT_ROOT/data/logs.
        log_file: Log file name.
    """
    if log_dir is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        log_dir = str(project_root / "data" / "logs")

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    root = logging.getLogger()
    root.setLevel(level)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    # File handler
    file_path = str(Path(log_dir) / log_file)
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Quiet down noisy third-party loggers
    for noisy in ("chromadb", "urllib3", "httpx", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
