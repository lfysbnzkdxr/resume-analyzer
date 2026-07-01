"""Application entry point — CLI, healthcheck, and Streamlit launcher."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import APP_VERSION
from src.core.logging_config import setup_logging
from src.ui.app import main

setup_logging()
logger = logging.getLogger(__name__)


def cli() -> int:
    """Parse CLI arguments and dispatch."""
    parser = argparse.ArgumentParser(description="AI 简历智能分析")
    parser.add_argument(
        "--version",
        action="store_true",
        help="打印版本号并退出",
    )
    parser.add_argument(
        "--healthcheck",
        action="store_true",
        help="运行健康检查并退出",
    )
    args, _ = parser.parse_known_args()

    if args.version:
        print(APP_VERSION)
        return 0

    if args.healthcheck:
        from src.core.healthcheck import main as healthcheck_main

        return healthcheck_main()

    logger.info("Starting resume-analyzer")
    main()
    return 0


if __name__ == "__main__":
    sys.exit(cli())
