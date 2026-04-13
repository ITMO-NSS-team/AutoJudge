"""Shared setup for example scripts.

Import this module at the top of any example to configure the Python path
and load environment variables:

    from common import setup
    setup()
"""

import sys
from pathlib import Path


def setup():
    """Add the src directory to sys.path and load .env."""
    # Walk up from the examples/ directory to find the project root (contains src/)
    current = Path(__file__).resolve().parent
    while current != current.parent:
        src_dir = current / "src"
        if src_dir.is_dir():
            if str(src_dir) not in sys.path:
                sys.path.insert(0, str(src_dir))
            break
        current = current.parent

    from dotenv import load_dotenv

    load_dotenv(".env")
