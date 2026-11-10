from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger

# Global state
_session_dir: Optional[Path] = None
_initialized = False


def setup_logging(
    base_log_dir: Path = Path("autojudge_logs"),
    session_name: Optional[str] = None,
    log_level: str = "INFO",
    enable_console: bool = True,
    enable_file: bool = True,
    force_reinit: bool = False,
) -> None:
    global _session_dir, _initialized

    if _initialized and not force_reinit:
        return

    # Create session directory
    if session_name is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        session_name = f"session_{timestamp}"

    _session_dir = base_log_dir / session_name
    _session_dir.mkdir(parents=True, exist_ok=True)

    # Configure loguru
    logger.remove()  # Remove default handler

    # Configure custom colors for log levels
    logger.level("INFO", color="<fg 92,120,226>")  # #5c78e2
    logger.level("DEBUG", color="<fg 102,159,89>")  # #669f59

    if enable_console:
        logger.add(
            sys.stderr,
            format="<fg 141,182,212>[{time:MM/DD/YY HH:mm:ss}]</fg 141,182,212> <level>{level: <8}</level> <cyan>{name}:{line}</cyan> {message}",
            level=log_level,
            colorize=True,
        )

    if enable_file:
        logger.add(
            _session_dir / "autojudge.log",
            format="[{time:MM/DD/YY HH:mm:ss}] {level: <8} {name}:{line} {message}",
            level=log_level,
            rotation="10 MB",
            retention=5,
            compression="zip",
            encoding="utf-8",
        )

    _initialized = True


def reset_logging() -> None:
    """Reset logging state to allow reconfiguration with new session directory."""
    global _session_dir, _initialized

    logger.remove()  # Remove all handlers
    _session_dir = None
    _initialized = False


# We can't make output type to be Logger because Loguru won't let us import it
def get_logger(name: Optional[str] = None) -> Any:
    """Get loguru logger. Name parameter kept for API compatibility."""
    if not _initialized:
        setup_logging()
    return logger


def get_session_dir() -> Path:
    """Get current session directory."""
    if _session_dir is None:
        setup_logging()
    assert _session_dir is not None
    return _session_dir


def get_artifact_path(filename: str) -> Path:
    """Get path for saving artifacts in session directory."""
    return get_session_dir() / filename


def save_json(data: Any, filename: str) -> Path:
    """Save data as JSON in session directory."""
    if not filename.endswith(".json"):
        filename += ".json"

    filepath = get_artifact_path(filename)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Saved JSON to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save JSON to {filepath}: {e}")
        raise

    return filepath


def save_text(content: str, filename: str) -> Path:
    """Save text content in session directory."""
    filepath = get_artifact_path(filename)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Saved text to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save text to {filepath}: {e}")
        raise

    return filepath
