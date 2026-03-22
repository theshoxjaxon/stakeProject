"""Central logging: console + rotating file under ``data/logs``."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import Any

_configured = False


def setup_logging(
    level: str | None = None,
    log_dir: Any = None,
    log_file: str | None = None,
    max_bytes: int | None = None,
    backup_count: int | None = None,
) -> None:
    """Configure root logger once (console + optional rotating file)."""
    global _configured
    if _configured:
        return

    from src.config import LOG_BACKUP_COUNT, LOG_DIR, LOG_FILE, LOG_LEVEL, LOG_MAX_BYTES

    lvl = (level or LOG_LEVEL).upper()
    numeric = getattr(logging, lvl, logging.INFO)

    root = logging.getLogger()
    root.setLevel(numeric)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(numeric)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    ld = log_dir or LOG_DIR
    lf = log_file or LOG_FILE
    mb = max_bytes if max_bytes is not None else LOG_MAX_BYTES
    bc = backup_count if backup_count is not None else LOG_BACKUP_COUNT

    try:
        ld.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            ld / lf,
            maxBytes=mb,
            backupCount=bc,
            encoding="utf-8",
        )
        fh.setLevel(numeric)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError:
        logging.getLogger(__name__).warning(
            "Could not create log file under %s; using console only", ld
        )

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger (calls setup on first use)."""
    setup_logging()
    return logging.getLogger(name)
