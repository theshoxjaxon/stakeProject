#!/usr/bin/env python3
"""Development database seed — runs ``seed_dev.py`` (same behavior)."""

from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "seed_dev.py"), run_name="__main__")
