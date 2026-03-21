#!/usr/bin/env python3
"""Dump INSERT-friendly SQL from SQLite tables (optional dev backup / seed generation)."""

from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import DATABASE_PATH, PROJECT_ROOT


def dump_with_sqlite3_cli(db_path: Path) -> None:
    """Full SQL dump via sqlite3 CLI (.dump); best fidelity for restores."""
    subprocess.run(
        ["sqlite3", str(db_path), ".dump"],
        check=True,
        stdout=sys.stdout,
    )


def dump_inserts_only(db_path: Path, tables: list[str]) -> None:
    """Emit INSERT statements for named tables only."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        for table in tables:
            cur = conn.execute(f'SELECT * FROM "{table}"')
            rows = cur.fetchall()
            if not rows:
                continue
            cols = [d[0] for d in cur.description]
            placeholders = ",".join("?" * len(cols))
            col_list = ",".join(f'"{c}"' for c in cols)
            print(f"-- {table} ({len(rows)} rows)")
            for row in rows:
                vals = tuple(row[c] for c in cols)
                print(
                    f'INSERT INTO "{table}" ({col_list}) VALUES ('
                    + ",".join(_sql_literal(v) for v in vals)
                    + ");"
                )
    finally:
        conn.close()


def _sql_literal(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, (int, float)):
        return str(value)
    # str / bytes — escape single quotes
    s = str(value)
    return "'" + s.replace("'", "''") + "'"


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump SQLite data as SQL.")
    parser.add_argument(
        "--database",
        type=Path,
        default=DATABASE_PATH,
        help=f"SQLite file (default: {DATABASE_PATH})",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Emit complete .dump (schema + data) via sqlite3 CLI.",
    )
    parser.add_argument(
        "--tables",
        nargs="*",
        default=["teams", "matches", "odds", "team_ratings"],
        help="For default mode: tables to export as INSERTs.",
    )
    args = parser.parse_args()
    db_path = args.database
    if not db_path.is_absolute():
        db_path = (PROJECT_ROOT / db_path).resolve()

    if not db_path.is_file():
        print(f"Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    if args.full:
        dump_with_sqlite3_cli(db_path)
    else:
        dump_inserts_only(db_path, args.tables)


if __name__ == "__main__":
    main()
