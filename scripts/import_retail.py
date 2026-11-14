"""Create a fresh, verified SQLite environment. Existing files are never overwritten."""

import argparse
from pathlib import Path

from sqlalchemy import func, select

from aftersalesagent.business.database import import_pinned_database, orders, products, users

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=ROOT / ".local/databases/retail_template.sqlite"
    )
    args = parser.parse_args()
    try:
        store = import_pinned_database(ROOT, args.output)
    except FileExistsError:
        parser.exit(2, "Database exists; choose a new output path. No file was overwritten.\n")
    with store.transaction() as connection:
        counts = {
            table.name: connection.scalar(select(func.count()).select_from(table))
            for table in (users, products, orders)
        }
    print("Created:", store.path)
    print("Environment ID:", store.path.stem)
    print("Counts:", counts)
    store.close()
