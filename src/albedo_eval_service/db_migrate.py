from __future__ import annotations

import argparse
from pathlib import Path

import psycopg

from .config import get_settings


def apply_migrations(*, database_url: str, migrations_dir: Path) -> list[str]:
    applied: list[str] = []
    with psycopg.connect(database_url) as conn:
        with conn.transaction():
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )

        for migration_path in sorted(migrations_dir.glob("*.sql")):
            version = migration_path.name
            already_applied = conn.execute(
                "SELECT 1 FROM schema_migrations WHERE version = %s",
                (version,),
            ).fetchone()
            if already_applied:
                continue

            sql = migration_path.read_text(encoding="utf-8")
            with conn.transaction():
                conn.execute(sql)
                conn.execute(
                    "INSERT INTO schema_migrations (version) VALUES (%s)",
                    (version,),
                )
            applied.append(version)
    return applied


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply Albedo eval service Postgres migrations.")
    parser.add_argument(
        "--migrations-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "migrations",
        help="Directory containing SQL migration files.",
    )
    args = parser.parse_args()

    settings = get_settings()
    applied = apply_migrations(database_url=settings.database_url, migrations_dir=args.migrations_dir)
    if applied:
        print("applied_migrations=" + ",".join(applied))
    else:
        print("applied_migrations=")
