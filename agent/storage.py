"""SQLite storage and deduplication for job listings."""

from pathlib import Path
import sqlite3
from typing import Optional

from agent.models import Listing

DEFAULT_DB_PATH = Path("data/jobs.db")


def get_db_connection(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path | str = DEFAULT_DB_PATH) -> None:
    """Create listings table if it does not exist."""
    with get_db_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS listings (
                source TEXT NOT NULL,
                id TEXT NOT NULL,
                title TEXT NOT NULL,
                company TEXT,
                location TEXT,
                remote INTEGER NOT NULL DEFAULT 0,
                work_type TEXT,
                salary TEXT,
                description TEXT,
                url TEXT,
                posted_at TEXT,
                status TEXT NOT NULL DEFAULT 'new',
                collected_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (source, id)
            )
            """
        )
        conn.commit()


def save_listings(
    listings: list[Listing], db_path: Path | str = DEFAULT_DB_PATH
) -> tuple[int, int]:
    """
    Save listings to SQLite with deduplication on (source, id).
    Returns (inserted_count, duplicate_count).
    """
    init_db(db_path)
    inserted = 0
    duplicates = 0

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        for listing in listings:
            cursor.execute(
                """
                INSERT OR IGNORE INTO listings (
                    source, id, title, company, location, remote,
                    work_type, salary, description, url, posted_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'new')
                """,
                (
                    listing.source,
                    listing.id,
                    listing.title,
                    listing.company,
                    listing.location,
                    1 if listing.remote else 0,
                    listing.work_type,
                    listing.salary,
                    listing.description,
                    listing.url,
                    listing.posted_at,
                ),
            )
            if cursor.rowcount > 0:
                inserted += 1
            else:
                duplicates += 1
        conn.commit()

    return inserted, duplicates


def count_listings(db_path: Path | str = DEFAULT_DB_PATH, status: Optional[str] = None) -> int:
    """Return total count of listings, optionally filtered by status."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        if status:
            cursor.execute("SELECT COUNT(*) FROM listings WHERE status = ?", (status,))
        else:
            cursor.execute("SELECT COUNT(*) FROM listings")
        row = cursor.fetchone()
        return row[0] if row else 0


def fetch_listings(
    db_path: Path | str = DEFAULT_DB_PATH,
    status: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[Listing]:
    """Fetch stored listings from database."""
    init_db(db_path)
    query = "SELECT * FROM listings"
    params: list = []
    if status:
        query += " WHERE status = ?"
        params.append(status)
    query += " ORDER BY collected_at DESC"
    if limit:
        query += " LIMIT ?"
        params.append(limit)

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        return [
            Listing(
                id=row["id"],
                source=row["source"],
                title=row["title"],
                company=row["company"] or "",
                location=row["location"] or "",
                remote=bool(row["remote"]),
                work_type=row["work_type"],
                salary=row["salary"],
                description=row["description"] or "",
                url=row["url"] or "",
                posted_at=row["posted_at"] or "",
            )
            for row in rows
        ]
