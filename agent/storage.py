"""SQLite storage and deduplication for job listings."""

import json
from pathlib import Path
import sqlite3
from typing import Any, Optional

from agent.models import Listing

DEFAULT_DB_PATH = Path("data/jobs.db")


def get_db_connection(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path | str = DEFAULT_DB_PATH) -> None:
    """Create listings table and ensure all required columns exist."""
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
                rejection_reason TEXT,
                score REAL,
                skills_match REAL,
                experience_fit REAL,
                role_fit REAL,
                gaps TEXT,
                reason TEXT,
                model TEXT,
                prompt_version TEXT,
                scored_at TEXT,
                collected_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (source, id)
            )
            """
        )
        # Migrate existing tables to ensure all new columns exist
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(listings)")
        existing_cols = {row["name"] for row in cursor.fetchall()}

        new_columns = {
            "rejection_reason": "TEXT",
            "score": "REAL",
            "skills_match": "REAL",
            "experience_fit": "REAL",
            "role_fit": "REAL",
            "gaps": "TEXT",
            "reason": "TEXT",
            "model": "TEXT",
            "prompt_version": "TEXT",
            "scored_at": "TEXT",
        }
        for col_name, col_type in new_columns.items():
            if col_name not in existing_cols:
                cursor.execute(f"ALTER TABLE listings ADD COLUMN {col_name} {col_type}")
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


def update_listing_status(
    source: str,
    id: str,
    status: str,
    rejection_reason: Optional[str] = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> None:
    """Update status and rejection_reason of a listing."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        conn.execute(
            """
            UPDATE listings
            SET status = ?, rejection_reason = ?
            WHERE source = ? AND id = ?
            """,
            (status, rejection_reason, source, id),
        )
        conn.commit()


def save_score(
    source: str,
    id: str,
    score_data: dict[str, Any],
    db_path: Path | str = DEFAULT_DB_PATH,
) -> None:
    """Store LLM score results and mark listing as scored."""
    init_db(db_path)
    gaps_raw = score_data.get("gaps", [])
    gaps_str = json.dumps(gaps_raw) if isinstance(gaps_raw, list) else str(gaps_raw)

    with get_db_connection(db_path) as conn:
        conn.execute(
            """
            UPDATE listings
            SET status = 'scored',
                score = ?,
                skills_match = ?,
                experience_fit = ?,
                role_fit = ?,
                gaps = ?,
                reason = ?,
                model = ?,
                prompt_version = ?,
                scored_at = CURRENT_TIMESTAMP
            WHERE source = ? AND id = ?
            """,
            (
                float(score_data["score"]),
                float(score_data.get("skills_match", 0.0)),
                float(score_data.get("experience_fit", 0.0)),
                float(score_data.get("role_fit", 0.0)),
                gaps_str,
                str(score_data.get("reason", "")),
                str(score_data.get("model", "")),
                str(score_data.get("prompt_version", "")),
                source,
                id,
            ),
        )
        conn.commit()


def save_scoring_error(
    source: str,
    id: str,
    error_message: str,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> None:
    """Record scoring failure without crashing."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        conn.execute(
            """
            UPDATE listings
            SET rejection_reason = ?
            WHERE source = ? AND id = ?
            """,
            (f"scoring_error: {error_message}", source, id),
        )
        conn.commit()


def count_listings(
    db_path: Path | str = DEFAULT_DB_PATH, status: Optional[str] = None
) -> int:
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


def fetch_passed_unscored_listings(
    prompt_version: str,
    limit: Optional[int] = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> list[Listing]:
    """Fetch listings that passed filtering and need scoring for this prompt version."""
    init_db(db_path)
    query = """
        SELECT * FROM listings
        WHERE status = 'passed'
          AND (score IS NULL OR prompt_version IS NULL OR prompt_version != ?)
        ORDER BY collected_at DESC
    """
    params: list[Any] = [prompt_version]
    if limit is not None:
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


def fetch_top_scored_listings(
    limit: int = 10,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Fetch top scored listings ordered by score descending."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT source, id, title, company, location, remote, work_type,
                   salary, url, score, skills_match, experience_fit, role_fit,
                   gaps, reason, model, prompt_version, scored_at
            FROM listings
            WHERE status = 'scored' AND score IS NOT NULL
            ORDER BY score DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        results = []
        for r in rows:
            gaps_val = r["gaps"]
            if gaps_val:
                try:
                    gaps_list = json.loads(gaps_val)
                except Exception:
                    gaps_list = [gaps_val]
            else:
                gaps_list = []

            results.append({
                "source": r["source"],
                "id": r["id"],
                "title": r["title"],
                "company": r["company"] or "",
                "location": r["location"] or "",
                "remote": bool(r["remote"]),
                "work_type": r["work_type"] or "",
                "salary": r["salary"] or "",
                "url": r["url"] or "",
                "score": r["score"],
                "skills_match": r["skills_match"],
                "experience_fit": r["experience_fit"],
                "role_fit": r["role_fit"],
                "gaps": gaps_list,
                "reason": r["reason"] or "",
                "model": r["model"] or "",
                "prompt_version": r["prompt_version"] or "",
                "scored_at": r["scored_at"] or "",
            })
        return results


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
