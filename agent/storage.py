"""SQLite storage and deduplication for job listings."""

import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sqlite3
from typing import Any, Optional

from agent.models import Listing

DEFAULT_DB_PATH = Path("data/jobs.db")
VALID_PIPELINE_STATUSES = {"new", "shortlisted", "applied", "replied", "rejected"}


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
                scoring_error_text TEXT,
                scoring_error_type TEXT,
                scoring_error_at TEXT,
                scoring_attempts INTEGER NOT NULL DEFAULT 0,
                pipeline_status TEXT NOT NULL DEFAULT 'new',
                status_updated_at TEXT,
                notes TEXT,
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
            "scoring_error_text": "TEXT",
            "scoring_error_type": "TEXT",
            "scoring_error_at": "TEXT",
            "scoring_attempts": "INTEGER NOT NULL DEFAULT 0",
            "pipeline_status": "TEXT NOT NULL DEFAULT 'new'",
            "status_updated_at": "TEXT",
            "notes": "TEXT",
            "duplicate_of": "TEXT",
            "similarity_score": "REAL",
        }
        for col_name, col_type in new_columns.items():
            if col_name not in existing_cols:
                cursor.execute(f"ALTER TABLE listings ADD COLUMN {col_name} {col_type}")

        conn.execute("CREATE INDEX IF NOT EXISTS idx_listings_duplicate_of ON listings(duplicate_of)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_listings_status ON listings(status)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS collector_runs (
                source TEXT PRIMARY KEY,
                last_run_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def migrate_tracking_columns(db_path: Path | str = DEFAULT_DB_PATH) -> int:
    """
    Migration: ensure pipeline_status, status_updated_at, notes columns exist,
    back up database first, and set pipeline_status = 'new' for existing rows.
    """
    path = Path(db_path)
    if not path.exists():
        return 0

    bak_path = path.with_suffix(".db.bak")
    shutil.copy2(path, bak_path)
    print(f"[Migration] Backed up database to {bak_path}")

    with get_db_connection(path) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(listings)")
        existing_cols = {row["name"] for row in cursor.fetchall()}
        has_tracking = "pipeline_status" in existing_cols

        cursor.execute("SELECT count(*) FROM listings")
        total_rows = cursor.fetchone()[0]

    init_db(path)

    with get_db_connection(path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM listings WHERE pipeline_status IS NULL")
        null_count = cursor.fetchone()[0]
        if null_count > 0:
            cursor.execute("UPDATE listings SET pipeline_status = 'new' WHERE pipeline_status IS NULL")
            conn.commit()

        migrated = total_rows if not has_tracking else null_count
        if migrated > 0:
            print(f"[Migration] Migrated tracking columns on {migrated} row(s).")
        return migrated


def migrate_bad_scored_listings(db_path: Path | str = DEFAULT_DB_PATH) -> int:
    """
    Reset to status 'passed' every listing that has status 'scored'
    but an empty score or a non-empty scoring_error.
    """
    path = Path(db_path)
    if not path.exists():
        return 0
    init_db(path)

    with get_db_connection(path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT count(*) FROM listings
            WHERE status = 'scored'
              AND (
                  score IS NULL
                  OR (rejection_reason IS NOT NULL AND rejection_reason LIKE '%scoring_error%')
                  OR (scoring_error_text IS NOT NULL AND scoring_error_text != '')
              )
            """
        )
        count = cursor.fetchone()[0]

        if count > 0:
            bak_path = path.with_suffix(".db.bak")
            shutil.copy2(path, bak_path)

            cursor.execute(
                """
                UPDATE listings
                SET status = 'passed',
                    score = NULL,
                    skills_match = NULL,
                    experience_fit = NULL,
                    role_fit = NULL,
                    gaps = NULL,
                    reason = NULL,
                    scored_at = NULL,
                    scoring_attempts = CASE WHEN scoring_attempts > 0 THEN scoring_attempts ELSE 1 END,
                    scoring_error_text = COALESCE(scoring_error_text, rejection_reason),
                    scoring_error_type = COALESCE(scoring_error_type, 'transient'),
                    scoring_error_at = CURRENT_TIMESTAMP,
                    rejection_reason = NULL
                WHERE status = 'scored'
                  AND (
                      score IS NULL
                      OR (rejection_reason IS NOT NULL AND rejection_reason LIKE '%scoring_error%')
                      OR (scoring_error_text IS NOT NULL AND scoring_error_text != '')
                  )
                """
            )
            conn.commit()
            print(f"[Migration] Backed up database to {bak_path}")
            print(f"[Migration] Reset {count} row(s) with scoring errors from 'scored' to 'passed'.")

        return count


def get_last_collector_run(
    source: str, db_path: Path | str = DEFAULT_DB_PATH
) -> Optional[str]:
    """Return ISO-8601 UTC timestamp of the last recorded run for a collector, or None."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT last_run_at FROM collector_runs WHERE source = ?",
            (source,),
        )
        row = cursor.fetchone()
        return row["last_run_at"] if row else None


def record_collector_run(
    source: str,
    db_path: Path | str = DEFAULT_DB_PATH,
    run_at: Optional[str] = None,
) -> None:
    """Record the latest collector run timestamp (UTC ISO-8601)."""
    init_db(db_path)
    ts = run_at or datetime.now(timezone.utc).isoformat()
    with get_db_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO collector_runs (source, last_run_at)
            VALUES (?, ?)
            ON CONFLICT(source) DO UPDATE SET last_run_at = excluded.last_run_at
            """,
            (source, ts),
        )
        conn.commit()


def normalize_url(url: Optional[str]) -> str:
    """Normalize a listing URL for deduplication."""
    if not url:
        return ""
    return url.strip().rstrip("/").lower()


def normalize_title_company(title: Optional[str], company: Optional[str]) -> tuple[str, str]:
    """Normalize title and company pair for deduplication."""
    norm_title = " ".join((title or "").strip().lower().split())
    norm_company = " ".join((company or "").strip().lower().split())
    return norm_title, norm_company


def fetch_dedup_index(db_path: Path | str = DEFAULT_DB_PATH) -> dict[str, Any]:
    """Return existing (source, id) keys, normalized URLs, and (title, company) -> sources map."""
    init_db(db_path)
    keys: set[tuple[str, str]] = set()
    urls: set[str] = set()
    title_company_sources: dict[tuple[str, str], set[str]] = {}

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT source, id, url, title, company FROM listings")
        for row in cursor.fetchall():
            src = row["source"]
            keys.add((src, row["id"]))
            norm_u = normalize_url(row["url"])
            if norm_u:
                urls.add(norm_u)
            tc = normalize_title_company(row["title"], row["company"])
            if tc[0] and tc[1]:
                title_company_sources.setdefault(tc, set()).add(src)

    return {
        "keys": keys,
        "urls": urls,
        "title_company_sources": title_company_sources,
    }


def save_listings(
    listings: list[Listing], db_path: Path | str = DEFAULT_DB_PATH
) -> tuple[int, int]:
    """
    Save listings to SQLite with deduplication on (source, id), non-empty URL,
    and normalized (title, company) across sources (and within company:* boards).
    Returns (inserted_count, duplicate_count).
    """
    init_db(db_path)
    inserted = 0
    duplicates = 0

    dedup = fetch_dedup_index(db_path)
    seen_keys: set[tuple[str, str]] = dedup["keys"]
    seen_urls: set[str] = dedup["urls"]
    tc_sources: dict[tuple[str, str], set[str]] = dedup["title_company_sources"]

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        for listing in listings:
            key = (listing.source, listing.id)
            norm_u = normalize_url(listing.url)
            tc = normalize_title_company(listing.title, listing.company)

            is_pk_dupe = key in seen_keys
            is_url_dupe = bool(norm_u and norm_u in seen_urls)
            existing_tc_srcs = tc_sources.get(tc, ())
            is_tc_dupe = bool(
                tc[0]
                and tc[1]
                and (
                    any(s != listing.source for s in existing_tc_srcs)
                    or (listing.source.startswith("company:") and listing.source in existing_tc_srcs)
                )
            )

            if is_pk_dupe or is_url_dupe or is_tc_dupe:
                duplicates += 1
                continue

            cursor.execute(
                """
                INSERT OR IGNORE INTO listings (
                    source, id, title, company, location, remote,
                    work_type, salary, description, url, posted_at, status, pipeline_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'new', 'new')
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
                seen_keys.add(key)
                if norm_u:
                    seen_urls.add(norm_u)
                if tc[0] and tc[1]:
                    tc_sources.setdefault(tc, set()).add(listing.source)
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
    """Update technical status and rejection_reason of a listing."""
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


def set_pipeline_status(
    listing_id: str,
    status: str,
    notes: Optional[str] = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> bool:
    """
    Update personal tracking pipeline_status and optional notes for a listing by ID.
    Allowed statuses: new, shortlisted, applied, replied, rejected.
    """
    count, _ = set_pipeline_status_by_query(
        query=listing_id,
        status=status,
        notes=notes,
        match_all=False,
        db_path=db_path,
    )
    return count > 0


def find_listings_by_query(
    query: str,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """
    Find listings matching query by exact ID, or substring of ID, company, or title.
    Returns list of listing dicts ordered by score DESC, collected_at DESC.
    """
    init_db(db_path)
    clean_query = query.strip()
    if not clean_query:
        return []

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        # 1. Exact ID match check first
        cursor.execute(
            """
            SELECT id, source, title, company, location, remote, work_type,
                   salary, url, score, pipeline_status, notes, status_updated_at
            FROM listings
            WHERE id = ?
            """,
            (clean_query,),
        )
        exact_matches = [dict(r) for r in cursor.fetchall()]
        if exact_matches:
            return exact_matches

        # 2. Substring match across id, company, and title (case-insensitive in SQLite LIKE)
        pattern = f"%{clean_query}%"
        cursor.execute(
            """
            SELECT id, source, title, company, location, remote, work_type,
                   salary, url, score, pipeline_status, notes, status_updated_at
            FROM listings
            WHERE id LIKE ? OR company LIKE ? OR title LIKE ?
            ORDER BY score DESC, collected_at DESC
            """,
            (pattern, pattern, pattern),
        )
        return [dict(r) for r in cursor.fetchall()]


def set_pipeline_status_by_query(
    query: str,
    status: str,
    notes: Optional[str] = None,
    match_all: bool = False,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> tuple[int, list[dict[str, Any]]]:
    """
    Update pipeline_status for listings matching query.
    - If no listings match, returns (0, []).
    - If multiple listings match and match_all is False, returns (0, matches) without updating.
    - If match_all is True or exactly 1 listing matches, updates matching listings and returns (updated_count, matches).
    """
    status_clean = status.lower().strip()
    if status_clean not in VALID_PIPELINE_STATUSES:
        raise ValueError(
            f"Invalid pipeline status '{status}'. Must be one of: {sorted(VALID_PIPELINE_STATUSES)}"
        )

    matches = find_listings_by_query(query, db_path=db_path)
    if not matches:
        return 0, []

    if len(matches) > 1 and not match_all:
        return 0, matches

    matched_ids = [m["id"] for m in matches]
    placeholders = ",".join(["?"] * len(matched_ids))
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        if notes is not None:
            cursor.execute(
                f"""
                UPDATE listings
                SET pipeline_status = ?,
                    status_updated_at = CURRENT_TIMESTAMP,
                    notes = ?
                WHERE id IN ({placeholders})
                """,
                [status_clean, notes] + matched_ids,
            )
        else:
            cursor.execute(
                f"""
                UPDATE listings
                SET pipeline_status = ?,
                    status_updated_at = CURRENT_TIMESTAMP
                WHERE id IN ({placeholders})
                """,
                [status_clean] + matched_ids,
            )
        conn.commit()
        return cursor.rowcount, matches


def fetch_tracked_listings(
    status: Optional[str] = None,
    limit: Optional[int] = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Fetch listings filtered by personal pipeline_status."""
    init_db(db_path)
    query = """
        SELECT id, source, title, company, location, remote, work_type,
               salary, url, score, pipeline_status, notes, status_updated_at
        FROM listings
    """
    params: list[Any] = []
    if status:
        status_clean = status.lower().strip()
        if status_clean not in VALID_PIPELINE_STATUSES:
            raise ValueError(
                f"Invalid pipeline status '{status}'. Must be one of: {sorted(VALID_PIPELINE_STATUSES)}"
            )
        query += " WHERE pipeline_status = ?"
        params.append(status_clean)

    query += " ORDER BY score DESC, collected_at DESC"
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def get_pipeline_stats(db_path: Path | str = DEFAULT_DB_PATH) -> dict[str, int]:
    """Return count of listings per personal tracking status."""
    init_db(db_path)
    stats = {s: 0 for s in ["new", "shortlisted", "applied", "replied", "rejected"]}
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT pipeline_status, count(*) FROM listings GROUP BY pipeline_status")
        for row in cursor.fetchall():
            status_val = row[0]
            if status_val in stats:
                stats[status_val] = row[1]
            elif status_val is None:
                stats["new"] += row[1]
    return stats


def save_score(
    source: str,
    id: str,
    score_data: dict[str, Any],
    db_path: Path | str = DEFAULT_DB_PATH,
) -> None:
    """
    Store valid parsed LLM score and set status to 'scored'.
    Clears any prior scoring error fields.
    """
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
                scored_at = CURRENT_TIMESTAMP,
                scoring_error_text = NULL,
                scoring_error_type = NULL,
                scoring_error_at = NULL,
                rejection_reason = NULL
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


def record_scoring_failure(
    source: str,
    id: str,
    error_text: str,
    error_type: str,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> None:
    """
    Record transient scoring failure on listing.
    Status remains 'passed' and scoring_attempts counter is incremented.
    """
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        conn.execute(
            """
            UPDATE listings
            SET status = 'passed',
                scoring_attempts = COALESCE(scoring_attempts, 0) + 1,
                scoring_error_text = ?,
                scoring_error_type = ?,
                scoring_error_at = CURRENT_TIMESTAMP
            WHERE source = ? AND id = ?
            """,
            (error_text, error_type, source, id),
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
    max_attempts: int = 3,
    limit: Optional[int] = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> list[Listing]:
    """
    Fetch listings that passed filtering, need scoring for this prompt version,
    and have not exceeded max_attempts.
    """
    init_db(db_path)
    query = """
        SELECT * FROM listings
        WHERE status = 'passed'
          AND (score IS NULL OR prompt_version IS NULL OR prompt_version != ?)
          AND (scoring_attempts IS NULL OR scoring_attempts < ?)
        ORDER BY collected_at DESC
    """
    params: list[Any] = [prompt_version, max_attempts]
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


def count_skipped_max_attempts(
    prompt_version: str,
    max_attempts: int = 3,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> int:
    """Return number of passed listings skipped because they exceeded max_attempts."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT count(*) FROM listings
            WHERE status = 'passed'
              AND (score IS NULL OR prompt_version IS NULL OR prompt_version != ?)
              AND scoring_attempts >= ?
            """,
            (prompt_version, max_attempts),
        )
        row = cursor.fetchone()
        return row[0] if row else 0


def fetch_top_scored_listings(
    limit: int = 10,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """
    Fetch top scored listings ordered by score descending.
    Excludes listings with pipeline_status IN ('applied', 'replied', 'rejected').
    """
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT source, id, title, company, location, remote, work_type,
                   salary, url, score, skills_match, experience_fit, role_fit,
                   gaps, reason, model, prompt_version, scored_at,
                   pipeline_status, notes
            FROM listings
            WHERE status = 'scored'
              AND score IS NOT NULL
              AND (pipeline_status IS NULL OR pipeline_status NOT IN ('applied', 'replied', 'rejected'))
            ORDER BY score DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        canonical_ids = [r["id"] for r in rows]
        dup_map = fetch_duplicate_urls_map(canonical_ids, db_path=db_path)
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
                "pipeline_status": r["pipeline_status"] or "new",
                "notes": r["notes"] or "",
                "duplicate_urls": dup_map.get(r["id"], []),
            })
        return results


def mark_listing_duplicate(
    source: str,
    id: str,
    canonical_id: str,
    similarity_score: float,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> None:
    """Mark a listing as duplicate of canonical_id, storing similarity score and rejection reason."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        conn.execute(
            """
            UPDATE listings
            SET status = 'duplicate',
                duplicate_of = ?,
                similarity_score = ?,
                rejection_reason = ?
            WHERE source = ? AND id = ?
            """,
            (
                canonical_id,
                float(similarity_score),
                f"duplicate of {canonical_id} (similarity: {similarity_score:.2f})",
                source,
                id,
            ),
        )
        conn.commit()


def fetch_duplicate_urls_for_canonical(
    canonical_id: str, db_path: Path | str = DEFAULT_DB_PATH
) -> list[str]:
    """Fetch all URLs of listings marked as duplicates of canonical_id."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT url FROM listings
            WHERE status = 'duplicate' AND duplicate_of = ? AND url IS NOT NULL AND url != ''
            ORDER BY collected_at ASC
            """,
            (canonical_id,),
        )
        return [row["url"] for row in cursor.fetchall()]


def fetch_duplicate_urls_map(
    canonical_ids: list[str], db_path: Path | str = DEFAULT_DB_PATH
) -> dict[str, list[str]]:
    """Fetch duplicate URLs for a batch of canonical IDs."""
    if not canonical_ids:
        return {}
    init_db(db_path)
    placeholders = ",".join(["?"] * len(canonical_ids))
    result: dict[str, list[str]] = {cid: [] for cid in canonical_ids}
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT duplicate_of, url FROM listings
            WHERE status = 'duplicate' AND duplicate_of IN ({placeholders})
              AND url IS NOT NULL AND url != ''
            ORDER BY collected_at ASC
            """,
            canonical_ids,
        )
        for row in cursor.fetchall():
            c_id = row["duplicate_of"]
            if c_id in result:
                result[c_id].append(row["url"])
    return result


def fetch_dedup_candidates(
    only_remote: bool = True, db_path: Path | str = DEFAULT_DB_PATH
) -> list[dict[str, Any]]:
    """
    Fetch candidate 'passed' listings and target 'scored' listings for duplicate detection.
    If only_remote is True, only listings with remote=1 are fetched.
    """
    init_db(db_path)
    query = """
        SELECT source, id, title, company, location, remote, work_type,
               salary, description, url, posted_at, collected_at, status, score
        FROM listings
        WHERE status IN ('passed', 'scored')
    """
    params: list[Any] = []
    if only_remote:
        query += " AND remote = 1"
    query += " ORDER BY posted_at ASC, collected_at ASC, id ASC"

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, tuple(params))
        return [dict(r) for r in cursor.fetchall()]



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
