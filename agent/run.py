"""End-to-end pipeline runner for job_search_agent."""

import argparse
from pathlib import Path
import sys

from agent.collect import run_collection
from agent.dedup import run_dedup
from agent.digest import print_digest
from agent.filter import run_filter
from agent.score import run_scoring
from agent.storage import DEFAULT_DB_PATH, init_db


def run_pipeline(limit: int | None = None, db_path: Path = DEFAULT_DB_PATH) -> None:
    """Run collect, filter, dedup, score, and digest in sequence."""
    print("=" * 70)
    print("  STARTING FULL JOB SEARCH PIPELINE")
    print("=" * 70)

    init_db(db_path)

    # Step 1: Collect
    print("\n[Step 1/5] Collecting listings from sources...")
    run_collection(db_path=db_path)

    # Step 2: Rule-based filter
    print("\n[Step 2/5] Running rule-based filter...")
    run_filter(db_path=db_path)

    # Step 3: Duplicate detection
    print("\n[Step 3/5] Running duplicate detection...")
    run_dedup(db_path=db_path)

    # Step 4: LLM Scoring
    print("\n[Step 4/5] Running LLM scoring...")
    run_scoring(db_path=db_path, limit=limit)

    # Step 5: Terminal Digest
    print("\n[Step 5/5] Generating daily digest of top matches...")
    print_digest(db_path=db_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the end-to-end job search pipeline.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of listings to score with LLM in this run (useful for testing).",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to SQLite database file.",
    )
    args = parser.parse_args()

    run_pipeline(limit=args.limit, db_path=args.db)


if __name__ == "__main__":
    main()
