"""CLI entrypoint for collecting job listings."""

import sys
from pathlib import Path

from agent.collectors.justjoin_it import JustJoinITCollector
from agent.collectors.nofluffjobs import NoFluffJobsCollector
from agent.storage import count_listings, init_db, save_listings

DEFAULT_DB = Path("data/jobs.db")


def run_collection(db_path: Path = DEFAULT_DB) -> None:
    print(f"=== Job Search Collection Run ===")
    print(f"Database target: {db_path}")
    init_db(db_path)

    collectors = [
        ("JustJoinIT", JustJoinITCollector()),
        ("NoFluffJobs", NoFluffJobsCollector()),
    ]

    total_new = 0
    total_dupes = 0

    for name, collector in collectors:
        try:
            print(f"Fetching from {name}...", end=" ", flush=True)
            listings = collector.collect()
            inserted, dupes = save_listings(listings, db_path=db_path)
            total_new += inserted
            total_dupes += dupes
            print(f"done! ({len(listings)} fetched -> {inserted} new, {dupes} duplicate)")
        except Exception as e:
            print(f"failed! Error: {e}")

    total_in_db = count_listings(db_path=db_path)
    print(f"\nCollection Summary:")
    print(f"  New listings added:       {total_new}")
    print(f"  Duplicate listings skipped:{total_dupes}")
    print(f"  Total listings in DB:     {total_in_db}")


def main() -> None:
    db_path = DEFAULT_DB
    if len(sys.argv) > 1:
        db_path = Path(sys.argv[1])
    run_collection(db_path=db_path)


if __name__ == "__main__":
    main()
