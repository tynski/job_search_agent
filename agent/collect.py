"""CLI entrypoint for collecting job listings."""

import argparse
from pathlib import Path

from agent.collectors.company_boards import CompanyBoardsCollector
from agent.collectors.justjoin_it import JustJoinITCollector
from agent.collectors.nofluffjobs import NoFluffJobsCollector
from agent.storage import count_listings, init_db, save_listings

DEFAULT_DB = Path("data/jobs.db")

# source_name -> factory, so --source matches collector.source_name.
def _collectors(db_path: Path, force: bool = False):
    return {
        "justjoin_it": ("JustJoinIT", lambda: JustJoinITCollector()),
        "nofluffjobs": ("NoFluffJobs", lambda: NoFluffJobsCollector()),
        "company_boards": ("CompanyBoards", lambda: CompanyBoardsCollector(db_path=db_path, force=force)),
    }


def run_collection(db_path: Path = DEFAULT_DB, source: str | None = None, force: bool = False) -> None:
    print(f"=== Job Search Collection Run ===")
    print(f"Database target: {db_path}")
    init_db(db_path)

    factories = _collectors(db_path, force)
    if source:
        if source not in factories:
            print(f"Unknown source: {source}")
            print(f"Available sources: {', '.join(factories)}")
            return
        name, factory = factories[source]
        collectors = [(name, factory())]
    else:
        collectors = [(name, factory()) for name, factory in factories.values()]

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
    parser = argparse.ArgumentParser(description="Collect job listings from job boards.")
    parser.add_argument(
        "db_path", nargs="?", default=None,
        help="SQLite database path (default: data/jobs.db)",
    )
    parser.add_argument(
        "--source", default=None,
        help="Run only this source (e.g. justjoin_it, nofluffjobs, company_boards)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Bypass the company_boards cooldown check",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else DEFAULT_DB
    run_collection(db_path=db_path, source=args.source, force=args.force)


if __name__ == "__main__":
    main()
