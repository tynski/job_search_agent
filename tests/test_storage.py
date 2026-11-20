"""Tests for SQLite storage and deduplication."""

from pathlib import Path
from agent.models import Listing
from agent.storage import count_listings, fetch_listings, init_db, save_listings


def test_storage_and_deduplication(tmp_path: Path):
    db_file = tmp_path / "test_jobs.db"
    init_db(db_file)

    l1 = Listing(
        id="offer-1",
        source="justjoin_it",
        title="Python Engineer",
        company="Tech Corp",
        location="Warszawa",
        remote=True,
        work_type="b2b",
        salary="20k-25k PLN",
        description="Python, SQL",
        url="https://justjoin.it/offers/offer-1",
        posted_at="2026-11-20T10:00:00Z",
    )
    l2 = Listing(
        id="offer-2",
        source="nofluffjobs",
        title="Data Engineer",
        company="Data Corp",
        location="Remote",
        remote=True,
        work_type="permanent",
        salary=None,
        description="ETL pipelines",
        url="https://nofluffjobs.com/job/offer-2",
        posted_at="2026-11-20T11:00:00Z",
    )

    # First save: 2 new, 0 duplicates
    inserted, dupes = save_listings([l1, l2], db_path=db_file)
    assert inserted == 2
    assert dupes == 0
    assert count_listings(db_path=db_file) == 2

    # Second save with 1 existing and 1 new
    l3 = Listing(
        id="offer-3",
        source="hackernews",
        title="AI Engineer",
        company="OpenAI",
        location="Remote",
        remote=True,
        work_type="contract",
        salary="$200k",
        description="LLM",
        url="https://news.ycombinator.com/item?id=3",
        posted_at="2026-11-20T12:00:00Z",
    )
    inserted, dupes = save_listings([l1, l3], db_path=db_file)
    assert inserted == 1
    assert dupes == 1
    assert count_listings(db_path=db_file) == 3

    # Check fetched listings
    all_listings = fetch_listings(db_path=db_file)
    assert len(all_listings) == 3
    ids = {item.id for item in all_listings}
    assert ids == {"offer-1", "offer-2", "offer-3"}
