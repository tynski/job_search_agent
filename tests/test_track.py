"""Tests for Step 4: personal pipeline tracking status."""

from pathlib import Path
import pytest
import sqlite3

from agent.models import Listing
from agent.storage import (
    count_listings,
    fetch_top_scored_listings,
    fetch_tracked_listings,
    get_db_connection,
    get_pipeline_stats,
    init_db,
    migrate_tracking_columns,
    save_listings,
    save_score,
    set_pipeline_status,
)
from agent.digest import format_digest_table
from agent.track import main as track_main


def _create_sample_listing(listing_id: str, title: str = "Test Job") -> Listing:
    return Listing(
        id=listing_id,
        source="test_src",
        title=title,
        company="Company X",
        location="Warszawa",
        remote=True,
        work_type="b2b",
        salary="25k PLN",
        description="Python data engineer",
        url=f"https://example.com/job/{listing_id}",
        posted_at="2026-11-20T10:00:00Z",
    )


def test_migrate_tracking_columns(tmp_path: Path):
    db_file = tmp_path / "legacy_jobs.db"
    
    # 1. Non-existent db returns 0
    assert migrate_tracking_columns(tmp_path / "non_existent.db") == 0

    # 2. Simulate older table where pipeline_status might be NULL
    with sqlite3.connect(str(db_file)) as conn:
        conn.execute(
            """
            CREATE TABLE listings (
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
        conn.execute(
            "INSERT INTO listings (source, id, title) VALUES ('s1', 'item-1', 'Legacy Job 1')"
        )
        conn.execute(
            "INSERT INTO listings (source, id, title) VALUES ('s1', 'item-2', 'Legacy Job 2')"
        )
        conn.commit()

    # Run migration
    changed = migrate_tracking_columns(db_file)
    assert changed == 2

    # Check backup created
    bak_file = tmp_path / "legacy_jobs.db.bak"
    assert bak_file.exists()

    # Check columns exist and values are 'new'
    with get_db_connection(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, pipeline_status, status_updated_at, notes FROM listings")
        rows = cursor.fetchall()
        assert len(rows) == 2
        for r in rows:
            assert r["pipeline_status"] == "new"
            assert r["notes"] is None


def test_set_pipeline_status(tmp_path: Path):
    db_file = tmp_path / "test_set.db"
    init_db(db_file)

    l1 = _create_sample_listing("job-1", "AI Engineer")
    save_listings([l1], db_path=db_file)

    # 1. Update to shortlisted with notes
    ok = set_pipeline_status("job-1", "shortlisted", notes="Reach out to recruiter", db_path=db_file)
    assert ok is True

    items = fetch_tracked_listings(status="shortlisted", db_path=db_file)
    assert len(items) == 1
    assert items[0]["id"] == "job-1"
    assert items[0]["pipeline_status"] == "shortlisted"
    assert items[0]["notes"] == "Reach out to recruiter"
    assert items[0]["status_updated_at"] is not None

    # 2. Update status without altering notes
    ok = set_pipeline_status("job-1", "applied", db_path=db_file)
    assert ok is True

    items = fetch_tracked_listings(status="applied", db_path=db_file)
    assert len(items) == 1
    assert items[0]["pipeline_status"] == "applied"
    assert items[0]["notes"] == "Reach out to recruiter"

    # 3. Non-existent listing returns False
    assert set_pipeline_status("nonexistent-id", "rejected", db_path=db_file) is False

    # 4. Invalid status raises ValueError
    with pytest.raises(ValueError, match="Invalid pipeline status 'invalid_status'"):
        set_pipeline_status("job-1", "invalid_status", db_path=db_file)


def test_fetch_tracked_listings_and_stats(tmp_path: Path):
    db_file = tmp_path / "test_stats.db"
    init_db(db_file)

    listings = [
        _create_sample_listing("j1"),
        _create_sample_listing("j2"),
        _create_sample_listing("j3"),
        _create_sample_listing("j4"),
        _create_sample_listing("j5"),
    ]
    save_listings(listings, db_path=db_file)

    set_pipeline_status("j1", "shortlisted", db_path=db_file)
    set_pipeline_status("j2", "applied", db_path=db_file)
    set_pipeline_status("j3", "replied", db_path=db_file)
    set_pipeline_status("j4", "rejected", db_path=db_file)
    # j5 remains 'new'

    # Check stats
    stats = get_pipeline_stats(db_path=db_file)
    assert stats == {
        "new": 1,
        "shortlisted": 1,
        "applied": 1,
        "replied": 1,
        "rejected": 1,
    }

    # Check fetch with filter and limit
    all_tracked = fetch_tracked_listings(db_path=db_file)
    assert len(all_tracked) == 5

    shortlisted = fetch_tracked_listings(status="shortlisted", db_path=db_file)
    assert len(shortlisted) == 1
    assert shortlisted[0]["id"] == "j1"

    limited = fetch_tracked_listings(limit=2, db_path=db_file)
    assert len(limited) == 2

    with pytest.raises(ValueError, match="Invalid pipeline status 'bogus'"):
        fetch_tracked_listings(status="bogus", db_path=db_file)


def test_digest_excludes_applied_replied_rejected(tmp_path: Path):
    db_file = tmp_path / "test_digest_tracking.db"
    init_db(db_file)

    # 5 listings scored with various pipeline statuses
    for i in range(1, 6):
        l = _create_sample_listing(f"offer-{i}", f"Title {i}")
        save_listings([l], db_path=db_file)
        save_score(
            source="test_src",
            id=f"offer-{i}",
            score_data={
                "score": 5.0 + i,
                "skills_match": 8.0,
                "experience_fit": 8.0,
                "role_fit": 8.0,
                "gaps": ["None"],
                "reason": f"Fit reason {i}",
                "model": "gemini",
                "prompt_version": "v1.0",
            },
            db_path=db_file,
        )

    # offer-1: 'new' (score 6.0)
    # offer-2: 'shortlisted' (score 7.0)
    # offer-3: 'applied' (score 8.0) -> must be excluded
    # offer-4: 'replied' (score 9.0) -> must be excluded
    # offer-5: 'rejected' (score 10.0) -> must be excluded
    set_pipeline_status("offer-2", "shortlisted", notes="Top choice", db_path=db_file)
    set_pipeline_status("offer-3", "applied", db_path=db_file)
    set_pipeline_status("offer-4", "replied", db_path=db_file)
    set_pipeline_status("offer-5", "rejected", db_path=db_file)

    digest_items = fetch_top_scored_listings(limit=10, db_path=db_file)

    # Only offer-2 (shortlisted, score 7.0) and offer-1 (new, score 6.0) should be returned
    assert len(digest_items) == 2
    assert digest_items[0]["id"] == "offer-2"
    assert digest_items[0]["pipeline_status"] == "shortlisted"
    assert digest_items[0]["notes"] == "Top choice"
    assert digest_items[1]["id"] == "offer-1"
    assert digest_items[1]["pipeline_status"] == "new"

    # Verify table formatting contains [SHORTLISTED] tag, ID, and note
    table_str = format_digest_table(digest_items)
    assert "[SHORTLISTED]" in table_str
    assert "Top choice" in table_str
    assert "Title 2" in table_str
    assert "ID:        offer-2" in table_str


def test_cli_track_stats(tmp_path: Path, monkeypatch, capsys):
    db_file = tmp_path / "cli_test.db"
    init_db(db_file)
    save_listings([_create_sample_listing("item-1")], db_path=db_file)

    monkeypatch.setattr(
        "sys.argv",
        ["track", "--db", str(db_file), "stats"],
    )
    track_main()
    captured = capsys.readouterr().out
    assert "Personal Pipeline Tracking Stats" in captured
    assert "new" in captured
    assert "Total Listings" in captured


def test_cli_track_set_and_list(tmp_path: Path, monkeypatch, capsys):
    db_file = tmp_path / "cli_test.db"
    init_db(db_file)
    save_listings([_create_sample_listing("item-cli")], db_path=db_file)

    # 1. CLI set status
    monkeypatch.setattr(
        "sys.argv",
        [
            "track",
            "--db",
            str(db_file),
            "set",
            "item-cli",
            "shortlisted",
            "--note",
            "Great company culture",
        ],
    )
    track_main()
    out = capsys.readouterr().out
    assert "Successfully updated listing 'item-cli'" in out
    assert "status: 'shortlisted'" in out

    # 2. CLI list
    monkeypatch.setattr(
        "sys.argv",
        [
            "track",
            "--db",
            str(db_file),
            "list",
            "--status",
            "shortlisted",
        ],
    )
    track_main()
    out = capsys.readouterr().out
    assert "Tracked Job Listings ('shortlisted': 1)" in out
    assert "SHORTLISTED" in out
    assert "item-cli" in out
    assert "Great company culture" in out


def test_cli_track_set_invalid_id_exits(tmp_path: Path, monkeypatch):
    db_file = tmp_path / "cli_test.db"
    init_db(db_file)

    monkeypatch.setattr(
        "sys.argv",
        [
            "track",
            "--db",
            str(db_file),
            "set",
            "missing-id",
            "applied",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        track_main()
    assert exc.value.code == 1


def test_find_and_smart_set_with_all_flag(tmp_path: Path, monkeypatch, capsys):
    from agent.storage import find_listings_by_query, set_pipeline_status_by_query

    db_file = tmp_path / "smart_test.db"
    init_db(db_file)

    l1 = Listing(id="cog-lodz", source="nfj", title="Python Engineer", company="Cognitiva", location="Lodz", remote=True, work_type="b2b", salary="20k", description="", url="", posted_at="")
    l2 = Listing(id="cog-waw", source="nfj", title="Python Engineer", company="Cognitiva", location="Warsaw", remote=True, work_type="b2b", salary="20k", description="", url="", posted_at="")
    l3 = Listing(id="other-job", source="jjit", title="Data Engineer", company="Acme Corp", location="Krakow", remote=False, work_type="b2b", salary="22k", description="", url="", posted_at="")
    save_listings([l1, l2, l3], db_path=db_file)

    # 1. find_listings_by_query
    matches = find_listings_by_query("cognitiva", db_path=db_file)
    assert len(matches) == 2
    assert {m["id"] for m in matches} == {"cog-lodz", "cog-waw"}

    # 2. set_pipeline_status_by_query: multiple matches without match_all
    count, matched = set_pipeline_status_by_query("cognitiva", "applied", match_all=False, db_path=db_file)
    assert count == 0
    assert len(matched) == 2

    # 3. set_pipeline_status_by_query: multiple matches with match_all
    count, matched = set_pipeline_status_by_query("cognitiva", "applied", notes="Applied batch", match_all=True, db_path=db_file)
    assert count == 2
    assert len(matched) == 2

    # Verify both are applied
    applied = fetch_tracked_listings(status="applied", db_path=db_file)
    assert len(applied) == 2
    assert {a["id"] for a in applied} == {"cog-lodz", "cog-waw"}
    assert applied[0]["notes"] == "Applied batch"

    # 4. CLI find command
    monkeypatch.setattr(
        "sys.argv",
        ["track", "--db", str(db_file), "find", "cognitiva"],
    )
    track_main()
    captured = capsys.readouterr().out
    assert "Search Results for 'cognitiva' (2 match(es))" in captured
    assert "cog-lodz" in captured
    assert "cog-waw" in captured

    # 5. CLI set multiple shows warning and exits 1
    monkeypatch.setattr(
        "sys.argv",
        ["track", "--db", str(db_file), "set", "cognitiva", "shortlisted"],
    )
    with pytest.raises(SystemExit) as exc:
        track_main()
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "Found 2 listings matching 'cognitiva'" in out
    assert "use its exact ID" in out


def test_cli_track_set_mutual_exclusion(tmp_path: Path, monkeypatch, capsys):
    db_file = tmp_path / "cli_mutex.db"
    init_db(db_file)

    # 1. Both --all and listing_id
    monkeypatch.setattr(
        "sys.argv",
        ["track", "--db", str(db_file), "set", "--all", "item-1", "shortlisted"],
    )
    with pytest.raises(SystemExit) as exc:
        track_main()
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "--all is mutually exclusive with <listing_id>" in out
    assert "Usage:" in out

    # 2. Neither --all nor listing_id
    monkeypatch.setattr(
        "sys.argv",
        ["track", "--db", str(db_file), "set"],
    )
    with pytest.raises(SystemExit) as exc:
        track_main()
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "Must specify either <listing_id> <status> or --all <status>" in out
    assert "Usage:" in out

    # 3. --all with no status
    monkeypatch.setattr(
        "sys.argv",
        ["track", "--db", str(db_file), "set", "--all"],
    )
    with pytest.raises(SystemExit) as exc:
        track_main()
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "Missing <status> for --all" in out


def test_cli_track_set_all_dry_run_and_confirmation(tmp_path: Path, monkeypatch, capsys):
    db_file = tmp_path / "cli_all_test.db"
    init_db(db_file)

    # Seed two scored listings in the digest selection
    l1 = _create_sample_listing("d1", "Senior Python Dev")
    l2 = _create_sample_listing("d2", "AI Solutions Engineer")
    save_listings([l1, l2], db_path=db_file)
    save_score(
        source="test_src", id="d1",
        score_data={"score": 9.0, "skills_match": 9.0, "experience_fit": 9.0, "role_fit": 9.0, "gaps": [], "reason": "fit"},
        db_path=db_file,
    )
    save_score(
        source="test_src", id="d2",
        score_data={"score": 8.5, "skills_match": 8.5, "experience_fit": 8.5, "role_fit": 8.5, "gaps": [], "reason": "fit"},
        db_path=db_file,
    )

    # 1. Dry run
    monkeypatch.setattr(
        "sys.argv",
        ["track", "--db", str(db_file), "set", "--all", "shortlisted", "--dry-run"],
    )
    track_main()
    out = capsys.readouterr().out
    assert "[Dry run] 2 listing(s) in digest selection would be updated to status 'shortlisted'" in out
    assert "d1" in out
    assert "d2" in out

    # Verify database unchanged
    stats = get_pipeline_stats(db_path=db_file)
    assert stats["shortlisted"] == 0
    assert stats["new"] == 2

    # 2. Confirmation declined (user inputs 'n')
    monkeypatch.setattr(
        "sys.argv",
        ["track", "--db", str(db_file), "set", "--all", "shortlisted"],
    )
    monkeypatch.setattr("builtins.input", lambda _: "n")
    track_main()
    out = capsys.readouterr().out
    assert "This will update 2 listing(s)" in out
    assert "Aborted." in out

    # Verify database still unchanged
    stats = get_pipeline_stats(db_path=db_file)
    assert stats["shortlisted"] == 0

    # 3. Confirmation accepted (user inputs 'y')
    monkeypatch.setattr(
        "sys.argv",
        ["track", "--db", str(db_file), "set", "--all", "shortlisted"],
    )
    monkeypatch.setattr("builtins.input", lambda _: "y")
    track_main()
    out = capsys.readouterr().out
    assert "Successfully updated 2 listing(s) -> status: 'shortlisted'" in out

    # Verify database updated
    stats = get_pipeline_stats(db_path=db_file)
    assert stats["shortlisted"] == 2


def test_cli_track_set_all_with_yes_and_only_status(tmp_path: Path, monkeypatch, capsys):
    db_file = tmp_path / "cli_yes_test.db"
    init_db(db_file)

    l1 = _create_sample_listing("d1", "Python Lead")
    l2 = _create_sample_listing("d2", "AI Architect")
    save_listings([l1, l2], db_path=db_file)
    save_score(
        source="test_src", id="d1",
        score_data={"score": 9.0, "skills_match": 9.0, "experience_fit": 9.0, "role_fit": 9.0, "gaps": [], "reason": "fit"},
        db_path=db_file,
    )
    save_score(
        source="test_src", id="d2",
        score_data={"score": 8.0, "skills_match": 8.0, "experience_fit": 8.0, "role_fit": 8.0, "gaps": [], "reason": "fit"},
        db_path=db_file,
    )

    # Set d1 to 'shortlisted', d2 remains 'new'
    set_pipeline_status("d1", "shortlisted", db_path=db_file)

    # Use --all with --only-status shortlisted to transition only d1 to applied
    monkeypatch.setattr(
        "sys.argv",
        [
            "track",
            "--db",
            str(db_file),
            "set",
            "--all",
            "applied",
            "--only-status",
            "shortlisted",
            "--note",
            "Auto applied to shortlisted",
            "--yes",
        ],
    )
    track_main()
    out = capsys.readouterr().out
    assert "Successfully updated 1 listing(s) -> status: 'applied'" in out
    assert "(note: 'Auto applied to shortlisted')" in out

    # Verify d1 is applied, d2 is still new
    stats = get_pipeline_stats(db_path=db_file)
    assert stats["applied"] == 1
    assert stats["new"] == 1

    applied_items = fetch_tracked_listings(status="applied", db_path=db_file)
    assert applied_items[0]["id"] == "d1"
    assert applied_items[0]["notes"] == "Auto applied to shortlisted"

