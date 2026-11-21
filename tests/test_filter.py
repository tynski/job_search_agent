"""Tests for rule-based filter."""

from pathlib import Path
from agent.filter import evaluate_listing
from agent.models import Listing
from agent.storage import fetch_listings, init_db, save_listings, update_listing_status


def get_mock_filter_config():
    return {
        "location": {
            "allowed_countries_and_cities": [
                "poland",
                "polska",
                "warszawa",
                "krakow",
                "kraków",
                "wroclaw",
                "wrocław",
                "remote",
                "zdalnie",
                "eu",
            ],
            "allow_any_remote": True,
        },
        "work_type": {
            "allowed": ["b2b", "employment", "permanent", "uop", "contract"],
            "allow_missing": True,
        },
        "exclude": {
            "titles": [
                "lead",
                "manager",
                "director",
                "sales",
                "frontend",
                "react",
                "php",
                "ruby",
                "ios",
                "android",
            ],
            "keywords": ["unpaid", "volunteering"],
        },
        "salary": {"min_pln": 10000},
        "include_keywords": ["Python", "SQL", "Data Engineer", "Solutions Engineer", "ETL", "Azure", "AWS"],
    }


def test_filter_passing_listing():
    cfg = get_mock_filter_config()
    listing = Listing(
        id="pass-1",
        source="justjoin_it",
        title="Python Data Engineer",
        company="GoodTech",
        location="Warszawa",
        remote=True,
        work_type="b2b",
        salary="22000 - 28000 PLN (b2b)",
        description="We need Python and SQL skills for cloud ETL.",
        url="https://example.com/1",
        posted_at="2026-11-20T10:00:00Z",
    )
    passed, reason = evaluate_listing(listing, cfg)
    assert passed is True
    assert reason is None


def test_filter_rejected_by_excluded_title():
    cfg = get_mock_filter_config()
    listing = Listing(
        id="rej-title",
        source="justjoin_it",
        title="Lead Engineering Manager",
        company="BigCorp",
        location="Warszawa",
        remote=True,
        work_type="b2b",
        salary="30000 PLN",
        description="Python experience required.",
        url="https://example.com/2",
        posted_at="2026-11-20T10:00:00Z",
    )
    passed, reason = evaluate_listing(listing, cfg)
    assert passed is False
    assert "excluded_title" in reason


def test_filter_rejected_by_location():
    cfg = get_mock_filter_config()
    listing = Listing(
        id="rej-loc",
        source="nofluffjobs",
        title="Python Developer",
        company="US Corp",
        location="San Francisco, CA",
        remote=False,
        work_type="b2b",
        salary="15000 PLN",
        description="Python and SQL.",
        url="https://example.com/3",
        posted_at="2026-11-20T10:00:00Z",
    )
    passed, reason = evaluate_listing(listing, cfg)
    assert passed is False
    assert reason == "location_mismatch"


def test_filter_polish_location_passes():
    cfg = get_mock_filter_config()
    listing = Listing(
        id="pass-pl-loc",
        source="nofluffjobs",
        title="Python Engineer",
        company="Wroclaw Tech",
        location="Wrocław",
        remote=False,
        work_type="b2b",
        salary=None,
        description="ETL pipelines with Python.",
        url="https://example.com/4",
        posted_at="2026-11-20T10:00:00Z",
    )
    passed, reason = evaluate_listing(listing, cfg)
    assert passed is True
    assert reason is None


def test_filter_rejected_by_work_type():
    cfg = get_mock_filter_config()
    listing = Listing(
        id="rej-wt",
        source="justjoin_it",
        title="Python Developer",
        company="Agency",
        location="Remote",
        remote=True,
        work_type="internship",
        salary=None,
        description="Python programming.",
        url="https://example.com/5",
        posted_at="2026-11-20T10:00:00Z",
    )
    passed, reason = evaluate_listing(listing, cfg)
    assert passed is False
    assert reason == "work_type_mismatch"


def test_filter_missing_work_type_and_salary_does_not_reject():
    cfg = get_mock_filter_config()
    listing = Listing(
        id="pass-missing-fields",
        source="hackernews",
        title="Solutions Engineer",
        company="Startup",
        location="Remote",
        remote=True,
        work_type=None,
        salary=None,
        description="Python and AWS integration engineer.",
        url="https://example.com/6",
        posted_at="2026-11-20T10:00:00Z",
    )
    passed, reason = evaluate_listing(listing, cfg)
    assert passed is True
    assert reason is None


def test_filter_rejected_by_missing_keywords():
    cfg = get_mock_filter_config()
    listing = Listing(
        id="rej-no-keywords",
        source="justjoin_it",
        title="DevOps Specialist",
        company="InfraCorp",
        location="Kraków",
        remote=True,
        work_type="b2b",
        salary=None,
        description="Kubernetes, Terraform, Bash and Linux administration only.",
        url="https://example.com/7",
        posted_at="2026-11-20T10:00:00Z",
    )
    passed, reason = evaluate_listing(listing, cfg)
    assert passed is False
    assert reason == "no_profile_keywords"


def test_filter_salary_below_minimum():
    cfg = get_mock_filter_config()
    listing = Listing(
        id="rej-low-salary",
        source="justjoin_it",
        title="Junior Python Specialist",
        company="SmallCo",
        location="Warszawa",
        remote=True,
        work_type="b2b",
        salary="5,000 - 8,000 PLN (b2b)",
        description="Python scripts.",
        url="https://example.com/8",
        posted_at="2026-11-20T10:00:00Z",
    )
    passed, reason = evaluate_listing(listing, cfg)
    assert passed is False
    assert "salary_below_minimum" in reason


def test_db_filter_status_update(tmp_path: Path):
    db_file = tmp_path / "filter_test.db"
    init_db(db_file)

    l1 = Listing(
        id="item-1",
        source="test",
        title="Python Engineer",
        company="A",
        location="Remote",
        remote=True,
        work_type="b2b",
        salary=None,
        description="Python",
        url="url",
        posted_at="",
    )
    save_listings([l1], db_path=db_file)

    # Initially new
    new_items = fetch_listings(db_path=db_file, status="new")
    assert len(new_items) == 1

    # Update to passed
    update_listing_status(source="test", id="item-1", status="passed", rejection_reason=None, db_path=db_file)
    assert len(fetch_listings(db_path=db_file, status="new")) == 0
    passed_items = fetch_listings(db_path=db_file, status="passed")
    assert len(passed_items) == 1
