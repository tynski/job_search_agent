"""Tests for job collectors."""

import json
from pathlib import Path

from agent.collectors.hackernews import HackerNewsCollector
from agent.collectors.justjoin_it import JustJoinITCollector
from agent.collectors.nofluffjobs import NoFluffJobsCollector

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_justjoin_it_normalization():
    fixture_path = FIXTURES_DIR / "sample_jjit.json"
    with open(fixture_path, encoding="utf-8") as f:
        data = json.load(f)

    listing = JustJoinITCollector.normalize(data[0])
    assert listing.id == "acme-corp-python-data-engineer-warszawa"
    assert listing.source == "justjoin_it"
    assert listing.title == "Python Data Engineer"
    assert listing.company == "Acme Corp"
    assert listing.location == "Warszawa"
    assert listing.remote is True
    assert listing.work_type == "b2b"
    assert "22,000 - 28,000 PLN (b2b)" in listing.salary
    assert "Python" in listing.description
    assert listing.url == "https://justjoin.it/offers/acme-corp-python-data-engineer-warszawa"
    assert listing.posted_at == "2026-11-20T12:00:00.000Z"


def test_nofluffjobs_normalization():
    fixture_path = FIXTURES_DIR / "sample_nfj.json"
    with open(fixture_path, encoding="utf-8") as f:
        data = json.load(f)

    listing = NoFluffJobsCollector.normalize(data["postings"][0])
    assert listing.id == "nfj-python-dev-123"
    assert listing.source == "nofluffjobs"
    assert listing.title == "Senior Python Backend Developer"
    assert listing.company == "DataFlow Tech"
    assert listing.location == "Kraków"
    assert listing.remote is True
    assert listing.work_type == "b2b"
    assert "24,000 - 30,000 PLN (b2b)" in listing.salary
    assert "Python" in listing.description
    assert "https://nofluffjobs.com/job/senior-python-backend-developer-dataflow-tech" in listing.url
    assert listing.posted_at.startswith("2024-") or listing.posted_at.startswith("202")


def test_hackernews_normalization():
    fixture_path = FIXTURES_DIR / "sample_hn.json"
    with open(fixture_path, encoding="utf-8") as f:
        data = json.load(f)

    listing = HackerNewsCollector.normalize(data[0])
    assert listing is not None
    assert listing.id == "hn_41234567"
    assert listing.source == "hackernews"
    assert listing.company == "Stripe"
    assert listing.title == "AI Solutions Engineer"
    assert "Remote" in listing.location
    assert listing.remote is True
    assert listing.work_type == "full-time"
    assert "$160k - $210k" in listing.salary
    assert "Stripe" in listing.description
    assert listing.url == "https://news.ycombinator.com/item?id=41234567"
