"""Tests for the company_boards collector. Fixture-based, no live requests."""

import json
import logging
from pathlib import Path

from agent.collectors.company_boards import CompanyBoardsCollector
from agent.models import Listing
from agent.storage import get_last_collector_run, record_collector_run

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load(name: str):
    with open(FIXTURES_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def _write_config(tmp_path: Path, companies: list[dict], cooldown_hours: float = 12.0) -> Path:
    lines = ["companies:"]
    for c in companies:
        lines.append(f"  - name: {c['name']!r}")
        lines.append(f"    ats: {c['ats']!r}")
        lines.append(f"    board_id: {c['board_id']!r}")
    lines += [
        "collectors:",
        "  company_boards:",
        "    request_delay: 0",
        f"    cooldown_hours: {cooldown_hours}",
        "filter:",
        "  location:",
        "    allowed_countries_and_cities:",
        '      - "warszawa"',
        '      - "poland"',
        '      - "kraków"',
        '      - "gdańsk"',
        '      - "remote"',
        '      - "eu"',
        "    allow_any_remote: true",
    ]
    path = tmp_path / "config.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class FakeResponse:
    def __init__(self, status_code: int = 200, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def _make_listing(location: str, remote: bool = False) -> Listing:
    return Listing(
        id="1",
        source="company:X",
        title="Some Title",
        company="X",
        location=location,
        remote=remote,
        work_type=None,
        salary=None,
        description="desc",
        url="https://x.example.com/job",
        posted_at="",
    )


# ---- Per-ATS normalization ----

def test_greenhouse_normalize():
    listing = CompanyBoardsCollector.normalize_greenhouse(_load("greenhouse_jobs.json")["jobs"][0], "Vaulttec")
    assert listing.source == "company:Vaulttec"
    assert listing.title == "Senior Data Engineer"
    assert listing.location == "Warszawa, Poland"
    assert listing.remote is False
    assert listing.url == "https://boards.greenhouse.io/vaulttec/jobs/150293821"
    assert "Python, SQL" in listing.description


def test_lever_normalize():
    listing = CompanyBoardsCollector.normalize_lever(_load("lever_postings.json")[0], "FictionalCo")
    assert listing.source == "company:FictionalCo"
    assert listing.title == "Data Engineer"
    assert listing.location == "Poland"
    assert listing.remote is False
    assert listing.url == "https://jobs.lever.co/fictionalco/5ac21346-8e0c-4494-8e7a-3eb92ff77902"
    assert "ETL pipelines" in listing.description


def test_ashby_normalize():
    listing = CompanyBoardsCollector.normalize_ashby(_load("ashby_jobs.json")["jobs"][0], "AshbyCo")
    assert listing.source == "company:AshbyCo"
    assert listing.title == "Solutions Engineer"
    assert listing.location == "Kraków, Poland"
    assert listing.remote is False
    assert listing.url == "https://jobs.ashbyhq.com/fictionalco/c1f0d2a3"
    assert "deploy our platform" in listing.description


def test_smartrecruiters_normalize():
    posting = _load("smartrecruiters_postings.json")["content"][0]
    listing = CompanyBoardsCollector.normalize_smartrecruiters(posting, "SR Co")
    assert listing.source == "company:SR Co"
    assert listing.title == "Senior Python Engineer"
    assert listing.location == "Gdańsk, Pomorskie"
    assert listing.remote is False
    assert listing.url == "https://api.smartrecruiters.com/v1/companies/fictionalco/postings/8122334455"
    remote_posting = _load("smartrecruiters_postings.json")["content"][1]
    assert CompanyBoardsCollector.normalize_smartrecruiters(remote_posting, "SR Co").remote is True


# ---- Fetch: documented endpoints, User-Agent header, 404 handling ----

def test_fetch_greenhouse_uses_documented_endpoint(tmp_path, monkeypatch):
    calls = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.update({"url": url, "params": params, "headers": headers})
        return FakeResponse(200, _load("greenhouse_jobs.json"))

    monkeypatch.setattr("agent.collectors.company_boards.requests.get", fake_get)
    collector = CompanyBoardsCollector(config_path=_write_config(tmp_path, []))
    entries = collector._fetch_greenhouse("vaulttec")

    assert calls["url"] == "https://boards-api.greenhouse.io/v1/boards/vaulttec/jobs"
    assert calls["params"] == {"content": "true"}
    assert calls["headers"]["User-Agent"]
    assert len(entries) == 2


def test_fetch_returns_none_on_404(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(
        "agent.collectors.company_boards.requests.get",
        lambda *a, **k: FakeResponse(404),
    )
    collector = CompanyBoardsCollector(config_path=_write_config(tmp_path, []))
    assert collector._get_json("https://boards-api.greenhouse.io/v1/boards/nope/jobs") is None
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_fetch_returns_empty_on_empty_board(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "agent.collectors.company_boards.requests.get",
        lambda *a, **k: FakeResponse(200, {"jobs": []}),
    )
    collector = CompanyBoardsCollector(config_path=_write_config(tmp_path, []))
    assert collector._fetch_greenhouse("vaulttec") == []


# ---- Location filter ----

def test_location_filter():
    cfg = {
        "allowed_countries_and_cities": ["warszawa", "remote", "eu"],
        "allow_any_remote": True,
    }
    assert CompanyBoardsCollector._matches_location(_make_listing("Warszawa"), cfg)
    assert CompanyBoardsCollector._matches_location(_make_listing("Remote", remote=True), cfg)
    assert not CompanyBoardsCollector._matches_location(_make_listing("Berlin"), cfg)


# ---- End-to-end collect via fixtures ----

def _fake_get_dispatch(monkeypatch):
    def fake_get(url, params=None, headers=None, timeout=None):
        if "boards-api.greenhouse.io" in url:
            return FakeResponse(200, _load("greenhouse_jobs.json"))
        if "api.lever.co" in url:
            return FakeResponse(200, _load("lever_postings.json"))
        if "ashbyhq.com" in url:
            return FakeResponse(200, _load("ashby_jobs.json"))
        if "api.smartrecruiters.com" in url:
            return FakeResponse(200, _load("smartrecruiters_postings.json"))
        return FakeResponse(404)

    monkeypatch.setattr("agent.collectors.company_boards.requests.get", fake_get)


def test_collect_end_to_end(tmp_path, monkeypatch):
    _fake_get_dispatch(monkeypatch)
    config = _write_config(
        tmp_path,
        [
            {"name": "Vaulttec", "ats": "greenhouse", "board_id": "vaulttec"},
            {"name": "FictionalCo", "ats": "lever", "board_id": "fictionalco"},
            {"name": "AshbyCo", "ats": "ashby", "board_id": "ashbyco"},
            {"name": "SR Co", "ats": "smartrecruiters", "board_id": "sico"},
        ],
    )
    db_path = tmp_path / "jobs.db"
    collector = CompanyBoardsCollector(config_path=config, db_path=db_path)
    listings = collector.collect()

    sources = {l.source for l in listings}
    assert sources == {"company:Vaulttec", "company:FictionalCo", "company:AshbyCo", "company:SR Co"}
    assert len(listings) == 6  # 2 Greenhouse (incl. remote) + 1 Lever + 1 Ashby + 2 SmartRecruiters
    sr_remote = [l for l in listings if l.title == "DevOps Engineer (remote)"]
    assert sr_remote and sr_remote[0].remote is True


def test_collect_dedups_by_url(tmp_path, monkeypatch):
    data = _load("greenhouse_jobs.json")
    data["jobs"] = [data["jobs"][0], data["jobs"][0]]  # same posting twice
    monkeypatch.setattr(
        "agent.collectors.company_boards.requests.get",
        lambda *a, **k: FakeResponse(200, data),
    )
    config = _write_config(tmp_path, [{"name": "Vaulttec", "ats": "greenhouse", "board_id": "vaulttec"}])
    collector = CompanyBoardsCollector(config_path=config, db_path=tmp_path / "jobs.db")
    assert len(collector.collect()) == 1


def test_404_continues_with_other_companies(tmp_path, monkeypatch):
    def fake_get(url, params=None, headers=None, timeout=None):
        if "boards-api.greenhouse.io" in url:
            return FakeResponse(404)
        if "api.lever.co" in url:
            return FakeResponse(200, _load("lever_postings.json"))
        return FakeResponse(404)

    monkeypatch.setattr("agent.collectors.company_boards.requests.get", fake_get)
    config = _write_config(
        tmp_path,
        [
            {"name": "GoneCo", "ats": "greenhouse", "board_id": "gone"},
            {"name": "FictionalCo", "ats": "lever", "board_id": "fictionalco"},
        ],
    )
    collector = CompanyBoardsCollector(config_path=config, db_path=tmp_path / "jobs.db")
    listings = collector.collect()
    assert [l.source for l in listings] == ["company:FictionalCo"]


# ---- Cooldown ----

def test_cooldown_skips_run(tmp_path, monkeypatch):
    _fake_get_dispatch(monkeypatch)
    config = _write_config(tmp_path, [{"name": "Vaulttec", "ats": "greenhouse", "board_id": "vaulttec"}])
    db_path = tmp_path / "jobs.db"
    record_collector_run("company_boards", db_path=db_path)
    collector = CompanyBoardsCollector(config_path=config, db_path=db_path)
    assert collector.collect() == []


def test_force_bypasses_cooldown(tmp_path, monkeypatch):
    _fake_get_dispatch(monkeypatch)
    config = _write_config(tmp_path, [{"name": "Vaulttec", "ats": "greenhouse", "board_id": "vaulttec"}])
    db_path = tmp_path / "jobs.db"
    record_collector_run("company_boards", db_path=db_path)
    collector = CompanyBoardsCollector(config_path=config, db_path=db_path, force=True)
    assert len(collector.collect()) == 2


def test_no_previous_run_collects(tmp_path, monkeypatch):
    _fake_get_dispatch(monkeypatch)
    config = _write_config(tmp_path, [{"name": "Vaulttec", "ats": "greenhouse", "board_id": "vaulttec"}])
    collector = CompanyBoardsCollector(config_path=config, db_path=tmp_path / "jobs.db")
    assert len(collector.collect()) == 2


def test_collect_records_run(tmp_path, monkeypatch):
    _fake_get_dispatch(monkeypatch)
    config = _write_config(tmp_path, [{"name": "Vaulttec", "ats": "greenhouse", "board_id": "vaulttec"}])
    db_path = tmp_path / "jobs.db"
    collector = CompanyBoardsCollector(config_path=config, db_path=db_path)
    collector.collect()
    assert get_last_collector_run("company_boards", db_path=db_path)
