"""Tests for duplicate detection module (agent.dedup)."""

from pathlib import Path
import sqlite3

from agent.dedup import (
    build_shingles,
    check_is_duplicate,
    jaccard_similarity,
    normalize_company,
    normalize_description,
    normalize_title,
    run_dedup,
)
from agent.digest import format_listing_block
from agent.models import Listing
from agent.storage import (
    fetch_duplicate_urls_for_canonical,
    fetch_listings,
    fetch_top_scored_listings,
    init_db,
    save_listings,
    save_score,
)


def _create_test_config(tmp_path: Path, only_remote: bool = True, threshold: float = 0.85) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"""
dedup:
  enabled: true
  only_remote: {str(only_remote).lower()}
  similarity_threshold: {threshold}
  shingle_size: 5
""",
        encoding="utf-8",
    )
    return cfg


def test_normalize_company_suffixes():
    assert normalize_company("Acme Sp. z o.o.") == "acme"
    assert normalize_company("Acme sp. z o. o.") == "acme"
    assert normalize_company("Acme Spółka z o.o.") == "acme"
    assert normalize_company("Acme Spółka z ograniczoną odpowiedzialnością") == "acme"
    assert normalize_company("Acme S.A.") == "acme"
    assert normalize_company("Acme Sp. k.") == "acme"
    assert normalize_company("Acme LLC") == "acme"
    assert normalize_company("Acme Ltd.") == "acme"
    assert normalize_company("Acme Inc.") == "acme"
    assert normalize_company("Acme GmbH") == "acme"
    assert normalize_company("Acme B.V.") == "acme"


def test_normalize_description_html_and_punctuation():
    raw = "<p>Senior <b>Python</b> Engineer with 5+ years of experience! &amp; cloud skills.</p>"
    norm = normalize_description(raw)
    assert norm == "senior python engineer with 5 years of experience cloud skills"


def test_identical_text_different_urls_exact_hash_duplicate(tmp_path):
    db_path = tmp_path / "jobs.db"
    cfg_path = _create_test_config(tmp_path)
    init_db(db_path)

    desc = "We are seeking a Senior Python Engineer to design, build, and maintain our scalable cloud data platform using Azure."
    l1 = Listing(
        id="nfj-warszawa-1",
        source="nofluffjobs",
        title="Senior Python Engineer",
        company="DataFlow Tech Sp. z o.o.",
        location="Warszawa",
        remote=True,
        work_type="b2b",
        salary="25000 PLN",
        description=desc,
        url="https://nofluffjobs.com/job/senior-python-engineer-warszawa",
        posted_at="2026-11-20T10:00:00Z",
    )
    l2 = Listing(
        id="nfj-krakow-2",
        source="nofluffjobs",
        title="Senior Python Engineer",
        company="DataFlow Tech LLC",
        location="Kraków",
        remote=True,
        work_type="b2b",
        salary="25000 PLN",
        description=desc,
        url="https://nofluffjobs.com/job/senior-python-engineer-krakow",
        posted_at="2026-11-20T11:00:00Z",
    )

    save_listings([l1, l2], db_path=db_path)
    # Set status to 'passed' (as if rule-based filter ran)
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE listings SET status = 'passed'")

    result = run_dedup(db_path=db_path, config_path=cfg_path)
    assert result["duplicates_found"] == 1
    assert result["llm_calls_saved"] == 1

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        r1 = conn.execute("SELECT * FROM listings WHERE id = 'nfj-warszawa-1'").fetchone()
        r2 = conn.execute("SELECT * FROM listings WHERE id = 'nfj-krakow-2'").fetchone()

        # Earlier posted_at should be canonical
        assert r1["status"] == "passed"
        assert r1["duplicate_of"] is None

        # Later posted_at should be marked duplicate
        assert r2["status"] == "duplicate"
        assert r2["duplicate_of"] == "nfj-warszawa-1"
        assert r2["similarity_score"] == 1.0

    dupe_urls = fetch_duplicate_urls_for_canonical("nfj-warszawa-1", db_path=db_path)
    assert dupe_urls == ["https://nofluffjobs.com/job/senior-python-engineer-krakow"]


def test_near_identical_text_jaccard_duplicate(tmp_path):
    db_path = tmp_path / "jobs.db"
    cfg_path = _create_test_config(tmp_path, threshold=0.85)
    init_db(db_path)

    # Base description of realistic length
    words = (
        "We are looking for an experienced Senior Python Engineer to join our high performing data team. "
        "In this role you will be responsible for designing, building, and deploying highly scalable ETL and data pipelines in the cloud. "
        "You will collaborate closely with machine learning engineers and product managers to deliver robust AI solutions and analytics. "
        "Requirements include at least five years of professional commercial experience with Python 3, strong SQL knowledge, Docker, "
        "and hands-on cloud experience with Azure or AWS. We offer competitive B2B rates, flexible remote working hours, and modern tooling."
    ).split()
    desc1 = " ".join(words)
    # Near identical: replace one minor word
    words2 = list(words)
    words2[-3] = "collaborative"
    desc2 = " ".join(words2)

    l1 = Listing(
        id="nfj-ai-1",
        source="nofluffjobs",
        title="AI Solutions Engineer",
        company="Cognitive Labs",
        location="Remote",
        remote=True,
        work_type="b2b",
        salary=None,
        description=desc1,
        url="https://nofluffjobs.com/job/ai-solutions-engineer-1",
        posted_at="2026-11-21T09:00:00Z",
    )
    l2 = Listing(
        id="nfj-ai-2",
        source="nofluffjobs",
        title="AI Solutions Engineer",
        company="Cognitive Labs",
        location="Remote",
        remote=True,
        work_type="b2b",
        salary=None,
        description=desc2,
        url="https://nofluffjobs.com/job/ai-solutions-engineer-2",
        posted_at="2026-11-21T10:00:00Z",
    )

    save_listings([l1, l2], db_path=db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE listings SET status = 'passed'")

    result = run_dedup(db_path=db_path, config_path=cfg_path)
    assert result["duplicates_found"] == 1

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        r2 = conn.execute("SELECT * FROM listings WHERE id = 'nfj-ai-2'").fetchone()
        assert r2["status"] == "duplicate"
        assert r2["duplicate_of"] == "nfj-ai-1"
        assert r2["similarity_score"] >= 0.85


def test_different_jobs_at_same_company_not_merged(tmp_path):
    db_path = tmp_path / "jobs.db"
    cfg_path = _create_test_config(tmp_path)
    init_db(db_path)

    l1 = Listing(
        id="job-python",
        source="nofluffjobs",
        title="Python Data Engineer",
        company="TechCorp",
        location="Remote",
        remote=True,
        work_type="b2b",
        salary=None,
        description="Build scalable data pipelines with Python, Spark, and Azure SQL database.",
        url="https://example.com/python",
        posted_at="2026-11-20T10:00:00Z",
    )
    l2 = Listing(
        id="job-frontend",
        source="nofluffjobs",
        title="Frontend React Developer",
        company="TechCorp",
        location="Remote",
        remote=True,
        work_type="b2b",
        salary=None,
        description="Develop modern web user interfaces using TypeScript, React, Next.js, and CSS.",
        url="https://example.com/frontend",
        posted_at="2026-11-20T11:00:00Z",
    )

    save_listings([l1, l2], db_path=db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE listings SET status = 'passed'")

    result = run_dedup(db_path=db_path, config_path=cfg_path)
    assert result["duplicates_found"] == 0

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT status, duplicate_of FROM listings").fetchall()
        for r in rows:
            assert r["status"] == "passed"
            assert r["duplicate_of"] is None


def test_non_remote_listings_not_merged_when_only_remote_true(tmp_path):
    db_path = tmp_path / "jobs.db"
    cfg_path = _create_test_config(tmp_path, only_remote=True)
    init_db(db_path)

    desc = "On-site office role maintaining internal reporting systems and infrastructure."
    l1 = Listing(
        id="onsite-warszawa",
        source="nofluffjobs",
        title="Data Specialist",
        company="Bank Polska",
        location="Warszawa",
        remote=False,
        work_type="employment",
        salary=None,
        description=desc,
        url="https://example.com/onsite-waw",
        posted_at="2026-11-20T10:00:00Z",
    )
    l2 = Listing(
        id="onsite-krakow",
        source="nofluffjobs",
        title="Data Specialist",
        company="Bank Polska",
        location="Kraków",
        remote=False,
        work_type="employment",
        salary=None,
        description=desc,
        url="https://example.com/onsite-krk",
        posted_at="2026-11-20T11:00:00Z",
    )

    save_listings([l1, l2], db_path=db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE listings SET status = 'passed'")

    # When only_remote is true, neither is a candidate
    result = run_dedup(db_path=db_path, config_path=cfg_path)
    assert result["duplicates_found"] == 0

    # When only_remote is false, they should be merged
    cfg_all = _create_test_config(tmp_path, only_remote=False)
    result2 = run_dedup(db_path=db_path, config_path=cfg_all)
    assert result2["duplicates_found"] == 1


def test_canonical_selection_prefers_scored_listing(tmp_path):
    db_path = tmp_path / "jobs.db"
    cfg_path = _create_test_config(tmp_path)
    init_db(db_path)

    desc = "Develop cloud microservices and distributed ETL architectures in Python."
    l_scored = Listing(
        id="scored-offer-1",
        source="nofluffjobs",
        title="Python Cloud Engineer",
        company="CloudScale",
        location="Remote",
        remote=True,
        work_type="b2b",
        salary=None,
        description=desc,
        url="https://example.com/scored-1",
        posted_at="2026-11-22T10:00:00Z",  # Note: posted LATER than new candidate
    )
    l_new = Listing(
        id="new-candidate-2",
        source="justjoin_it",
        title="Python Cloud Engineer",
        company="CloudScale Sp. z o.o.",
        location="Remote",
        remote=True,
        work_type="b2b",
        salary=None,
        description=desc,
        url="https://example.com/new-2",
        posted_at="2026-11-20T10:00:00Z",  # Earlier posted_at!
    )

    save_listings([l_scored, l_new], db_path=db_path)
    save_score(
        source="nofluffjobs",
        id="scored-offer-1",
        score_data={
            "score": 9.0,
            "skills_match": 9.0,
            "experience_fit": 9.0,
            "role_fit": 9.0,
            "gaps": [],
            "reason": "Great match",
        },
        db_path=db_path,
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE listings SET status = 'passed' WHERE id = 'new-candidate-2'")

    result = run_dedup(db_path=db_path, config_path=cfg_path)
    assert result["duplicates_found"] == 1

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        scored_row = conn.execute("SELECT status, score FROM listings WHERE id = 'scored-offer-1'").fetchone()
        new_row = conn.execute("SELECT status, duplicate_of, similarity_score FROM listings WHERE id = 'new-candidate-2'").fetchone()

        # Scored listing must NEVER be modified or rescored
        assert scored_row["status"] == "scored"
        assert scored_row["score"] == 9.0

        # New passed listing must be marked duplicate of scored listing
        assert new_row["status"] == "duplicate"
        assert new_row["duplicate_of"] == "scored-offer-1"
        assert new_row["similarity_score"] == 1.0


def test_dedup_idempotency(tmp_path):
    db_path = tmp_path / "jobs.db"
    cfg_path = _create_test_config(tmp_path)
    init_db(db_path)

    desc = "Identical offer posted across multiple Polish job portals for remote work."
    l1 = Listing(
        id="idemp-1",
        source="nofluffjobs",
        title="Backend Python Developer",
        company="SoftServe",
        location="Remote",
        remote=True,
        work_type="b2b",
        salary=None,
        description=desc,
        url="https://example.com/idemp-1",
        posted_at="2026-11-20T10:00:00Z",
    )
    l2 = Listing(
        id="idemp-2",
        source="nofluffjobs",
        title="Backend Python Developer",
        company="SoftServe",
        location="Remote",
        remote=True,
        work_type="b2b",
        salary=None,
        description=desc,
        url="https://example.com/idemp-2",
        posted_at="2026-11-20T11:00:00Z",
    )

    save_listings([l1, l2], db_path=db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE listings SET status = 'passed'")

    # Run 1
    res1 = run_dedup(db_path=db_path, config_path=cfg_path)
    assert res1["duplicates_found"] == 1

    # Run 2 immediately after
    res2 = run_dedup(db_path=db_path, config_path=cfg_path)
    assert res2["duplicates_found"] == 0

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        r1 = conn.execute("SELECT status FROM listings WHERE id = 'idemp-1'").fetchone()
        r2 = conn.execute("SELECT status, duplicate_of FROM listings WHERE id = 'idemp-2'").fetchone()
        assert r1["status"] == "passed"
        assert r2["status"] == "duplicate"
        assert r2["duplicate_of"] == "idemp-1"


def test_digest_displays_duplicate_urls():
    item = {
        "id": "canon-100",
        "title": "Senior Solutions Engineer",
        "company": "DeepAI",
        "location": "Warszawa",
        "remote": True,
        "score": 9.5,
        "pipeline_status": "new",
        "url": "https://example.com/main",
        "duplicate_urls": [
            "https://example.com/alt-krakow",
            "https://example.com/alt-wroclaw",
        ],
    }
    lines = format_listing_block(item)
    block_text = "\n".join(lines)
    assert "Also posted at (2 other URLs):" in block_text
    assert "https://example.com/alt-krakow" in block_text
    assert "https://example.com/alt-wroclaw" in block_text
