"""Tests for LLM scoring and terminal digest (mocking LiteLLM)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from litellm.exceptions import AuthenticationError, NotFoundError, ServiceUnavailableError
import pytest

from agent.digest import format_digest_table
from agent.models import Listing
from agent.score import (
    FatalScoringError,
    _execute_completion,
    _parse_and_validate_score_json,
    is_fatal_error,
    run_scoring,
    score_listing,
)
from agent.storage import (
    fetch_listings,
    fetch_top_scored_listings,
    init_db,
    migrate_bad_scored_listings,
    save_listings,
    save_score,
    update_listing_status,
)


def test_parse_and_validate_score_json_valid():
    sample = """{
        "score": 8.5,
        "skills_match": 8.0,
        "experience_fit": 9.0,
        "role_fit": 8.5,
        "gaps": ["Snowflake", "dbt"],
        "reason": "Strong match for Python ETL and cloud requirements."
    }"""
    data = _parse_and_validate_score_json(sample)
    assert data["score"] == 8.5
    assert data["skills_match"] == 8.0
    assert data["gaps"] == ["Snowflake", "dbt"]
    assert "Python ETL" in data["reason"]


def test_parse_and_validate_score_json_with_markdown_fence():
    sample = """```json
    {
        "score": 7.0,
        "skills_match": 7.0,
        "experience_fit": 7.0,
        "role_fit": 7.0,
        "gaps": [],
        "reason": "Decent fit."
    }
    ```"""
    data = _parse_and_validate_score_json(sample)
    assert data["score"] == 7.0
    assert data["gaps"] == []


def test_parse_and_validate_score_json_missing_keys():
    with pytest.raises(ValueError, match="Missing required numeric field"):
        _parse_and_validate_score_json('{"score": 5.0}')


def test_parse_and_validate_score_json_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        _parse_and_validate_score_json("""{
            "score": 15.0,
            "skills_match": 5.0,
            "experience_fit": 5.0,
            "role_fit": 5.0,
            "gaps": [],
            "reason": "Test"
        }""")


def make_mock_response(content: str):
    mock_resp = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_resp.choices = [mock_choice]
    mock_resp.usage.prompt_tokens = 250
    mock_resp.usage.completion_tokens = 80
    return mock_resp


@patch("litellm.completion")
@patch("litellm.completion_cost", return_value=0.00015)
def test_score_listing_success(mock_cost, mock_completion):
    mock_completion.return_value = make_mock_response("""{
        "score": 9.0,
        "skills_match": 9.5,
        "experience_fit": 8.5,
        "role_fit": 9.0,
        "gaps": [],
        "reason": "Great match for Python solutions engineer."
    }""")

    listing = Listing(
        id="test-1",
        source="justjoin_it",
        title="Solutions Engineer",
        company="AI Corp",
        location="Warszawa",
        remote=True,
        work_type="b2b",
        salary="25000 PLN",
        description="Python, SQL, integrations",
        url="https://example.com/1",
        posted_at="",
    )

    score_data, p_tokens, c_tokens, cost = score_listing(
        listing=listing,
        resume_text="Resume details",
        profile_yaml_text="Profile details",
        model="gemini/gemini-2.5-flash",
    )

    assert score_data["score"] == 9.0
    assert score_data["skills_match"] == 9.5
    assert p_tokens == 250
    assert c_tokens == 80
    assert cost == 0.00015
    assert mock_completion.call_count == 1
    call_kwargs = mock_completion.call_args.kwargs
    assert call_kwargs.get("num_retries") == 3
    assert call_kwargs.get("retry_strategy") == "exponential_backoff_retry"


@patch("litellm.completion")
def test_score_listing_503_then_fallback_to_local_model(mock_completion):
    err_503 = ServiceUnavailableError(
        message="503 Service Unavailable",
        model="gemini/gemini-2.5-flash",
        llm_provider="gemini",
    )
    local_resp = make_mock_response("""{
        "score": 7.8,
        "skills_match": 8.0,
        "experience_fit": 7.5,
        "role_fit": 8.0,
        "gaps": ["dbt"],
        "reason": "Scored via fallback local model."
    }""")
    mock_completion.side_effect = [err_503, local_resp]

    listing = Listing(
        id="test-fallback",
        source="justjoin_it",
        title="Python Engineer",
        company="LocalCorp",
        location="Remote",
        remote=True,
        work_type="b2b",
        salary=None,
        description="Python and SQL",
        url="https://example.com/fallback",
        posted_at="",
    )

    score_data, _, _, _ = score_listing(
        listing=listing,
        resume_text="Resume",
        profile_yaml_text="Profile",
        model="gemini/gemini-2.5-flash",
        fallback_model="ollama/qwen3.5:4b",
        local_api_base="http://localhost:11434",
    )

    assert mock_completion.call_count == 2
    fallback_kwargs = mock_completion.call_args_list[1].kwargs
    assert fallback_kwargs["model"] == "ollama/qwen3.5:4b"
    assert fallback_kwargs["api_base"] == "http://localhost:11434"
    assert fallback_kwargs.get("thinking") == {"type": "disabled"}
    assert score_data["score"] == 7.8
    assert score_data["model"] == "ollama/qwen3.5:4b"


@patch("litellm.completion")
def test_score_listing_provider_local_direct(mock_completion):
    local_resp = make_mock_response("""{
        "score": 8.2,
        "skills_match": 8.0,
        "experience_fit": 8.5,
        "role_fit": 8.0,
        "gaps": [],
        "reason": "Direct local scoring."
    }""")
    mock_completion.return_value = local_resp

    listing = Listing(
        id="test-local-direct",
        source="test",
        title="Python Dev",
        company="A",
        location="Remote",
        remote=True,
        work_type="b2b",
        salary=None,
        description="Python",
        url="",
        posted_at="",
    )

    score_data, _, _, _ = score_listing(
        listing=listing,
        resume_text="Resume",
        profile_yaml_text="Profile",
        model="gemini/gemini-2.5-flash",
        fallback_model="ollama/qwen3.5:4b",
        local_api_base="http://localhost:11434",
        provider="local",
    )

    assert mock_completion.call_count == 1
    call_kwargs = mock_completion.call_args.kwargs
    assert call_kwargs["model"] == "ollama/qwen3.5:4b"
    assert call_kwargs["api_base"] == "http://localhost:11434"
    assert call_kwargs.get("thinking") == {"type": "disabled"}
    assert score_data["score"] == 8.2


@patch("litellm.completion")
@patch("litellm.completion_cost", return_value=0.0002)
def test_score_listing_retries_on_invalid_json(mock_cost, mock_completion):
    bad_resp = make_mock_response("I cannot provide a score right now.")
    good_resp = make_mock_response("""{
        "score": 7.5,
        "skills_match": 8.0,
        "experience_fit": 7.0,
        "role_fit": 7.5,
        "gaps": ["dbt"],
        "reason": "Solid match with minor dbt gap."
    }""")
    mock_completion.side_effect = [bad_resp, good_resp]

    listing = Listing(
        id="test-retry",
        source="justjoin_it",
        title="Data Engineer",
        company="Data Corp",
        location="Remote",
        remote=True,
        work_type="b2b",
        salary="20000 PLN",
        description="Python ETL",
        url="https://example.com/2",
        posted_at="",
    )

    score_data, _, _, _ = score_listing(
        listing=listing,
        resume_text="Resume",
        profile_yaml_text="Profile",
        model="gemini/gemini-2.5-flash",
        fallback_model=None,
    )

    assert mock_completion.call_count == 2
    assert score_data["score"] == 7.5
    assert score_data["gaps"] == ["dbt"]


@patch("litellm.completion")
def test_scoring_failure_leaves_status_passed(mock_completion, tmp_path: Path):
    """On scoring failure, listing status must remain 'passed' and error details recorded."""
    mock_completion.return_value = make_mock_response("Malformed json forever")

    db_file = tmp_path / "test_status_passed.db"
    init_db(db_file)

    l1 = Listing(
        id="fail-transient",
        source="test",
        title="Python Dev",
        company="A",
        location="Remote",
        remote=True,
        work_type="b2b",
        salary=None,
        description="Python",
        url="",
        posted_at="",
    )
    save_listings([l1], db_path=db_file)
    update_listing_status("test", "fail-transient", "passed", db_path=db_file)

    with patch("agent.score.DEFAULT_DB_PATH", db_file):
        run_scoring(db_path=db_file, limit=1)

    with sqlite3_connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status, score, scoring_attempts, scoring_error_type FROM listings WHERE id = 'fail-transient'")
        row = cursor.fetchone()
        assert row[0] == "passed"  # Status must remain passed!
        assert row[1] is None     # Score must be None
        assert row[2] == 1        # Attempts incremented
        assert row[3] == "invalid_json"


@patch("litellm.completion")
def test_fatal_error_aborts_run(mock_completion, tmp_path: Path):
    """Fatal auth or config error aborts the run immediately without touching status."""
    auth_err = AuthenticationError(
        message="Invalid API Key",
        model="gemini/gemini-2.5-flash",
        llm_provider="gemini",
    )
    mock_completion.side_effect = auth_err

    db_file = tmp_path / "test_fatal.db"
    init_db(db_file)

    l1 = Listing(
        id="item-fatal",
        source="test",
        title="Python Dev",
        company="A",
        location="Remote",
        remote=True,
        work_type="b2b",
        salary=None,
        description="Python",
        url="",
        posted_at="",
    )
    save_listings([l1], db_path=db_file)
    update_listing_status("test", "item-fatal", "passed", db_path=db_file)

    with pytest.raises(FatalScoringError):
        run_scoring(db_path=db_file, limit=1)

    # Verify status was NOT modified
    with sqlite3_connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status, scoring_attempts, scoring_error_text FROM listings WHERE id = 'item-fatal'")
        row = cursor.fetchone()
        assert row[0] == "passed"
        assert row[1] == 0  # Attempts counter untouched!
        assert row[2] is None


@patch("litellm.completion")
def test_transient_error_continues_with_next_listing(mock_completion, tmp_path: Path):
    """A transient error on one listing logs failure, keeps status passed, and continues."""
    err_503 = ServiceUnavailableError(
        message="503 Service Unavailable",
        model="gemini/gemini-2.5-flash",
        llm_provider="gemini",
    )
    good_resp = make_mock_response("""{
        "score": 8.0,
        "skills_match": 8.0,
        "experience_fit": 8.0,
        "role_fit": 8.0,
        "gaps": [],
        "reason": "Success"
    }""")
    # Listing 1 fails on primary and fallback, Listing 2 succeeds on primary
    mock_completion.side_effect = [err_503, err_503, good_resp]

    db_file = tmp_path / "test_continue.db"
    init_db(db_file)

    l1 = Listing(id="1", source="t", title="Dev 1", company="A", location="", remote=True, work_type="", salary="", description="", url="", posted_at="")
    l2 = Listing(id="2", source="t", title="Dev 2", company="B", location="", remote=True, work_type="", salary="", description="", url="", posted_at="")
    save_listings([l1, l2], db_path=db_file)
    update_listing_status("t", "1", "passed", db_path=db_file)
    update_listing_status("t", "2", "passed", db_path=db_file)

    run_scoring(db_path=db_file, limit=2)

    with sqlite3_connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, status, score, scoring_attempts FROM listings ORDER BY id")
        rows = cursor.fetchall()
        # Item 1 remained passed
        assert rows[0][0] == "1"
        assert rows[0][1] == "passed"
        assert rows[0][2] is None
        assert rows[0][3] == 1

        # Item 2 succeeded and was scored
        assert rows[1][0] == "2"
        assert rows[1][1] == "scored"
        assert rows[1][2] == 8.0


@patch("litellm.completion")
def test_skip_listings_after_max_attempts(mock_completion, tmp_path: Path):
    """Listings that have reached max_attempts are skipped."""
    db_file = tmp_path / "test_max_attempts.db"
    init_db(db_file)

    l1 = Listing(id="maxed", source="t", title="Dev", company="A", location="", remote=True, work_type="", salary="", description="", url="", posted_at="")
    save_listings([l1], db_path=db_file)
    update_listing_status("t", "maxed", "passed", db_path=db_file)

    with sqlite3_connect(db_file) as conn:
        conn.execute("UPDATE listings SET scoring_attempts = 3 WHERE id = 'maxed'")

    run_scoring(db_path=db_file, limit=5)
    # LLM should never be called because listing exceeded max_attempts=3
    assert mock_completion.call_count == 0


def test_migration_fixes_bad_scored_rows(tmp_path: Path):
    """One-off migration resets listings with status 'scored' and empty score or scoring_error."""
    db_file = tmp_path / "test_mig.db"
    init_db(db_file)

    l1 = Listing(id="1", source="t", title="No Score", company="A", location="", remote=True, work_type="", salary="", description="", url="", posted_at="")
    l2 = Listing(id="2", source="t", title="Has Error", company="B", location="", remote=True, work_type="", salary="", description="", url="", posted_at="")
    l3 = Listing(id="3", source="t", title="Valid Scored", company="C", location="", remote=True, work_type="", salary="", description="", url="", posted_at="")
    save_listings([l1, l2, l3], db_path=db_file)

    # 1. status=scored but score is NULL
    with sqlite3_connect(db_file) as conn:
        conn.execute("UPDATE listings SET status = 'scored', score = NULL WHERE id = '1'")
        # 2. status=scored with score but has rejection_reason with scoring_error
        conn.execute("UPDATE listings SET status = 'scored', score = 7.0, rejection_reason = 'scoring_error: 404' WHERE id = '2'")
        # 3. status=scored with score and no error
        conn.execute("UPDATE listings SET status = 'scored', score = 9.0, rejection_reason = NULL WHERE id = '3'")

    changed = migrate_bad_scored_listings(db_file)
    assert changed == 2

    # Verify backup exists
    assert (tmp_path / "test_mig.db.bak").exists()

    with sqlite3_connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, status, score, scoring_attempts FROM listings ORDER BY id")
        rows = cursor.fetchall()
        # Item 1 reset to passed
        assert rows[0][0] == "1" and rows[0][1] == "passed" and rows[0][2] is None
        # Item 2 reset to passed
        assert rows[1][0] == "2" and rows[1][1] == "passed" and rows[1][2] is None and rows[1][3] >= 1
        # Item 3 remains scored
        assert rows[2][0] == "3" and rows[2][1] == "scored" and rows[2][2] == 9.0


@patch("litellm.completion")
def test_prompt_version_caching(mock_completion, tmp_path: Path):
    mock_completion.return_value = make_mock_response("""{
        "score": 8.0,
        "skills_match": 8.0,
        "experience_fit": 8.0,
        "role_fit": 8.0,
        "gaps": [],
        "reason": "Good."
    }""")

    db_file = tmp_path / "test_cache.db"
    init_db(db_file)

    l1 = Listing(
        id="cache-item",
        source="test",
        title="Python Engineer",
        company="A",
        location="Remote",
        remote=True,
        work_type="b2b",
        salary=None,
        description="Python",
        url="",
        posted_at="",
    )
    save_listings([l1], db_path=db_file)
    update_listing_status("test", "cache-item", "passed", db_path=db_file)

    # First run scores the listing
    run_scoring(db_path=db_file, limit=5)
    assert mock_completion.call_count == 1

    # Second run with same prompt version should skip (0 new calls)
    run_scoring(db_path=db_file, limit=5)
    assert mock_completion.call_count == 1


def test_digest_ranking_order(tmp_path: Path):
    db_file = tmp_path / "test_digest.db"
    init_db(db_file)

    l1 = Listing(id="1", source="t", title="Job Low", company="A", location="", remote=True, work_type="", salary="", description="", url="", posted_at="")
    l2 = Listing(id="2", source="t", title="Job High", company="B", location="", remote=True, work_type="", salary="", description="", url="", posted_at="")
    l3 = Listing(id="3", source="t", title="Job Mid", company="C", location="", remote=True, work_type="", salary="", description="", url="", posted_at="")
    save_listings([l1, l2, l3], db_path=db_file)

    save_score("t", "1", {"score": 6.5, "skills_match": 6, "experience_fit": 6, "role_fit": 6, "gaps": [], "reason": "Low", "model": "m", "prompt_version": "v1.0"}, db_path=db_file)
    save_score("t", "2", {"score": 9.5, "skills_match": 9, "experience_fit": 9, "role_fit": 9, "gaps": [], "reason": "High", "model": "m", "prompt_version": "v1.0"}, db_path=db_file)
    save_score("t", "3", {"score": 8.0, "skills_match": 8, "experience_fit": 8, "role_fit": 8, "gaps": [], "reason": "Mid", "model": "m", "prompt_version": "v1.0"}, db_path=db_file)

    top = fetch_top_scored_listings(limit=10, db_path=db_file)
    assert len(top) == 3
    assert top[0]["id"] == "2" and top[0]["score"] == 9.5
    assert top[1]["id"] == "3" and top[1]["score"] == 8.0
    assert top[2]["id"] == "1" and top[2]["score"] == 6.5

    table_text = format_digest_table(top)
    assert "Job High" in table_text
    assert "9.5" in table_text


@patch("litellm.completion")
def test_score_listing_no_temperature_when_unconfigured(mock_completion):
    mock_completion.return_value = make_mock_response("""{
        "score": 8.0,
        "skills_match": 8.0,
        "experience_fit": 8.0,
        "role_fit": 8.0,
        "gaps": [],
        "reason": "Good."
    }""")

    listing = Listing(
        id="test-no-temp",
        source="test",
        title="Python Dev",
        company="A",
        location="Remote",
        remote=True,
        work_type="b2b",
        salary=None,
        description="Python",
        url="",
        posted_at="",
    )

    score_listing(
        listing=listing,
        resume_text="Resume",
        profile_yaml_text="Profile",
        model="gemini/gemini-3.8-flash",
        temperature=None,
        fallback_model=None,
    )

    assert mock_completion.call_count == 1
    call_kwargs = mock_completion.call_args.kwargs
    assert "temperature" not in call_kwargs
    assert call_kwargs.get("drop_params") is True


@patch("litellm.completion")
def test_score_listing_passes_temperature_when_set(mock_completion):
    mock_completion.return_value = make_mock_response("""{
        "score": 8.0,
        "skills_match": 8.0,
        "experience_fit": 8.0,
        "role_fit": 8.0,
        "gaps": [],
        "reason": "Good."
    }""")

    listing = Listing(
        id="test-with-temp",
        source="test",
        title="Python Dev",
        company="A",
        location="Remote",
        remote=True,
        work_type="b2b",
        salary=None,
        description="Python",
        url="",
        posted_at="",
    )

    score_listing(
        listing=listing,
        resume_text="Resume",
        profile_yaml_text="Profile",
        model="gemini/gemini-3.8-flash",
        temperature=0.3,
        fallback_model=None,
    )

    assert mock_completion.call_count == 1
    call_kwargs = mock_completion.call_args.kwargs
    assert call_kwargs.get("temperature") == 0.3
    assert call_kwargs.get("drop_params") is True


@patch("litellm.completion")
def test_score_listing_passes_local_timeout(mock_completion):
    mock_completion.return_value = make_mock_response("""{
        "score": 8.0,
        "skills_match": 8.0,
        "experience_fit": 8.0,
        "role_fit": 8.0,
        "gaps": [],
        "reason": "Good."
    }""")

    listing = Listing(
        id="test-local-timeout",
        source="test",
        title="Python Dev",
        company="A",
        location="Remote",
        remote=True,
        work_type="b2b",
        salary=None,
        description="Python",
        url="",
        posted_at="",
    )

    score_listing(
        listing=listing,
        resume_text="Resume",
        profile_yaml_text="Profile",
        model="gemini/gemini-2.5-flash",
        fallback_model="ollama/qwen3.5:4b",
        provider="local",
        local_timeout=45.0,
    )

    assert mock_completion.call_count == 1
    call_kwargs = mock_completion.call_args.kwargs
    assert call_kwargs.get("timeout") == 45.0


@patch("litellm.completion")
def test_score_listing_fallback_passes_local_timeout(mock_completion):
    err_503 = ServiceUnavailableError(
        message="503 Service Unavailable",
        model="gemini/gemini-2.5-flash",
        llm_provider="gemini",
    )
    local_resp = make_mock_response("""{
        "score": 7.5,
        "skills_match": 7.5,
        "experience_fit": 7.5,
        "role_fit": 7.5,
        "gaps": [],
        "reason": "Fallback good."
    }""")
    mock_completion.side_effect = [err_503, local_resp]

    listing = Listing(
        id="test-fallback-timeout",
        source="test",
        title="Python Dev",
        company="A",
        location="Remote",
        remote=True,
        work_type="b2b",
        salary=None,
        description="Python",
        url="",
        posted_at="",
    )

    score_listing(
        listing=listing,
        resume_text="Resume",
        profile_yaml_text="Profile",
        model="gemini/gemini-2.5-flash",
        fallback_model="ollama/qwen3.5:4b",
        local_timeout=30.0,
    )

    assert mock_completion.call_count == 2
    fallback_kwargs = mock_completion.call_args_list[1].kwargs
    assert fallback_kwargs.get("timeout") == 30.0


@patch("litellm.completion")
def test_execute_completion_fallback_when_tenacity_fails(mock_completion):
    tenacity_err = Exception("tenacity import failed please run `pip install tenacity`")
    err_503 = ServiceUnavailableError(
        message="503 Service Unavailable",
        model="gemini/gemini-2.5-flash",
        llm_provider="gemini",
    )
    good_resp = make_mock_response('{"score": 9.0}')
    mock_completion.side_effect = [tenacity_err, err_503, good_resp]

    kwargs = {
        "model": "gemini/gemini-2.5-flash",
        "messages": [{"role": "user", "content": "hi"}],
        "num_retries": 3,
        "retry_strategy": "exponential_backoff_retry",
    }
    resp = _execute_completion(
        call_kwargs=kwargs,
        num_retries=2,
        retry_backoff=0.01,
        has_tenacity=True,
    )
    assert resp == good_resp
    assert mock_completion.call_count == 3


def sqlite3_connect(path: Path):
    import sqlite3
    return sqlite3.connect(path)
