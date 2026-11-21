"""Tests for LLM scoring and terminal digest (mocking LiteLLM)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from agent.digest import format_digest_table
from agent.models import Listing
from agent.score import _parse_and_validate_score_json, run_scoring, score_listing
from agent.storage import fetch_top_scored_listings, init_db, save_listings, save_score


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


@patch("litellm.completion")
@patch("litellm.completion_cost", return_value=0.0002)
def test_score_listing_retries_on_invalid_json(mock_cost, mock_completion):
    # First call returns invalid JSON, second call returns valid JSON
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
    )

    assert mock_completion.call_count == 2
    assert score_data["score"] == 7.5
    assert score_data["gaps"] == ["dbt"]


@patch("litellm.completion")
def test_scoring_failure_does_not_crash_run(mock_completion, tmp_path: Path):
    # Always fails
    mock_completion.return_value = make_mock_response("Malformed json forever")

    db_file = tmp_path / "test_score_fail.db"
    init_db(db_file)

    l1 = Listing(
        id="fail-1",
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

    # Mark as passed
    with patch("agent.storage.DEFAULT_DB_PATH", db_file):
        from agent.storage import update_listing_status
        update_listing_status("test", "fail-1", "passed", db_path=db_file)

        # Run scoring - must not raise exception
        run_scoring(db_path=db_file, limit=1)

        # Listing should not be scored
        scored = fetch_top_scored_listings(limit=10, db_path=db_file)
        assert len(scored) == 0


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

    from agent.storage import update_listing_status
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
    )

    assert mock_completion.call_count == 1
    call_kwargs = mock_completion.call_args.kwargs
    assert call_kwargs.get("temperature") == 0.3
    assert call_kwargs.get("drop_params") is True

