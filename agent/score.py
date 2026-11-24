"""LLM Scoring module using LiteLLM with fallback and retry resilience."""

import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Optional
from dotenv import load_dotenv
import litellm

from agent.config import load_scoring_config, load_text
from agent.models import Listing
from agent.storage import (
    DEFAULT_DB_PATH,
    count_skipped_max_attempts,
    fetch_passed_unscored_listings,
    init_db,
    migrate_bad_scored_listings,
    record_scoring_failure,
    save_score,
)

# Load environment variables (.env)
load_dotenv()


class FatalScoringError(Exception):
    """Fatal error (authentication, 404, invalid model) that aborts the run immediately."""
    pass


def is_fatal_error(exc: Exception) -> bool:
    """Check if exception is a fatal config or authentication error."""
    if isinstance(exc, FatalScoringError):
        return True
    if isinstance(exc, (
        litellm.exceptions.AuthenticationError,
        litellm.exceptions.NotFoundError,
        litellm.exceptions.PermissionDeniedError,
    )):
        return True
    status_code = getattr(exc, "status_code", None)
    if status_code in (401, 403, 404):
        return True
    msg = str(exc).lower()
    if any(k in msg for k in ["missing gemini api key", "missing api key", "invalid api key", "not_found", "is no longer available"]):
        if not any(trans in msg for trans in ["503", "429", "timeout"]):
            return True
    return False


def classify_error_type(exc: Exception) -> str:
    """Classify transient error type."""
    if isinstance(exc, litellm.exceptions.RateLimitError):
        return "rate_limit_429"
    if isinstance(exc, litellm.exceptions.ServiceUnavailableError):
        return "service_unavailable_503"
    if isinstance(exc, litellm.exceptions.Timeout):
        return "timeout"
    if isinstance(exc, (litellm.exceptions.APIConnectionError, ConnectionError)):
        return "connection_error"
    if isinstance(exc, (ValueError, json.JSONDecodeError)):
        return "invalid_json"
    msg = str(exc).lower()
    if "429" in msg:
        return "rate_limit_429"
    if "503" in msg:
        return "service_unavailable_503"
    if "timeout" in msg:
        return "timeout"
    if "json" in msg:
        return "invalid_json"
    return "transient_error"


def build_scoring_prompt(
    listing: Listing,
    resume_text: str,
    profile_yaml_text: str,
) -> str:
    """Construct strict scoring prompt comparing listing against resume and profile."""
    return f"""You are an IT recruiting evaluator scoring job listings for candidate Bartosz Tyński.
Your task is to objectively evaluate how well the following job listing matches the candidate's verified profile and resume.

### CANDIDATE RESUME (Source of Truth):
{resume_text}

### STRUCTURED PROFILE:
{profile_yaml_text}

### JOB LISTING TO EVALUATE:
Title: {listing.title}
Company: {listing.company}
Location: {listing.location} (Remote: {listing.remote})
Work Type: {listing.work_type or 'Not specified'}
Salary: {listing.salary or 'Not specified'}
Description:
{listing.description}

### RULES & CONSTRAINTS:
1. Strict source of truth: compare ONLY against the candidate's verified experience in the resume and profile. NEVER invent or assume skills, tools, or years of experience.
2. Scoring rubric (0.0 to 10.0 scale):
   - skills_match (0.0 - 10.0):
     * 9.0 - 10.0: Strong match with candidate's primary stack (Python, SQL, ETL, Cloud Azure/AWS, REST APIs, Pytest).
     * 7.0 - 8.5: Matches primary language/framework, but lacks secondary required tools (e.g., Docker, dbt, Snowflake).
     * 4.0 - 6.5: Partial skill match; core stack differs or missing multiple essential technologies.
     * 0.0 - 3.5: Minimal or no relevant technical overlap.
   - experience_fit (0.0 - 10.0):
     * The candidate has ~4.5 years of experience (Mid-to-Senior level).
     * 8.0 - 9.5: Candidate meets or exceeds the required experience level (Junior, Mid, Regular ~0-5 years) — fully capable of executing the work. Exceeding requirements is a positive fit, never a disqualifier.
     * 7.0 - 8.0: Senior level (~5+ years) where candidate's ~4.5 years is a close stretch.
     * 1.0 - 3.0: Lead / Principal / Director (>8-10 years) where candidate lacks required depth.
   - role_fit (0.0 - 10.0):
     * 8.0 - 10.0: Core target engineering roles (Python Engineer, Solutions Engineer, Data Engineer, Applied AI, ETL), regardless of junior/regular level.
     * 6.0 - 7.5: Adjacent technical delivery or cloud roles.
     * 0.0 - 4.0: Unrelated specializations (Frontend, Mobile, QA, Sales, Management).
   - gaps: Array of strings representing specific required technologies, tools, or domain experience that the candidate lacks.
   - reason: Exactly one concise sentence explaining the score and key match/gap factor.
3. You must respond with a SINGLE valid JSON object ONLY. No markdown wrapper, no conversational preamble, no trailing text.

Expected JSON format:
{{
  "skills_match": 8.0,
  "experience_fit": 9.0,
  "role_fit": 8.5,
  "gaps": ["Docker"],
  "reason": "Strong Python skills and exceeds experience requirements; minor gap in Docker."
}}
"""


def _parse_and_validate_score_json(raw_text: str) -> dict[str, Any]:
    """Parse raw LLM response text and validate required fields and ranges."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
        cleaned = cleaned.strip()

    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("Response is not a JSON object")

    required_numeric = ["skills_match", "experience_fit", "role_fit"]
    for key in required_numeric:
        if key not in data:
            raise ValueError(f"Missing required numeric field: '{key}'")
        try:
            val = float(data[key])
            if not (0.0 <= val <= 10.0):
                raise ValueError(f"Field '{key}' ({val}) is out of range 0.0 - 10.0")
            data[key] = round(val, 2)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid value for '{key}': {e}")

    if "score" in data:
        try:
            raw_score = float(data["score"])
            if not (0.0 <= raw_score <= 10.0):
                raise ValueError(f"Field 'score' ({raw_score}) is out of range 0.0 - 10.0")
        except (TypeError, ValueError) as e:
            if "out of range" in str(e):
                raise
            raise ValueError(f"Invalid value for 'score': {e}")

    # Deterministic weighted composite calculation: (0.4 * skills_match) + (0.3 * role_fit) + (0.3 * experience_fit)
    data["score"] = round(
        (0.4 * data["skills_match"]) + (0.3 * data["role_fit"]) + (0.3 * data["experience_fit"]),
        2,
    )

    if "gaps" not in data or not isinstance(data["gaps"], list):
        raise ValueError("Missing or invalid 'gaps' array")
    data["gaps"] = [str(g).strip() for g in data["gaps"] if str(g).strip()]

    if "reason" not in data or not isinstance(data["reason"], str) or not data["reason"].strip():
        raise ValueError("Missing or empty 'reason' string")
    data["reason"] = data["reason"].strip()

    return data


def _execute_completion(
    call_kwargs: dict[str, Any],
    num_retries: int = 3,
    retry_backoff: float = 2.0,
) -> Any:
    """Execute litellm.completion with exponential backoff on transient errors."""
    last_exc = None
    for attempt in range(num_retries + 1):
        try:
            return litellm.completion(**call_kwargs)
        except Exception as e:
            last_exc = e
            if is_fatal_error(e):
                raise
            if attempt < num_retries:
                sleep_sec = retry_backoff * (2 ** attempt)
                time.sleep(sleep_sec)
    if last_exc:
        raise last_exc


def _invoke_completion_with_validation(
    model: str,
    prompt: str,
    temperature: Optional[float] = 0.0,
    seed: Optional[int] = 42,
    num_retries: int = 3,
    retry_backoff: float = 2.0,
    prompt_version: str = "v1.1",
) -> tuple[dict[str, Any], int, int, float]:
    """Execute completion call with exponential retry for 503/429 and 1 retry on malformed JSON."""
    messages = [{"role": "user", "content": prompt}]
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0

    call_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "drop_params": True,
    }
    if temperature is not None:
        call_kwargs["temperature"] = temperature
        call_kwargs["allowed_openai_params"] = ["temperature"]
    if seed is not None:
        call_kwargs["seed"] = seed

    # Attempt 1
    resp = _execute_completion(
        call_kwargs=call_kwargs,
        num_retries=num_retries,
        retry_backoff=retry_backoff,
    )

    if hasattr(resp, "usage") and resp.usage:
        total_prompt_tokens += getattr(resp.usage, "prompt_tokens", 0) or 0
        total_completion_tokens += getattr(resp.usage, "completion_tokens", 0) or 0
    try:
        cost = litellm.completion_cost(completion_response=resp)
        if cost:
            total_cost += float(cost)
    except Exception:
        pass

    raw_text = resp.choices[0].message.content or ""

    try:
        score_data = _parse_and_validate_score_json(raw_text)
        score_data["model"] = model
        score_data["prompt_version"] = prompt_version
        return score_data, total_prompt_tokens, total_completion_tokens, total_cost
    except Exception as first_err:
        # Corrective retry for invalid JSON
        retry_msg = (
            f"The previous output was invalid: {first_err}.\n"
            f"Please return ONLY a valid JSON object matching the exact requested schema."
        )
        messages.append({"role": "assistant", "content": raw_text})
        messages.append({"role": "user", "content": retry_msg})

        retry_kwargs = dict(call_kwargs)
        retry_kwargs["messages"] = messages
        retry_resp = _execute_completion(
            call_kwargs=retry_kwargs,
            num_retries=num_retries,
            retry_backoff=retry_backoff,
        )

        if hasattr(retry_resp, "usage") and retry_resp.usage:
            total_prompt_tokens += getattr(retry_resp.usage, "prompt_tokens", 0) or 0
            total_completion_tokens += getattr(retry_resp.usage, "completion_tokens", 0) or 0
        try:
            cost = litellm.completion_cost(completion_response=retry_resp)
            if cost:
                total_cost += float(cost)
        except Exception:
            pass

        raw_retry_text = retry_resp.choices[0].message.content or ""
        score_data = _parse_and_validate_score_json(raw_retry_text)
        score_data["model"] = model
        score_data["prompt_version"] = prompt_version
        return score_data, total_prompt_tokens, total_completion_tokens, total_cost


def score_listing(
    listing: Listing,
    resume_text: str,
    profile_yaml_text: str,
    model: str,
    temperature: Optional[float] = 0.0,
    seed: Optional[int] = 42,
    prompt_version: str = "v1.1",
    num_retries: int = 3,
    retry_backoff: float = 2.0,
) -> tuple[dict[str, Any], int, int, float]:
    """
    Score a single listing using cloud LLM via LiteLLM.
    Transient errors retry automatically with exponential backoff.
    Fatal config/auth errors immediately abort.
    """
    prompt = build_scoring_prompt(listing, resume_text, profile_yaml_text)

    try:
        return _invoke_completion_with_validation(
            model=model,
            prompt=prompt,
            temperature=temperature,
            seed=seed,
            num_retries=num_retries,
            retry_backoff=retry_backoff,
            prompt_version=prompt_version,
        )
    except Exception as exc:
        if is_fatal_error(exc):
            raise FatalScoringError(f"Fatal error on model '{model}': {exc}") from exc
        raise


def run_scoring(
    db_path: Path = DEFAULT_DB_PATH,
    limit: Optional[int] = None,
    config_path: Path = Path("config.yaml"),
    profile_path: Path = Path("data/profile.yaml"),
    resume_path: Path = Path("data/resume.md"),
) -> None:
    print(f"=== Job Search LLM Scoring Run ===")
    print(f"Database: {db_path}")

    init_db(db_path)
    # One-off migration check to fix any previously malformed 'scored' listings
    migrate_bad_scored_listings(db_path=db_path)

    cfg = load_scoring_config(config_path)
    prompt_version = cfg["prompt_version"]
    model = cfg.get("model", "gemini/gemini-3.5-flash-lite")
    num_retries = cfg.get("num_retries", 3)
    retry_backoff = cfg.get("retry_backoff", 2.0)
    max_attempts = cfg.get("max_attempts", 3)
    temperature = cfg.get("temperature", 0.0)
    seed = cfg.get("seed", 42)
    max_batch = limit if limit is not None else cfg["max_listings_per_run"]

    resume_text = load_text(resume_path)
    profile_yaml_text = load_text(profile_path)

    unscored = fetch_passed_unscored_listings(
        prompt_version=prompt_version,
        max_attempts=max_attempts,
        limit=max_batch,
        db_path=db_path,
    )
    skipped_max_attempts = count_skipped_max_attempts(
        prompt_version=prompt_version,
        max_attempts=max_attempts,
        db_path=db_path,
    )

    print(
        f"Found {len(unscored)} passed listing(s) to score "
        f"(model: {model}, prompt_version: {prompt_version})."
    )
    if not unscored:
        if skipped_max_attempts > 0:
            print(f"Notice: {skipped_max_attempts} listing(s) skipped because they exceeded max attempts ({max_attempts}).")
        print("No passed listings require scoring.")
        return

    scored_count = 0
    failed_count = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0

    for idx, listing in enumerate(unscored, start=1):
        print(f"[{idx}/{len(unscored)}] Scoring '{listing.title}' ({listing.company})...", end=" ", flush=True)
        try:
            score_data, p_tokens, c_tokens, cost = score_listing(
                listing=listing,
                resume_text=resume_text,
                profile_yaml_text=profile_yaml_text,
                model=model,
                temperature=temperature,
                seed=seed,
                prompt_version=prompt_version,
                num_retries=num_retries,
                retry_backoff=retry_backoff,
            )
            # Only set status to 'scored' on valid parsed score
            save_score(source=listing.source, id=listing.id, score_data=score_data, db_path=db_path)
            scored_count += 1
            total_prompt_tokens += p_tokens
            total_completion_tokens += c_tokens
            total_cost += cost
            print(f"done! (Score: {score_data['score']}/10 via {score_data['model']})")
        except Exception as e:
            if is_fatal_error(e):
                print(f"\n[FATAL ERROR] {e}")
                print("Aborting scoring run immediately. No listing status modified.")
                raise FatalScoringError(f"Fatal scoring error: {e}") from e

            # Transient error: mark only this listing as failed, keep status 'passed'
            failed_count += 1
            err_type = classify_error_type(e)
            record_scoring_failure(
                source=listing.source,
                id=listing.id,
                error_text=str(e),
                error_type=err_type,
                db_path=db_path,
            )
            print(f"failed! ({err_type}: {e}) - listing status remains 'passed'")

    print(f"\nScoring Summary:")
    print(f"  Successfully scored: {scored_count}")
    print(f"  Transient failures:  {failed_count}")
    if skipped_max_attempts > 0:
        print(f"  Skipped (>{max_attempts} failed attempts): {skipped_max_attempts}")
    print(f"  Tokens used:         {total_prompt_tokens + total_completion_tokens} (prompt: {total_prompt_tokens}, completion: {total_completion_tokens})")
    if total_cost > 0.0:
        print(f"  Estimated cost:      ${total_cost:.5f}")


def main() -> None:
    limit = None
    args = sys.argv[1:]
    if "--limit" in args:
        idx = args.index("--limit")
        if idx + 1 < len(args):
            limit = int(args[idx + 1])
    run_scoring(limit=limit)


if __name__ == "__main__":
    main()
