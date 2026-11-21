"""LLM Scoring module using LiteLLM."""

import json
from pathlib import Path
import re
import sys
from typing import Any, Optional
from dotenv import load_dotenv
import litellm

from agent.config import load_scoring_config, load_text, load_yaml
from agent.models import Listing
from agent.storage import (
    DEFAULT_DB_PATH,
    fetch_passed_unscored_listings,
    init_db,
    save_score,
    save_scoring_error,
)

# Load environment variables (.env)
load_dotenv()


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
   - skills_match (0.0 - 10.0): Match with Python, SQL, ETL, Cloud (Azure/AWS), REST APIs, Pytest, Data modelling, etc.
   - experience_fit (0.0 - 10.0): Candidate has ~4.5 years mid-to-senior software/data engineering experience.
   - role_fit (0.0 - 10.0): Alignment with Solutions Engineer, AI/Applied ML, Data Engineer, or customer-facing technical delivery.
   - score (0.0 - 10.0): Overall weighted match score: (0.4 * skills_match) + (0.3 * role_fit) + (0.3 * experience_fit).
   - gaps: Array of strings representing specific required technologies, tools, or domain experience that the candidate lacks.
   - reason: Exactly one concise sentence explaining the score and key match/gap factor.
3. You must respond with a SINGLE valid JSON object ONLY. No markdown wrapper, no conversational preamble, no trailing text.

Expected JSON format:
{{
  "score": 8.5,
  "skills_match": 8.0,
  "experience_fit": 9.0,
  "role_fit": 8.5,
  "gaps": ["Snowflake", "dbt"],
  "reason": "Strong match for Python ETL and cloud requirements; minor gap in dbt."
}}
"""


def _parse_and_validate_score_json(raw_text: str) -> dict[str, Any]:
    """Parse raw LLM response text and validate required fields and ranges."""
    cleaned = raw_text.strip()
    # Strip markdown code blocks if model included them
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
        cleaned = cleaned.strip()

    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("Response is not a JSON object")

    required_numeric = ["score", "skills_match", "experience_fit", "role_fit"]
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

    if "gaps" not in data or not isinstance(data["gaps"], list):
        raise ValueError("Missing or invalid 'gaps' array")
    data["gaps"] = [str(g).strip() for g in data["gaps"] if str(g).strip()]

    if "reason" not in data or not isinstance(data["reason"], str) or not data["reason"].strip():
        raise ValueError("Missing or empty 'reason' string")
    data["reason"] = data["reason"].strip()

    return data


def score_listing(
    listing: Listing,
    resume_text: str,
    profile_yaml_text: str,
    model: str,
    temperature: Optional[float] = None,
    prompt_version: str = "v1.0",
) -> tuple[dict[str, Any], int, int, float]:
    """
    Call LiteLLM to score a single listing.
    Retries once on invalid JSON.
    Returns: (score_data, prompt_tokens, completion_tokens, estimated_cost)
    """
    prompt = build_scoring_prompt(listing, resume_text, profile_yaml_text)
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

    # First attempt
    resp = litellm.completion(**call_kwargs)

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
        # Retry once with explicit error feedback
        retry_msg = (
            f"The previous output was invalid: {first_err}.\n"
            f"Please return ONLY a valid JSON object matching the exact requested schema."
        )
        messages.append({"role": "assistant", "content": raw_text})
        messages.append({"role": "user", "content": retry_msg})

        retry_kwargs = dict(call_kwargs)
        retry_kwargs["messages"] = messages
        retry_resp = litellm.completion(**retry_kwargs)

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
    cfg = load_scoring_config(config_path)
    prompt_version = cfg["prompt_version"]
    model = cfg["model"]
    temperature = cfg["temperature"]
    max_batch = limit if limit is not None else cfg["max_listings_per_run"]

    resume_text = load_text(resume_path)
    profile_yaml_text = load_text(profile_path)

    unscored = fetch_passed_unscored_listings(
        prompt_version=prompt_version,
        limit=max_batch,
        db_path=db_path,
    )

    print(f"Found {len(unscored)} passed listing(s) to score (model: {model}, prompt_version: {prompt_version}).")
    if not unscored:
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
                prompt_version=prompt_version,
            )
            save_score(source=listing.source, id=listing.id, score_data=score_data, db_path=db_path)
            scored_count += 1
            total_prompt_tokens += p_tokens
            total_completion_tokens += c_tokens
            total_cost += cost
            print(f"done! (Score: {score_data['score']}/10)")
        except Exception as e:
            failed_count += 1
            save_scoring_error(source=listing.source, id=listing.id, error_message=str(e), db_path=db_path)
            print(f"failed! ({e})")

    print(f"\nScoring Summary:")
    print(f"  Successfully scored: {scored_count}")
    print(f"  Failed / Skipped:   {failed_count}")
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
