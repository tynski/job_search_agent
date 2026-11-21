"""Rule-based filtering for job listings."""

from pathlib import Path
import re
import sys
from typing import Any, Optional

from agent.config import load_filter_config
from agent.models import Listing
from agent.storage import fetch_listings, get_db_connection, init_db, update_listing_status

DEFAULT_DB = Path("data/jobs.db")


def _contains_keyword(text: str, keyword: str) -> bool:
    """Case-insensitive word-boundary keyword check."""
    if not text or not keyword:
        return False
    # Use word boundaries; for symbols like c# or .net, adjust boundaries
    escaped = re.escape(keyword)
    pattern = rf"(?<!\w){escaped}(?!\w)"
    return bool(re.search(pattern, text, re.IGNORECASE))


def _extract_salary_max_pln(salary_text: Optional[str]) -> Optional[int]:
    """Extract max PLN salary amount if identifiable."""
    if not salary_text:
        return None
    # Look for PLN amounts
    if "pln" in salary_text.lower() or "zł" in salary_text.lower():
        numbers = re.findall(r"\b\d{1,3}(?:[ ,]\d{3})*\b", salary_text)
        clean_nums = []
        for n in numbers:
            try:
                val = int(n.replace(" ", "").replace(",", ""))
                if val >= 1000:  # Ignore days/hours, focus on monthly amounts
                    clean_nums.append(val)
            except ValueError:
                continue
        if clean_nums:
            return max(clean_nums)
    return None


def evaluate_listing(
    listing: Listing, filter_cfg: dict[str, Any]
) -> tuple[bool, Optional[str]]:
    """
    Evaluate listing against rule-based filter criteria.
    Returns (True, None) if passed, or (False, rejection_reason) if rejected.
    """
    title = listing.title or ""
    description = listing.description or ""
    location = listing.location or ""
    work_type = listing.work_type or ""
    salary = listing.salary or ""

    # 1. Exclude titles check
    exclude_cfg = filter_cfg.get("exclude", {})
    for excl_title in exclude_cfg.get("titles", []):
        if _contains_keyword(title, excl_title):
            return False, f"excluded_title: {excl_title}"

    # 2. Exclude description keywords check
    for excl_kw in exclude_cfg.get("keywords", []):
        if _contains_keyword(title, excl_kw) or _contains_keyword(description, excl_kw):
            return False, f"excluded_keyword: {excl_kw}"

    # 3. Location check
    loc_cfg = filter_cfg.get("location", {})
    allow_any_remote = loc_cfg.get("allow_any_remote", True)
    allowed_locations = [l.lower() for l in loc_cfg.get("allowed_countries_and_cities", [])]

    is_location_valid = False
    if listing.remote and allow_any_remote:
        is_location_valid = True
    else:
        loc_lower = location.lower()
        title_desc_lower = (title + " " + description).lower()
        # Check explicit location or remote mention in title/description
        for allowed in allowed_locations:
            if allowed in loc_lower or (allowed in ["remote", "zdalnie"] and allowed in title_desc_lower):
                is_location_valid = True
                break

    if not is_location_valid:
        return False, "location_mismatch"

    # 4. Work type check
    wt_cfg = filter_cfg.get("work_type", {})
    allow_missing_wt = wt_cfg.get("allow_missing", True)
    allowed_wts = [w.lower() for w in wt_cfg.get("allowed", [])]

    if not work_type:
        if not allow_missing_wt:
            return False, "work_type_missing"
    else:
        wt_lower = work_type.lower()
        if not any(allowed in wt_lower for allowed in allowed_wts):
            return False, "work_type_mismatch"

    # 5. Include profile keywords check (must match at least one in title or description)
    include_keywords = filter_cfg.get("include_keywords", [])
    has_keyword_match = False
    text_to_check = title + "\n" + description
    for kw in include_keywords:
        if _contains_keyword(text_to_check, kw):
            has_keyword_match = True
            break

    if not has_keyword_match:
        return False, "no_profile_keywords"

    # 6. Optional minimum salary check (skipped when salary is missing/unspecified)
    min_pln = filter_cfg.get("salary", {}).get("min_pln", 0) or 0
    if min_pln > 0 and salary:
        max_sal = _extract_salary_max_pln(salary)
        if max_sal is not None and max_sal < min_pln:
            return False, f"salary_below_minimum ({max_sal} < {min_pln} PLN)"

    return True, None


def run_filter(
    db_path: Path = DEFAULT_DB,
    config_path: Path = Path("config.yaml"),
    profile_path: Path = Path("data/profile.yaml"),
) -> None:
    print(f"=== Job Search Filter Run ===")
    print(f"Database: {db_path}")

    init_db(db_path)
    filter_cfg = load_filter_config(config_path=config_path, profile_path=profile_path)
    new_listings = fetch_listings(db_path=db_path, status="new")

    total_evaluated = len(new_listings)
    print(f"Evaluating {total_evaluated} new listing(s)...")

    if total_evaluated == 0:
        print("No new listings to filter.")
        return

    passed_count = 0
    rejected_count = 0
    rejection_reasons: dict[str, int] = {}

    for listing in new_listings:
        passed, reason = evaluate_listing(listing, filter_cfg)
        new_status = "passed" if passed else "rejected"
        update_listing_status(
            source=listing.source,
            id=listing.id,
            status=new_status,
            rejection_reason=reason,
            db_path=db_path,
        )

        if passed:
            passed_count += 1
        else:
            rejected_count += 1
            reason_key = reason or "unknown"
            rejection_reasons[reason_key] = rejection_reasons.get(reason_key, 0) + 1

    print(f"\nFilter Summary:")
    print(f"  Total evaluated: {total_evaluated}")
    print(f"  Passed:          {passed_count}")
    print(f"  Rejected:        {rejected_count}")

    if rejection_reasons:
        print(f"\nRejection Breakdown:")
        for r, count in sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {r}: {count}")


def main() -> None:
    db_path = DEFAULT_DB
    if len(sys.argv) > 1:
        db_path = Path(sys.argv[1])
    run_filter(db_path=db_path)


if __name__ == "__main__":
    main()
