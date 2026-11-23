"""Terminal digest viewer for top-scored job listings."""

from pathlib import Path
import re
import sys
from typing import Optional

from agent.config import load_scoring_config
from agent.storage import DEFAULT_DB_PATH, fetch_top_scored_listings, init_db


def format_listing_block(item: dict, index: Optional[int] = None) -> list[str]:
    """Format a single listing into readable key:value lines (Layout 2)."""
    lines = []
    lines.append("-" * 80)
    if index is not None:
        lines.append(f"#{index}")

    score_val = f"{item['score']:.1f}/10" if item.get("score") is not None else "N/A"
    status_val = item.get("pipeline_status", "new").upper()
    lines.append(f"Score:     {score_val} [{status_val}]")
    lines.append(f"Title:     {item.get('title', '')}")
    lines.append(f"Company:   {item.get('company', '')}")

    location = item.get("location") or ""
    if item.get("remote"):
        location = "Remote" if not location or location.lower() == "remote" else f"Remote ({location})"
    else:
        location = location or "Not specified"
    lines.append(f"Location:  {location}")

    work_type = item.get("work_type") or ""
    salary = item.get("salary") or ""
    if work_type and salary:
        clean_salary = re.sub(rf"\s*\({re.escape(work_type)}\)", "", salary, flags=re.IGNORECASE).strip()
        work_str = f"{work_type} ({clean_salary})"
    elif work_type:
        work_str = work_type
    elif salary:
        work_str = salary
    else:
        work_str = "Not specified"
    lines.append(f"Work:      {work_str}")

    if item.get("reason"):
        lines.append(f"Reason:    {item['reason']}")

    gaps = item.get("gaps")
    if gaps:
        gaps_str = ", ".join(gaps) if isinstance(gaps, list) else str(gaps)
        lines.append(f"Gaps:      {gaps_str}")

    if item.get("notes"):
        lines.append(f"Notes:     {item['notes']}")

    lines.append(f"ID:        {item.get('id', '')}")
    if item.get("url"):
        lines.append(f"URL:       {item['url']}")

    return lines


def format_digest_table(listings: list[dict]) -> str:
    """Format scored listings into clean key:value blocks."""
    if not listings:
        return "No actionable scored job listings found in database."

    lines = []
    for idx, item in enumerate(listings, start=1):
        lines.extend(format_listing_block(item, index=idx))
    lines.append("-" * 80)
    return "\n".join(lines)


def print_digest(top_n: Optional[int] = None, db_path: Path = DEFAULT_DB_PATH) -> None:
    init_db(db_path)
    cfg = load_scoring_config()
    limit = top_n if top_n is not None else cfg.get("top_n_digest", 10)

    listings = fetch_top_scored_listings(limit=limit, db_path=db_path)
    print(f"\n=== Top {len(listings)} Actionable IT Job Matches (by LLM Match Score) ===")
    print(format_digest_table(listings))
    print()


def main() -> None:
    top_n = None
    args = sys.argv[1:]
    if "--top" in args:
        idx = args.index("--top")
        if idx + 1 < len(args):
            top_n = int(args[idx + 1])
    print_digest(top_n=top_n)


if __name__ == "__main__":
    main()
