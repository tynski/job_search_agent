"""Terminal digest viewer for top-scored job listings."""

from pathlib import Path
import sys
from typing import Optional

from agent.config import load_scoring_config
from agent.storage import DEFAULT_DB_PATH, fetch_top_scored_listings, init_db


def format_digest_table(listings: list[dict]) -> str:
    """Format scored listings into a clean, readable ASCII table."""
    if not listings:
        return "No actionable scored job listings found in database."

    lines = []
    lines.append("=" * 110)
    lines.append(f"{'#':<3} | {'SCORE':<5} | {'TITLE':<30} | {'COMPANY':<18} | {'LOCATION':<14} | {'WORK TYPE':<10}")
    lines.append("-" * 110)

    for idx, item in enumerate(listings, start=1):
        title = (item["title"][:28] + "..") if len(item["title"]) > 30 else item["title"]
        company = (item["company"][:16] + "..") if len(item["company"]) > 18 else item["company"]
        location = item["location"]
        if item["remote"]:
            location = "Remote" if not location else f"{location} (Remote)"
        location = (location[:12] + "..") if len(location) > 14 else location
        work_type = (item["work_type"][:8] + "..") if len(item["work_type"]) > 10 else item["work_type"]
        score_val = f"{item['score']:.1f}"

        lines.append(f"{idx:<3} | {score_val:<5} | {title:<30} | {company:<18} | {location:<14} | {work_type:<10}")

        # Sub-row with Reason, Gaps, Status tag, and URL
        gaps_str = ", ".join(item.get("gaps", [])) or "None"
        if len(gaps_str) > 60:
            gaps_str = gaps_str[:57] + "..."
        reason_str = item.get("reason", "")
        if len(reason_str) > 85:
            reason_str = reason_str[:82] + "..."

        pipeline_status = item.get("pipeline_status", "new")
        status_tag = f" [{pipeline_status.upper()}]" if pipeline_status != "new" else ""

        lines.append(f"    Gaps:   {gaps_str}")
        lines.append(f"    Reason: {reason_str}{status_tag}")
        if item.get("notes"):
            lines.append(f"    Notes:  {item['notes']}")
        if item.get("salary"):
            lines.append(f"    Salary: {item['salary']}")
        lines.append(f"    ID:     {item.get('id', '')}")
        lines.append(f"    URL:    {item.get('url', '')}")
        lines.append("-" * 110)

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
