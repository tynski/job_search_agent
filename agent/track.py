"""CLI for personal pipeline status tracking."""

import argparse
from pathlib import Path
import sys

from agent.storage import (
    DEFAULT_DB_PATH,
    VALID_PIPELINE_STATUSES,
    fetch_tracked_listings,
    find_listings_by_query,
    get_pipeline_stats,
    init_db,
    migrate_tracking_columns,
    set_pipeline_status,
    set_pipeline_status_by_query,
)


def format_listings_table(listings: list[dict]) -> str:
    """Format tracked listings into a readable ASCII table."""
    if not listings:
        return "No listings found matching criteria."

    lines = []
    lines.append("=" * 115)
    lines.append(f"{'SCORE':<5} | {'STATUS':<11} | {'COMPANY':<18} | {'LOCATION':<16} | {'TITLE'}")
    lines.append("-" * 115)

    for item in listings:
        score_val = f"{item['score']:.1f}" if item.get("score") is not None else "N/A"
        status_val = item.get("pipeline_status", "new").upper()
        title = (item["title"][:40] + "..") if len(item["title"]) > 42 else item["title"]
        company = (item["company"][:16] + "..") if len(item["company"]) > 18 else item["company"]
        location = item.get("location") or ""
        if item.get("remote"):
            location = "Remote" if not location else f"{location} (Remote)"
        location = (location[:14] + "..") if len(location) > 16 else location

        lines.append(f"{score_val:<5} | {status_val:<11} | {company:<18} | {location:<16} | {title}")
        lines.append(f"      ID:     {item['id']}")
        if item.get("notes"):
            lines.append(f"      Notes:  {item['notes']}")
        if item.get("url"):
            lines.append(f"      URL:    {item['url']}")
        lines.append("-" * 115)

    return "\n".join(lines)


def cmd_list(args: argparse.Namespace) -> None:
    db_path = Path(args.db)
    init_db(db_path)
    listings = fetch_tracked_listings(status=args.status, limit=args.limit, db_path=db_path)
    header_status = f"'{args.status}'" if args.status else "All"
    print(f"\n=== Tracked Job Listings ({header_status}: {len(listings)}) ===")
    print(format_listings_table(listings))
    print()


def cmd_find(args: argparse.Namespace) -> None:
    db_path = Path(args.db)
    init_db(db_path)
    matches = find_listings_by_query(args.query, db_path=db_path)
    print(f"\n=== Search Results for '{args.query}' ({len(matches)} match(es)) ===")
    print(format_listings_table(matches))
    print()


def cmd_set(args: argparse.Namespace) -> None:
    db_path = Path(args.db)
    init_db(db_path)
    query = args.id_or_query.strip()
    status = args.status.lower().strip()

    if status not in VALID_PIPELINE_STATUSES:
        print(f"Error: Invalid status '{status}'. Must be one of: {sorted(VALID_PIPELINE_STATUSES)}")
        sys.exit(1)

    updated_count, matches = set_pipeline_status_by_query(
        query=query,
        status=status,
        notes=args.note,
        match_all=args.all,
        db_path=db_path,
    )

    if updated_count == 0:
        if not matches:
            print(f"Error: No listings found matching '{query}'.")
            sys.exit(1)
        else:
            # Multiple matches found and --all was not passed
            print(f"\nFound {len(matches)} listings matching '{query}':")
            print(format_listings_table(matches))
            print(f"Multiple listings matched! To update a specific listing, use its exact ID:")
            print(f"  python -m agent.track set <exact_id> {status}")
            print(f"\nOr to update all {len(matches)} matching listings at once, re-run with --all:")
            note_flag = f' --note "{args.note}"' if args.note else ""
            print(f"  python -m agent.track set \"{query}\" {status} --all{note_flag}\n")
            sys.exit(1)

    note_str = f" (note: '{args.note}')" if args.note is not None else ""
    if updated_count == 1:
        item = matches[0]
        print(f"Successfully updated listing '{item['id']}' ({item.get('company', '')} - {item.get('title', '')}) -> status: '{status}'{note_str}")
    else:
        print(f"Successfully updated {updated_count} listings matching '{query}' -> status: '{status}'{note_str}")


def cmd_stats(args: argparse.Namespace) -> None:
    db_path = Path(args.db)
    init_db(db_path)
    stats = get_pipeline_stats(db_path=db_path)
    total = sum(stats.values())

    print("\n=== Personal Pipeline Tracking Stats ===")
    print(f"{'STATUS':<15} | {'COUNT':<8}")
    print("-" * 26)
    for s in ["new", "shortlisted", "applied", "replied", "rejected"]:
        print(f"{s:<15} | {stats.get(s, 0):<8}")
    print("-" * 26)
    print(f"{'Total Listings':<15} | {total:<8}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Personal IT job search pipeline tracking CLI.")
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to SQLite database file.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Subcommand: list
    list_parser = subparsers.add_parser("list", help="List listings by tracking status.")
    list_parser.add_argument(
        "--status",
        type=str,
        default=None,
        choices=sorted(VALID_PIPELINE_STATUSES),
        help="Filter by pipeline tracking status.",
    )
    list_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum listings to display.",
    )
    list_parser.set_defaults(func=cmd_list)

    # Subcommand: find
    find_parser = subparsers.add_parser("find", help="Find listings by company name, title, or ID keyword.")
    find_parser.add_argument("query", type=str, help="Search keyword (e.g. company name, title, or ID snippet).")
    find_parser.set_defaults(func=cmd_find)

    # Subcommand: set
    set_parser = subparsers.add_parser("set", help="Set pipeline tracking status for a listing.")
    set_parser.add_argument(
        "id_or_query",
        type=str,
        help="ID, company name, or title keyword of the listing to update.",
    )
    set_parser.add_argument(
        "status",
        type=str,
        choices=sorted(VALID_PIPELINE_STATUSES),
        help=f"Target pipeline status ({', '.join(sorted(VALID_PIPELINE_STATUSES))}).",
    )
    set_parser.add_argument(
        "--note",
        type=str,
        default=None,
        help="Optional note to attach to the listing.",
    )
    set_parser.add_argument(
        "--all",
        action="store_true",
        help="Apply update to all matching listings if multiple are found (e.g. multi-location duplicates).",
    )
    set_parser.set_defaults(func=cmd_set)

    # Subcommand: stats
    stats_parser = subparsers.add_parser("stats", help="Display statistics of pipeline tracking statuses.")
    stats_parser.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
