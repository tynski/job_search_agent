"""CLI for personal pipeline status tracking."""

import argparse
from pathlib import Path
import sys

from agent.config import load_scoring_config
from agent.digest import format_listing_block
from agent.storage import (
    DEFAULT_DB_PATH,
    VALID_PIPELINE_STATUSES,
    fetch_top_scored_listings,
    fetch_tracked_listings,
    find_listings_by_query,
    get_pipeline_stats,
    init_db,
    migrate_tracking_columns,
    set_pipeline_status,
    set_pipeline_status_by_query,
)


def format_listings_table(listings: list[dict]) -> str:
    """Format tracked listings into readable key:value blocks."""
    if not listings:
        return "No listings found matching criteria."

    lines = []
    for idx, item in enumerate(listings, start=1):
        lines.extend(format_listing_block(item, index=idx))
    lines.append("-" * 80)
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

    pos_args = args.target_or_status
    usage_help = (
        "Usage:\n"
        "  python -m agent.track set --all <status> [--only-status <pipeline_status>] [--note \"...\"] [--yes] [--dry-run]\n"
        "  python -m agent.track set <listing_id> <status> [--note \"...\"]"
    )

    if args.all:
        if len(pos_args) == 0:
            print(f"Error: Missing <status> for --all.\n\n{usage_help}")
            sys.exit(1)
        elif len(pos_args) > 1:
            print(f"Error: --all is mutually exclusive with <listing_id>.\n\n{usage_help}")
            sys.exit(1)
        status = pos_args[0].lower().strip()
    else:
        if len(pos_args) == 0:
            print(f"Error: Must specify either <listing_id> <status> or --all <status>.\n\n{usage_help}")
            sys.exit(1)
        elif len(pos_args) == 1:
            print(f"Error: Missing <status> for listing '{pos_args[0]}'.\n\n{usage_help}")
            sys.exit(1)
        elif len(pos_args) == 2:
            listing_id = pos_args[0].strip()
            status = pos_args[1].lower().strip()
        else:
            print(f"Error: Unexpected arguments {pos_args[2:]}.\n\n{usage_help}")
            sys.exit(1)

    if status not in VALID_PIPELINE_STATUSES:
        print(f"Error: Invalid status '{status}'. Must be one of: {sorted(VALID_PIPELINE_STATUSES)}")
        sys.exit(1)

    note_str = f" (note: '{args.note}')" if args.note is not None else ""

    if args.all:
        only_status = (args.only_status or "new").lower().strip()
        if only_status not in VALID_PIPELINE_STATUSES:
            print(f"Error: Invalid --only-status '{args.only_status}'. Must be one of: {sorted(VALID_PIPELINE_STATUSES)}")
            sys.exit(1)

        cfg = load_scoring_config()
        limit = cfg.get("top_n_digest", 10)
        digest_listings = fetch_top_scored_listings(limit=limit, db_path=db_path)

        matching_items = [
            item for item in digest_listings
            if (item.get("pipeline_status") or "new").lower() == only_status
        ]

        if not matching_items:
            print(f"No listings in the current digest selection matched status '{only_status}'.")
            return

        if args.dry_run:
            print(f"[Dry run] {len(matching_items)} listing(s) in digest selection would be updated to status '{status}'{note_str}:")
            for item in matching_items[:5]:
                company = item.get("company", "")
                title = item.get("title", "")
                print(f"  - {item['id']}: {title} ({company})")
            if len(matching_items) > 5:
                print(f"  ... and {len(matching_items) - 5} more.")
            return

        print(f"This will update {len(matching_items)} listing(s) in digest selection to status '{status}'{note_str}.")
        for item in matching_items[:5]:
            company = item.get("company", "")
            title = item.get("title", "")
            print(f"  - {item['id']}: {title} ({company})")
        if len(matching_items) > 5:
            print(f"  ... and {len(matching_items) - 5} more.")

        if not args.yes:
            try:
                reply = input(f"Proceed with updating {len(matching_items)} listing(s)? (y/N): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nAborted.")
                sys.exit(1)
            if reply not in ("y", "yes"):
                print("Aborted.")
                return

        for item in matching_items:
            set_pipeline_status(item["id"], status, notes=args.note, db_path=db_path)
        print(f"Successfully updated {len(matching_items)} listing(s) -> status: '{status}'{note_str}")
        return

    # Single listing update (Form B)
    if args.dry_run:
        matches = find_listings_by_query(listing_id, db_path=db_path)
        if not matches:
            print(f"Error: No listings found matching '{listing_id}'.")
            sys.exit(1)
        item = matches[0]
        print(f"[Dry run] Listing '{item['id']}' ({item.get('company', '')} - {item.get('title', '')}) would be updated to status '{status}'{note_str}")
        return

    updated_count, matches = set_pipeline_status_by_query(
        query=listing_id,
        status=status,
        notes=args.note,
        match_all=False,
        db_path=db_path,
    )

    if updated_count == 0:
        if not matches:
            print(f"Error: No listings found matching '{listing_id}'.")
            sys.exit(1)
        else:
            print(f"\nFound {len(matches)} listings matching '{listing_id}':")
            print(format_listings_table(matches))
            print(f"Multiple listings matched! To update a specific listing, use its exact ID:")
            print(f"  python -m agent.track set <exact_id> {status}")
            sys.exit(1)

    item = matches[0]
    print(f"Successfully updated listing '{item['id']}' ({item.get('company', '')} - {item.get('title', '')}) -> status: '{status}'{note_str}")



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
    set_parser = subparsers.add_parser(
        "set",
        usage="""python -m agent.track set --all <status> [--only-status <pipeline_status>] [--note "..."] [--yes] [--dry-run]
       python -m agent.track set <listing_id> <status> [--note "..."]""",
        help="Set pipeline tracking status for a listing or all digest listings.",
    )
    set_parser.add_argument(
        "--all",
        action="store_true",
        help="Apply status to all listings in the current digest selection (mutually exclusive with <listing_id>).",
    )
    set_parser.add_argument(
        "--only-status",
        type=str,
        default="new",
        help="Narrow --all update to digest listings currently in this pipeline status (default: new).",
    )
    set_parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt when using --all.",
    )
    set_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show how many rows would change and a preview of titles without writing.",
    )
    set_parser.add_argument(
        "--note",
        type=str,
        default=None,
        help="Optional note to attach to the listing(s).",
    )
    set_parser.add_argument(
        "target_or_status",
        nargs="*",
        metavar="ARGS",
        help="<status> (with --all) OR <listing_id> <status>",
    )
    set_parser.set_defaults(func=cmd_set)

    # Subcommand: stats
    stats_parser = subparsers.add_parser("stats", help="Display statistics of pipeline tracking statuses.")
    stats_parser.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
