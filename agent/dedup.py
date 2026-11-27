"""Duplicate detection module for job listings.

Deduplicates remote job listings (e.g. from No Fluff Jobs or cross-posted sources)
after the rule-based filter and before LLM scoring to save tokens and avoid
redundant evaluations.
"""

from collections import defaultdict
from datetime import datetime
import html
import logging
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Optional

from agent.config import DEFAULT_CONFIG_PATH, load_dedup_config
from agent.storage import (
    DEFAULT_DB_PATH,
    fetch_dedup_candidates,
    init_db,
    mark_listing_duplicate,
)

logger = logging.getLogger(__name__)

# Common legal suffixes in Poland and internationally to strip during company normalization
LEGAL_SUFFIXES_PATTERNS = [
    r"sp[oó]łka z (?:ograniczon[aą] odpowiedzialno[sś]ci[aą]|o\.?\s*o\.?)",
    r"sp\.?\s*z\s*o\.?\s*o\.?",
    r"sp\.?\s*[kjpa]\.?",
    r"s\.?\s*a\.?",
    r"gmbh",
    r"ltd\.?",
    r"llc",
    r"inc\.?",
    r"corp\.?",
    r"b\.?v\.?",
    r"co\.?",
]

LEGAL_SUFFIXES_REGEX = re.compile(
    r"(?:\b|\s)(?:" + "|".join(LEGAL_SUFFIXES_PATTERNS) + r")(?=\s|[.,;]|$)",
    re.IGNORECASE,
)



def normalize_company(company: Optional[str]) -> str:
    """Normalize company name: lowercase, trim, remove legal suffixes and punctuation."""
    if not company:
        return ""
    text = company.lower()
    text = LEGAL_SUFFIXES_REGEX.sub(" ", text)
    # Remove remaining punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def normalize_title(title: Optional[str]) -> str:
    """Normalize job title: lowercase, trim, strip punctuation and collapse whitespace."""
    if not title:
        return ""
    text = title.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def normalize_description(desc: Optional[str]) -> str:
    """
    Normalize description: strip HTML tags, unescape HTML entities, lowercase,
    strip punctuation and collapse whitespace.
    """
    if not desc:
        return ""
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", desc)
    # Unescape HTML entities (&amp;, &nbsp;, etc.)
    text = html.unescape(text)
    # Lowercase
    text = text.lower()
    # Strip punctuation (keep letters, digits, whitespace)
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse extra whitespace
    return " ".join(text.split())


def build_shingles(norm_text: str, shingle_size: int = 5) -> set[tuple[str, ...]]:
    """Generate word n-gram shingles from normalized description."""
    words = norm_text.split()
    if not words:
        return set()
    if len(words) < shingle_size:
        return {tuple(words)}
    return {tuple(words[i : i + shingle_size]) for i in range(len(words) - shingle_size + 1)}


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def check_is_duplicate(
    norm_desc_a: str,
    norm_desc_b: str,
    shingles_a: set,
    shingles_b: set,
    similarity_threshold: float = 0.85,
) -> tuple[bool, float]:
    """
    Check if two listings have duplicate descriptions:
    1. Exact text match -> duplicate (similarity 1.0).
    2. Word 5-gram shingles Jaccard similarity >= similarity_threshold.
    """
    if norm_desc_a and norm_desc_a == norm_desc_b:
        return True, 1.0
    sim = jaccard_similarity(shingles_a, shingles_b)
    if sim >= similarity_threshold:
        return True, sim
    return False, sim


def backup_database(db_path: Path | str) -> Optional[Path]:
    """Create a backup of the SQLite database before running deduplication."""
    path = Path(db_path)
    if not path.exists():
        return None
    bak_path = path.with_suffix(".db.bak")
    shutil.copy2(path, bak_path)
    return bak_path


def deduplicate_listings_group(
    group_listings: list[dict[str, Any]],
    similarity_threshold: float = 0.85,
    shingle_size: int = 5,
) -> list[tuple[dict[str, Any], dict[str, Any], float]]:
    """
    Deduplicate listings within a single company+title group.
    Returns list of tuples: (duplicate_listing, canonical_listing, similarity_score).

    Rules:
    - Scored listings are comparison targets only; never marked duplicate.
    - If a candidate matches a scored listing, the scored listing is canonical.
    - Otherwise, earliest posted_at among candidates is canonical.
    """
    candidates = [l for l in group_listings if l["status"] == "passed"]
    if not candidates:
        return []

    scored_targets = [l for l in group_listings if l["status"] == "scored"]

    # Precompute normalized descriptions and shingles
    processed_items: dict[str, dict[str, Any]] = {}
    for item in group_listings:
        norm_desc = normalize_description(item.get("description") or "")
        shingles = build_shingles(norm_desc, shingle_size=shingle_size)
        processed_items[item["id"]] = {
            "item": item,
            "norm_desc": norm_desc,
            "shingles": shingles,
        }

    # Sort candidates by posted_at ASC, collected_at ASC, id ASC to prefer earliest
    def sort_key(item: dict[str, Any]):
        return (
            item.get("posted_at") or "9999-99-99",
            item.get("collected_at") or "9999-99-99",
            item.get("id") or "",
        )

    candidates.sort(key=sort_key)
    scored_targets.sort(key=sort_key)

    duplicates: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    # Candidates that remain canonical within this group
    active_canonicals: list[dict[str, Any]] = []

    for cand in candidates:
        cand_p = processed_items[cand["id"]]
        matched_canonical = None
        best_sim = 0.0

        # 1. Compare against scored targets first (prefer already scored)
        for scored in scored_targets:
            scored_p = processed_items[scored["id"]]
            is_dup, sim = check_is_duplicate(
                cand_p["norm_desc"],
                scored_p["norm_desc"],
                cand_p["shingles"],
                scored_p["shingles"],
                similarity_threshold=similarity_threshold,
            )
            if is_dup and sim > best_sim:
                matched_canonical = scored
                best_sim = sim

        # 2. If not matched to a scored listing, compare against active canonical candidates
        if matched_canonical is None:
            for canon in active_canonicals:
                canon_p = processed_items[canon["id"]]
                is_dup, sim = check_is_duplicate(
                    cand_p["norm_desc"],
                    canon_p["norm_desc"],
                    cand_p["shingles"],
                    canon_p["shingles"],
                    similarity_threshold=similarity_threshold,
                )
                if is_dup and sim > best_sim:
                    matched_canonical = canon
                    best_sim = sim

        if matched_canonical is not None:
            duplicates.append((cand, matched_canonical, best_sim))
        else:
            # Candidate becomes a canonical listing for subsequent candidates in this group
            active_canonicals.append(cand)

    return duplicates


def run_dedup(
    db_path: Path | str = DEFAULT_DB_PATH,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """
    Run duplicate detection pipeline step on listings with status 'passed'.
    Backs up database first and marks duplicates in SQLite.
    """
    db_file = Path(db_path)
    print(f"=== Job Search Duplicate Detection Run ===")
    print(f"Database target: {db_file}")

    init_db(db_file)
    cfg = load_dedup_config(config_path)

    if not cfg["enabled"]:
        print("Duplicate detection is disabled in config.")
        return {"enabled": False, "duplicates_found": 0, "llm_calls_saved": 0}

    # 1. Back up database file
    bak_path = backup_database(db_file)
    if bak_path:
        print(f"Backed up database to: {bak_path}")

    # 2. Fetch candidate listings
    only_remote = cfg["only_remote"]
    candidates_and_targets = fetch_dedup_candidates(only_remote=only_remote, db_path=db_file)
    passed_count = sum(1 for l in candidates_and_targets if l["status"] == "passed")
    print(
        f"Evaluating {passed_count} 'passed' listing(s) "
        f"({'remote only' if only_remote else 'all work types'})..."
    )

    if passed_count == 0:
        print("No passed listings to deduplicate.")
        return {
            "enabled": True,
            "evaluated": 0,
            "duplicates_found": 0,
            "llm_calls_saved": 0,
        }

    # 3. Group by normalized company and normalized title
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for listing in candidates_and_targets:
        c_norm = normalize_company(listing.get("company"))
        t_norm = normalize_title(listing.get("title"))
        if c_norm and t_norm:
            groups[(c_norm, t_norm)].append(listing)

    # 4. Deduplicate within each group
    total_dupes = 0
    canonical_set: set[str] = set()
    similarity_threshold = cfg["similarity_threshold"]
    shingle_size = cfg["shingle_size"]

    for group_key, group_items in groups.items():
        dupe_tuples = deduplicate_listings_group(
            group_items,
            similarity_threshold=similarity_threshold,
            shingle_size=shingle_size,
        )
        for dupe, canon, sim in dupe_tuples:
            mark_listing_duplicate(
                source=dupe["source"],
                id=dupe["id"],
                canonical_id=canon["id"],
                similarity_score=sim,
                db_path=db_file,
            )
            total_dupes += 1
            canonical_set.add(canon["id"])

    # 5. Print summary
    print(f"\nDuplicate Detection Summary:")
    print(f"  Passed listings evaluated: {passed_count}")
    print(f"  Duplicates found & marked: {total_dupes}")
    print(f"  Canonical listings kept:   {len(canonical_set)}")
    print(f"  Estimated LLM calls saved: {total_dupes}")

    return {
        "enabled": True,
        "evaluated": passed_count,
        "duplicates_found": total_dupes,
        "canonical_count": len(canonical_set),
        "llm_calls_saved": total_dupes,
    }


def main() -> None:
    db_path = DEFAULT_DB_PATH
    if len(sys.argv) > 1:
        db_path = Path(sys.argv[1])
    run_dedup(db_path=db_path)


if __name__ == "__main__":
    main()
