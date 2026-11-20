"""Hacker News 'Who is hiring' collector using official Algolia REST API."""

import html
import re
from typing import Any, Optional
import requests

from agent.collectors.base import BaseCollector
from agent.models import Listing


class HackerNewsCollector(BaseCollector):
    source_name = "hackernews"
    HN_ALGOLIA_SEARCH = "https://hn.algolia.com/api/v1/search_by_date"
    HN_ALGOLIA_ITEMS = "https://hn.algolia.com/api/v1/search"

    def __init__(self, limit: int = 100, timeout: int = 15):
        self.limit = limit
        self.timeout = timeout

    @staticmethod
    def _clean_html(text: str) -> str:
        if not text:
            return ""
        # Convert <p> to newlines
        cleaned = re.sub(r"</?p[^>]*>", "\n", text)
        cleaned = re.sub(r"<br\s*/?>", "\n", cleaned)
        cleaned = re.sub(r"<[^>]+>", "", cleaned)
        cleaned = html.unescape(cleaned)
        # Strip excessive blank lines
        lines = [line.strip() for line in cleaned.splitlines()]
        return "\n".join([l for l in lines if l])

    @staticmethod
    def normalize(hit: dict[str, Any]) -> Optional[Listing]:
        object_id = str(hit.get("objectID") or "")
        if not object_id:
            return None

        raw_comment = hit.get("comment_text") or ""
        clean_text = HackerNewsCollector._clean_html(raw_comment)
        if not clean_text:
            return None

        # Standard HN Who is Hiring format for first line: Company | Role | Location | Remote/Onsite | Salary
        lines = clean_text.splitlines()
        first_line = lines[0] if lines else clean_text
        parts = [p.strip() for p in first_line.split("|")]

        company = parts[0] if parts else "Unknown"
        title = parts[1] if len(parts) > 1 else "Software / Data Engineer"
        location = parts[2] if len(parts) > 2 else ""

        full_lower = clean_text.lower()
        is_remote = "remote" in full_lower

        # Extract salary if mentioned (e.g. $160k - $210k, $120k-$160k, €80k, etc.)
        salary_match = re.search(
            r"(\$|€|£)\s*\d+[\d,\.]*[kK]?(?:\s*[-–]\s*(?:\$|€|£)?\s*\d+[\d,\.]*[kK]?)?",
            clean_text,
        )
        salary = salary_match.group(0).strip() if salary_match else None

        work_type = None
        if "contract" in full_lower or "freelance" in full_lower or "b2b" in full_lower:
            work_type = "contract"
        elif "full-time" in full_lower or "full time" in full_lower:
            work_type = "full-time"

        url = f"https://news.ycombinator.com/item?id={object_id}"
        posted_at = hit.get("created_at") or ""

        return Listing(
            id=f"hn_{object_id}",
            source=HackerNewsCollector.source_name,
            title=title,
            company=company,
            location=location,
            remote=is_remote,
            work_type=work_type,
            salary=salary,
            description=clean_text,
            url=url,
            posted_at=posted_at,
        )

    def _get_latest_who_is_hiring_story_id(self) -> Optional[int]:
        params = {
            "tags": "story,author_whoishiring",
            "query": "Ask HN: Who is hiring?",
            "hitsPerPage": 1,
        }
        try:
            resp = requests.get(self.HN_ALGOLIA_SEARCH, params=params, timeout=self.timeout)
            if resp.status_code != 200:
                return None
            hits = resp.json().get("hits", [])
            if hits:
                return int(hits[0]["objectID"])
        except Exception:
            return None
        return None

    def collect(self) -> list[Listing]:
        story_id = self._get_latest_who_is_hiring_story_id()
        if not story_id:
            return []

        params = {
            "tags": f"comment,story_{story_id}",
            "hitsPerPage": self.limit,
        }
        try:
            resp = requests.get(self.HN_ALGOLIA_ITEMS, params=params, timeout=self.timeout)
            if resp.status_code != 200:
                return []
            hits = resp.json().get("hits", [])
            results: list[Listing] = []
            for hit in hits:
                listing = self.normalize(hit)
                if listing:
                    results.append(listing)
            return results
        except Exception:
            return []
