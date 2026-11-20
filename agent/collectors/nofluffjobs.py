"""No Fluff Jobs collector using public search API."""

from datetime import datetime, timezone
from typing import Any, Optional
import requests

from agent.collectors.base import BaseCollector
from agent.models import Listing


class NoFluffJobsCollector(BaseCollector):
    source_name = "nofluffjobs"
    SEARCH_URL = "https://nofluffjobs.com/api/search/posting"

    def __init__(self, requirements: Optional[list[str]] = None, timeout: int = 15):
        self.requirements = requirements or ["python", "data-engineering", "sql"]
        self.timeout = timeout

    @staticmethod
    def _format_salary(salary_data: dict[str, Any]) -> Optional[str]:
        if not salary_data:
            return None
        from_val = salary_data.get("from")
        to_val = salary_data.get("to")
        currency = salary_data.get("currency", "PLN")
        sal_type = salary_data.get("type", "")
        if from_val and to_val:
            return f"{int(from_val):,} - {int(to_val):,} {currency} ({sal_type})"
        if from_val:
            return f"from {int(from_val):,} {currency} ({sal_type})"
        if to_val:
            return f"up to {int(to_val):,} {currency} ({sal_type})"
        return None

    @staticmethod
    def normalize(posting: dict[str, Any]) -> Listing:
        post_id = str(posting.get("id") or posting.get("url") or "")
        title = (posting.get("title") or "").strip()
        company = (posting.get("name") or "").strip()

        location_data = posting.get("location", {})
        places = location_data.get("places", [])
        city = places[0].get("city", "") if places else ""
        is_remote = bool(location_data.get("fullyRemote", False))

        salary_data = posting.get("salary", {})
        salary = NoFluffJobsCollector._format_salary(salary_data)
        work_type = salary_data.get("type") if salary_data else None

        tiles = posting.get("tiles", {}).get("values", [])
        tag_values = []
        for t in tiles:
            if isinstance(t, dict):
                v = t.get("value")
                if v:
                    tag_values.append(str(v))
            elif isinstance(t, str):
                tag_values.append(t)

        technology = posting.get("technology", "")
        desc_parts = []
        if technology:
            desc_parts.append(f"Tech: {technology}")
        if tag_values:
            desc_parts.append("Tags: " + ", ".join(tag_values))
        description = "\n".join(desc_parts)

        raw_url = posting.get("url", "")
        url = f"https://nofluffjobs.com/job/{raw_url}" if raw_url else ""

        posted_ms = posting.get("posted")
        posted_at = ""
        if isinstance(posted_ms, (int, float)):
            try:
                posted_at = datetime.fromtimestamp(posted_ms / 1000.0, tz=timezone.utc).isoformat()
            except Exception:
                posted_at = ""

        return Listing(
            id=post_id,
            source=NoFluffJobsCollector.source_name,
            title=title,
            company=company,
            location=city,
            remote=is_remote,
            work_type=work_type,
            salary=salary,
            description=description,
            url=url,
            posted_at=posted_at,
        )

    def collect(self) -> list[Listing]:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko)",
            "Content-Type": "application/json",
            "Accept": "application/json, text/plain, */*",
        }
        params = {
            "salaryCurrency": "PLN",
            "salaryPeriod": "month",
            "region": "pl",
        }
        results: list[Listing] = []
        seen_ids: set[str] = set()

        for req in self.requirements:
            payload = {
                "criteriaSearch": {
                    "requirement": [req],
                },
                "page": 1,
                "pageSize": 50,
            }
            try:
                resp = requests.post(
                    self.SEARCH_URL,
                    params=params,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                if resp.status_code != 200:
                    continue
                data = resp.json()
                postings = data.get("postings", [])
                for p in postings:
                    listing = self.normalize(p)
                    if listing.id and listing.id not in seen_ids:
                        seen_ids.add(listing.id)
                        results.append(listing)
            except Exception:
                continue

        return results
