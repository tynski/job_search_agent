"""Just Join IT collector using public candidate API."""

from typing import Any, Optional
import requests

from agent.collectors.base import BaseCollector
from agent.models import Listing


class JustJoinITCollector(BaseCollector):
    source_name = "justjoin_it"
    BASE_URL = "https://justjoin.it/api/candidate-api/offers"

    def __init__(self, categories: Optional[list[str]] = None, timeout: int = 15):
        self.categories = categories or ["data", "python", "devops"]
        self.timeout = timeout

    @staticmethod
    def _format_salary(employment_types: list[dict[str, Any]]) -> Optional[str]:
        if not employment_types:
            return None
        parts = []
        for emp in employment_types:
            emp_type = emp.get("type", "")
            currency = emp.get("currency", "PLN")
            from_val = emp.get("from")
            to_val = emp.get("to")
            if from_val and to_val:
                parts.append(f"{int(from_val):,} - {int(to_val):,} {currency} ({emp_type})")
            elif from_val:
                parts.append(f"from {int(from_val):,} {currency} ({emp_type})")
            elif to_val:
                parts.append(f"up to {int(to_val):,} {currency} ({emp_type})")
        return ", ".join(parts) if parts else None

    @staticmethod
    def normalize(item: dict[str, Any]) -> Listing:
        slug = item.get("slug") or item.get("guid") or ""
        title = item.get("title", "").strip()
        company = item.get("companyName", "").strip()
        city = item.get("city", "").strip()
        workplace_type = (item.get("workplaceType") or "").lower()
        is_remote = workplace_type in ["remote", "partly_remote"]

        emp_types = item.get("employmentTypes", [])
        work_type_list = [e.get("type") for e in emp_types if e.get("type")]
        work_type = ", ".join(sorted(set(work_type_list))) if work_type_list else None
        salary = JustJoinITCollector._format_salary(emp_types)

        req_skills = [s.get("name") for s in item.get("requiredSkills", []) if s.get("name")]
        nice_skills = [s.get("name") for s in item.get("niceToHaveSkills", []) if s.get("name")]
        desc_parts = []
        if req_skills:
            desc_parts.append("Required: " + ", ".join(req_skills))
        if nice_skills:
            desc_parts.append("Nice to have: " + ", ".join(nice_skills))
        description = "\n".join(desc_parts)

        url = f"https://justjoin.it/offers/{slug}" if slug else ""
        posted_at = item.get("publishedAt") or item.get("lastPublishedAt") or ""

        return Listing(
            id=slug,
            source=JustJoinITCollector.source_name,
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
            "Accept": "application/json",
        }
        listings: list[Listing] = []
        seen_ids: set[str] = set()

        for category in self.categories:
            params = {"category": category, "page": 1, "perPage": 50}
            try:
                resp = requests.get(self.BASE_URL, headers=headers, params=params, timeout=self.timeout)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                items = data.get("data", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
                for item in items:
                    listing = self.normalize(item)
                    if listing.id and listing.id not in seen_ids:
                        seen_ids.add(listing.id)
                        listings.append(listing)
            except Exception:
                continue

        return listings
