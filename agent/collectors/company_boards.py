"""Collector for public job boards of companies on applicant tracking systems.

Reads the documented public endpoints of Greenhouse, Lever, Ashby and
SmartRecruiters (no login, no API key) for companies configured under the
``companies`` key of config.yaml.
"""

from datetime import datetime, timezone
import logging
from pathlib import Path
import time
from typing import Any, Optional

import requests

from agent.collectors.base import BaseCollector
from agent.config import DEFAULT_CONFIG_PATH, load_companies_config
from agent.models import Listing
from agent.storage import (
    DEFAULT_DB_PATH,
    get_last_collector_run,
    normalize_title_company,
    normalize_url,
    record_collector_run,
)

logger = logging.getLogger(__name__)

USER_AGENT = "job_search_agent/1.0"


class CompanyBoardsCollector(BaseCollector):
    """Collect open postings from configured companies' ATS public boards."""

    source_name = "company_boards"

    # Documented public endpoints (official docs, no auth):
    # - Greenhouse: docs.greenhouse.io/job-board.html
    # - Lever: github.com/lever/postings-api
    # - Ashby: developers.ashbyhq.com/docs/public-job-posting-api
    # - SmartRecruiters: developers.smartrecruiters.com/docs/endpoints
    GREENHOUSE_JOBS_URL = "https://boards-api.greenhouse.io/v1/boards/{board_id}/jobs"
    LEVER_POSTINGS_URL = "https://api.lever.co/v0/postings/{board_id}"
    ASHBY_JOBS_URL = "https://api.ashbyhq.com/posting-api/job-board/{board_id}"
    SMARTRECRUITERS_POSTINGS_URL = (
        "https://api.smartrecruiters.com/v1/companies/{board_id}/postings"
    )

    def __init__(
        self,
        config_path: Optional[Path | str] = None,
        db_path: Optional[Path | str] = None,
        timeout: int = 15,
        force: bool = False,
    ):
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.timeout = timeout
        self.force = force

        cfg = load_companies_config(self.config_path)
        self.companies: list[dict[str, Any]] = cfg["companies"]
        self.request_delay: float = cfg["request_delay"]
        self.cooldown_hours: float = cfg["cooldown_hours"]
        self.location_filter: dict[str, Any] = cfg["location_filter"]

    # ---- HTTP (politeness: User-Agent header, delay between requests) ----

    def _get_json(
        self,
        url: str,
        params: Optional[dict[str, Any]] = None,
        company: str = "",
        ats: str = "",
    ) -> Optional[Any]:
        headers = {"User-Agent": USER_AGENT}
        context = f"[{company} ({ats})] " if (company or ats) else ""
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=self.timeout)
        except requests.RequestException as exc:
            logger.warning("company_boards: %srequest failed for %s: %s", context, url, exc)
            return None
        if resp.status_code != 200:
            logger.warning(
                "company_boards: %sHTTP %s for %s", context, resp.status_code, url
            )
            return None
        try:
            return resp.json()
        except ValueError:
            logger.warning("company_boards: %sinvalid JSON from %s", context, url)
            return None

    def _fetch_greenhouse(self, board_id: str, company: str = "") -> list[dict[str, Any]]:
        data = self._get_json(
            self.GREENHOUSE_JOBS_URL.format(board_id=board_id),
            params={"content": "true"},
            company=company,
            ats="greenhouse",
        )
        return data.get("jobs", []) if isinstance(data, dict) else []

    def _fetch_lever(self, board_id: str, company: str = "") -> list[dict[str, Any]]:
        # Lever returns all active postings by default when mode=json is requested in a single call
        data = self._get_json(
            self.LEVER_POSTINGS_URL.format(board_id=board_id),
            params={"mode": "json"},
            company=company,
            ats="lever",
        )
        return data if isinstance(data, list) else []

    def _fetch_ashby(self, board_id: str, company: str = "") -> list[dict[str, Any]]:
        data = self._get_json(
            self.ASHBY_JOBS_URL.format(board_id=board_id),
            params={"includeCompensation": "true"},
            company=company,
            ats="ashby",
        )
        return data.get("jobs", []) if isinstance(data, dict) else []

    def _fetch_smartrecruiters(self, board_id: str, company: str = "") -> list[dict[str, Any]]:
        # Request maximum page size of 100 to avoid default 10-item truncation
        data = self._get_json(
            self.SMARTRECRUITERS_POSTINGS_URL.format(board_id=board_id),
            params={"limit": 100},
            company=company,
            ats="smartrecruiters",
        )
        return data.get("content", []) if isinstance(data, dict) else []

    def _fetch(self, ats: str, board_id: str, company: str = "") -> list[dict[str, Any]]:
        fetchers = {
            "greenhouse": self._fetch_greenhouse,
            "lever": self._fetch_lever,
            "ashby": self._fetch_ashby,
            "smartrecruiters": self._fetch_smartrecruiters,
        }
        fetcher = fetchers.get(ats)
        if fetcher is None:
            logger.warning("company_boards: [%s] unsupported ATS %r", company, ats)
            return []
        return fetcher(board_id, company=company)

    # ---- Normalization to the shared Listing schema ----

    @staticmethod
    def normalize_greenhouse(job: dict[str, Any], company: str) -> Optional[Listing]:
        location_data = job.get("location") or {}
        location = str(location_data.get("name") or "")
        offices = job.get("offices") or []
        office_names = [
            str(o.get("name") or o.get("location") or "")
            for o in offices
            if isinstance(o, dict)
        ]
        office_names = [o for o in office_names if o]

        all_locations = [location] if location else []
        for oname in office_names:
            if oname and oname not in all_locations:
                all_locations.append(oname)
        display_location = ", ".join(all_locations) if all_locations else location

        url = str(job.get("absolute_url") or "")
        title = str(job.get("title") or "")
        if not url or not title:
            return None

        is_remote = "remote" in display_location.lower()
        return Listing(
            id=str(job.get("id") or url),
            source=f"company:{company}",
            title=title,
            company=company,
            location=display_location,
            remote=is_remote,
            work_type=None,
            salary=None,
            description=str(job.get("content") or ""),
            url=url,
            posted_at=str(job.get("updated_at") or ""),
        )

    @staticmethod
    def normalize_lever(posting: dict[str, Any], company: str) -> Optional[Listing]:
        categories = posting.get("categories") or {}
        location = str(categories.get("location") or "")
        all_locations = categories.get("allLocations") or []
        if not location and all_locations:
            location = str(all_locations[0])
        country = str(posting.get("country") or "")
        if country and country.lower() not in location.lower():
            location = f"{location}, {country}" if location else country

        workplace_type = str(posting.get("workplaceType") or "").lower()
        remote = workplace_type == "remote" or "remote" in location.lower()

        url = str(posting.get("hostedUrl") or "")
        title = str(posting.get("text") or "")
        if not url or not title:
            return None

        # Build full description: intro paragraph + list sections (duties, tech stack, requirements) + closing
        desc_parts: list[str] = []
        desc_plain = str(posting.get("descriptionPlain") or "").strip()
        if desc_plain:
            desc_parts.append(desc_plain)
        lists = posting.get("lists") or []
        for item in lists:
            if isinstance(item, dict):
                hdr = str(item.get("text") or "").strip()
                content = str(item.get("content") or "").strip()
                if hdr:
                    desc_parts.append(f"\n{hdr}")
                if content:
                    desc_parts.append(content)
        additional = str(posting.get("additionalPlain") or "").strip()
        if additional:
            desc_parts.append(f"\n{additional}")
        description = "\n".join(desc_parts)

        # Parse createdAt timestamp (Lever returns epoch ms as int)
        created_at = posting.get("createdAt")
        if isinstance(created_at, (int, float)) and created_at > 0:
            posted_at = datetime.fromtimestamp(created_at / 1000.0, tz=timezone.utc).isoformat()
        else:
            posted_at = str(created_at or "")

        return Listing(
            id=str(posting.get("id") or url),
            source=f"company:{company}",
            title=title,
            company=company,
            location=location,
            remote=remote,
            work_type=None,
            salary=None,
            description=description,
            url=url,
            posted_at=posted_at,
        )

    @staticmethod
    def normalize_ashby(job: dict[str, Any], company: str) -> Optional[Listing]:
        url = str(job.get("jobUrl") or "")
        title = str(job.get("title") or "")
        if not url or not title:
            return None

        # Primary location + secondary locations
        primary_loc = str(job.get("location") or "")
        postal = ((job.get("address") or {}).get("postalAddress") or {})
        if not primary_loc and postal.get("addressLocality"):
            primary_loc = str(postal["addressLocality"])

        sec_locs: list[str] = []
        for sec in job.get("secondaryLocations") or []:
            if isinstance(sec, dict) and sec.get("location"):
                sec_locs.append(str(sec["location"]))

        all_locations = [primary_loc] if primary_loc else []
        for sloc in sec_locs:
            if sloc and sloc not in all_locations:
                all_locations.append(sloc)
        location = "; ".join(all_locations) if all_locations else primary_loc

        workplace_type = str(job.get("workplaceType") or "").lower()
        remote = bool(job.get("isRemote")) or workplace_type == "remote" or "remote" in location.lower()

        # Ashby provides unique UUID job id
        job_id = str(job.get("id") or job.get("jobUrl") or title)

        return Listing(
            id=job_id,
            source=f"company:{company}",
            title=title,
            company=company,
            location=location,
            remote=remote,
            work_type=str(job.get("employmentType") or "") or None,
            salary=None,
            description=str(job.get("descriptionPlain") or ""),
            url=url,
            posted_at=str(job.get("publishedAt") or ""),
        )

    @staticmethod
    def normalize_smartrecruiters(
        posting: dict[str, Any], company: str, board_id: str = ""
    ) -> Optional[Listing]:
        posting_id = str(posting.get("id") or "")
        title = str(posting.get("name") or "")
        if not title:
            return None

        # Build public job posting URL: https://jobs.smartrecruiters.com/{company_identifier}/{posting_id}
        comp_info = posting.get("company") or {}
        comp_id = str(comp_info.get("identifier") or board_id or "")
        if comp_id and posting_id:
            url = f"https://jobs.smartrecruiters.com/{comp_id}/{posting_id}"
        else:
            url = str(posting.get("ref") or "")
        if not url:
            return None

        location_data = posting.get("location") or {}
        city = str(location_data.get("city") or "")
        region = str(location_data.get("region") or "")
        country = str(location_data.get("country") or "")
        full_location = str(location_data.get("fullLocation") or "")

        loc_parts = [p for p in (city, region, country) if p]
        location = full_location or ", ".join(loc_parts)

        remote = (
            bool(location_data.get("remote"))
            or country.lower() == "remote"
            or "remote" in location.lower()
        )

        type_of_emp = posting.get("typeOfEmployment") or {}
        work_type = str(type_of_emp.get("label") or type_of_emp.get("id") or "") or None

        return Listing(
            id=posting_id or url,
            source=f"company:{company}",
            title=title,
            company=company,
            location=location,
            remote=remote,
            work_type=work_type,
            salary=None,
            description="",  # not included in the SmartRecruiters list endpoint
            url=url,
            posted_at=str(posting.get("releasedDate") or ""),
        )

    def _normalize(
        self, ats: str, raw: dict[str, Any], company: str, board_id: str = ""
    ) -> Optional[Listing]:
        if ats == "greenhouse":
            return self.normalize_greenhouse(raw, company)
        if ats == "lever":
            return self.normalize_lever(raw, company)
        if ats == "ashby":
            return self.normalize_ashby(raw, company)
        if ats == "smartrecruiters":
            return self.normalize_smartrecruiters(raw, company, board_id=board_id)
        return None

    # ---- Location rules from config (Poland or remote EU) ----

    @staticmethod
    def _matches_location(listing: Listing, loc_cfg: dict[str, Any]) -> bool:
        """Location rules mirror agent/filter.evaluate_listing (location part only)."""
        allow_any_remote = loc_cfg.get("allow_any_remote", True)
        allowed = [str(x).lower() for x in loc_cfg.get("allowed_countries_and_cities", [])]

        if listing.remote and allow_any_remote:
            return True

        loc_lower = (listing.location or "").lower()
        text_lower = ((listing.title or "") + " " + (listing.description or "")).lower()
        for allowed_loc in allowed:
            if allowed_loc in loc_lower or (
                allowed_loc in ("remote", "zdalnie") and allowed_loc in text_lower
            ):
                return True
        return False

    # ---- Cooldown (no more than one run per day) ----

    def _cooldown_active(self) -> bool:
        if self.force:
            return False
        last = get_last_collector_run(self.source_name, db_path=self.db_path)
        if not last:
            return False
        try:
            last_dt = datetime.fromisoformat(last)
        except ValueError:
            return False
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - last_dt).total_seconds() / 3600
        return age_hours < self.cooldown_hours

    # ---- Main entry ----

    def collect(self) -> list[Listing]:
        if self._cooldown_active():
            print(
                f"\n[company_boards] Cooldown active: last run was less than {self.cooldown_hours:.1f}h ago. "
                f"Skipping run (use --force to bypass)."
            )
            logger.info(
                "company_boards: last run less than %.1f h ago; skipping.",
                self.cooldown_hours,
            )
            return []

        listings: list[Listing] = []
        seen_urls: set[str] = set()
        seen_title_company: set[tuple[str, str]] = set()

        for company in self.companies:
            name = str(company.get("name") or "")
            ats = str(company.get("ats") or "").strip().lower()
            board_id = str(company.get("board_id") or "")

            raw_entries = self._fetch(ats, board_id, company=name)
            if not raw_entries:
                logger.warning(
                    "company_boards: [%s (%s)] board %r returned no postings or was not found; continuing",
                    name,
                    ats,
                    board_id,
                )
                continue

            for raw in raw_entries:
                listing = self._normalize(ats, raw, name, board_id=board_id)
                if listing is None:
                    continue
                if not self._matches_location(listing, self.location_filter):
                    continue
                norm_url = normalize_url(listing.url)
                if norm_url and norm_url in seen_urls:
                    continue
                tc = normalize_title_company(listing.title, listing.company)
                if tc[0] and tc[1] and tc in seen_title_company:
                    continue
                listings.append(listing)
                if norm_url:
                    seen_urls.add(norm_url)
                if tc[0] and tc[1]:
                    seen_title_company.add(tc)

            if company is not self.companies[-1]:
                time.sleep(self.request_delay)

        record_collector_run(self.source_name, db_path=self.db_path)
        return listings
