"""Collector for public job boards of companies on applicant tracking systems.

Reads the documented public endpoints of Greenhouse, Lever, Ashby and
SmartRecruiters (no login, no API key) for companies configured under the
``companies`` key of config.yaml.
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

from agent.collectors.base import BaseCollector
from agent.config import load_companies_config
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
    LEVER_PAGE_SIZE = 100

    def __init__(
        self,
        config_path: Optional[Path | str] = None,
        db_path: Optional[Path | str] = None,
        timeout: int = 15,
        force: bool = False,
    ):
        self.config_path = config_path
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.timeout = timeout
        self.force = force

        cfg = load_companies_config(self.config_path)
        self.companies: list[dict[str, Any]] = cfg["companies"]
        self.request_delay: float = cfg["request_delay"]
        self.cooldown_hours: float = cfg["cooldown_hours"]
        self.location_filter: dict[str, Any] = cfg["location_filter"]

    # ---- HTTP (politeness: User-Agent header, delay between requests) ----

    def _get_json(self, url: str, params: Optional[dict[str, Any]] = None) -> Optional[Any]:
        headers = {"User-Agent": USER_AGENT}
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=self.timeout)
        except requests.RequestException as exc:
            logger.warning("company_boards: request failed for %s: %s", url, exc)
            return None
        if resp.status_code != 200:
            if resp.status_code == 404:
                logger.warning("company_boards: board not found: %s", url)
            else:
                logger.warning(
                    "company_boards: HTTP %s for %s", resp.status_code, url
                )
            return None
        try:
            return resp.json()
        except ValueError:
            logger.warning("company_boards: invalid JSON from %s", url)
            return None

    def _fetch_greenhouse(self, board_id: str) -> list[dict[str, Any]]:
        data = self._get_json(
            self.GREENHOUSE_JOBS_URL.format(board_id=board_id), params={"content": "true"}
        )
        return data.get("jobs", []) if isinstance(data, dict) else []

    def _fetch_lever(self, board_id: str) -> list[dict[str, Any]]:
        url = self.LEVER_POSTINGS_URL.format(board_id=board_id)
        postings: list[dict[str, Any]] = []
        skip = 0
        while True:
            data = self._get_json(
                url,
                params={
                    "mode": "json",
                    "skip": skip,
                    "limit": self.LEVER_PAGE_SIZE,
                },
            )
            if not isinstance(data, list) or not data:
                break
            postings.extend(data)
            if len(data) < self.LEVER_PAGE_SIZE:
                break
            skip += len(data)
            time.sleep(self.request_delay)
        return postings

    def _fetch_ashby(self, board_id: str) -> list[dict[str, Any]]:
        data = self._get_json(
            self.ASHBY_JOBS_URL.format(board_id=board_id),
            params={"includeCompensation": "true"},
        )
        return data.get("jobs", []) if isinstance(data, dict) else []

    def _fetch_smartrecruiters(self, board_id: str) -> list[dict[str, Any]]:
        data = self._get_json(self.SMARTRECRUITERS_POSTINGS_URL.format(board_id=board_id))
        return data.get("content", []) if isinstance(data, dict) else []

    def _fetch(self, ats: str, board_id: str) -> list[dict[str, Any]]:
        fetchers = {
            "greenhouse": self._fetch_greenhouse,
            "lever": self._fetch_lever,
            "ashby": self._fetch_ashby,
            "smartrecruiters": self._fetch_smartrecruiters,
        }
        fetcher = fetchers.get(ats)
        if fetcher is None:
            logger.warning("company_boards: unsupported ATS %r", ats)
            return []
        return fetcher(board_id)

    # ---- Normalization to the shared Listing schema ----

    @staticmethod
    def normalize_greenhouse(job: dict[str, Any], company: str) -> Optional[Listing]:
        location_data = job.get("location") or {}
        location = str(location_data.get("name") or "")
        url = str(job.get("absolute_url") or "")
        title = str(job.get("title") or "")
        if not url or not title:
            return None
        return Listing(
            id=str(job.get("id") or url),
            source=f"company:{company}",
            title=title,
            company=company,
            location=location,
            remote="remote" in location.lower(),
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
        url = str(posting.get("hostedUrl") or "")
        title = str(posting.get("text") or "")
        if not url or not title:
            return None
        return Listing(
            id=str(posting.get("id") or url),
            source=f"company:{company}",
            title=title,
            company=company,
            location=location,
            remote="remote" in location.lower(),
            work_type=None,
            salary=None,
            description=str(posting.get("descriptionPlain") or ""),
            url=url,
            posted_at=str(posting.get("createdAt") or ""),
        )

    @staticmethod
    def normalize_ashby(job: dict[str, Any], company: str) -> Optional[Listing]:
        url = str(job.get("jobUrl") or "")
        title = str(job.get("title") or "")
        if not url or not title:
            return None
        location = str(job.get("location") or "")
        postal = ((job.get("address") or {}).get("postalAddress") or {})
        if not location and postal.get("addressLocality"):
            location = str(postal["addressLocality"])
        return Listing(
            id=str(job.get("jobUrl") or job.get("title")),
            source=f"company:{company}",
            title=title,
            company=company,
            location=location,
            remote=bool(job.get("isRemote")) or "remote" in location.lower(),
            work_type=None,
            salary=None,
            description=str(job.get("descriptionPlain") or ""),
            url=url,
            posted_at=str(job.get("publishedAt") or ""),
        )

    @staticmethod
    def normalize_smartrecruiters(posting: dict[str, Any], company: str) -> Optional[Listing]:
        url = str(posting.get("ref") or "")
        title = str(posting.get("name") or "")
        if not url or not title:
            return None
        location_data = posting.get("location") or {}
        location = str(location_data.get("city") or "")
        if location_data.get("region"):
            location = f"{location}, {location_data['region']}" if location else str(
                location_data["region"]
            )
        return Listing(
            id=str(posting.get("id") or url),
            source=f"company:{company}",
            title=title,
            company=company,
            location=location,
            remote=bool(location_data.get("remote")) or "remote" in location.lower(),
            work_type=None,
            salary=None,
            description="",  # not included in the SmartRecruiters list endpoint
            url=url,
            posted_at=str(posting.get("releasedDate") or ""),
        )

    def _normalize(self, ats: str, raw: dict[str, Any], company: str) -> Optional[Listing]:
        normalizers = {
            "greenhouse": self.normalize_greenhouse,
            "lever": self.normalize_lever,
            "ashby": self.normalize_ashby,
            "smartrecruiters": self.normalize_smartrecruiters,
        }
        normalizer = normalizers.get(ats)
        if normalizer is None:
            return None
        return normalizer(raw, company)

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

            raw_entries = self._fetch(ats, board_id)
            if not raw_entries:
                logger.warning(
                    "company_boards: company %r (%s) board %r returned no postings or 404; continuing",
                    name, ats, board_id,
                )
                continue

            for raw in raw_entries:
                listing = self._normalize(ats, raw, name)
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
