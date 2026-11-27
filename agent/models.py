"""Data models for job search agent."""

from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass
class Listing:
    id: str
    source: str
    title: str
    company: str
    location: str
    remote: bool
    work_type: Optional[str]
    salary: Optional[str]
    description: str
    url: str
    posted_at: str
    duplicate_of: Optional[str] = None
    similarity_score: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
