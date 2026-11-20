"""Base collector class."""

from abc import ABC, abstractmethod

from agent.models import Listing


class BaseCollector(ABC):
    source_name: str = "base"

    @abstractmethod
    def collect(self) -> list[Listing]:
        """Fetch and return normalized job listings."""
        pass
