"""Configuration and profile loader."""

from pathlib import Path
from typing import Any
import yaml

DEFAULT_CONFIG_PATH = Path("config.yaml")
DEFAULT_PROFILE_PATH = Path("data/profile.yaml")
DEFAULT_RESUME_PATH = Path("data/resume.md")


def load_yaml(path: Path | str) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        return data or {}


def load_text(path: Path | str) -> str:
    file_path = Path(path)
    if not file_path.exists():
        return ""
    return file_path.read_text(encoding="utf-8")


def extract_keywords_from_profile(profile_data: dict[str, Any]) -> list[str]:
    """Extract distinct include keywords (skills, roles) from profile.yaml."""
    keywords: set[str] = set()

    # Target roles
    for role in profile_data.get("targets", {}).get("roles", []):
        if role:
            keywords.add(role.strip())

    # Skills categories
    skills_obj = profile_data.get("skills", {})
    for category_skills in skills_obj.values():
        if isinstance(category_skills, list):
            for skill in category_skills:
                if isinstance(skill, dict) and "name" in skill:
                    raw_name = skill["name"]
                    # Extract main skill words (e.g., 'Azure (Functions...)' -> 'Azure', 'Python')
                    clean_name = raw_name.split("(")[0].strip()
                    if clean_name:
                        keywords.add(clean_name)
                    # Also include specific key items if present
                    if "ETL" in raw_name.upper():
                        keywords.add("ETL")
                    if "ML" in raw_name.upper():
                        keywords.add("ML")
                        keywords.add("Machine Learning")

    # Add core base keywords if not already present
    core = ["Python", "SQL", "Data Engineer", "Solutions Engineer", "ETL", "Azure", "AWS"]
    for c in core:
        keywords.add(c)

    # Sort descending by length so multi-word phrases match first
    return sorted(list(keywords), key=lambda k: len(k), reverse=True)


def load_filter_config(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    profile_path: Path | str = DEFAULT_PROFILE_PATH,
) -> dict[str, Any]:
    """Load merged filter configuration and include keywords."""
    config_data = load_yaml(config_path)
    profile_data = load_yaml(profile_path)

    filter_cfg = config_data.get("filter", {})
    profile_keywords = extract_keywords_from_profile(profile_data)

    return {
        "location": filter_cfg.get("location", {}),
        "work_type": filter_cfg.get("work_type", {}),
        "exclude": filter_cfg.get("exclude", {}),
        "salary": filter_cfg.get("salary", {}),
        "include_keywords": profile_keywords,
    }


def load_scoring_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load scoring configuration with defaults."""
    config_data = load_yaml(config_path)
    scoring_cfg = config_data.get("scoring", {})
    temp_raw = scoring_cfg.get("temperature")
    temperature = float(temp_raw) if temp_raw is not None else None

    provider = str(scoring_cfg.get("provider", "cloud")).lower()
    primary_model = scoring_cfg.get("primary_model") or scoring_cfg.get("model", "gemini/gemini-2.5-flash")
    fallback_model = scoring_cfg.get("fallback_model", "ollama/qwen3.5:4b")
    local_api_base = scoring_cfg.get("local_api_base", "http://localhost:11434")
    num_retries = int(scoring_cfg.get("num_retries", 3))
    retry_backoff = float(scoring_cfg.get("retry_backoff", 2.0))

    return {
        "provider": provider,
        "primary_model": primary_model,
        "model": primary_model,  # Backwards compatibility
        "fallback_model": fallback_model,
        "local_api_base": local_api_base,
        "num_retries": num_retries,
        "retry_backoff": retry_backoff,
        "temperature": temperature,
        "max_listings_per_run": int(scoring_cfg.get("max_listings_per_run", 20)),
        "prompt_version": str(scoring_cfg.get("prompt_version", "v1.0")),
        "top_n_digest": int(scoring_cfg.get("top_n_digest", 10)),
    }

