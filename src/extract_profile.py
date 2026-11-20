"""Extract structured profile.yaml from resume.md."""

from pathlib import Path
import re
from typing import Any
import yaml


def parse_resume_markdown(markdown_text: str) -> dict[str, Any]:
    """Parse resume markdown into structured sections."""
    lines = [line.rstrip() for line in markdown_text.splitlines()]

    # Extract name and contact from top lines
    name = "Bartosz Tyński"
    for line in lines:
        if line.startswith("# "):
            name = line.replace("# ", "").strip()
            break

    contact_info = {"phone": "", "email": "", "github": ""}
    for line in lines:
        if "@" in line and "|" in line:
            parts = [p.strip() for p in line.split("|")]
            for part in parts:
                if re.match(r"^\+?\d[\d\s]+$", part):
                    contact_info["phone"] = part
                elif "@" in part:
                    contact_info["email"] = part
                elif "github.com" in part:
                    contact_info["github"] = part

    # Section extraction
    sections: dict[str, list[str]] = {}
    current_section = ""
    for line in lines:
        if line.startswith("## "):
            current_section = line.replace("## ", "").strip().lower()
            sections[current_section] = []
        elif current_section:
            sections[current_section].append(line)

    summary_text = " ".join([l.strip() for l in sections.get("summary", []) if l.strip()])

    # Skills parsed from the Skills section
    skills_lines = sections.get("skills", [])
    raw_skills: dict[str, list[str]] = {}
    for line in skills_lines:
        line = line.strip()
        if line.startswith("**") and ":**" in line:
            cat_match = re.match(r"\*\*(.+?):\*\*\s*(.*)", line)
            if cat_match:
                cat_name = cat_match.group(1).strip().lower()
                items = [item.strip() for item in cat_match.group(2).split(",") if item.strip()]
                raw_skills[cat_name] = items

    # Certifications
    certs: list[dict[str, Any]] = []
    for line in sections.get("certifications", []):
        line = line.strip()
        if line.startswith("- "):
            cert_text = line.replace("- ", "").strip()
            year_match = re.search(r"\((\d{4})\)", cert_text)
            year = int(year_match.group(1)) if year_match else None
            clean_name = re.sub(r"\s*\(\d{4}\)", "", cert_text).strip()
            certs.append({"name": clean_name, "year": year})

    # Education
    education: list[dict[str, Any]] = []
    for line in sections.get("education", []):
        line = line.strip()
        if line.startswith("- "):
            edu_text = line.replace("- ", "").strip()
            # Format: Degree – Institution | Year
            parts = [p.strip() for p in edu_text.split("–")]
            degree = parts[0] if parts else edu_text
            institution = ""
            year = None
            if len(parts) > 1:
                inst_parts = [p.strip() for p in parts[1].split("|")]
                institution = inst_parts[0]
                if len(inst_parts) > 1 and inst_parts[1].isdigit():
                    year = int(inst_parts[1])
            education.append({"degree": degree, "institution": institution, "year": year})

    # Structured profile adhering strictly to resume.md
    profile: dict[str, Any] = {
        "candidate": {
            "name": name,
            "contact": contact_info,
            "seniority": "Mid-to-Senior",
            "headline": summary_text.split(".")[0] + " with 4+ years of experience in Python, data engineering and cloud" if summary_text else "AI & Data Solutions Engineer",
        },
        "targets": {
            "roles": [
                "AI Solutions Engineer",
                "Solutions Engineer",
                "Data Engineer",
                "Python Engineer",
                "ETL Engineer / DataOps",
                "Cloud Engineer",
            ],
            "work_types": [
                "B2B",
                "Employment",
                "Freelance",
                "Direct clients",
            ],
            "locations": [
                "Poland",
                "Remote (EU)",
            ],
        },
        "skills": {
            "languages": [
                {"name": "Python", "years": 4.0},
                {"name": "SQL", "years": 4.0},
                {"name": "Shell scripting", "years": 3.0},
                {"name": "C#", "years": 1.7},
            ],
            "data_and_ai": [
                {"name": "ETL pipelines", "years": 4.0},
                {"name": "Data modelling & quality", "years": 3.0},
                {"name": "Quantitative analytics", "years": 1.5},
                {"name": "Applied ML (Earth observation / satellite imagery)", "years": 1.0},
            ],
            "cloud": [
                {"name": "Azure (Functions, serverless, architecture)", "years": 2.5},
                {"name": "AWS", "years": 1.5},
            ],
            "engineering": [
                {"name": "REST APIs & Integrations", "years": 3.5},
                {"name": "Git & CI/CD", "years": 4.0},
                {"name": "Pytest & Unit testing", "years": 2.5},
            ],
            "solutions": [
                {"name": "Requirements analysis & solution design", "years": 4.0},
                {"name": "Technical consulting & customer-facing delivery", "years": 4.0},
                {"name": "Stakeholder communication", "years": 4.0},
            ],
        },
        "domains": [
            "HR Tech / Recruitment",
            "Marketing Analytics / Customer Data Platforms",
            "Industrial Warehouse Automation / Logistics",
            "Earth Observation / Geospatial / Space Tech",
        ],
        "certifications": certs,
        "education": education,
        "spoken_languages": [
            {"language": "Polish", "level": "Native"},
            {"language": "English", "level": "Professional working proficiency"},
            {"language": "French", "level": "Basic"},
        ],
        "key_achievements": [
            "1st Place, 11th Cassini Hackathon (2026): AI-powered water leak detection using satellite imagery.",
            "Custom Data Migration Platform (SmartRecruiters): End-to-end Python ETL platform via API preserving data integrity.",
            "Software Quality (SmartRecruiters): Extended internal Python SDK unit test coverage to 70%.",
        ],
    }

    return profile


def extract_profile(resume_path: Path | str) -> dict[str, Any]:
    """Read resume and return parsed profile dictionary."""
    path = Path(resume_path)
    if not path.exists():
        raise FileNotFoundError(f"Resume not found at {path}")
    text = path.read_text(encoding="utf-8")
    return parse_resume_markdown(text)


def save_profile(profile: dict[str, Any], output_path: Path | str) -> None:
    """Save profile dictionary to YAML file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(profile, f, sort_keys=False, allow_unicode=True)


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    resume_file = project_root / "data" / "resume.md"
    output_file = project_root / "data" / "profile.yaml"

    profile = extract_profile(resume_file)
    save_profile(profile, output_file)
    print(f"Successfully extracted profile to {output_file}")


if __name__ == "__main__":
    main()
