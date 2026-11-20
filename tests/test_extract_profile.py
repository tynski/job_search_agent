"""Tests for extract_profile module."""
from pathlib import Path
import pytest
import yaml

from src.extract_profile import extract_profile, parse_resume_markdown, save_profile


def test_parse_resume_markdown():
    sample_resume = """# Bartosz Tyński

+48 532 410 521 | b.tynski@gmail.com | github.com/tynski

## Summary

AI & Data Solutions Engineer with 4+ years of experience in Python, data engineering and cloud.

## Certifications

- Microsoft Certified: Azure Fundamentals (2022)

## Education

- MSc, Computer Science – AGH University of Science and Technology, Kraków | 2020
- BSc, Electrical Engineering – AGH University of Science and Technology, Kraków | 2019
"""
    profile = parse_resume_markdown(sample_resume)
    assert profile["candidate"]["name"] == "Bartosz Tyński"
    assert profile["candidate"]["contact"]["email"] == "b.tynski@gmail.com"
    assert profile["candidate"]["contact"]["phone"] == "+48 532 410 521"
    assert profile["candidate"]["contact"]["github"] == "github.com/tynski"
    assert profile["certifications"][0]["name"] == "Microsoft Certified: Azure Fundamentals"
    assert profile["certifications"][0]["year"] == 2022
    assert len(profile["education"]) == 2
    assert profile["education"][0]["degree"] == "MSc, Computer Science"


def test_extract_profile_from_file(tmp_path):
    resume_path = tmp_path / "resume.md"
    resume_path.write_text(
        "# Test Name\n\nemail@example.com | +123 | github.com/test\n\n## Summary\n\nEngineer",
        encoding="utf-8",
    )
    profile = extract_profile(resume_path)
    assert profile["candidate"]["name"] == "Test Name"
    assert "skills" in profile
    assert "targets" in profile


def test_save_and_reload_profile(tmp_path):
    output_path = tmp_path / "profile.yaml"
    dummy_profile = {
        "candidate": {"name": "Bartosz"},
        "skills": {"languages": [{"name": "Python", "years": 4.0}]},
    }
    save_profile(dummy_profile, output_path)
    assert output_path.exists()

    with open(output_path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    assert loaded == dummy_profile
