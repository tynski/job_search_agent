# job_search_agent

Personal agent that finds IT jobs, contracts and clients for user. 
MVP, personal use only. Keep it simple.

## Source of truth
- data/resume.md is the only source of profile. Never invent skills or experience.
- Search keywords and titles are generated from profile.yaml.
- Scoring compares each listing against profile.yaml and the resume text.
- Backlog is in BACKLOG.md; work only on items in Now unless I say otherwise.

## Target
- Roles: AI / Solutions Engineer, data engineering, Python, cloud (Azure/AWS).
- Work types: employment, B2B contracts, freelance, direct clients.
- Location: Poland or remote (EU).

## Stack
Python 3.14, SQLite, requests, feedparser, LiteLLM (provider-neutral), PyYAML, pytest.

## Pipeline
0. Extract a structured profile.yaml from resume.md (skills with years, roles,
  domains, certifications, languages, seniority, achievements). I review it by hand.
1. Collect listings (2-3 sources via API/RSS)
2. Rule-based filter (location, work type, keywords)
3. LLM scoring 0-10, JSON output: score, skills_match, experience_fit, role_fit,
   gaps (list), reason (one line)
4. Store in SQLite (status: new / shortlisted / applied / replied / rejected)
5. Daily digest of top matches (terminal first, email/Telegram later)
Later: outreach draft generation for shortlisted items only.

## Rules
- Small modules, minimal dependencies, no over-engineering.
- Config in config.yaml, secrets in .env (never commit; both in .gitignore).
- Rule-based filter first; call the LLM only for listings that pass.
- Cache results; never score the same listing twice.
- Cheap model for scoring (configurable in config.yaml), stronger model only for drafting.
- Respect site terms: prefer official APIs/RSS over scraping.
- Human in the loop: the agent never sends applications or messages on its own.
- Write a pytest test for each module.
- Commit after each working step with a clear message.

## Working style
- Plan first, code after I approve.
- One pipeline step per session. Show the diff and how to run it.
- Ask before adding a new dependency.
- Keep updated README.md file.
