# Job Search Agent

A lightweight, personal automation tool that discovers, filters, scores, and tracks IT job opportunities, freelance contracts, and client projects matching a candidate's profile.

Operating strictly with a human in the loop, the agent evaluates job listings against a verified resume and structured profile, saving candidates hours of manual searching while never submitting applications or contacting recruiters autonomously.

---

## How It Works

The agent runs as a sequential, multi-stage pipeline stored in a local SQLite database (`data/jobs.db`):

```text
 +-----------------------------------------------------------+
 | 0. Profile Preparation (data/resume.md -> profile.yaml)   |
 +-----------------------------+-----------------------------+
                               |
                               v
 +-----------------------------------------------------------+
 | 1. Collect (JustJoinIT, NoFluffJobs -> DB)                |
 +-----------------------------+-----------------------------+
                               |
                               v
 +-----------------------------------------------------------+
 | 2. Rule-Based Filter (Location, Work Type, Exclusions)    |
 +-----------------------------+-----------------------------+
                               |
                               v
 +-----------------------------------------------------------+
 | 3. LLM Scoring (LiteLLM: Gemini / Local Ollama Fallback)  |
 +-----------------------------+-----------------------------+
                               |
                               v
 +-----------------------------------------------------------+
 | 4. Terminal Digest (Ranked matches, gaps, reasons)        |
 +-----------------------------+-----------------------------+
                               |
                               v
 +-----------------------------------------------------------+
 | 5. Personal Tracking (shortlisted, applied, replied, ...)  |
 +-----------------------------------------------------------+
```

1. **Profile Preparation**: Extracts verified skills, experience, and target roles from `data/resume.md` into `data/profile.yaml`.
2. **Collection**: Fetches raw postings via public REST endpoints (JustJoinIT, NoFluffJobs) and deduplicates records into SQLite.
3. **Rule-Based Filter**: Fast, zero-cost pre-filtering based on location, contract type, excluded titles/keywords, minimum salary, and required profile keywords.
4. **LLM Scoring**: Passed listings are scored from 0 to 10 against the candidate's profile using LiteLLM (Google Gemini or local Ollama), returning subscores, missing skill gaps, and a concise reason. **Scoring strictly prioritizes the freshest offers first** (ordered by collection timestamp descending).
5. **Terminal Digest**: Formats top-scored opportunities into an ASCII summary table, filtering out jobs you have already acted on.
6. **Personal Tracking**: Built-in CLI tool to record application statuses (`new`, `shortlisted`, `applied`, `replied`, `rejected`) and personal notes.

---

## Requirements

- **Python**: `3.14` (also compatible with `3.11+`)
- **Operating System**: Linux (tested on Ubuntu 24.04 / Debian)
- **Optional**: [Ollama](https://ollama.com/) for local offline model inference (e.g. `qwen3.5:4b`)

---

## Installation

Clone the repository and set up a virtual environment:

```bash
git clone <repository_url> job_search_agent
cd job_search_agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Configuration

### 1. API Keys (`.env`)

Copy the environment template and configure your LLM provider key:

```bash
cp .env.example .env
```

Edit `.env`:
```bash
# Free API key from Google AI Studio (https://aistudio.google.com/)
GEMINI_API_KEY=your_gemini_api_key_here
```

### 2. Application Config (`config.yaml`)

Copy the example configuration:

```bash
cp config.example.yaml config.yaml
```

Key configuration sections:

| Section / Key | Description |
| :--- | :--- |
| **`filter.location.allowed_countries_and_cities`** | List of allowed location strings in lowercase (e.g., `poland`, `warszawa`, `remote`, `eu`). |
| **`filter.location.allow_any_remote`** | When `true`, any listing flagged as remote passes the location filter. |
| **`filter.work_type.allowed`** | Accepted contract types (e.g., `b2b`, `employment`, `permanent`, `contract`, `freelance`). |
| **`filter.work_type.allow_missing`** | When `true`, listings that do not specify a work type are not rejected. |
| **`filter.exclude.titles`** | Job title substrings that trigger immediate rejection (e.g., `lead`, `manager`, `sales`, `intern`). |
| **`filter.exclude.keywords`** | Description keywords that trigger rejection. |
| **`filter.salary.min_pln`** | Minimum monthly salary in PLN. Set to `0` to disable. Missing salary data is never rejected. |
| **`scoring.provider`** | Set to `"cloud"` to use external APIs, or `"local"` to run fully offline via Ollama. |
| **`scoring.primary_model`** | LiteLLM model identifier for cloud scoring (e.g., `gemini/gemini-2.5-flash` or `gemini/gemini-3.5-flash-lite`). |
| **`scoring.fallback_model`** | Model used automatically when the primary model fails or experiences service errors (e.g., `ollama/qwen3.5:4b`). |
| **`scoring.local_api_base`** | Endpoint URL for the local inference server (default: `http://localhost:11434`). |
| **`scoring.local_timeout`** | Timeout in seconds for local inference calls before aborting (default: `60.0`). |
| **`scoring.num_retries`** | Maximum exponential backoff retries for transient HTTP errors (429, 503). |
| **`scoring.retry_backoff`** | Backoff multiplier in seconds between retries. |
| **`scoring.max_listings_per_run`** | Maximum passed listings to evaluate per scoring run. |
| **`scoring.max_attempts`** | Threshold of transient failures after which a listing is skipped to prevent wasted quota. |
| **`scoring.prompt_version`** | Version identifier for cache validation. Changing this invalidates cached scores and triggers re-scoring. |
| **`scoring.top_n_digest`** | Default number of listings displayed in the terminal digest. |

> [!NOTE]
> **Listing Sources**: Collectors are currently active for JustJoinIT and NoFluffJobs. Enabling/disabling individual sources via config is slated in the roadmap.
> **Waves / Depth**: Ingesting older listings in sequential waves is currently tracked in [BACKLOG.md](BACKLOG.md).

### 3. Candidate Profile (`profile.yaml`)

Your candidate profile is the source of truth for keywords and scoring.

You can initialize it in one of two ways:

1. **Extract from Markdown Resume**:
   Place your resume markdown file at `data/resume.md` and run:
   ```bash
   python -m src.extract_profile
   ```
2. **Copy Example Profile**:
   ```bash
   cp profile.example.yaml data/profile.yaml
   ```

Review `data/profile.yaml` by hand to adjust your skills, years of experience, target roles, and certifications.

### 4. Privacy & Personal Data

Create the local data directory:
```bash
mkdir -p data
```

> [!IMPORTANT]
> The following items contain private candidate information or operational state and are strictly excluded via `.gitignore`:
> - `.env`
> - `config.yaml`
> - `data/` (including `data/resume.md`, `data/profile.yaml`, and `data/jobs.db`)
> - `*.db` and `*.bak` files
> 
> Never commit these files to version control.

---

## Usage

### Quick Test Run

Run the end-to-end pipeline with a cap of 5 LLM scoring evaluations:

```bash
python -m agent.run --limit 5
```

### Full Pipeline Run

Execute collection, rule filtering, scoring, and digest generation:

```bash
python -m agent.run
```

### CLI Command Reference

| Command | Options | Purpose | Example |
| :--- | :--- | :--- | :--- |
| `agent.run` | `--limit <N>`<br>`--db <path>` | Runs collection, filter, scoring, and digest end-to-end. | `python -m agent.run --limit 10` |
| `agent.collect` | `[db_path]` | Fetches listings from all collectors into SQLite. | `python -m agent.collect` |
| `agent.filter` | `[db_path]` | Applies rule-based filter to `new` listings. | `python -m agent.filter` |
| `agent.score` | `--limit <N>` | Scores passed listings with configured LLM. | `python -m agent.score --limit 10` |
| `agent.digest` | `--top <N>` | Displays top scored actionable opportunities in ASCII table. | `python -m agent.digest --top 15` |
| `agent.track stats` | `--db <path>` | Prints counts of listings across all tracking statuses. | `python -m agent.track stats` |
| `agent.track find` | `query`<br>`--db <path>` | Searches listings by company, title, or ID snippet. | `python -m agent.track find cognitiva` |
| `agent.track list` | `--status <status>`<br>`--limit <N>`<br>`--db <path>` | Lists tracked listings filtered by status (`new`, `shortlisted`, `applied`, `replied`, `rejected`). | `python -m agent.track list --status shortlisted` |
| `agent.track set` | `id_or_query`<br>`status`<br>`--note "<note>"`<br>`--all`<br>`--db <path>` | Updates tracking status and note by exact ID or company/title query (use `--all` for multi-location duplicates). | `python -m agent.track set cognitiva applied --all --note "Sent CV"` |
| `src.extract_profile` | *(none)* | Parses `data/resume.md` to `data/profile.yaml`. | `python -m src.extract_profile` |

### Offer Freshness & Incremental Runs

- **Freshest-First Scoring**: When scoring passed listings, the agent queries SQLite with `ORDER BY collected_at DESC`. This prioritizes the newest offers ingested from the job boards over older backlogs.
- **Incremental & Cached**: The agent never scores the same listing twice under the same `prompt_version`. Once scored, the listing is marked `status = 'scored'`, and subsequent runs automatically evaluate the next freshest unscored offers.
- **Batch Processing with `--limit N`**: Passing `--limit 10` evaluates exactly the 10 freshest unscored offers that passed your rule-based filter. On your next run with `--limit 10`, it skips those already scored and proceeds to the next 10 freshest offers.
- **Live Database Updates**: The database and digest table update dynamically on every run:
  - New postings from collectors are inserted with status `new`.
  - Rule filtering updates evaluated rows to `passed` or `rejected`.
  - Newly scored offers are assigned numeric scores, gap analysis, and reasons.
  - If a newly scored fresh offer achieves a higher match score than older listings, it immediately appears on the top digest leaderboard.

---

## Local Model Option (Ollama)

You can run the scoring step completely locally without cloud API keys:

1. **Install Ollama**:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```
2. **Pull a Model**:
   ```bash
   ollama pull qwen3.5:4b
   ```
3. **Configure in `config.yaml`**:
   ```yaml
   scoring:
     provider: "local"
     fallback_model: "ollama/qwen3.5:4b"
     local_api_base: "http://localhost:11434"
   ```
4. **Automatic Cloud Fallback**:
   If `provider: "cloud"` is kept, the agent automatically falls back to your local Ollama instance if the cloud provider encounters persistent 503 or 429 errors.

> [!TIP]
> Running local LLMs on CPU is significantly slower than cloud APIs. For CPU-only setups, use small quantized models (e.g., 3B-4B parameters) and set `--limit 5` when evaluating batches.

---

## Reading the Digest

When running `python -m agent.digest` or completing `python -m agent.run`, the terminal displays an ASCII table:

```text
=== Top 3 Actionable IT Job Matches (by LLM Match Score) ===
--------------------------------------------------------------------------------
#1
Score:     8.5/10 [NEW]
Title:     Senior Python Engineer
Company:   Example Corp
Location:  Remote (Warsaw)
Work:      b2b (24,000 - 28,000 PLN)
Reason:    Strong match for core Python, SQL, and AWS; lacks Snowflake data warehouse experience.
Gaps:      Snowflake, dbt
ID:        example-corp-senior-python-engineer
URL:       https://example.com/job/senior-python-engineer
--------------------------------------------------------------------------------
```

### Scoring Rubric

- **`score` (0.0 - 10.0)**: Overall weighted score:
  $$\text{score} = (0.4 \times \text{skills\_match}) + (0.3 \times \text{role\_fit}) + (0.3 \times \text{experience\_fit})$$
- **`skills_match` (0.0 - 10.0)**: Match between required listing skills and verified profile skills.
- **`experience_fit` (0.0 - 10.0)**: Alignment between listing seniority requirements and candidate years of experience.
- **`role_fit` (0.0 - 10.0)**: Alignment with candidate target roles (e.g. Solutions Engineer, Data Engineer).
- **`gaps`**: Specific technologies, tools, or domain experience mentioned in the job description that the candidate lacks.
- **`reason`**: One-line rationale summarizing the evaluation.
- **Actionable Filtering**: Any listing marked as `applied`, `replied`, or `rejected` via `agent.track set` is automatically excluded from future digest displays.
- **Leaderboard Ranking vs. Freshness**: While the scoring step prioritizes evaluating the *most recently collected offers* first, the digest table displays the *highest-scoring offers overall* (`ORDER BY score DESC`). This produces an evolving leaderboard that automatically refreshes after every run whenever a newly evaluated offer earns a top match score.

---

## Troubleshooting

| Issue | Cause | Resolution |
| :--- | :--- | :--- |
| **`404 NotFoundError`** | The model identifier in `config.yaml` is deprecated or unavailable for your API tier. | Verify available models in Google AI Studio or LiteLLM documentation (e.g., use `gemini/gemini-2.5-flash` or `gemini/gemini-3.5-flash-lite`). |
| **`429 RateLimitError` / `503 ServiceUnavailable`** | Cloud provider quota limits or temporary outages. | The agent automatically backs off and retries (`num_retries`, `retry_backoff`). If errors persist, it falls back to the configured local model. |
| **Invalid JSON from local model** | Smaller local LLMs may occasionally omit JSON formatting. | The agent validates response schema and automatically retries once. If invalid after retry, the listing is recorded as a transient failure and skipped without crashing the run. |
| **Zero listings pass filter** | Strict filter rules or missing keywords. | Check `config.yaml` exclusion titles (`exclude.titles`) and allowed locations (`filter.location`). Ensure `data/profile.yaml` contains relevant target roles and skills. |

---

## Limitations

- **Free-Tier API Quotas**: Google AI Studio free tier enforces RPM (requests per minute) and daily token ceilings. Use `--limit` to throttle batch sizes during testing.
- **Heuristic Scoring**: LLM match scores are automated estimates based strictly on text comparison. Always review listings before applying.
- **Terms of Service**: The tool uses official public candidate APIs and RSS/Algolia endpoints to ensure compliance with source platform terms. Do not alter collectors to perform aggressive scraping.

---

## Project Structure & Tests

```text
job_search_agent/
├── agent/
│   ├── collectors/          # Platform collectors (JustJoinIT, NoFluffJobs)
│   ├── collect.py           # Ingestion CLI
│   ├── config.py            # YAML & profile loaders
│   ├── digest.py            # Terminal table formatter
│   ├── filter.py            # Rule-based filter
│   ├── models.py            # Listing data classes
│   ├── run.py               # End-to-end pipeline runner
│   ├── score.py             # LiteLLM scoring & error handling
│   ├── storage.py           # SQLite migrations, persistence & queries
│   └── track.py             # Personal status tracking CLI
├── src/
│   └── extract_profile.py   # Resume markdown parser
├── tests/                   # Automated pytest suite (40+ unit & integration tests)
├── BACKLOG.md               # Task backlog and upcoming roadmap
├── GEMINI.md                # Project architecture & core principles
├── config.example.yaml      # Configuration template
├── profile.example.yaml     # Fictional profile template
├── .env.example             # Environment variables template
├── requirements.txt         # Python package dependencies
└── pyproject.toml           # Pytest & project metadata
```

### Running Tests

Execute the complete test suite:

```bash
pytest -v
```

All network calls and LLM completions in tests are mocked, requiring no external credentials or network connectivity.

---

## Roadmap

Planned enhancements, upcoming sources, and evaluation experiments are tracked in [BACKLOG.md](BACKLOG.md).
