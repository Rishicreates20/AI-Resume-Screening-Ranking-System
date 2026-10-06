# AI Resume Screening & Ranking System

Takes a folder of resumes and processes all of them in one run. It applies a rule-based Python + AI eligibility filter, then scores the eligible candidates out of 100. Scoring combines evidence from projects (judged by an LLM, with a rules fallback) with public GitHub activity. The output is a ranked shortlist in which **every point traces back to a quote from the resume**.

```
resumes/ ─► ingest ─► extract ─► hard filter ─┬─► LLM analysis ─► GitHub ─► score ─► rank ─► results.json / .csv / .html
           (PDF/DOCX/TXT)       (rules only)  │   (rules fallback)
                                              └─► rejected (with reasons)
```

## Quick start

Requires Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # Windows: copy .env.example .env
# put GROQ_API_KEY (free at console.groq.com) and optionally GITHUB_TOKEN in .env

# put the resume PDFs in ./resumes (not committed: they contain candidates' personal data)
python main.py --input ./resumes --output ./output/results.json
```

Then open **`output/results.html`** (double-click, no server needed) for the dashboard, or read `results.json` / `results.csv`.

On Groq's free tier the LLM step is limited to about 8k tokens/minute, so 40 resumes take roughly 15–25 minutes. Progress is logged for every resume. Completed analyses are cached, so an interrupted run resumes where it stopped and a re-run makes no LLM calls. Try `--limit 3` first to check your key and model in a few seconds. `--no-llm` produces a complete result in about five seconds.

| Flag | Effect |
|---|---|
| `--no-llm` | Rules-only analysis: no API key needed, fully offline and deterministic |
| `--no-github` | Skip GitHub enrichment |
| `--limit N` | Process only the first N files (cheap trial run before a long LLM batch) |
| `--config path` | Use another config file (weights, term lists, model) |

Running without an API key works too. The system logs that the key is missing and scores every resume with the rules analyzer.

**Tests** (offline: no network, no API key, 58 tests): `python -m pytest -q`

**Optional API:** `uvicorn api:app`, then `POST /screen {"input_dir": "./resumes"}`, `GET /results` (JSON) and `GET /` (the HTML dashboard). The API calls the same `run_pipeline()` function the CLI uses.

## Output

`output/results.json` (with `results.csv` and `results.html` alongside it) contains:

- `summary`: total files, parsed, failed/unreadable, duplicates, eligible, rejected, LLM vs. fallback counts, GitHub status counts, runtime and per-stage seconds
- `ranked_candidates`: eligible candidates, highest score first. Each entry has:
  - `score_breakdown`, plus `score_evidence` (the quoted resume line behind each point)
  - `matched_skills`, `project_summary`, `github_summary`
  - `strengths`, `concerns`, `analysis_mode`
- `rejected_candidates`: candidates who failed the gate, with explicit `rejection_reasons` and their matched skills
- `failed_files` and `duplicates`: each with a reason

The console prints a ranked table and the reason for each rejection.

**On the 50 provided resumes** the gate is deterministic, so these counts hold in every mode: 50 parsed, 0 failed, **40 eligible, 10 rejected**. All 10 rejections were checked against the resume text: 7 have no Python at all (JavaScript, Java or Node only, two of them with LLM/RAG projects built in TypeScript), and 3 have Python but no AI project (one lists only "Machine Learning Basics" in a skills line).

**Generated results.** The results for those 50 resumes are in [`results/`](results/results.json): `results.json`, `results.csv` and the HTML dashboard `results.html`. This copy is anonymised (names, emails, phone numbers, GitHub usernames and repository names removed; scores, rankings and evidence unchanged) because the repository is public.

## Bonus features

| Bonus idea from the brief | Where |
|---|---|
| DOCX (and TXT/MD) parsing in addition to PDF | `ingest.py`; tested with generated DOCX files |
| Simple FastAPI endpoints | `api.py`: `POST /screen`, `GET /results`, `GET /` (dashboard), `GET /health` |
| Bounded concurrency with a measured speed-up | `pipeline.py` thread pools; `scripts/benchmark.py` (table below) |
| Caching GitHub and LLM results | `cache.py`: on-disk, keyed by model + prompt version + resume text; GitHub 24 h TTL |
| A small HTML report with score breakdown | `report_html.py` + `templates/dashboard.html` |
| Tests with synthetic resumes | `tests/`: eligibility, scoring, ingestion, LLM adapter over real HTTP, API, report |

### Concurrency benchmark

`python scripts/benchmark.py --input ./resumes` runs the full pipeline at 1, 2, 4 and 8 workers for the LLM and GitHub stages. Latency is **simulated** (0.3 s per LLM call, 0.2 s per GitHub call) so the benchmark is deterministic and free; it measures the pipeline's scheduling, not Groq's speed. With real calls the speed-up is capped by the provider's rate limits, which is why the default is only 2 LLM workers.

| workers | LLM stage (s) | GitHub stage (s) | total (s) | speed-up |
|---:|---:|---:|---:|---:|
| 1 | 13.86 | 7.23 | 21.09 | 1.0x |
| 2 | 7.10 | 3.61 | 10.71 | 2.0x |
| 4 | 3.78 | 1.81 | 5.59 | 3.8x |
| 8 | 2.33 | 1.01 | 3.34 | 6.3x |

### Dashboard (`results.html`)

A single self-contained HTML file: no server, no build step, no network, works offline. It opens on a summary band: a proportional outcome bar with the reasons behind the rejections, a score-distribution strip where every eligible candidate is a clickable dot (with the median marked), and the run facts with per-stage timings. Below it, a ruled candidate list (eligible / rejected / skipped tabs, search, minimum-score filter, a stacked bar of points per category) sits next to an inspector with the total score, one bar per category with expandable “why these points” quotes, GitHub enrichment, strengths and concerns, penalties, and the evidence behind the hard filter. The address bar keeps the tab, selected candidate, search and minimum score, so a link opens the same view. Arrow keys move through the list (and through the strip when it has focus), `/` jumps to search, and it follows the system light/dark setting. Category colours come from a validated colour-blind-safe palette and always come with a text label. Resume text is untrusted, so everything is HTML-escaped and the embedded data cannot break out of its `<script>` tag (covered by a test). CSV cells that begin with `=`, `+`, `-` or `@` are neutralised so a resume cannot plant a spreadsheet formula.

The full design, with screenshots, the data contract and the list of changes from the original spec, is in `docs/AI_Resume_Screening_UI_Workflow_Design_v2.docx` (it ships in the submission zip and is not committed to git; see Repository notes).

It deliberately does **not** include the spec's Next.js stack, WebSocket progress, in-browser upload or a settings modal for API keys. The brief says no frontend is required and that API keys must come from environment variables, so those would add weight and risk without improving the screening. The CLI and API cover running a batch.

## Design Decisions

### Filtering strategy (hard gate, rules only)
- **The gate is pure rules and stays outside the LLM.** It is deterministic, cheap and testable. A well-written resume cannot talk its way past it.
- **Eligible = Python evidence AND AI evidence.**
  - Python evidence is "Python" itself or a Python-only library (FastAPI, Django, PyTorch, Pandas…).
  - AI evidence is an LLM/RAG/agentic term from a configurable list (LangChain, LangGraph, ADK, LlamaIndex, RAG, embeddings, vector DBs, tool calling, multi-agent, LLM APIs…).
- **The gate checks for evidence that is present, never for stacks that are absent.** Java, React or Next.js never cause a rejection.
- **Evidence only counts outside education, certification and interest sections.** A "B.Tech (AI & ML)" degree name or a Coursera Python certificate is not a project. The rejection reason says so explicitly.
- **Matching is word-bounded and case-aware:**
  - "ai" does not match inside "maintain"
  - "RAG" does not match "drag"
  - "LL.M." (law degree) is not "LLM"
  - "LoRa" (IoT radio) is not "LoRA"
- **Classical ML only (CNNs, scikit-learn) passes the gate, but its AI-depth points are capped at 15/40.** It *is* AI, and a wrongly rejected candidate is never seen by a human, whereas a low-ranked one still is. To reject these profiles instead, set `allow_classical_ml_only: false`.
- **Classical ML must be shown in a project or job.** The vocabulary is broad ("Machine Learning Basics" in a skills line), so a skills-line mention alone is rejected with an explicit reason (`classical_ml_requires_applied_evidence`). LLM/RAG/agent terms are specific enough to count anywhere, because naming a framework is evidence.
- **Scrambled PDFs are repaired, not skipped.** Designer templates store text out of reading order, which hides the section headings. When no project/experience section is found, the PDF is re-read in visual order and that version is used if it helps. This recovered two resumes in the provided set.

### Scoring strategy (100 points, weights in `config.yaml`)
**The LLM never produces a score.** It reports *signals* per project, each with a verbatim quote, and plain Python turns those signals into points. The same evidence always gives the same score.

| Category | Max | How points are earned |
|---|---|---|
| AI / agentic project depth | 40 | Per project: retrieval 8, tool calling 7, orchestration 7, evaluation 6, state/memory 5, business logic 5, shipped/impact 2. The candidate gets their **best** project's score plus up to 5 for a second AI project. Classical-ML projects are capped at 15. AI that appears only in a skills list earns 3. |
| Python & backend | 30 | Python, FastAPI, async, PostgreSQL, Redis: 6 each. **Full points only when used in a project or job; half when only listed in skills.** Partial credit for near-equivalents (Flask/Django 4, other databases 3). |
| Cloud / deployment / full stack | 15 | GCP 5 (AWS/Azure 3), Docker 4, deployment evidence 4. React/Next.js add 2 **only** inside a project that also has a backend. |
| GitHub | 10 | See below |
| Engineering depth | 5 | 1 each for testing, caching, queues, observability, concurrency, failure handling or architecture, counted only from projects and experience |

**Penalties and caps** (each shown as its own line in the breakdown):
- −10 if every LLM project is a thin API wrapper; −5 if some are.
- −2 per project listed without implementation details, capped at −5. Total penalties are capped at −15.
- **No meaningful LLM/agentic project (one with a real depth signal) caps the total at 30.** This keeps a strong Python generalist below candidates who have actually built AI systems, as the brief requires.

**Ties** are broken by AI depth, then Python/backend score, then name, so the ranking is reproducible.

### LLM usage
- **Provider-agnostic adapter.** It targets any OpenAI-compatible `/chat/completions` API (Groq by default, since it has a free tier with no card). Switching provider means changing `base_url`, `model` and `api_key_env` in config. All provider code lives behind the `LLMClient` protocol in `src/screener/llm/`.
- **Structured output.** The request carries a Pydantic-derived JSON schema (`ResumeAnalysis`), and the response is validated again locally with Pydantic. The prompt asks for facts and quotes, not judgments about the person.
- **Anti-hallucination check.** Every quote the model returns must actually appear in the resume text (normalised, with a fuzzy match at ≥ 80%):
  - A signal whose quote can't be found is dropped.
  - A project that can't be found in the resume is dropped.
  - Each drop is recorded in `analysis_notes`.
- **Failure handling, per resume:**
  - Retries with `Retry-After` or exponential backoff on 429, 5xx, timeouts and invalid JSON.
  - If the model rejects JSON-schema mode or extra parameters, the adapter downgrades once and continues.
  - **A bad key or model name (401/403/404) disables the LLM for the rest of the run with one clear message** that names the setting to check, instead of failing the same way 40 times.
  - If a resume still fails, it is scored with the rules analyzer, which fills the same schema. `analysis_mode` records which path was used, and the console warns when any resume fell back.
- **Nothing important is silently truncated.** Long resumes first drop education, certification and similar sections, and only then are cut, so projects and experience always reach the model.
- **Cost and speed:**
  - Only **eligible** resumes reach the LLM.
  - Concurrency is bounded (`max_concurrency`, default 2, because free tiers limit tokens per minute).
  - Responses are cached on disk by model + prompt version + resume text, so re-runs make no calls and give identical results.
- **Low temperature.** Temperature is 0, and a reasoning model runs at low reasoning effort.
- **Tested over real HTTP.** `tests/test_llm_http.py` runs the adapter against a local OpenAI-compatible server: request payload and auth header, schema response format, 429/500 retries, fail-fast on 401/404, and caching.

### GitHub scoring (0–10, never a hard requirement)
- **Username extraction.** The username comes from PDF link annotations as well as the text, since many resumes show "GitHub" as clickable text and the URL only exists in the link. Profile links are preferred over repo links. Neighbouring labels in a header such as "GitHub | LinkedIn" are never read as a username.
- **Activity, 0–5 points.** Based on days since the last public push or event: ≤30 days = 5, ≤90 = 3, ≤180 = 1.
- **Maintained, relevant repos, 0–5 points.** One point per non-fork, non-archived repo pushed in the last 365 days that is Python or AI-related (by language, name, description or topics).
- **API budget.** Unauthenticated GitHub allows 60 requests/hour, so the default is **one call per candidate** (`/users/{u}/repos?sort=pushed`). With `GITHUB_TOKEN` (read from the environment) it also reads public events.
- **Failures:**
  - A 404 is recorded as `not_found`.
  - A rate limit trips a **circuit breaker**, and the remaining candidates are marked `rate_limited` without further calls.
  - Other errors become `error`.
  - In every case GitHub contributes 0 points and the failure is listed under `concerns`. The batch never fails because of GitHub.
- **Caching.** Results are cached on disk for 24 hours and memoised within a run.
- **Ownership signal.** Resume projects that match a repo name are listed as a strength. This is not scored.

### Reliability
- **Per-file isolation.** Each file is read in its own try/except. The following become `failed` with a reason, and the run continues:
  - corrupted or password-protected PDFs
  - image-only (scanned) PDFs
  - unsupported file types
- **Duplicates.** A duplicate is detected by identical bytes, identical normalised text, or the same email as an earlier file. The first file, by name, is kept.
- **Separated configuration.** Weights, thresholds, term lists and model settings live in `config.yaml`; secrets come only from environment variables.

## Project layout

```
main.py                    CLI
api.py                     optional FastAPI wrapper (also serves the dashboard)
config.yaml                weights, thresholds, term lists, model settings
scripts/benchmark.py       concurrency speed-up measurement
docs/                      UI workflow design document (v2, as built)
src/screener/
  ingest.py                PDF/DOCX/TXT reading, hashing, text normalisation, visual-order fallback
  extract.py               sections, contacts/GitHub, name, skills, project entries
  eligibility.py           hard filter (rules)
  analysis.py              LLM call + evidence verification + fallback
  llm/                     LLMClient protocol, OpenAI-compatible adapter, prompt, rules fallback
  scoring.py               signals and evidence -> points, strengths, concerns
  github.py                GitHub client (cache, circuit breaker) + scoring
  pipeline.py              orchestration with bounded concurrency and stage timings
  report.py                JSON/CSV writer, console table
  report_html.py           self-contained HTML dashboard (templates/dashboard.html)
tests/                     eligibility, scoring, ingestion, LLM adapter (mocked and real HTTP), GitHub, API, report, batch
```

## Known limitations
- No OCR, so image-only PDFs are reported as unreadable rather than parsed.
- Section detection is heuristic. When no project or experience section is found, the scorer falls back to the whole text (minus skills) and flags this in `concerns`.
- The rules fallback judges project depth by keywords. It is useful as a safety net, but the LLM is much better at spotting thin wrappers.
- Without a token, GitHub activity is measured from pushes to the candidate's own repos, so work done only in forks or other people's repositories is not seen.

## If I Had More Time
1. **A labelled evaluation set.** Have a recruiter rank about 20 resumes, measure rank correlation (Spearman/NDCG) against this system, and tune weights against it instead of by intuition.
2. **OCR fallback** (Tesseract or a vision model) for scanned PDFs, plus true layout-aware parsing for two-column resumes instead of the visual-order fallback.
3. **Deeper GitHub verification.** Check that repos linked to resume projects have real commit history by the candidate (commit counts, README, tests), not just a matching name.
4. **A client-side token-bucket limiter** matched to the provider's tokens-per-minute budget (so calls are paced instead of retried after a 429), plus a fallback model chain and async I/O with `httpx`.

## Repository notes
- `output/` and the design document in `docs/` are git-ignored: they show real candidate names, emails and scores from the company's dataset, and this repository is public. Run `python main.py --input ./resumes` to regenerate the results locally. `results/` holds the anonymised copy of the generated results that is committed instead.
- `resumes/` is ignored for the same reason, `.env` holds your API keys and is never committed (copy `.env.example`), and `.cache/` holds cached LLM and GitHub responses.
