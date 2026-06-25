# Denominator Assessment Portal

An interactive Streamlit dashboard for **cross-comparing population-target denominators** (children aged **1–59 months**) across five independent data sources covering Nigeria's northern states. Designed for programme managers, M&E officers, and data analysts working in immunisation, MDA (Mass Drug Administration), and primary-health-care planning.

The portal includes an optional **AI chat assistant** (Groq Llama 3.3 70B via LangChain) that lets users ask natural-language questions directly against the underlying dataframes.

---

## Table of Contents

1. [Why this exists](#why-this-exists)
2. [Datasets](#datasets)
3. [Features](#features)
4. [Architecture](#architecture)
5. [Requirements](#requirements)
6. [Quick start](#quick-start)
7. [Configuration](#configuration)
8. [Running the dashboard](#running-the-dashboard)
9. [Using the chat assistant](#using-the-chat-assistant)
10. [Project structure](#project-structure)
11. [Data conventions](#data-conventions)
12. [Deployment](#deployment)
13. [Troubleshooting](#troubleshooting)
14. [Security notes](#security-notes)
15. [Contributing](#contributing)
16. [License](#license)

---

## Why this exists

Programmes that target children under five (immunisation campaigns, polio MDA rounds, nutrition surveys) all need a **denominator** — the number of eligible children in each Local Government Area (LGA). Different programmes derive that denominator from different sources:

- Household enumeration teams count children door-to-door.
- The Identify & Enumerate (IE) methodology samples and extrapolates.
- WorldPop publishes modelled projections.
- Immunisation programmes set their own targets.
- MDA campaigns record the children they treat.

When these numbers diverge — sometimes by **>2× for the same LGA** — campaigns over- or under-allocate vaccines, drugs, and field staff. This portal puts all five denominators side-by-side so the gap is visible at a glance.

---

## Datasets

All five datasets share the columns `state`, `local_government_area`, and `1_59m` (target population aged 1–59 months). The MDA file carries an additional `round` column.

| Dataset | File | States covered | Notes |
|---|---|---|---|
| **Enumeration** | `enumeration_kaduna_bauchi_adamawa_gombe_yobe.csv` | Kaduna, Bauchi, Adamawa, Gombe, Yobe | Household enumeration ground-truth counts |
| **IE (Identify & Enumerate)** | `ie_jigawa_katsina_kebbi_zamfara.csv` | Jigawa, Katsina, Kebbi, Zamfara | Alternative enumeration methodology |
| **MDA Round** | `mda_round_1.csv` | All nine states | Mass Drug Administration target population — includes `round` campaign number |
| **World Pop** | `world_pop_2026.csv` | All nine states | WorldPop 2026 modelled projection |
| **Immunisation** | `immunisation.csv` | All nine states | Immunisation programme target population |

### Schema

```
state                  : str   # Nigerian state, Title Case (e.g. "Kaduna")
local_government_area  : str   # LGA, UPPERCASE, hyphen-normalised (e.g. "ZANGO-KATAF")
1_59m                  : int   # target population aged 1–59 months
round                  : int   # MDA file only — campaign round number
```

---

## Features

- **State-level comparison** — grouped bar chart of total 1–59m targets across all five datasets, with in-bar labels formatted with thousands separators.
- **LGA drill-down** — line chart comparing each LGA within a selected state across the five sources, ordered alphabetically for stable visual scanning.
- **Unified comparison table** — pivoted LGA-by-source view with per-state totals (highlighted) and MDA round tracking. Multi-select state filter.
- **AI chat assistant** — natural-language questions answered by a LangChain pandas-dataframe agent. The agent is primed with column semantics, normalisation rules, and formatting conventions.
- **Premium UI** — DM Sans / Syne typography, gradient header, floating chat popover (bottom-right), hover-elevated metric cards, and a polished plotly theme.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit (app.py)                   │
├─────────────────────────────────────────────────────────┤
│  load_data()                                            │
│    └─ pd.read_csv × 5  →  normalize_lga()  →  concat    │
│                                                          │
│  ┌──────────────┐   ┌───────────────┐   ┌────────────┐  │
│  │  Metric row  │   │  Plotly bar   │   │   Pivoted  │  │
│  │  (5 columns) │   │  + line tabs  │   │   table    │  │
│  └──────────────┘   └───────────────┘   └────────────┘  │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Floating chat popover                           │   │
│  │    └─ ChatOpenAI (Groq endpoint, Llama 3.3 70B)  │   │
│  │       └─ create_pandas_dataframe_agent           │   │
│  │          └─ runs Python against df1…df5          │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

Key implementation details:

- **`normalize_lga()`** (app.py) collapses whitespace, uppercases, and unifies hyphen variants (`-`, `–`, `—`) so that joins across sources line up.
- **`build_comparison_table()`** pivots long-form data into one row per LGA with one column per source, then appends a per-state `TOTAL` row.
- The chat agent is given an explicit system prompt covering dataset purposes, column semantics, formatting rules, and an instruction to flag missing data rather than return `0`.

---

## Requirements

- **Python 3.9+** (tested on 3.10 / 3.11)
- **Streamlit ≥ 1.32** — required for the floating chat popover (older versions fall back to an expander)
- A **Groq API key** ([console.groq.com](https://console.groq.com), free tier available) — only needed if you want the chat assistant

Python packages (also listed in `requirements.txt`):

```
streamlit>=1.32
pandas
plotly
python-dotenv
openpyxl
langchain
langchain-experimental
langchain-openai
```

---

## Quick start

```bash
# 1. Clone the repo
git clone <repo-url>
cd Baseline_comparison

# 2. Create and activate a virtual environment
python -m venv venv

# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Windows cmd
venv\Scripts\activate.bat

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create a .env file (see Configuration below)

# 5. Launch
streamlit run app.py
```

The dashboard opens at <http://localhost:8501>.

---

## Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> The app also accepts the legacy `GROK_API_KEY` name for backwards compatibility, but `GROQ_API_KEY` is preferred.

Without an API key the dashboard still works fully — only the chat panel is disabled.

---

## Running the dashboard

```bash
streamlit run app.py
```

To run on a custom port or host (useful when deploying behind a reverse proxy):

```bash
streamlit run app.py --server.port 8080 --server.address 0.0.0.0
```

To suppress the "Welcome to Streamlit" email prompt in headless deploys, create `~/.streamlit/credentials.toml` with:

```toml
[general]
email = ""
```

---

## Using the chat assistant

Click the floating **💬 Ask your data a question** button (bottom-right). Example questions:

- *"Which LGA in Kaduna has the largest gap between WorldPop and Enumeration?"*
- *"What is the total 1–59m target in Bauchi according to the Immunisation dataset?"*
- *"Compare the MDA Round target for Birnin Kudu against the IE figure."*
- *"List the top 5 LGAs by WorldPop target across all states."*
- *"Where is the Immunisation target more than 50% higher than Enumeration?"*

The agent will:

1. Normalise your input (state → Title Case, LGA → UPPERCASE).
2. Identify the relevant dataframe(s).
3. Run pandas operations against them.
4. Return a concise answer with thousands-separated numbers, flagging any missing data.

---

## Project structure

```
Baseline_comparison/
├── app.py                                            # Main Streamlit application (~800 lines)
├── README.md                                         # This file
├── requirements.txt                                  # Python dependencies
├── .env                                              # Local secrets (NOT committed)
├── .gitignore                                        # Excludes venv/, .env, caches, etc.
│
├── enumeration_kaduna_bauchi_adamawa_gombe_yobe.csv  # Household enumeration counts
├── ie_jigawa_katsina_kebbi_zamfara.csv               # IE survey counts
├── mda_round_1.csv                                   # MDA Round 1 targets (+ round col)
├── world_pop_2026.csv                                # WorldPop 2026 projection
├── immunisation.csv                                  # Immunisation programme targets
├── Denomerator COmbined.xlsx                         # Source workbook (reference only)
│
└── venv/                                             # Local virtualenv (NOT committed)
```

---

## Data conventions

- **State names** are stored in Title Case (e.g. `Kaduna`, not `KADUNA` or `kaduna`).
- **LGA names** are UPPERCASE with normalised hyphens. The `normalize_lga()` function applied at load time enforces:
  - `strip().upper()`
  - `[-–—]+` collapsed to a single `-`
  - whitespace around hyphens removed (`ZANGO - KATAF` → `ZANGO-KATAF`)
  - multiple spaces collapsed to one
- **`1_59m`** is always an integer count, never a rate or percentage.
- **Missing data** for a given LGA in a given source is left absent (NaN), not zero-filled. The chat agent is instructed to flag missingness explicitly.

If you add a new dataset, follow the same column names and conventions so that `pd.concat` and the pivot tables continue to work.

---

## Deployment

### Streamlit Community Cloud

1. Push the repo to GitHub.
2. Connect at <https://share.streamlit.io>.
3. Add `GROQ_API_KEY` as a secret under **Settings → Secrets**.

### Docker

A minimal Dockerfile:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
```

Build and run:

```bash
docker build -t denominator-portal .
docker run -p 8501:8501 --env-file .env denominator-portal
```

### Behind a reverse proxy

Streamlit uses WebSockets — make sure your proxy (nginx, Traefik, Caddy) forwards the `Upgrade` and `Connection` headers.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Chatbot disabled — add your GROQ_API_KEY…` | Ensure `.env` exists in the project root and contains `GROQ_API_KEY=...`; restart the app. |
| `LangChain packages not found` | `pip install langchain langchain-experimental langchain-openai` |
| `Error loading datasets` | Run `streamlit run app.py` from the project root — the script uses relative paths. |
| Chat button not floating (appears inline) | Streamlit ≥ 1.32 required for `st.popover`; upgrade with `pip install -U streamlit`. |
| LGA names not matching across datasets | Check that the source CSV's LGA column passes through `normalize_lga()` consistently. |
| Chart bars unreadable on small screens | Use the LGA tab to drill into a single state at a time. |
| Groq rate limit | The free tier is shared per-key per-minute — wait 60s or upgrade. |

---

## Security notes

- The LangChain agent uses `allow_dangerous_code=True`, which permits **LLM-generated Python to execute** against your dataframes inside the Streamlit process. This is acceptable for local / trusted use, but **must be sandboxed** before any public deployment. Consider:
  - Running the app in a container with no network egress except to Groq.
  - Replacing the dataframe agent with a query-only SQL agent or a restricted tool.
- `.env` is in `.gitignore` — never commit your API key.
- The five CSVs do not contain PII (state and LGA aggregates only), but the workbook may; treat it accordingly.

---

## Contributing

1. Fork and branch off `main`.
2. Keep changes scoped — UI tweaks separate from data-pipeline changes where possible.
3. Run the app locally and verify both the State and LGA tabs render and the comparison table totals match before opening a PR.
4. If you add a new dataset, update:
   - `load_data()` in `app.py`
   - `SOURCE_COLORS` (add a hex code)
   - The "Datasets" and "Schema" sections in this README

---

## License

Internal / unlicensed — adjust as needed for your organisation.
