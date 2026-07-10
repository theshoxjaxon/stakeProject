# Quantitative Football Betting Engine — Full Documentation

> **Status:** working end-to-end as of 2026-07-10 · Python 3.11 · SQLite · The Odds API v4
> **Disclaimer:** educational/research software. Betting involves risk; nothing here is financial advice.

---

## Table of contents

1. [What this project does](#1-what-this-project-does)
2. [Quick start — the exact commands](#2-quick-start--the-exact-commands)
3. [Architecture](#3-architecture)
4. [How a prediction is made](#4-how-a-prediction-is-made)
5. [Tournament auto-detection (World Cup / UCL / Euros)](#5-tournament-auto-detection)
6. [The AI advisor (Gemini)](#6-the-ai-advisor-gemini)
7. [Web API (FastAPI)](#7-web-api-fastapi)
8. [Helper scripts](#8-helper-scripts)
9. [Configuration reference](#9-configuration-reference)
10. [Database schema](#10-database-schema)
11. [Testing & CI](#11-testing--ci)
12. [Honest assessment — how to actually improve prediction accuracy](#12-honest-assessment--how-to-actually-improve-prediction-accuracy)
13. [Code health — what was broken and what remains](#13-code-health--what-was-broken-and-what-remains)
14. [Security incident log](#14-security-incident-log)

---

## 1. What this project does

The engine estimates **1X2 probabilities** (home / draw / away) for upcoming football
matches, compares them with bookmaker odds, and flags **value bets** with a
**fractional-Kelly stake** suggestion.

The chain in one sentence:

> Fetch odds & results from The Odds API → fit a **Dixon-Coles** goal model on
> completed matches → adjust expected goals with **form / H2H / rest / xG / injury**
> multipliers → de-margin the bookmaker line → report sides where the model's
> probability beats the fair implied probability by more than `EDGE_THRESHOLD` →
> optionally let a **Gemini AI advisor** confirm or veto the pick → store every
> prediction for later settlement and ROI tracking.

---

## 2. Quick start — the exact commands

### One-time setup

```bash
cd ~/Desktop/stakeProject

# Create the virtualenv (uv manages the Python 3.11 toolchain)
uv venv --python 3.11 venv

# Install dependencies
uv pip install --python venv/bin/python -r requirements.txt

# Configure secrets — copy the template and fill in your keys
cp .env.example .env        # then edit .env
```

`.env` needs at minimum:

```
ODDS_API_KEY=<your key from https://the-odds-api.com/>
GEMINI_API_KEY=<optional — omit to run pure-quant>
PREDICTION_HORIZON_DAYS=7
```

### Run the full pipeline (the main command)

```bash
venv/bin/python main.py
```

This is the single command that executes the code fully: it syncs odds
(leagues **plus any tournament currently in season**), fits the model, prints
the prediction/value table to the console, and saves predictions to the DB.

### Everything else

```bash
venv/bin/python -m uvicorn src.api:app --reload   # REST API on :8000 (frontend backend)
venv/bin/python view_predictions.py               # dashboard of saved predictions
venv/bin/python -m src.settle_bets                # settle finished matches → win/loss
venv/bin/python -m src.accountant                 # ROI / win-rate report
venv/bin/python update_results.py                 # legacy result updater (Bet rows)
venv/bin/python -m pytest                         # test suite
```

Typical daily loop: `main.py` (morning, get picks) → matches finish →
`main.py` again next day (its score fetch marks results) → `-m src.settle_bets`
→ `-m src.accountant`.

---

## 3. Architecture

```
stakeProject/
├── main.py                  # CLI pipeline orchestrator (start here)
├── src/
│   ├── config.py            # ALL tunables, loaded from .env — single source of config
│   ├── logger.py            # console + rotating file log, silences noisy HTTP libs
│   ├── models.py            # SQLAlchemy ORM — single source of schema truth
│   ├── database.py          # engine factory, SessionLocal, get_session(), init_db()
│   │
│   ├── fetch_data.py        # The Odds API: odds + scores sync, credit-protector cache
│   ├── tournaments.py       # NEW: in-season tournament detection (/v4/sports, cached)
│   ├── backfill.py          # seed matches + rebuild Elo from recent scores
│   │
│   ├── poisson_model.py     # Dixon-Coles GoalEngine (MLE fit, 6×6 score matrix)
│   ├── elo.py               # standalone in-memory Elo engine (used by backfill)
│   ├── elo_model.py         # DB-backed Elo helpers + Elo→xG cold-start bridge
│   ├── feature_engineering.py  # form / H2H / rest / midweek λ multipliers
│   ├── advanced_features.py     # xG regression-to-mean + injury penalty multipliers
│   ├── fetch_advanced.py        # FBref xG scraper (soccerdata) — opt-in, slow
│   │
│   ├── value_detector.py    # margin removal + fractional Kelly staking
│   ├── ai_advisor.py        # optional Gemini risk-manager verdict on value picks
│   │
│   ├── prediction_saver.py  # legacy save/settle helpers (Bet-table flow)
│   ├── settle_bets.py       # settle Prediction rows against final scores
│   ├── accountant.py        # win-rate / P&L report
│   ├── match_queries.py     # shared future-match SELECTs + bet-history sorting
│   ├── predict.py           # alternate pure-Elo value scan (research path)
│   ├── report.py            # BTTS / projected-score text reports
│   └── api.py               # FastAPI app for the React trading-terminal frontend
├── alembic/                 # migrations (schema history)
├── tests/                   # pytest suite (16 tests)
├── scripts/                 # dev seeding helpers
├── data/betting.db          # SQLite (gitignored)
└── DOCUMENTATION.md         # this file
```

**Data flow:**

```
The Odds API ──odds──▶ matches + odds tables ──▶ GoalEngine.fit_from_matches()
      │                                                      │
      └──scores──▶ completed matches ──▶ Elo backfill        ▼
                                              predict_match(home, away)
FBref (optional) ──xG──▶ match_advanced ──▶ λ multipliers ──▶ 6×6 score matrix
                                                              │
bookmaker line ──▶ remove_margin() ──▶ fair probs ──▶ edge ──▶ Kelly stake
                                                              │
                                            Gemini advisor (confirm / veto)
                                                              │
                                                              ▼
                                            predictions table + console table
```

---

## 4. How a prediction is made

### 4.1 Dixon-Coles goal model (`src/poisson_model.py`)

The core is the Dixon-Coles (1997) extension of the independent-Poisson model:

- Each team gets an **attack** and **defence** parameter, estimated jointly by
  **maximum likelihood** (SLSQP with a sum-to-zero attack constraint).
- A global **home advantage** term is added to the home side's log-λ.
- The **ρ (rho)** parameter corrects the well-known Poisson bias on low-scoring
  results (0-0, 1-0, 0-1, 1-1) — Poisson underestimates draws.

Expected goals for a fixture:

```
λ_home = exp(attack_home_team + defence_away_team + home_advantage)
λ_away = exp(attack_away_team + defence_home_team)
```

The 6×6 **score matrix** `P(home=i, away=j)` for i,j ∈ 0..5 is built from the
two Poissons with the Dixon-Coles τ adjustment, then normalised. From the
matrix: P(home win) = lower triangle, P(draw) = trace, P(away win) = upper
triangle, plus Over 2.5 and BTTS.

**Cold start:** when a team was not in the fitted data, the engine falls back
to **Elo → expected goals** (`src/elo_model.py`). When *neither* team has any
rating history at all, `main.py` refuses to recommend a bet on that match and
labels it *"edge ignored — no team history (cold start)"* — an edge computed
from an uninformative prior is noise, not value.

### 4.2 Feature multipliers (`src/feature_engineering.py`)

When enabled, small multiplicative nudges are applied to λ before building
the matrix (each typically ±a few %; combined result clamped to [0.75, 1.25]):

| Feature | Signal | Default weight |
|---|---|---|
| **Form** | points-per-game over last `FORM_WINDOW` vs ~1.5 baseline | `FORM_WEIGHT=0.08` |
| **H2H** | win balance in last 5 meetings (geometric-mean normalised) | `H2H_WEIGHT=0.06` |
| **Rest** | last match < `REST_SHORT_DAYS` days ago | ×`0.97` |
| **Midweek** | Tue–Thu UTC kickoff (congestion proxy) | ×`0.99` |
| **xG regression** | scoring above/below xG over `XG_WINDOW` games regresses to mean | `XG_WEIGHT` |
| **Injuries** | compounding penalty per key player out/suspended | `INJURY_WEIGHT^n` |

xG and injuries only activate when `match_advanced` / `player_injuries` tables
have data (xG requires the opt-in FBref sync).

### 4.3 Value detection & staking (`src/value_detector.py`)

1. **Margin removal** — bookmaker 1X2 implied probabilities sum to >1
   (the overround). We subtract a constant `c` from each implied probability
   such that `Σ max(pᵢ − c, 0) = 1` (additive de-margining, solved by binary
   search) — this yields the **fair** implied probabilities.
2. **Edge** — `edge_side = model_prob − fair_prob`. A side is a value
   candidate when `edge > EDGE_THRESHOLD` (default 5%).
3. **Stake** — fractional Kelly (`KELLY_FRACTION=0.25` of full Kelly),
   computed from the **model's own probability** and the *actual market odds*
   you would bet at: `f = kelly_fraction × (b·p − q)/b` with `b = odds − 1`.

### 4.4 Persistence

Every displayed match gets exactly **one live `Prediction` row** (the
unsettled row for that match is updated in place on re-runs, so re-running
`main.py` never duplicates). `src/settle_bets.py` later compares the
recommendation with the final score and writes win/loss + profit.

---

## 5. Tournament auto-detection

**The feature:** when a major tournament is in season — **FIFA World Cup,
UEFA Euro, Champions League, Europa League, Copa América** — its matches
automatically join the odds sync, predictions, and API output. During July
2026, for example, the console shows FIFA World Cup fixtures alongside (empty
off-season) domestic leagues with zero configuration.

**How it works** (`src/tournaments.py`):

1. The Odds API's `GET /v4/sports` endpoint lists every competition with an
   `active` flag reflecting real season state. This endpoint is **free** — it
   does not consume request quota.
2. The result is cached in `data/sports_cache.json` for
   `SPORTS_CACHE_TTL_HOURS` (default 12 h).
3. `resolve_sport_keys()` = `DEFAULT_SPORTS` ∪ (candidate tournaments marked
   active). This resolved list drives `run_update_cycle()`,
   `fetch_historical_scores()` and `run_backfill()` — one code path, no
   special-casing per tournament.
4. The frontend can render competition tabs via `GET /tournaments/active`.

Configuration:

```
TOURNAMENT_AUTO_DETECT=true          # master switch
TOURNAMENT_SPORT_KEYS=soccer_fifa_world_cup,soccer_uefa_european_championship,soccer_uefa_champs_league,soccer_uefa_europa_league,soccer_conmebol_copa_america
SPORTS_CACHE_TTL_HOURS=12
```

Add any Odds-API soccer key to the candidate list (e.g.
`soccer_conmebol_copa_libertadores`) and it will light up automatically when
in season.

> **Accuracy caveat:** tournament matches involve national teams the club
> model has never seen. Until international results are backfilled, those
> matches show honest probabilities from an uninformative prior and are
> excluded from bet recommendations (see §4.1 cold start).

---

## 6. The AI advisor (Gemini)

`src/ai_advisor.py` sends the model probabilities, derived markets (BTTS,
away-win combos from the score matrix) and the bookmaker line to Gemini
(`GEMINI_MODEL`, default `gemini-3.1-pro-preview`) acting as a
*risk manager*. It answers with strict JSON: `Bet`/`Skip`, a reasoning
sentence, and a side.

Design rules (enforced in `main.py`):

- The advisor is only consulted for **quant value candidates** — it can
  **veto** a pick or switch the side, but can never invent a bet the model
  didn't find. This caps API cost at (number of value candidates) calls.
- **Fail-open:** if the Gemini API errors (quota, network), the quant pick
  stands, annotated *"AI unavailable — quant only"*.
- **Circuit breaker:** two consecutive API failures disable the advisor for
  the rest of the run.
- No `GEMINI_API_KEY` (or `AI_ADVISOR_ENABLED=false`) → clean pure-quant mode.

---

## 7. Web API (FastAPI)

Start: `venv/bin/python -m uvicorn src.api:app --reload` → docs at
`http://127.0.0.1:8000/docs`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness + number of fitted teams |
| GET | `/tournaments/active` | in-season tournaments (for frontend tabs) |
| GET | `/upcoming?horizon_days=7` | future fixtures incl. `sport_key` |
| GET | `/history?sort_by=pnl&sort_dir=desc&settled_only=true` | settled predictions, DB-side sorting |
| POST | `/predict` `{home_team, away_team, kickoff?}` | on-demand Dixon-Coles prediction, full score matrix |
| POST | `/model/refit` | re-fit the engine without restarting |

The GoalEngine is fitted **once at startup** (lifespan hook) and shared across
requests; `/model/refit` refreshes it after new data lands. CORS defaults
cover common local dev ports; override with `CORS_ORIGINS`.

---

## 8. Helper scripts

| Command | What it does |
|---|---|
| `venv/bin/python view_predictions.py` | dashboard: recent predictions, bet status, 1/7/30-day P&L |
| `venv/bin/python -m src.settle_bets` | settles unsettled predictions against final scores (skips no-bet rows) |
| `venv/bin/python -m src.accountant` | totals: settled bets, win rate, profit |
| `venv/bin/python update_results.py` | legacy settler for the `bets` table flow |
| `venv/bin/python scripts/seed_dev.py` | seed a dev database |

---

## 9. Configuration reference

All settings live in `.env` (see `.env.example`). Key groups:

| Variable | Default | Meaning |
|---|---|---|
| `ODDS_API_KEY` | — | **required** for live fetches |
| `GEMINI_API_KEY` | — | optional; enables AI advisor |
| `AI_ADVISOR_ENABLED` | `true` | master switch for the advisor |
| `GEMINI_MODEL` | `gemini-3.1-pro-preview` | advisor model id |
| `DEFAULT_SPORTS` | EPL, La Liga, Bundesliga | always-on competitions |
| `TOURNAMENT_AUTO_DETECT` | `true` | add in-season tournaments automatically |
| `TOURNAMENT_SPORT_KEYS` | WC, Euro, UCL, UEL, Copa América | candidate tournaments |
| `SPORTS_CACHE_TTL_HOURS` | `12` | /sports detection cache |
| `PREDICTION_HORIZON_DAYS` | unset | only show kickoffs within N days |
| `EDGE_THRESHOLD` | `0.05` | min edge to flag value |
| `KELLY_FRACTION` | `0.25` | fraction of full Kelly |
| `SHOW_ONLY_VALUE_BETS` | `false` | compact value-only console view |
| `FEATURES_ENABLED` | `true` | form/H2H/rest/xG/injury multipliers |
| `FORM_WINDOW` / `FORM_WEIGHT` / `H2H_WEIGHT` | 10 / 0.08 / 0.06 | feature strengths |
| `REST_SHORT_DAYS` / `REST_FATIGUE_FACTOR` | 3 / 0.97 | fatigue rule |
| `MIDWEEK_FATIGUE_FACTOR` | 0.99 | Tue–Thu congestion proxy |
| `XG_SYNC_ENABLED` | `false` | opt-in FBref xG scrape (slow) |
| `XG_WEIGHT` / `XG_WINDOW` / `INJURY_WEIGHT` | 1.05 / 5 / 0.95 | advanced multipliers |
| `SCORES_DAYS_FROM` | `3` | scores look-back (API max 3) |
| `LOG_LEVEL` / `LOG_DIR` / `LOG_FILE` | INFO / data/logs / betting_engine.log | logging |
| `DATABASE_URL` | `sqlite:///data/betting.db` | SQLite only |

**Odds-fetch credit protector:** odds per sport are cached in the DB for
60 minutes — re-running `main.py` within the hour costs zero API credits.

---

## 10. Database schema

ORM models in `src/models.py`; Alembic migrations in `alembic/versions/`.

| Table | Purpose |
|---|---|
| `matches` | fixtures & results; `sport_key` ties a match to its competition |
| `odds` | one row per (match, bookmaker), updated in place with latest 1X2 line |
| `teams` | team registry + legacy `current_elo` |
| `team_ratings` | Elo + fitted attack/defence strengths |
| `predictions` | one live row per match: model probs, market snapshot, fair probs, edges, recommendation, stake, settlement outcome |
| `bets` | optional actual-bet tracking (legacy flow via `prediction_saver`) |
| `match_advanced` | per-match xG (FBref) |
| `team_stats` | rolling aggregates cache |
| `player_injuries` | manual injury/suspension registry feeding the injury penalty |

Fresh install: `venv/bin/python -m alembic upgrade head` (or let `init_db()`
create tables on first run; Alembic is the authoritative history).

---

## 11. Testing & CI

```bash
venv/bin/python -m pytest          # 16 tests, all passing
venv/bin/python -m flake8 --max-line-length=120 src/ main.py
```

Covered: Dixon-Coles matrix properties & τ adjustment, home advantage,
cold-start fallback, Elo updates, margin removal & Kelly, future-match
selection, form/H2H features. GitHub Actions runs migrations + black +
flake8 + pytest.

**Gap (be aware):** there are no tests for `fetch_data` parsing, the
tournament detector, `main.py` orchestration, or the API endpoints. These are
the highest-value next tests to write.

---

## 12. Honest assessment — how to actually improve prediction accuracy

My frank opinion, ordered by expected impact:

### 12.1 The #1 problem is data, not the model *(critical)*

The database currently holds **7 completed matches** while the Dixon-Coles
fit estimates ~118 parameters for 58 teams. That is mathematically hopeless —
the MLE is fitting noise, and in practice most predictions route through the
Elo fallback, which itself has almost no history to learn from. **No model
tweak matters until this is fixed.** The Odds API scores endpoint only looks
back 3 days, so it can never build a season of history.

**Fix (a weekend of work, transformative):** import full historical results —
[football-data.co.uk](https://www.football-data.co.uk/) ships free CSVs with
10+ seasons of every major league *including closing odds* (which also gives
you a backtesting dataset for free). One import script + team-name mapping
table. For national teams, Kaggle's international-results dataset covers
every fixture since 1872. Target ≥2 seasons (~1,500 matches for 3 leagues)
before trusting any fit.

### 12.2 You cannot know your accuracy without backtesting *(critical)*

There is no way today to answer "is the model any good?" Build a walk-forward
backtest: for each past matchweek, fit on everything before it, predict it,
record (a) **Brier score / log-loss** vs. the bookmaker's de-margined
probabilities as the benchmark, and (b) **simulated Kelly P&L at closing
odds**. If the model doesn't beat the de-margined market benchmark, the edge
threshold is just measuring bookmaker margin, not skill. This is the honest
scoreboard every other improvement gets judged against.

### 12.3 Time-decay the fit *(easy win)*

Dixon-Coles' own paper weights matches by `exp(−ξ·days_ago)`. Right now a
match from two years ago counts as much as last week's. Add the ξ weight to
the log-likelihood (~5 lines) and tune ξ on the backtest (typical ξ ≈
0.0018–0.0065/day). Teams change; the model should forget.

### 12.4 Blend with the market instead of fighting it *(the professional trick)*

The closing line is the strongest publicly available predictor. Shrink model
probabilities toward the de-margined market: `p = α·p_model + (1−α)·p_market`
with α tuned by backtest (usually 0.2–0.4 for a young model). This slashes
false "value" flags caused by model error — most of your current +14% "edges"
are the model being wrong, not the market.

### 12.5 Promote xG from a nudge to the signal *(medium effort)*

Goals are ~0.3-goals-per-match noise around xG. Fitting attack/defence on
**xG instead of goals** (once the FBref sync has data) roughly doubles the
effective information per match. Today xG is only a ±5% multiplier; used as
the fit target it's a different class of model.

### 12.6 Calibrate national-team ratings before the next tournament

The cold-start guard now (correctly) refuses to bet on World Cup matches.
To actually predict them: import international history (§12.1), maintain a
separate Elo (or use public Elo ratings as priors), and fit a
tournament-specific home-advantage term (hosts vs. neutral venues behave
differently).

### 12.7 Kill the remaining leakage risks in features

`compute_team_form` filters on `status='completed'` — make sure score updates
land *before* predictions in every code path (they do in `main.py`'s current
order, but nothing enforces it). When backtesting, always pass
`before=kickoff` (the plumbing exists and is correct — keep it that way).

### 12.8 Diversify markets only after 1X2 is calibrated

BTTS/over-under code paths exist, but selling more markets from a mis-calibrated
matrix multiplies losses. Order of operations: data → backtest → calibration
→ then extra markets.

**What I would *not* spend time on yet:** deep learning, player-level models,
more bookmakers, live betting. The marginal value is tiny compared with
§12.1–12.4, and each adds large complexity.

---

## 13. Code health — what was broken and what remains

### 13.1 Fixed in this pass (2026-07-10)

**Crash bugs (the pipeline could not run end-to-end before):**

| # | Bug | Fix |
|---|---|---|
| 1 | `main.py` read `result["odds_added"]` — key doesn't exist → **KeyError** on every run | use real summary keys |
| 2 | `update_results.py` / `view_predictions.py` imported non-existent `src.database.get_session` → **ImportError** | added `get_session()` context manager |
| 3 | `settle_bets.py` referenced `market_odds` before assignment on **every losing bet** → NameError | compute odds before the branch; skip no-bet predictions |
| 4 | `ai_advisor.py` **raised at import time** without `GEMINI_API_KEY`, killing pure-quant runs | lazy client init + `advisor_enabled()` |
| 5 | `fetch_advanced.py` used `session.merge` with autoincrement PK → **IntegrityError** on every re-sync | proper select-then-update upsert |
| 6 | `prediction_saver.get_todays_predictions` used `date.replace(day=day+1)` → **ValueError** on month-end | `timedelta(days=1)` |
| 7 | `tests/test_poisson.py` targeted the pre-rewrite API → ImportError | rewritten for Dixon-Coles |
| 8 | No venv existed; `requirements.txt` missing `scipy` and `google-genai` | env rebuilt with uv; requirements fixed |

**Correctness / robustness:**

- Kelly stake was computed from *raw* implied + *fair* edge — systematically
  over-staking by the margin share; now staked from the model probability.
- `remove_margin` binary search had a wrong upper bound for skewed books.
- Re-running `main.py` created duplicate `Prediction` rows forever; now one
  live row per match, updated in place.
- Odds sync ran inside an open read session (SQLite lock hazard); reordered.
- AI advisor was called for **every** match (cost/quota) and dumped raw JSON
  into the console; now value-candidates-only + fail-open + circuit breaker.
- Cold-start matches produced fake +15–25% "edges" and would have recommended
  bets; now suppressed with an explicit note.
- Elo was written to `team_ratings` but read from `teams.current_elo` (two
  disconnected stores); reads now prefer `team_ratings` with legacy fallback.
- FBref xG scrape (slow, ban-prone, 5–12 s sleeps) ran on **every** pipeline
  start; now opt-in via `XG_SYNC_ENABLED`.
- `httpx`/`google_genai` INFO spam silenced; stray junk files removed
  (`main.py.backup`, `update_results.py.save`, a mis-named module copy,
  `sqlite:/:memory:` artifact, `error.log`).

### 13.2 Known remaining issues (ranked)

1. **Datetime storage is mixed naive/aware UTC.** SQLite stores whatever
   string SQLAlchemy binds; comparisons work today because UTC prefixes sort
   correctly, but it's fragile. Normalize to naive UTC at every write
   boundary and add a regression test.
2. **Two settlement flows.** `predictions` (settle_bets) vs `bets`
   (prediction_saver/update_results) — plus mixed side encodings
   (`H/D/A` vs `home/draw/away`). Pick the `predictions` flow, standardize on
   `H/D/A`, delete the rest.
3. **`src/elo.py` vs `src/elo_model.py`** duplicate Elo math with different
   home-advantage handling. Merge into one module.
4. **Alembic vs `init_db()` drift.** `init_db` creates tables that bypass
   migration history; on a fresh clone, run Alembic first. Long-term: make
   Alembic the only schema creator.
5. **Thin test coverage** on I/O paths (fetchers, tournament detector, API).
6. **`predict.py` / `report.py`** research paths partially overlap the main
   pipeline; fine as scratchpads, but don't let them diverge further.
7. **`GEMINI_MODEL` default** points at a preview model that will eventually
   disappear; pin a stable model when one fits the budget.

---

## 14. Security incident log

**2026-07-10 — API keys exposed in public git history.** `.env` (with live
`ODDS_API_KEY` and `GEMINI_API_KEY`) was tracked from the first commits and
pushed to the public GitHub repo. Response: file untracked and pushed
(`fad8a88`), `.gitignore` verified. **Both keys must be treated as
compromised and rotated** at the-odds-api.com and Google AI Studio — history
still contains them (rewriting history was declined in favour of rotation;
that is fine *once rotation actually happens*). Never commit `.env`; use
`.env.example` as the template.

---

*Documentation generated 2026-07-10. Keep this file updated as the single
source of truth alongside code changes; the README stays a short landing page.*
