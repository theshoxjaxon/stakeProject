# Quantitative Football Betting Engine

Python 3.11+ project that estimates 1X2 probabilities with a **Poisson goal model**, compares them to bookmaker odds (with **margin removal**), and suggests **fractional Kelly** stakes. Optional **form, head-to-head, rest, and midweek** adjustments nudge expected goals (λ) before building the score matrix.

---

## Features

| Area | Description |
|------|-------------|
| **Odds & data** | The Odds API v4 for fixtures and h2h (1X2) lines; optional scores endpoint for recent results |
| **Poisson model** | `GoalEngine` fits attack/defense from completed matches, with **home vs away** strength split |
| **Feature adjustments** | Rolling **form** (PPG), **H2H** record, **short rest** fatigue, **midweek** kickoff proxy |
| **Value** | Additive de-margining of 1X2 implieds; edge vs “fair” implied; Kelly with configurable fraction |
| **Persistence** | SQLAlchemy + SQLite (`data/betting.db`), Alembic migrations |
| **Safety** | Future-only match selection (UTC-consistent), secrets via `.env` |

---

## Repository layout

```
├── main.py                 # CLI: backfill → fit model → fetch odds → value table
├── src/
│   ├── config.py           # Environment-driven settings (see below)
│   ├── logger.py           # Console + rotating file logging
│   ├── models.py           # ORM models (Alembic source of truth)
│   ├── database.py         # Engine helpers + re-exports
│   ├── match_queries.py    # Future-only match selection + horizon
│   ├── feature_engineering.py  # Form / H2H / rest multipliers
│   ├── poisson_model.py    # GoalEngine + Poisson matrix
│   ├── value_detector.py   # Margin removal + Kelly
│   ├── fetch_data.py       # Odds API sync
│   ├── predict.py          # Alternate Elo→xG path + value lines (logging)
│   └── report.py           # BTTS-style text reports (logging)
├── alembic/                # Migrations
├── tests/                  # pytest
├── scripts/                # seed_dev.py, seed_data.py, dump_seed_sql.py
└── .env.example            # Copy to `.env`
```

---

## Quick start

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # add ODDS_API_KEY
python -m alembic upgrade head
python main.py
```

**Development seed data** (optional):

```bash
python scripts/seed_dev.py
# or
python scripts/seed_data.py
```

---

## Configuration

All tunables live in **environment variables** (and optionally a **`.env`** file next to the project). See **`.env.example`** for names and defaults.

| Variable | Role |
|----------|------|
| `DATABASE_URL` | SQLite URL, e.g. `sqlite:///data/betting.db` |
| `ODDS_API_KEY` | The Odds API key (required for live fetches) |
| `PREDICTION_HORIZON_DAYS` | Only show kickoffs within the next *N* days (omit = no cap) |
| `EDGE_THRESHOLD` | Minimum edge vs fair implied (e.g. `0.05` = 5%); labels “value” in full output |
| `SHOW_ONLY_VALUE_BETS` | `true` = compact table with only sides above `EDGE_THRESHOLD` (default `false`) |
| `KELLY_FRACTION` | Fractional Kelly multiplier (default `0.25`) |
| `FEATURES_ENABLED` | `true`/`false` — apply form/H2H/rest to λ in `GoalEngine` |
| `FORM_WINDOW` | Last *N* games for form |
| `FORM_WEIGHT`, `H2H_WEIGHT` | Strength of λ nudges |
| `REST_SHORT_DAYS`, `REST_FATIGUE_FACTOR` | Fatigue if last match was fewer than *N* days ago |
| `MIDWEEK_FATIGUE_FACTOR` | Extra nudge for Tue–Thu UTC kickoffs |
| `LOG_LEVEL`, `LOG_DIR`, `LOG_FILE` | Logging |

**Never commit** `.env` or `*.db` (already gitignored).

---

## How prediction works

### 1) `main.py` pipeline

1. Initialize DB; if empty, **backfill** recent scored matches (Odds API scores window is small—see `src/backfill.py`).
2. **`GoalEngine.fit_from_matches`**: from completed fixtures, estimate team attack/defense; maintain **home vs away** splits when enough games exist.
3. **`run_update_cycle`**: pull upcoming events + h2h odds into `matches` / `odds`.
4. **`matches_for_prediction()`**: kickoff after now (UTC), excludes terminal statuses (`completed`, `finished`, `cancelled`, `postponed`, `abandoned`), optional `PREDICTION_HORIZON_DAYS` upper bound, **no SQL row limit**, sorted **soonest first**.
5. For **each** match with 1X2 odds: model **H/D/A** %, de-margined fair %, **edge %** on each side, **Kelly stake %** when edge exceeds `EDGE_THRESHOLD`. Use **`SHOW_ONLY_VALUE_BETS=true`** for the legacy compact “value lines only” view.
6. Matches **without** odds rows are skipped; the footer log reports how many were skipped.

### 2) Feature integration (form / H2H / rest)

When `FEATURES_ENABLED` is true and `session` + `kickoff` are passed:

- **Form**: last `FORM_WINDOW` completed games → points per game vs a ~1.5 PPG baseline → small λ multiplier.
- **H2H**: last few meetings between the two clubs → tilt toward the side that dominated (normalized so the geometric mean of multipliers stays ~1).
- **Rest**: if days since last completed game &lt; `REST_SHORT_DAYS`, apply `REST_FATIGUE_FACTOR`.
- **Midweek**: Tue–Thu UTC kickoffs apply `MIDWEEK_FATIGUE_FACTOR` to both sides (congestion proxy).

These are **conservative** multipliers (typically a few percent), to avoid overfitting thin samples.

### 3) Alternate path: `src/predict.py`

`run_value_detection()` uses the **Elo → expected goals** path (`elo_model`) with the same **future-only** match filter. It is useful for cross-checking or research; **`main.py`** is the integrated Poisson + features pipeline documented above.

---

## Database & migrations

- **ORM**: `src/models.py` (`Base`, `Team`, `Match`, `Odds`, `TeamRating`).
- **Alembic**: `alembic.ini` + `alembic/env.py`; metadata from `src.models`.
- **Fresh install**:

  ```bash
  python -m alembic upgrade head
  ```

- **Existing pre-Alembic DB** with the same tables: **`alembic stamp head`** once (see migration notes in repo history).

---

## Logging

`src/logger.py` configures the **root** logger:

- **Console** (stdout)
- **Rotating file** under `data/logs/` (default `betting_engine.log`)

`main.py`, `predict.py`, and `report.py` log through `get_logger(__name__)`. Adjust `LOG_LEVEL` for DEBUG during development.

---

## Testing & CI

```bash
python -m pytest
```

GitHub Actions (`.github/workflows/ci.yml`): install deps, **`alembic upgrade head`** on a temp SQLite DB, **black --check**, **flake8**, **pytest**.

---

## Design notes & limitations

- **SQLite** stores datetimes without time zone; the app compares using **UTC-consistent** “now” and normalizes where needed (`match_queries`, `feature_engineering`).
- **Form/H2H** need enough **completed** rows in `matches`; thin data → multipliers stay near 1.0.
- **Odds API** `/odds` returns **upcoming** events only; scores endpoint is **limited to a few days**—full season history needs another data source.
- **No** `predictions` / `bets` tables yet—extend `models.py` + Alembic when you want audited bet tracking.

---

## Answers to common design questions

1. **Integrating form/H2H into Poisson** — Apply **multipliers to λ_home / λ_away** after baseline λ from strengths; renormalize gently (see `compute_lambda_multipliers`).
2. **Storing rolling metrics** — Computed **on demand** from `matches` here; optional future table: `team_form_cache(team, window, json, updated_at)` if profiling shows hotspots.
3. **Performance** — Index on `matches.date` + filter by `status` and team names; batch predictions; add caching only after measuring.
4. **Pipeline** — `fit_from_matches` (offline) → per-match `predict_match` with `FeatureContext` (session + kickoff).
5. **Timezones** — Store **UTC**; compare with `datetime.now(timezone.utc)` or naive UTC consistently.
6. **Testing DB code** — In-memory SQLite + `Base.metadata.create_all` (see `tests/`).
7. **Caching** — Start without; add LRU or DB cache for `compute_team_form` if needed.

---

## License / disclaimer

This software is for **education and research**. Betting involves risk; past model performance does not guarantee future results. Comply with local laws and API terms of service.
