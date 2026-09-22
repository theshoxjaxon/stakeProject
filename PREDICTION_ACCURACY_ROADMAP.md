# Improving Prediction Accuracy — a grounded plan

> Written against your actual code: `poisson_model.py`, `feature_engineering.py`,
> `advanced_features.py`, `predict.py`, `elo.py`, `ai_advisor.py`, `models.py`.
> Goal: (1) **measure** how precise the model is, (2) make it **improve over time**,
> (3) add **deep squad / head-to-head / injury analysis** so the AI understands *context*.

---

## The one principle everything hangs on

> **You cannot improve what you don't measure.**

Right now your `Prediction` table stores `was_win` and `profit`, but nothing computes a
**proper score** of how good the probabilities were. So "how precise is the AI?" has no
answer yet, and "make it increase over time" has no signal to optimize. **Build the
measurement layer first.** Everything else in this doc is judged by it.

Think of it like adding a test suite before refactoring: the metrics are your unit tests
for the model. Without them, every "improvement" is a guess.

---

## Part 1 — Measure precision (build this first)

### 1.1 Why accuracy % is the wrong metric

"The model was right 55% of the time" is almost useless for a probabilistic model. If the
model says *home 48%* and home loses, was it wrong? No — it said the other outcomes were
52% combined. You need scores that reward **calibrated probabilities**, not just the
top pick. Three standard ones:

| Metric | What it measures | Good value | Use it for |
|---|---|---|---|
| **Log loss** | Penalizes confident wrong predictions hard | lower is better | primary training/selection metric |
| **Brier score** (multiclass) | Mean squared error of probabilities | lower is better | stable, interpretable |
| **RPS** (Ranked Probability Score) | Brier that respects *ordering* H→D→A | lower is better | **the football standard** for 1X2 |
| **Calibration** | Do "60%" predictions win ~60% of the time? | diagonal | trust / honesty (see §Part 4 of the product doc) |

RPS is the one football researchers report, because it knows that predicting a draw when
the away team wins is "less wrong" than predicting a home win. Report all three; optimize
log loss or RPS.

### 1.2 The scoring code (drop-in)

You already have everything needed in the `Prediction` table (`home_prob`, `draw_prob`,
`away_prob`, `actual_outcome`). Add `src/scoring.py`:

```python
import numpy as np

OUTCOMES = ["home", "draw", "away"]

def log_loss_one(probs: dict, actual: str) -> float:
    p = max(probs[actual], 1e-15)          # clip to avoid log(0)
    return -np.log(p)

def brier_one(probs: dict, actual: str) -> float:
    y = {o: 1.0 if o == actual else 0.0 for o in OUTCOMES}
    return sum((probs[o] - y[o]) ** 2 for o in OUTCOMES)

def rps_one(probs: dict, actual: str) -> float:
    """Ranked Probability Score for ordered outcomes H, D, A."""
    order = ["home", "draw", "away"]
    p_cum, y_cum, total = 0.0, 0.0, 0.0
    for o in order[:-1]:                    # sum over N-1 categories
        p_cum += probs[o]
        y_cum += 1.0 if o == actual else 0.0
        total += (p_cum - y_cum) ** 2
    return total / (len(order) - 1)
```

### 1.3 A `ModelScore` table (track it over time)

This is the table that answers "is it getting better?". Write one row per settled
prediction, then aggregate.

```python
class PredictionScore(Base):
    __tablename__ = "prediction_scores"
    id            = mapped_column(Integer, primary_key=True)
    prediction_id = mapped_column(ForeignKey("predictions.id"), unique=True, index=True)
    model_version = mapped_column(String(40), index=True)  # e.g. "dc-v3+form+inj"
    log_loss      = mapped_column(Float)
    brier         = mapped_column(Float)
    rps           = mapped_column(Float)
    top_pick_hit  = mapped_column(Boolean)   # was the argmax outcome correct
    created_at    = mapped_column(DateTime, default=datetime.utcnow, index=True)
```

Hook it into `settle_bets.py` — the moment a prediction gets its `actual_outcome`, compute
and store its scores. You already loop over pending predictions there; add three lines.

### 1.4 The baseline that keeps you honest

A model is only "good" relative to a baseline. Always compare against:

1. **The bookmaker** — de-margin the market odds into probabilities (you do this already in
   `predict.py` via `implied_probability`) and score *those*. **If your model can't beat
   the de-margined market's RPS, you have no edge — full stop.** This is the single most
   important benchmark; the market is the strongest public model.
2. **Home/draw/away base rates** (~45/27/28% in most leagues) — the dumbest possible model.
3. **Pure Dixon-Coles with no multipliers** — so you can prove each feature *helps*.

Put these side by side in a table and the whole project suddenly has direction.

---

## Part 2 — Make it improve over time

Measurement gives you a number. This part is the loop that drives it down.

### 2.1 Walk-forward backtesting (the correct way to test)

**Never** score the model on matches it was trained on — that's lookahead leakage and it
lies to you. Use **walk-forward validation**: train on everything up to date *T*, predict
match at *T+1*, roll forward.

```
for each match-week W in chronological order:
    fit Dixon-Coles on all completed matches BEFORE W
    predict every match in W
    score predictions against real results
report: rolling RPS / log-loss over the season
```

Your `feature_engineering.py` already respects a `before` cutoff (the `_normalize_before`
plumbing) — that's exactly the discipline walk-forward needs. Build `src/backtest.py` that
replays a season this way and prints the metric table from Part 1. **This becomes your
experiment harness:** change a weight, rerun, see if RPS drops. That is "improving over
time" made concrete.

### 2.2 A/B every change against the harness

Each idea in Part 3 is a hypothesis. Protocol:

1. Snapshot current backtest RPS (the champion).
2. Make one change (e.g. importance-weighted injuries).
3. Rerun backtest. Did RPS drop *and* hold on a held-out later period?
4. Keep it only if yes. Tag the `model_version` so `PredictionScore` remembers which
   version produced which results.

This is real ML engineering — a controlled experiment loop, not vibes. It's also great
FAANG-interview material (offline eval, champion/challenger, guarding against leakage).

### 2.3 Automated retrain + drift detection

- **Retrain cadence:** refit Dixon-Coles after each match-day (you already can). Teams
  change through a season; stale attack/defence params decay accuracy.
- **Drift alarm:** track rolling 30-match RPS. If it climbs above its trailing average by a
  threshold, something broke (data feed, a new season, a rule change) — alert yourself.
- **Weight tuning:** your multiplier weights (`FORM_WEIGHT`, `H2H_WEIGHT`, `INJURY_WEIGHT`,
  `XG_WEIGHT`, rest/midweek factors) are hand-set. Once the backtest exists, **fit them**:
  grid-search or Bayesian-optimize them to minimize walk-forward log loss. Hand-tuned →
  data-tuned is often the single biggest accuracy jump available to you.

### 2.4 Calibration correction (cheap, high-impact)

Models are usually over- or under-confident. After you have scored history, fit a simple
**calibration map** (isotonic regression or Platt scaling) that nudges raw probabilities
toward their empirically observed frequencies. It's a small post-processing step that
reliably improves log loss and makes your displayed percentages *true* — which matters for
the trust/sharing loop in the product doc.

---

## Part 3 — Deep squad / H2H / injury analysis (what you described)

This is where you asked for the AI to "analyse how they played before they met, when they
met, with the squad they met, if a superstar or top scorer is injured." Here's how to build
that into the model, in priority order.

### 3.1 Fix the injury model — use the `importance` column you already have

Your `PlayerInjury` table has an `importance` field (1–5), but `get_injury_penalty()`
**ignores it** and just counts bodies:

```python
# CURRENT — a benchwarmer hurts as much as a star:
penalty = INJURY_WEIGHT ** unavailable_players_count
```

Replace with an **importance-weighted** penalty so a top scorer out matters far more than a
squad player:

```python
def get_injury_penalty(team_name, db_session) -> float:
    rows = db_session.execute(
        select(PlayerInjury.importance).where(
            PlayerInjury.team_name == team_name,
            PlayerInjury.status.in_(["out", "suspended"]),
        )
    ).scalars().all()
    penalty = 1.0
    for importance in rows:
        # importance 5 (talisman) bites hard; 1 (fringe) barely moves λ
        penalty *= 1.0 - INJURY_UNIT * (importance / 5.0)
    return max(0.7, penalty)     # floor so one injury can't zero-out a team
```

**Bigger win — model the top scorer explicitly.** A team's expected goals should drop by
that player's *share of goal contribution*. If your striker scored 40% of the team's goals
and he's out, the attack λ should fall meaningfully — not by a generic constant.

```python
def scorer_absence_multiplier(team, db_session) -> float:
    """Reduce attack λ by the injured players' share of season goal contributions."""
    contributions = get_goal_contribution_shares(team, db_session)   # {player: share}
    lost = sum(share for player, share in contributions.items()
               if is_unavailable(player, db_session))
    # If 40% of goals are unavailable, damp attack — but not linearly (others step up)
    return max(0.65, 1.0 - GOAL_SHARE_WEIGHT * lost)
```

This requires **player-level goal data** — see §3.5 on data sources. This one feature
("is the guy who scores the goals actually playing?") is often more predictive than form.

### 3.2 Squad-aware, recency-weighted H2H

Your `compute_h2h()` uses a raw win-difference over the last 5 meetings. Two problems:
old meetings had **different squads** (players and managers turn over every 2–3 years), and
a 5-year-old result gets the same weight as last season's. Upgrade it:

- **Time-decay the meetings:** weight recent H2H exponentially more (`0.5 ** (years_ago)`).
  A meeting from this season tells you far more than one from 2019 with half the players
  gone.
- **Squad-continuity discount:** if you have lineup data, down-weight past meetings where
  few of today's starters were on the pitch. "This fixture" means little if it's really two
  different teams now. Even a rough proxy — squad turnover % since the meeting — helps.
- **Style/context, not just result:** store *how* the games went (xG, shots) so a team that
  lost but dominated on xG isn't punished for one unlucky result.

### 3.3 Richer "how they played before" form (you're halfway there)

`compute_team_form()` already does PPG, GF/GA, and a `WWDLW` string. Deepen it:

- **Weight by opponent strength.** 3 wins vs. relegation sides ≠ 3 wins vs. the top 4.
  Multiply each result by the opponent's Elo (you have `elo.py`) so form reflects
  *quality of opposition*, not just points.
- **Recency decay inside the window** — last match > 5 matches ago.
- **Use xG form, not just goals** (`get_rolling_stats` already gives `avg_xg_for/against`).
  A team winning while getting out-xG'd is riding luck and about to regress; xG form catches
  that earlier than goal form. This is the "played shit but got results" case you described.
- **Separate home/away form** — some teams are fortress-at-home, timid-away.
- **Momentum/trend:** is form rising or falling across the window, not just its average.

### 3.4 Turn on the manual context the AI advisor can see

You already call Gemini in `ai_advisor.py`. Right now it only gets model probabilities.
Feed it the **context signals** so it can reason about exactly the scenarios you listed —
"top scorer injured," "played poorly recently," "dominant in this fixture":

```
prompt context to add:
- home/away form (PPG, xG form, opponent-adjusted), last 5 with a one-line read
- H2H recency-weighted summary + squad-continuity note
- injury list WITH importance + goal-contribution share of anyone out
- rest days / midweek / travel
Ask it: does the statistical model miss any of this context? Confirm or flag, with reason.
```

Keep the statistical model as the source of truth and the LLM as a **context-aware
sanity check / explainer**, never the primary number. That gives you both rigor and the
human-readable "why" that makes users trust and share picks.

### 3.5 The data you'll need (and how to get it)

Deep squad analysis needs **player- and lineup-level data**, which your odds feed doesn't
carry. You already have `soccerdata` in `requirements.txt` — that's your route:

- **FBref / Understat (via `soccerdata`):** player goals, assists, minutes, xG, and match
  lineups. This unlocks §3.1 goal-share and §3.2 squad-continuity.
- **Injury/suspension feeds:** populate `PlayerInjury` reliably (right now it looks
  manually/seed-filled). A scheduled scraper or an injury API keeps it fresh — stale injury
  data is worse than none.
- Store lineups per match (a `MatchLineup` table) so H2H can compute squad overlap and so a
  future model can price *predicted starting XI* strength, not just team-level averages.

**Sequence:** you can't do §3.1/§3.2 well until the player data pipeline exists, so land
that pipeline before the fancy features. Injury-importance fix (§3.1 first half) and
opponent-adjusted form (§3.3) work with data you already have — do those first.

---

## Part 4 — Where accuracy gains actually come from (priority order)

Ranked by typical bang-for-buck on a Dixon-Coles setup like yours:

1. **Beat the market benchmark & fit your weights to it (Parts 1–2).** Measurement +
   data-tuned weights. Biggest, safest win. Do this first.
2. **Blend model with market.** The de-margined market is a strong predictor. A weighted
   average of *your* probs and *market* probs almost always scores better than either alone.
   Start ~50/50, tune the blend on the backtest.
3. **Importance/goal-share injury model (§3.1).** High signal, moderate effort once player
   data lands.
4. **Opponent-adjusted + xG form (§3.3).** Uses data you already have.
5. **Ensemble/stacking.** You already import `xgboost` and have `elo.py`. Train a small
   gradient-boosted model on your features (form, Elo diff, xG, rest, injuries) and
   **stack** it with Dixon-Coles + Elo — a meta-model learns when to trust which. This is
   the path to a genuinely strong system, but only *after* the measurement harness exists,
   or you're flying blind.
6. **Squad-continuity H2H (§3.2).** Nice, lower marginal gain — do it last.

> Reality check to keep expectations sane: good public football models land around
> **RPS ≈ 0.19–0.21** and only *narrowly* beat the closing market, if at all. The market
> aggregates enormous information. Your realistic aim is to be **competitive with the
> market and beat it in specific niches** (lower leagues, early lines before they sharpen),
> not to crush it everywhere. Measuring against the market (Part 1) keeps you honest about
> whether you're actually there.

---

## Part 5 — Concrete next steps

1. **`src/scoring.py`** — log loss, Brier, RPS (§1.2). Small, no dependencies.
2. **`PredictionScore` table** + Alembic revision; compute scores inside `settle_bets.py` (§1.3).
3. **`src/backtest.py`** — walk-forward replay of last season, print the metric table incl.
   the market baseline (§1.4, §2.1). This is your experiment harness.
4. **Fix `get_injury_penalty`** to use `importance` (§3.1) — one function, immediate.
5. **Opponent-adjust `compute_team_form`** using Elo (§3.3) — data you already have.
6. **Player-data pipeline** via `soccerdata` (FBref/Understat) → goal-share + lineups (§3.5).
7. **Fit the multiplier weights** to minimize backtest log loss (§2.3).
8. **Add market blend** and re-measure (§4.2).

Do 1–3 before anything else. Once you can see RPS move, every later change becomes a
measurable experiment instead of a guess — and *that* feedback loop is what makes the model
"increase over time."

*Note: this improves prediction quality, not certainty. Football is high-variance; even a
perfectly calibrated model loses plenty of individual bets. The metrics tell you the
probabilities are honest — they can't make an uncertain sport certain.*
