# EdgeAI — Product Improvement Roadmap

> **From:** single-user quant engine (Dixon-Coles model + value detector + accountant)
> **To:** multi-user prediction & betting product with accounts, single/multi bet slips, and social retention.
> **Stack target:** Python + FastAPI (your current backend) · SQLite → PostgreSQL · React frontend.
> **Status of this doc:** high-level roadmap. Schemas and endpoints are sketches to build from, not final migrations.
> **Disclaimer:** if this ever touches real money, betting is regulated — see §7 before writing a payment line.

---

## 0. TL;DR — what changes and why

You already have the hard part: a model that produces **probabilities, edges, and Kelly stakes** per match. Today it's single-user — one bankroll, one `Bet` per `Prediction`. To turn it into a product you add three layers on top of that engine:

1. **Identity** — every person gets an account and their own wallet/bankroll.
2. **Bet slip** — users assemble one or more selections into a *single* or *multiple (parlay)* slip before confirming.
3. **A retention loop** — the prediction itself becomes the hook: track user picks vs. the model, leaderboards, streaks, shareable results.

The model stays exactly where it is. You are wrapping it, not rewriting it.

```
                    ┌─────────────────────────────────────────┐
   EXISTING (keep)  │  Dixon-Coles → value_detector → Kelly    │
                    │  Prediction table (probs, edges, stake)  │
                    └───────────────────┬─────────────────────┘
                                        │ read-only
   NEW (build)      ┌───────────────────▼─────────────────────┐
                    │  Users · Wallets · BetSlips · Selections │
                    │  Auth · Settlement · Social · Notifs     │
                    └─────────────────────────────────────────┘
```

---

## 1. Multi-user accounts + wallets

### 1.1 The core idea (intuition first)

Right now "the bankroll" is a global number in `accountant.py`. In a multi-user app, **bankroll becomes a column on a user's wallet**, and everything the accountant does (stake sizing, P/L) gets scoped by `user_id`. Think of it like going from a single global variable to an instance per user.

### 1.2 New tables (sketch)

```python
# src/models.py — new models, same Base

class User(Base):
    __tablename__ = "users"
    id            = mapped_column(Integer, primary_key=True)
    email         = mapped_column(String(255), unique=True, index=True, nullable=False)
    username      = mapped_column(String(40), unique=True, index=True, nullable=False)
    password_hash = mapped_column(String(255), nullable=False)   # bcrypt/argon2 — never plaintext
    display_name  = mapped_column(String(80))
    avatar_url    = mapped_column(String(255))
    is_verified   = mapped_column(Boolean, default=False)
    created_at    = mapped_column(DateTime, default=datetime.utcnow, index=True)

class Wallet(Base):
    __tablename__ = "wallets"
    id            = mapped_column(Integer, primary_key=True)
    user_id       = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), unique=True, index=True)
    currency      = mapped_column(String(8), default="COINS")   # play-money by default
    balance       = mapped_column(Numeric(14, 2), default=1000) # starting bankroll
    locked        = mapped_column(Numeric(14, 2), default=0)    # staked but unsettled

class LedgerEntry(Base):
    """Append-only money log — never mutate balance without a row here."""
    __tablename__ = "ledger"
    id            = mapped_column(Integer, primary_key=True)
    wallet_id     = mapped_column(ForeignKey("wallets.id"), index=True)
    kind          = mapped_column(String(20))   # deposit, stake, payout, void, bonus, adjustment
    amount        = mapped_column(Numeric(14, 2))   # signed: -stake, +payout
    balance_after = mapped_column(Numeric(14, 2))
    ref_slip_id   = mapped_column(ForeignKey("bet_slips.id"), nullable=True)
    created_at    = mapped_column(DateTime, default=datetime.utcnow, index=True)
```

**Rule that saves you later:** never write `wallet.balance -= x` on its own. Every balance change is a `LedgerEntry`, and `balance_after` is recomputed from it. This is the single most important design decision for a money app — it makes bugs auditable and disputes resolvable. It's the double-entry accounting idea; treat the wallet as a cache of the ledger's sum.

### 1.3 Auth flow

Use **JWT access + refresh tokens** (FastAPI has first-class support via `fastapi.security`).

```
POST /auth/register  → create User + Wallet(balance=1000) in one transaction
POST /auth/login     → verify password (passlib/argon2) → return access(15min)+refresh(30d)
POST /auth/refresh   → new access token
GET  /me             → current user + wallet (Depends(get_current_user))
```

Every existing prediction/bet endpoint gains a `Depends(get_current_user)` guard. The `get_current_user` dependency decodes the JWT, loads the user, and injects it — that one dependency is how "single-user" becomes "per-user" across the whole API with minimal churn.

**Packages to add:** `python-jose[cryptography]` (JWT), `passlib[argon2]` (hashing), `pydantic[email]`.

### 1.4 Migration note

You're on SQLite with Alembic already. Add these as a new Alembic revision. **Before real users, move to PostgreSQL** — SQLite's single-writer lock will choke the moment two people place bets at once. Postgres also gives you `SELECT ... FOR UPDATE`, which you need for safe balance updates (§2.4).

---

## 2. Single & multiple bet slips

### 2.1 The model shift

Today: `Bet` is one row tied to one `Prediction` (`unique=True`). That's a *single*. A **multiple/parlay/accumulator** is several selections combined into one wager where **all legs must win** and the odds **multiply**. So one `Bet` row can't represent it. Split into two tables:

- **BetSlip** = the wager the user confirmed (has one stake, one payout).
- **BetSelection** = each leg (one per match/market on the slip).

A single = a slip with 1 selection. A multiple = a slip with 2+ selections.

```python
class BetSlip(Base):
    __tablename__ = "bet_slips"
    id             = mapped_column(Integer, primary_key=True)
    user_id        = mapped_column(ForeignKey("users.id"), index=True)
    slip_type      = mapped_column(String(12))   # 'single' | 'multiple'
    stake          = mapped_column(Numeric(14, 2))
    total_odds     = mapped_column(Float)         # product of leg odds (1.0 for pending calc)
    potential_payout = mapped_column(Numeric(14, 2))
    status         = mapped_column(String(12), default="pending")  # pending|won|lost|void|cashed_out
    profit         = mapped_column(Numeric(14, 2), default=0)
    placed_at      = mapped_column(DateTime, default=datetime.utcnow, index=True)
    settled_at     = mapped_column(DateTime, nullable=True)

class BetSelection(Base):
    __tablename__ = "bet_selections"
    id             = mapped_column(Integer, primary_key=True)
    slip_id        = mapped_column(ForeignKey("bet_slips.id", ondelete="CASCADE"), index=True)
    prediction_id  = mapped_column(ForeignKey("predictions.id"), index=True)  # link back to the model
    match_id       = mapped_column(ForeignKey("matches.id"), index=True)
    selection      = mapped_column(String(10))   # 'home'|'draw'|'away'
    odds           = mapped_column(Float)         # odds locked at placement time
    model_edge     = mapped_column(Float)         # snapshot of edge_used — for "did the model like this?"
    status         = mapped_column(String(12), default="pending")  # pending|won|lost|void
```

Keep the old `Bet` table for backward compatibility, or migrate its rows into single-selection slips and retire it. A single-selection slip fully subsumes the old model.

### 2.2 Single vs. multiple — the math

```
Single:    payout = stake × odds
Multiple:  total_odds = odds₁ × odds₂ × … × oddsₙ
           payout     = stake × total_odds
```

A €10 treble at 2.47 × 3.60 × 2.21 pays €196.60 — but **any single leg losing voids the whole slip**. That risk/reward asymmetry is exactly why parlays are engaging and why you should surface the combined model probability (see §3.3) so users see how unlikely the payout actually is.

### 2.3 Placement flow (endpoints)

```
POST /slips                 body: {slip_type, stake, selections:[{prediction_id, selection}]}
GET  /slips?status=pending  user's open slips
GET  /slips/{id}            one slip + legs
POST /slips/{id}/cashout    (optional, advanced — settle early at a discount)
```

Server-side placement must, **inside one DB transaction**:

1. Load each `prediction_id`, read the **current** odds for that selection (don't trust odds sent by the client — recompute or re-fetch; clients lie).
2. Reject if any match has already kicked off (`match.commence_time <= now`).
3. Compute `total_odds` and `potential_payout`.
4. Check `wallet.balance >= stake`; deduct stake → `locked`, write a `stake` ledger entry.
5. Create `BetSlip` + `BetSelection` rows.
6. Commit. If anything fails, roll back the whole thing.

### 2.4 Concurrency (the bug that bites everyone)

Two slips placed at once can both read `balance = 100` and both deduct `80`, leaving `-60`. Prevent it with a row lock:

```python
wallet = session.execute(
    select(Wallet).where(Wallet.user_id == uid).with_for_update()
).scalar_one()
if wallet.balance < stake:
    raise HTTPException(400, "Insufficient balance")
```

`with_for_update()` needs Postgres (another reason to migrate off SQLite).

### 2.5 Settlement (extend what you have)

Your `settle_bets.py` already settles single bets against `Match` results. Extend it:

```
for each pending selection whose match finished:
    selection.status = 'won' if selection.selection == match.outcome else 'lost'
for each pending slip:
    if any selection lost   → slip lost,  profit = -stake, release locked
    elif all selections won → slip won,   payout = stake × total_odds
                              → ledger 'payout' entry, balance += payout
    (void legs → drop from product, recompute odds)
```

Run it on the schedule/cron you already use for `update_results.py`.

---

## 3. Keeping an eye on the prediction (the core hook)

This is the part you flagged hardest. The prediction is your differentiator — a plain sportsbook doesn't tell you the model *disagrees* with your pick. Make the model visible at every step.

### 3.1 On the bet slip, per leg

When a user adds a selection, attach the model's read to it (you already compute all of this):

- **Model probability** for that outcome (the 48% / 21% / 31% bars in your UI).
- **Edge** (`+7.8%` H, etc.) — green if the model agrees, red if it doesn't.
- **Kelly stake** suggestion.
- A one-line verdict: *"Model backs this (+7.8% edge)"* or *"⚠️ Model rates this a poor bet (−11.6% edge)."*

This turns every bet into a teachable moment and nudges users toward +EV picks — which also improves their results, which improves retention.

### 3.2 A personal prediction feed

Give each user a `/feed` that ranks upcoming value bets **for them** — filtered by leagues they follow, sorted by edge. This is the "come back tomorrow" surface. Pair it with a scheduled morning push: *"3 new value bets in La Liga today, best edge +9.2%."*

### 3.3 Combined model probability for multiples

For a parlay, multiply the model probabilities of the legs (approx, assuming independence) and show it next to the payout:

```
Model says this treble hits ~14% of the time. Payout 19.7×.
Fair odds ≈ 7.1× → the 19.7× line is +EV.  (or: −EV, avoid)
```

Showing *"model thinks this is 14% likely"* next to a tempting payout is honest **and** sticky — users learn to trust the number.

### 3.4 Track user vs. model (the retention engine)

For every settled slip, store both outcomes so you can tell each user:

- Their **ROI** and record (you have `accountant.py` / `report.py` — scope it per user).
- **Model-agreement rate**: what % of their bets the model endorsed.
- **"Following the model" shadow P/L**: what they'd have made betting only model-endorsed picks. When that number beats their actual, it's a powerful nudge; when it's below, it's bragging rights. Either way they check it.

### 3.5 Prediction accuracy transparency

Publish the model's rolling calibration/ROI (from your prediction-tracking tables). Trust is the moat — a betting app people believe is honest gets shared. A public "model went 11–4 last week, +6.3% ROI" card is free marketing.

---

## 4. Retention & virality — make them stay and share

Grouped by the psychology each one hits.

### 4.1 Habit (come back daily)

- **Daily streaks** — place or log in N days running; streak freezes you can earn. (Duolingo's whole retention model.)
- **Daily free coins** (play-money) — a small login reward.
- **Morning value-bet push** — scheduled notification tied to §3.2.
- **Match-day live updates** — WebSocket score/settlement pushes so the app is worth opening during games.

### 4.2 Progress (feel yourself improving)

- **Levels / XP** from volume and accuracy.
- **Badges**: "Beat the model," "5-leg parlay hit," "10-day streak," "First green week."
- **Personal stats dashboard** — ROI curve, best pick, model-agreement %, biggest win.

### 4.3 Competition (social pressure)

- **Leaderboards** — weekly/monthly by ROI or profit; global and friends-only. Reset weekly so newcomers can win.
- **Private leagues** — invite friends, compete in a closed group (huge for organic growth; each league is an invite funnel).
- **Copy/tail a bet** — see what top users or the model picked and one-tap replicate.

### 4.4 Sharing (bring others in)

- **Shareable bet slips** — a clean image/card of a slip (legs, odds, payout, model verdict) with a deep link. Winners *want* to post these.
- **Result cards** — auto-generated "I went +€240 this week" graphics for socials.
- **Referral rewards** — both parties get bonus coins; ties into the wallet/ledger you already built.
- **Challenge a friend** — "I bet Alavés win, you take the draw" head-to-head.

### 4.5 Cheapest high-impact wins first

If you build only three of these, build: **private leagues** (viral invite loop), **shareable slip cards** (free acquisition), and **daily push tied to the model feed** (habit). Everything else compounds on those.

---

## 5. Server architecture (how to run it)

```
FastAPI app
├── /auth        register / login / refresh          (JWT)
├── /me          profile + wallet
├── /matches     upcoming + model probs (existing, now auth-guarded)
├── /predictions value bets, edges, Kelly (existing)
├── /slips       create / list / detail / cashout
├── /wallet      balance + ledger history
├── /social      leaderboards, leagues, follow, share
└── /ws          live scores + settlement (WebSocket)

Background workers (extend your existing scripts)
├── fetch odds & results        (existing cron)
├── refit Dixon-Coles           (existing)
├── settle_slips                (extend settle_bets.py → §2.5)
└── send notifications          (new)
```

### 5.1 Concrete recommendations

- **DB:** migrate SQLite → **PostgreSQL** before launch (row locks, concurrency, JSON columns for flexible stats). Keep Alembic.
- **Cache/queue:** add **Redis** for sessions, leaderboards (sorted sets are perfect for rankings), rate limiting, and as a **Celery/RQ** broker for the workers.
- **Money integrity:** append-only ledger (§1.2), row locks on wallets (§2.4), every mutation in a transaction. Add DB constraints: `balance >= 0`, `stake > 0`.
- **Idempotency:** give slip placement an idempotency key so a double-tap/retry doesn't place two bets.
- **Rate limiting** on `/auth` and `/slips` (slowapi).
- **Odds are server-authoritative** — never settle or price from client-supplied odds.
- **Observability:** structured logs (you have `logger.py`), plus per-user P/L and model-ROL dashboards.

### 5.2 Layering (keep it clean for FAANG-style review)

```
routers/   thin — parse, auth, return          (FastAPI endpoints)
services/  business logic — place_slip(), settle(), price()
repositories/  DB access — no business rules
models/    SQLAlchemy tables (you have this)
schemas/   Pydantic in/out DTOs
```

Your model code (`poisson_model`, `value_detector`, `elo`, …) stays untouched under `services/` as the "prediction engine" the betting layer calls into. This separation is exactly what a system-design interviewer wants to see.

---

## 6. UI architecture (how to manage the front end)

### 6.1 State — the bet slip is the hard part

The bet slip is **global, cross-page state**: a user adds a leg on the matches page, navigates, and the slip persists. Model it as one store (Zustand/Redux/Context):

```ts
betSlip: {
  type: 'single' | 'multiple',
  legs: Array<{ matchId, selection, odds, modelEdge, modelProb }>,
  stake: number,
  // derived (compute, don't store):
  totalOdds:  legs.reduce((a,l) => a * l.odds, 1),
  payout:     stake * totalOdds,
  modelProb:  legs.reduce((a,l) => a * l.modelProb, 1),  // combined model likelihood
}
```

- **Single vs multiple** is a tab/derived from `legs.length`: 1 leg = single, 2+ = multiple. Let users toggle "combine into multiple" vs "place as separate singles."
- Compute `totalOdds`/`payout` on the client for instant feedback, but the **server recomputes on placement** (§2.3) — the client number is a preview, not the truth.
- Show the model verdict per leg right in the slip (§3.1).

### 6.2 Suggested screens

```
Live Matches   (you have this) → tap outcome → adds leg to slip
Bet Slip       drawer/panel: legs, stake, single/multiple toggle, model verdict, place
My Bets        open + settled slips, live status
Wallet         balance, ledger history, deposit/claim coins
Profile/Stats  ROI curve, model-agreement %, badges, streak
Leaderboard    weekly/friends, private leagues
Feed           personalized value bets (§3.2)
```

### 6.3 Practical notes

- **React Query / SWR** for server data (matches, predictions, slips) — caching + background refetch for free.
- **WebSocket** channel for live scores and instant settlement toasts.
- **Optimistic UI** on "add to slip" and "place bet," reconcile with server response.
- **Reuse your model-probability bar** (the H/D/A component) everywhere a selection appears — in the slip, in the feed, on share cards. It's your signature visual.
- Keep the dark quant aesthetic; it reads as "serious tool," which builds the trust from §3.5.

---

## 7. Money model & compliance (read before you charge anyone)

You said "not sure yet," so here's the fork:

### Play-money / social (recommended to start)

- Virtual **coins**, no cash out. Leaderboards, streaks, bragging rights.
- **No gambling license needed** in most places → you can ship fast and iterate on the fun.
- The wallet/ledger design above is identical — you just never connect a payment rail.
- Monetize via cosmetics, premium stats, ad-free, or a Pro tier for deeper model insight.

### Real-money

Different universe. Before a single wager touches real cash you need, depending on jurisdiction:

- A **gambling/betting license** (per-country, often per-state; e.g. UKGC, MGA, US state-by-state). Non-negotiable and the long pole.
- **KYC/AML** — identity verification, age gating (18+/21+), sanctions screening.
- **Responsible-gambling** tooling — deposit limits, self-exclusion, reality checks. Often legally mandated.
- **Payment processing** built for gambling (many mainstream processors refuse it).
- **Geofencing** and per-region legality checks.

**Recommendation:** build and launch **play-money**. It exercises the entire product (accounts, slips, model hook, social) with zero regulatory exposure, proves retention, and — if you later go real-money — the architecture already fits because the ledger was money-safe from day one. Treat real-money as a business/legal decision, not a coding sprint.

*This is general information, not legal advice — consult a gambling-law specialist for your target markets before going real-money.*

---

## 8. Phased plan

**Phase 1 — Identity (make it multi-user).**
Users, Wallets, LedgerEntry, JWT auth, `get_current_user` guard on existing endpoints. Everyone gets 1000 starting coins. *Now two people can use it.*

**Phase 2 — Bet slips (single + multiple).**
BetSlip/BetSelection, placement transaction with row locks, extend `settle_bets.py` for parlays, My Bets screen, bet-slip store on the front end. *Now they can actually bet, solo or parlay.*

**Phase 3 — Prediction hook.**
Model verdict per leg, combined model probability for multiples, personal ROI + model-agreement tracking, the personalized feed. *Now the model is visible and sticky.*

**Phase 4 — Social & retention.**
Leaderboards + private leagues, shareable slip/result cards, referrals, streaks/badges, notifications. *Now it grows itself.*

**Phase 5 — Scale/harden (and optionally real-money).**
Postgres + Redis + Celery, rate limiting, idempotency, observability. Real-money only after the compliance work in §7.

Ship each phase end-to-end before starting the next. Phase 1+2 is a usable product; 3+4 is what makes it spread.

---

## 9. First concrete steps

1. New Alembic revision: `User`, `Wallet`, `LedgerEntry`.
2. `src/auth.py`: register/login/refresh + `get_current_user` dependency (`python-jose`, `passlib[argon2]`).
3. Add `Depends(get_current_user)` to existing routers in `src/api.py`.
4. New Alembic revision: `BetSlip`, `BetSelection`; write `services/place_slip()` with the transaction in §2.3.
5. Extend `settle_bets.py` to settle slips (§2.5).
6. Frontend: bet-slip store + Bet Slip panel + My Bets screen, reusing your H/D/A model bar.

Everything above sits *on top of* your existing engine — `poisson_model`, `value_detector`, `elo`, `accountant`, `settle_bets` keep doing their job; you're giving them users, wallets, slips, and a reason to come back.
