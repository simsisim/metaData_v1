# Analysis Framework: Three-Layer Approach

## Overview

The pipeline is conceptually split into three layers that run in sequence.
Each layer depends on the previous one.

```
Market Data (OHLCV)
        ↓
[Layer 2] Basic Calculations  ← runs once, enriches all tickers
        ↓
[Layer 1] Market Health       ← answers: "is it safe to buy?"
        +
[Layer 3] Screeners           ← answers: "which stocks specifically?"
        ↓
  Candidates filtered by market health regime
        ↓
  Position sizing (future module)
```

---

## Layer 1 — Market Health
*"Should I be in the market at all? How aggressively?"*

Top-down filter. Does not find stocks — determines the market regime and drives position sizing.

| What | Module in system | Output |
|---|---|---|
| Market trend | GMI / GMI2 (SPY, QQQ daily) | Score 0-4 / 0-9, bull/bear signal |
| Momentum confirmation | FTD / DD detection | Follow-through days vs distribution days |
| Breadth quality | Market Breadth (A/D line, % above MA, new H/L) | Is the rally broad or narrow? |
| Risk zones | Chillax MAs + MA Cycles (SPY/QQQ/IWM) | Color-coded trend zones |
| Intermarket | Sustainability Ratios / MMM | Sector rotation, relative strength of markets |

**Decision output:** `RISK_ON` / `RISK_OFF` / `CAUTIOUS` → drives position sizing

---

## Layer 2 — Basic Calculations
*"Classify each stock so screeners can use the data"*

Inputs to screeners, not screeners themselves. Run once per timeframe on all tickers.
Output is an enriched data matrix — every ticker gets all columns below — which feeds all screeners.

| What | Module in system | Output per ticker |
|---|---|---|
| Trend classification | Stage Analysis (Minervini 1-4) | Stage number + ATR volatility |
| Relative performance | RS IBD-style + RS MA (vs SPY, QQQ) | RS score + percentile |
| Volatility | ATR / ATRext / ATR percentile | ATR value, rank vs 100-day history |
| Price structure | EMA/SMA matrix (10, 20, 50, 150, 200, 250, 350) | MA alignment, distance from MA |
| Momentum | % change across all configured periods | Return per period |

### Section 6b — PCT CHANGE PERIODS (single source of truth)

Defined once in `user_data.csv` Section 6b. These periods drive **all four downstream consumers**:

```
Section 6b periods
       │
       ├── Basic Calc    → pct_change columns per period per ticker
       ├── RS            → stock/benchmark ratio per period (column suffix = period label)
       ├── PER           → percentile rank of each RS period within universe
       ├── Minervini     → weighted RS rating using period suffixes from rs_weights config
       └── Qullamaggie   → RS ≥ 97 check on configured periods (7d;22d;66d;132d)
```

**Current daily period configuration:**

| Group | Periods | Days | Purpose |
|---|---|---|---|
| Daily | `1d; 3d; 5d` | 1, 3, 5 | Short-term momentum |
| Weekly | `7d; 14d` | 7, 14 | Weekly momentum |
| Monthly | `22d; 44d` | 22, 44 | 1-month / 2-month |
| Quarterly | `66d; 132d` | 66, 132 | 1-quarter / 2-quarter |
| Yearly | `252d` | 252 | 1-year |

**Minervini RS weights** (user_data.csv `MINERVINI_rs_weights`):

| Period | Weight | Rationale |
|---|---|---|
| 1d | 5% | Very short term, low weight |
| 3d | 10% | Short term |
| 7d | 15% | Weekly |
| 22d | 20% | Monthly |
| 66d | 25% | Quarterly — primary Minervini horizon |
| 252d | 25% | Yearly — primary Minervini horizon |

Weights are auto-normalized by available periods — if a period is missing from the RS file it is skipped and remaining weights are rescaled.

**Qullamaggie RS periods** (user_data.csv `QULLAMAGGIE_SUITE_rs_periods`):

| Period | Days | Rationale |
|---|---|---|
| 7d | 7 | 1-week momentum |
| 22d | 22 | 1-month momentum |
| 66d | 66 | 1-quarter momentum |
| 132d | 132 | 2-quarter momentum |

Passes if **at least one** period scores ≥ 97 (top 3% of universe). No weighted average — pure threshold check.

---

## Layer 3 — Screeners
*"Given the classified universe, find actionable candidates"*

Each screener looks for a specific setup, using Layer 2 data as input.

| Screener | Setup it finds | Key Layer 2 dependency | Status |
|---|---|---|---|
| **PVB TWmodel** | Price + volume breaking above MA | Volume + price vs MA | ✅ active |
| **Minervini Template** | Classic trend template | RS≥70 weighted + MA alignment | ✅ implemented |
| **Qullamaggie** | Perfect MA stack + top RS + ATR volatility | RS≥97 + ATR RS≥50 + MA stack | ✅ implemented |
| **Gold Launch Pad** | MAs compressing before breakout | MA Z-score convergence | wired, path TBD |
| **Stockbee 9M** | Explosive volume surge | Absolute volume threshold | wired, path TBD |
| **Stockbee Weekly** | 20%+ weekly move with momentum | % change weekly | wired, path TBD |
| **Dr. Wish GLB** | Green Line Breakout (52w high pivot) | Price structure, stochastic | wired, path TBD |
| **Guppy GMMA** | EMA group compression → expansion | Short/long EMA group spread | wired, path TBD |
| **ADL** | Accumulation diverging from price | A/D line vs price | wired, path TBD |
| **RTI** | Volatility coiling before expansion | Range tightening measure | wired, path TBD |
| **Giusti** | Top momentum funnel (12m→6m→3m) | Multi-period % return | not yet wired |

### Minervini Template — criteria detail

| # | Criterion | Threshold |
|---|---|---|
| C1 | Price > SMA150 AND Price > SMA200 | ratio > 1 |
| C2 | SMA150 > SMA200 | ratio > 1 |
| C3 | SMA200 trending up | SMA200[today] > SMA200[22d ago] |
| C4 | EMA50 > SMA150 AND EMA50 > SMA200 | ratio > 1 |
| C5 | Price > EMA50 | ratio > 1 |
| C6 | Price ≥ 30% above 52w low | price / (1.3 × 52w_low) ≥ 1 |
| C7 | Price within 25% of 52w high | price / (0.75 × 52w_high) ≥ 1 |
| C8 | Weighted RS rating | ≥ 70 (default) |

All 8 must pass. Output sorted by pass_count then rs_rating_wa.

### Qullamaggie — criteria detail

| # | Criterion | Threshold | Data source |
|---|---|---|---|
| 1 | Market cap | ≥ $1B | basic_calc (loaded at init) |
| 2 | RS percentile | ≥ 97 on ≥ 1 of: 7d, 22d, 66d, 132d | PER file (loaded at init) |
| 3 | MA alignment | Price ≥ EMA10 ≥ SMA20 ≥ SMA50 ≥ SMA100 ≥ SMA200 | OHLCV (computed per ticker) |
| 4 | ATR RS vs $1B+ universe | ≥ 50th percentile | basic_calc atr_pct ranked at init |
| 5 | Range position | Price ≥ 50% of 20-day High/Low range | OHLCV (computed per ticker) |

Output sorted by ATR extension from SMA50 = `(Price − SMA50) / ATR14`:
- NORMAL < 7× · WARNING 7–11× · DANGER > 11× (very extended, reversal risk)

---

## Current pipeline state (user_data.csv)

| Layer | Flag | Status |
|---|---|---|
| Layer 1 | `MARKET_HEALTH` | TRUE — breadth + market pulse running |
| Layer 2 | `BASIC` | TRUE — basic calc + stage + RS + PER running |
| Layer 3 | `SCREENERS` | TRUE |
| Layer 3 | `PVB_TWmodel_enable` | TRUE — active |
| Layer 3 | `MINERVINI_enable` | FALSE — implemented, ready to enable |
| Layer 3 | `QULLAMAGGIE_SUITE_enable` | FALSE — implemented, ready to enable |

**Output directories:**
- Layer 1 → `results/layer1_market_health/`
- Layer 2 → `results/layer2_basic_calculations/`
- Layer 3 → `results/layer3_screeners/`

**Correct run order:**
1. `MARKET_HEALTH=TRUE` → market regime assessment
2. `BASIC=TRUE` → fresh pct_change, RS, stage, ATR, PER for all tickers
3. `SCREENERS=TRUE` → candidates (use Layer 2 output as input)
