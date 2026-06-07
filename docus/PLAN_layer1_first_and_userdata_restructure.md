# Plan: Layer 1 First + user_data.csv Restructure

## Overview

Three coordinated changes:
1. Add `MARKET_HEALTH` flag and move Layer 1 modules to run before Layer 2 in `main.py`
2. Add matching `MARKET_HEALTH` flag block to `user_data.csv`
3. Restructure `user_data.csv` to mirror execution order

> **Note:** Duplicate period definitions (`daily_daily_periods` ×3) are left as-is.
> All three map to the same Python attribute — last value wins — and currently hold
> identical values. Deferred to a future cleanup pass.

A backup of the current state already exists. No module logic changes — pure orchestration
and config reorganization.

---

## Part 1 — main.py: Layer 1 First

### Current execution order (inside single BASIC block)

```
PHASE 0: PRE_PROCESS
PHASE 1: BASIC  ← Layer 1 and Layer 2 mixed
  1. Sustainability Ratios        (Layer 1)
  2. Basic Calculations           (Layer 2)
  3. Stage Analysis               (Layer 2)
  4. Market Breadth               (Layer 1)
  5. Market Pulse                 (Layer 1)
  6. RS Analysis                  (Layer 2)
  7. PER / Percentile             (Layer 2)
PHASE 2: SCREENERS                (Layer 3)
PHASE 3: POST_PROCESS
```

### Target execution order

```
PHASE 0: PRE_PROCESS              (unchanged)

PHASE 1: MARKET_HEALTH            (new — Layer 1 only)
  1. Sustainability Ratios / SR   (intermarket, sector rotation)
  2. Market Breadth               (reads all component OHLCV directly)
  3. Market Pulse                 (GMI, GMI2, FTD/DD, Chillax, MA Cycles)
     ↑ Market Pulse MUST stay after Market Breadth (GMI reads breadth file output)

PHASE 2: BASIC                    (Layer 2 only — trimmed)
  1. Basic Calculations           (EMA, SMA, ATR, % change per ticker)
  2. Stage Analysis               (Minervini stage classification)
  3. RS Analysis                  (relative strength vs SPY, QQQ)
  4. PER / Percentile             (ranking across full universe)

PHASE 3: SCREENERS                (Layer 3 — unchanged)

PHASE 4: POST_PROCESS             (unchanged)
```

### Changes required in main.py

1. **Add new phase block** `PHASE 1: MARKET_HEALTH` with flag `user_config.MARKET_HEALTH`
2. **Move** `run_sr_analysis()` into MARKET_HEALTH block
3. **Move** `run_all_market_breadth()` into MARKET_HEALTH block
4. **Move** `run_all_market_pulse()` into MARKET_HEALTH block — after breadth (keep order)
5. **Remove** those three calls from the BASIC block
6. **BASIC block** keeps only: basic_calculations, stage_analysis, rs_analysis, per_analysis
7. **Add empty result dicts** in the else branches for the new block
   (same pattern already used for BASIC and SCREENERS skipped branches)

### No changes required
- All module files unchanged (`market_breadth_calculation.py`, `market_pulse/`, `sustainability_ratios/` etc.)
- All function signatures unchanged
- SCREENERS and POST_PROCESS phases unchanged

### Important dependency note
Market Breadth reads from `results/ticker_universes/ticker_universe_{name}.csv` files.
These are generated during **Universe Generation** (already runs before all phases at line 1673).
No dependency issue — universe files exist before MARKET_HEALTH phase starts.

---

## Part 2 — Breadth Universe vs ticker_choice clarification

`ticker_choice` and `MARKET_BREADTH_universe` are independent settings.

| Setting | Controls |
|---|---|
| `ticker_choice` | Which stocks go through Layer 2 (basic calcs, RS, stage) and Layer 3 (screeners) |
| `MARKET_BREADTH_universe` | Which pre-built universe files are used for breadth calculation |

**The hard constraint**: OHLCV files must physically exist in `downloadData_v1/data/market_data/daily/`.
If downloadData_v1 was run with NASDAQ100 only (ticker_choice=2), SP500-only stocks have no files.
Market Breadth silently skips missing files — breadth will be partial without any error.

**Rule**: `MARKET_BREADTH_universe` scope should never exceed what was downloaded.
Add a comment in user_data.csv to make this explicit.

---

## Part 3 — user_data.csv Restructure

### Problems with current file

| Problem | Detail |
|---|---|
| No alignment with execution order | Config jumps between Layer 1/2/3 concerns |
| `daily_daily_periods` defined 3 times | Lines 111, 151, 420 — same values, different sections |
| BASIC flag controls Layer 1 + Layer 2 | No way to run only market health |
| Deprecated entries mixed in | Commented-out legacy settings scattered throughout |
| No quick-reference for daily use | Must scroll through ~890 lines to find common toggles |
| Inconsistent section headers | Mix of `#` comments and `# ---` separators |

### Target structure (section order mirrors execution order)

```
SECTION 0 — QUICK REFERENCE              ← NEW: most-changed settings in one place
SECTION 1 — ENVIRONMENT & PATHS          ← existing, cleaned up
SECTION 2 — DATA INPUT SOURCES           ← existing, cleaned up
SECTION 3 — PIPELINE FLAGS               ← execution order: MARKET_HEALTH→BASIC→SCREENERS
SECTION 4 — SHARED PERIOD DEFINITIONS    ← NEW: single definition replacing 3 duplicates
SECTION 5 — LAYER 1: MARKET HEALTH
  5a. Sustainability Ratios (SR)
  5b. Market Breadth
  5c. Market Pulse (GMI, GMI2, FTD/DD, Chillax, MA Cycles)
SECTION 6 — LAYER 2: BASIC CALCULATIONS
  6a. Technical Indicators (EMA, SMA, ATR)
  6b. Stage Analysis
  6c. Relative Strength (RS)
  6d. Percentile (PER)
SECTION 7 — LAYER 3: SCREENERS
  7a.  PVB TWmodel
  7b.  Minervini Template
  7c.  Qullamaggie
  7d.  Stockbee Suite
  7e.  Dr. Wish (GLB, Blue Dot, Black Dot)
  7f.  Giusti Momentum
  7g.  ADL (Accumulation/Distribution)
  7h.  Guppy GMMA
  7i.  Gold Launch Pad
  7j.  RTI (Range Tightening)
  7k.  ATR1 / ATR2
  7l.  Volume Suite
SECTION 8 — OUTPUT & REPORTS
  8a. PDF Reports
  8b. Dashboard
SECTION 9 — POST-PROCESSING
```

### Section 0 — Quick Reference (new)

A short block at the top with only the settings changed regularly:

```
# ================================================================
# QUICK REFERENCE — settings changed most often
# ================================================================
ticker_choice          2        # 0=TW universe 1=SP500 2=NASDAQ100 3=all NASDAQ ...
MARKET_HEALTH         TRUE      # Layer 1: market breadth, pulse, SR
BASIC                 FALSE     # Layer 2: basic calcs, stage, RS, percentile
SCREENERS             TRUE      # Layer 3: all screeners
MARKET_BREADTH_universe  SP500;NASDAQ100   # must match downloaded data scope
PVB_TWmodel_enable    TRUE      # main active screener
```

### Section 4 — Shared Period Definitions (new)

Single canonical definition replacing the 3 duplicates:

```
# These period values are shared across RS, Basic Calculations, and Index Overview.
# Define once here; all modules read the same keys.
daily_daily_periods,1;3;5
daily_weekly_periods,7;14
daily_monthly_periods,22;44
daily_quarterly_periods,66;132
daily_yearly_periods,252
weekly_weekly_periods,1;3;5;10;15
weekly_monthly_periods,4;8
monthly_monthly_periods,2;3;6
```

### Section 3 — Pipeline Flags

Add `MARKET_HEALTH` flag between PRE_PROCESS and BASIC:

```
PRE_PROCESS,FALSE
MARKET_HEALTH,TRUE      ← new
BASIC,FALSE
SCREENERS,TRUE
POST_PROCESS,FALSE
BACKTESTING,FALSE
```

### What gets removed / cleaned up

| Item | Action |
|---|---|
| Duplicate `daily_daily_periods` (×3) | **DEFERRED — keep as-is** |
| Commented-out legacy `#SR_chart_generation` etc. | Remove — documented elsewhere |
| Commented-out legacy `#MARKET_PULSE_ftd_dd_enable` (old version) | Remove — superseded by active version |
| `database_enable,FALSE` / `database_type,disabled` | Remove — database functionality deleted from code |
| `Basic_calculation_file,DEPRECATED` line | Remove |

### Section-by-section content decisions

**Section 0 — Quick Reference**
New block. Contains only the settings changed daily:
```
ticker_choice
MARKET_HEALTH
BASIC
SCREENERS
MARKET_BREADTH_universe   ← with note: must match downloaded data scope
PVB_TWmodel_enable        ← currently the main active screener
```

**Section 1 — Environment & Paths**
Move from current top. Keep all local/colab path settings.
No changes to values.

**Section 2 — Data Input Sources**
`WEB_tickers_down`, `TW_tickers_down`, `TW_universe_file`,
`YF_hist_data`, `YF_daily_data`, `YF_weekly_data`, `YF_monthly_data`,
`TW_intraday_data`, financial data enrichment flags.
No changes to values.

**Section 3 — Pipeline Flags**
The master on/off switches in execution order:
```
PRE_PROCESS
MARKET_HEALTH    ← new
BASIC
SCREENERS
POST_PROCESS
BACKTESTING
```
Add `batch_size` and `ticker_choice` here (currently scattered).

**Section 4 — Ticker Universe**
`ticker_choice`, ticker group definitions (the `# N: description` comment lines),
`ticker_info_TW`, `ticker_info_YF`.
Currently split across top of file and middle — consolidate here.

**Section 5 — Layer 1: Market Health**

5a. Sustainability Ratios (SR)
- `SR_enable` and all `SR_*` settings
- Currently correct position in file, just moves to new section header

5b. Market Breadth
- `MARKET_BREADTH_enable`, universe, timeframes, thresholds
- Add comment: *universe scope must match downloadData_v1 download scope*

5c. Market Pulse
- `MARKET_PULSE_enable` and all sub-settings
- GMI, GMI2, FTD/DD, Chillax MAs, MA Cycles — each as a sub-block
- Keep internal order: Breadth → Market Pulse (GMI reads breadth file)

**Section 6 — Layer 2: Basic Calculations**

6a. Technical Indicators (EMA, SMA, ATR)
- `daily_ema_periods`, `daily_sma_periods`, ATR config

6b. Basic Calculations timeframe flags
- `Basic_calc_daily_enable`, `Basic_calc_weekly_enable`
- Period definitions (kept ×3 as-is per deferred decision)

6c. Stage Analysis
- `enable_stage_analysis`, all `stage_*` settings per timeframe

6d. Relative Strength (RS)
- `RS_enable_stocks`, `RS_benchmark_tickers`, all RS period settings
- `RS_ma_enable`, `RS_method_for_PER`

6e. Percentile (PER)
- `PER_output_dir`, `RS_percentile_universe_*` settings

**Section 7 — Layer 3: Screeners**
One sub-block per screener, each with:
- enable flag first
- timeframe flags
- parameters
- output dir last

Order (most commonly used first):
7a. PVB TWmodel
7b. Minervini Template
7c. Qullamaggie
7d. Stockbee Suite
7e. Dr. Wish (GLB, Blue Dot, Black Dot)
7f. Giusti Momentum
7g. ADL (Accumulation/Distribution)
7h. Guppy GMMA
7i. Gold Launch Pad
7j. RTI (Range Tightening)
7k. ATR1 / ATR2
7l. Volume Suite

**Section 8 — Output & Reports**
- PDF reports config
- Dashboard config
- Output directory overrides

**Section 9 — Post-Processing**
- `POST_PROCESS` sub-settings
- `user_data_pp.csv` reference

---

## Implementation sequence

| Step | File | Risk |
|---|---|---|
| 1 | `user_data.csv` — add `MARKET_HEALTH,FALSE` flag only | Zero — new key, no effect |
| 2 | `main.py` — add MARKET_HEALTH block, move 3 functions | Low — same functions, new if-block |
| 3 | Test run with `MARKET_HEALTH=TRUE, BASIC=FALSE, SCREENERS=FALSE` | Validates Layer 1 isolation |
| 4 | Test run with `MARKET_HEALTH=FALSE, BASIC=TRUE, SCREENERS=FALSE` | Validates Layer 2 isolation |
| 5 | `user_data.csv` — full restructure | Medium — cosmetic only, no value changes |
| 6 | Validate all keys still parse correctly (`read_user_data()`) | Catches any key rename issues |

Start with Step 1+2 only. Restructure user_data.csv (Step 5) only after Steps 3+4 pass.
