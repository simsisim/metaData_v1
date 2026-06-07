# Session Continuation — Where We Stopped

**Date**: 2026-06-04
**Session summary**: Restructured the pipeline into three layers, reorganized config,
and started migrating output to a layer-based folder structure.

---

## What was accomplished this session

### 1. Three-layer framework defined
See `docus/ANALYSIS_FRAMEWORK_THREE_LAYERS.md` for full documentation.
- **Layer 1** — Market Health (SR, Market Breadth, Market Pulse)
- **Layer 2** — Basic Calculations (EMA/SMA/ATR, Stage Analysis, RS, Percentile)
- **Layer 3** — Screeners (PVB, Minervini, Qullamaggie, etc.)

### 2. Layer 1 runs first — implemented and tested
**Files changed**:
- `user_data.csv` — added `MARKET_HEALTH` flag between `PRE_PROCESS` and `BASIC`
- `src/user_defined_data.py` — added `MARKET_HEALTH` attribute + parser mapping
- `main.py` — new `PHASE 1: MARKET_HEALTH` block (SR → Market Breadth → Market Pulse);
  `PHASE 2: BASIC` now contains Layer 2 only

**Current pipeline execution order**:
```
PRE_PROCESS → MARKET_HEALTH → BASIC → SCREENERS → POST_PROCESS
```

### 3. user_data.csv fully restructured
Old: ~890 lines, flat, inconsistent.
New: 786 lines, 9 sections in execution order:
- Section 0: Quick Reference (most-changed settings at top)
- Section 1: Environment & Paths
- Section 2: Data Input Sources
- Section 3: Pipeline Flags (authoritative — Section 0 is a mirror)
- Section 4: Ticker Universe
- Section 5: Layer 1 — Market Health (SR → Breadth → Pulse)
- Section 6: Layer 2 — Basic Calculations
- Section 7: Layer 3 — Screeners (12 screeners, each in own sub-block)
- Section 8: Output & Reports
- Section 9: Post-Processing

**Parser fix** (`src/user_defined_data.py`): the CSV had many multi-value fields written with
commas instead of semicolons (e.g. `20,50,200` instead of `20;50;200`), causing pandas to
detect 8 columns instead of 3 and silently return default values. Fixed by reading all columns
and rebuilding each row: col0=variable, cols 1..n-2 joined with `;` = value, col n-1 = description.

### 4. Results folder — layer-based structure (partial)
Output directories updated in `user_data.csv` and `src/user_defined_data.py` defaults:

| Layer | Folder |
|---|---|
| Layer 1 | `results/layer1_market_health/` |
| Layer 2 | `results/layer2_basic_calculations/` |
| Layer 3 | `results/layer3_screeners/` |

**What IS going to the new structure**:
- `layer1_market_health/market_breadth/` ← breadth CSVs ✅
- `layer1_market_health/reports/` ← market pulse PDFs + GMI reports ✅

**What is NOT yet migrated** (still going to old `results/` flat paths):
- Breadth PNG charts → `results/market_breadth/*.png`
- Breadth PDF reports → `results/reports/breadth_*.pdf`
- Market Pulse PNGs (chillax, ftd_dd, ma_cycles) → `results/market_pulse/*.png`

---

## Current config state (user_data.csv)

```
MARKET_HEALTH = TRUE    ← Layer 1 active
BASIC         = FALSE
SCREENERS     = FALSE
ticker_choice = 2       ← NASDAQ100
```

---

## What needs to be done next

### Priority 1 — Complete the results folder migration

The remaining PNGs and PDFs still go to old flat paths because the market pulse
calculators hardcode `config.directories['RESULTS_DIR'] / 'market_pulse'`.

**Files to fix** (all save to hardcoded `results/market_pulse/`):
- `src/market_pulse/indicators/chillax_mas.py` lines 728, 804
- `src/market_pulse/indicators/ftd_dd_analyzer.py` lines 660, 735, 1146
- `src/market_pulse/indicators/ma_cycles_analyzer.py` lines 625, 772
- `src/market_pulse/calculators/gmi2_calculator.py` line 1043

**Pattern to apply in each**:
```python
# Replace:
output_dir = self.config.directories['RESULTS_DIR'] / 'market_pulse'
# With:
output_dir = Path(getattr(self.user_config, 'market_pulse_output_dir', 'results/market_pulse'))
if not output_dir.is_absolute():
    output_dir = self.config.base_dir / output_dir
```
Note: these modules need `user_config` available as `self.user_config`. Check each one — some
may need user_config passed in from the manager.

**Breadth chart PNG path**: already correct in `market_breadth_visualizer.py` (saves next to CSV
using `csv_path.with_suffix('.png')`). But `breadth_analyzer.py` `_generate_breadth_charts_from_enhanced_csv`
may be passing a stale/old CSV path to the visualizer. Check what path it passes.

**Breadth PDF reports**: in `main.py` `run_all_market_breadth()`, the report loop does:
```python
csv_files = list(breadth_dir.glob('market_breadth_*.csv'))
for csv_file in csv_files:
    png_file = csv_file.with_suffix('.png')
    if png_file.exists():   # ← PNG must be next to CSV for this to work
```
Once breadth PNGs save to the correct layer1 location, breadth PDF reports will follow.

### Priority 2 — Enable and test Layer 2 (BASIC)

Set `BASIC=TRUE` in user_data.csv and run. Verify:
- Basic calculations → `results/layer2_basic_calculations/basic_calculation/`
- Stage analysis → `results/layer2_basic_calculations/stage_analysis/`
- RS → `results/layer2_basic_calculations/rs/`
- PER → `results/layer2_basic_calculations/per/`
- Stage analysis PDF reports → `results/layer2_basic_calculations/reports/`

### Priority 3 — Enable and test Layer 3 (SCREENERS)

Set `SCREENERS=TRUE` in user_data.csv and run. Verify:
- PVB results → `results/layer3_screeners/pvbTW/`

### Priority 4 — user_data.csv Section 0 note

Section 0 (Quick Reference) duplicates the phase flags from Section 3.
Since the parser uses last-value-wins, Section 3 is authoritative.
Both sections are currently kept in sync manually.
**Future improvement**: add a comment in Section 0 making this explicit, or remove
the duplicate flags from Section 0 and keep only the non-flag quick-reference items there.

---

## Known warnings (non-blocking)

```
WARNING - Breadth file not found: results/layer1_market_health/market_breadth/market_breadth_SP500_2-5_daily_...
```
This warning fires at startup from `breadth_analyzer_tornado.py` which looks for a ticker_choice
`2-5` breadth file (a combined choice) that doesn't exist. Non-blocking — safe to ignore.
It will go away once breadth is run with ticker_choice `2-5` or if that default is changed.

---

## Backup

A backup of the state before this session exists in the project folder.
Git history also captures all changes (branch: master).

---

## Reference documents created this session

| File | Content |
|---|---|
| `docus/ANALYSIS_FRAMEWORK_THREE_LAYERS.md` | Three-layer framework + current pipeline state |
| `docus/PLAN_layer1_first_and_userdata_restructure.md` | Implementation plan for all changes done |
| `docus/SESSION_CONTINUATION.md` | This file |
