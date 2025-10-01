# ADL (Accumulation/Distribution Line) Screener - Implementation Plan

**Status**: Planning Phase - No Code Implementation Yet
**Date**: 2025-09-30
**Location**: `src/screeners/ad_line/` (proposed folder structure)

---

## Executive Summary

This document outlines the comprehensive enhancement plan for the ADL (Accumulation/Distribution Line) screener module. The enhancement transforms the current basic divergence-detection screener into a sophisticated 5-step accumulation analysis system that combines long-term trends with short-term momentum signals.

**Current State**: Basic ADL calculation with divergence and breakout detection
**Target State**: Multi-dimensional accumulation analysis with composite scoring and ranking

---

## Table of Contents

1. [What the Current Code Calculates](#1-what-the-current-code-calculates)
2. [Enhancement Overview - 5-Step Methodology](#2-enhancement-overview---5-step-methodology)
3. [Current Configuration Parameters](#3-current-configuration-parameters)
4. [Proposed New Parameters](#4-proposed-new-parameters)
5. [Implementation Plan](#5-implementation-plan)
6. [Folder Structure](#6-folder-structure)
7. [Module Specifications](#7-module-specifications)
8. [Integration Points](#8-integration-points)
9. [Output Files](#9-output-files)
10. [Testing Strategy](#10-testing-strategy)
11. [Performance Considerations](#11-performance-considerations)
12. [Signal Interpretation Guide](#12-signal-interpretation-guide)

---

## 1. What the Current Code Calculates

### 1.1 Core ADL Calculation
**Location**: `src/indicators/ADLINE.py` (lines 61-157) and `src/screeners/adl_screener.py` (lines 188-231)

**Formula Components**:
1. **Money Flow Multiplier (MFM)**:
   ```
   MFM = ((Close - Low) - (High - Close)) / (High - Low)
   ```
   - Ranges from -1 to +1
   - Measures where the close is relative to the high-low range
   - Close near high → positive MFM (accumulation)
   - Close near low → negative MFM (distribution)

2. **Money Flow Volume (MFV)**:
   ```
   MFV = MFM × Volume
   ```
   - Volume-weighted measure of buying/selling pressure

3. **Accumulation/Distribution Line (ADL)**:
   ```
   ADL = Cumulative Sum of MFV
   ```
   - Running total showing net accumulation or distribution
   - Rising ADL = accumulation (buying pressure)
   - Falling ADL = distribution (selling pressure)

### 1.2 Current Signal Types

#### Divergence Detection
**Location**: `src/screeners/adl_screener.py` (lines 232-370)

**Bullish Divergence**:
- **Condition**: Price making lower lows while ADL making higher lows
- **Interpretation**: Price weakness not confirmed by volume → potential reversal up
- **Strength Calculation**: Average of price decline % and ADL improvement %
- **Minimum Threshold**: 0.7% (configurable via `min_divergence_strength`)

**Bearish Divergence**:
- **Condition**: Price making higher highs while ADL making lower highs
- **Interpretation**: Price strength not confirmed by volume → potential reversal down
- **Strength Calculation**: Average of price advance % and ADL weakness %
- **Minimum Threshold**: 0.7% (configurable via `min_divergence_strength`)

#### Breakout/Breakdown Signals
**Location**: `src/screeners/adl_screener.py` (lines 372-488)

**ADL Breakout**:
- **Condition**: Current ADL breaks above highest ADL in lookback period
- **Interpretation**: Strong accumulation signal
- **Strength Calculation**: `(Current ADL - Previous High ADL) / Previous High ADL`
- **Minimum Threshold**: 1.2 (120% breakout strength)
- **Confirmation**: Requires above-average volume

**ADL Breakdown**:
- **Condition**: Current ADL breaks below lowest ADL in lookback period
- **Interpretation**: Strong distribution signal
- **Strength Calculation**: `|Current ADL - Previous Low ADL| / Previous Low ADL`
- **Minimum Threshold**: 1.2 (120% breakdown strength)
- **Confirmation**: Requires above-average volume

### 1.3 Current Limitations

1. **No Month-over-Month Analysis**: Cannot detect consistent accumulation trends over multiple months
2. **No Short-term Momentum**: Missing 5-day, 10-day, 20-day percentage change tracking
3. **No Moving Average Analysis**: No MA overlay or crossover detection on ADL line
4. **No Composite Scoring**: Signals are independent; no combined ranking system
5. **Limited Trend Analysis**: Basic slope calculation only; no sophisticated trend confirmation

---

## 2. Enhancement Overview - 5-Step Methodology

### Step 1: Calculate ADL Line (Current Implementation)
**Status**: ✅ Already Implemented
**Output**: Daily ADL values for each stock

### Step 2: Month-over-Month Percentage Change Analysis
**Status**: 🔴 Needs Implementation
**Purpose**: Identify consistent accumulation patterns over extended periods

**Process**:
1. Calculate ADL values at monthly intervals (every 22 trading days)
2. Compute month-over-month percentage change
3. Check if % change falls within target range (15%-30% default)
4. Track consecutive months meeting criteria
5. Flag stocks with 3+ consecutive months of consistent accumulation

**Key Metric**: Accumulation Consistency Score (0-100)
- 100 = Maximum months with consistent accumulation in threshold range
- 0 = No consistent accumulation pattern

### Step 3: Short-term Percentage Changes
**Status**: 🔴 Needs Implementation
**Purpose**: Detect momentum shifts and acceleration

**Process**:
1. Calculate 5-day, 10-day, 20-day ADL percentage changes
2. Compare current short-term change to historical average
3. Detect momentum inflection points
4. Identify acceleration/deceleration patterns

**Key Metrics**:
- **5-day % change**: Immediate momentum
- **10-day % change**: Short-term trend
- **20-day % change**: Medium-term trend
- **Momentum Score**: Weighted combination of all three

### Step 4: Moving Average Overlay
**Status**: 🔴 Needs Implementation
**Purpose**: Confirm upward momentum with trend alignment

**Process**:
1. Calculate 20-period, 50-period, 100-period MAs of ADL line
2. Check for bullish MA alignment (20 > 50 > 100)
3. Detect MA crossovers (golden cross, death cross)
4. Calculate MA slope to confirm direction
5. Measure distance between MAs (tight vs. wide separation)

**Key Metrics**:
- **MA Alignment Score**: 0 (bearish) to 100 (perfect bullish alignment)
- **Crossover Signal**: Recent golden/death cross detection
- **MA Slope**: Positive/negative/flat for each MA period

### Step 5: Combined Confirmation Signals
**Status**: 🔴 Needs Implementation
**Purpose**: Rank stocks by overall accumulation strength

**Process**:
1. Calculate long-term accumulation score (Step 2 output)
2. Calculate short-term momentum score (Step 3 output)
3. Calculate MA alignment score (Step 4 output)
4. Apply configurable weights to each component
5. Generate composite score (0-100 scale)
6. Rank all stocks by composite score
7. Filter stocks meeting minimum composite threshold

**Composite Score Formula**:
```
Composite = (w₁ × Long_Term_Score) + (w₂ × Short_Term_Score) + (w₃ × MA_Alignment_Score)

Default weights:
w₁ = 0.4 (long-term accumulation)
w₂ = 0.3 (short-term momentum)
w₃ = 0.3 (MA alignment)
```

---

## 3. Current Configuration Parameters

**Location**: `user_data.csv` (lines 587-600)

```csv
ADL_SCREENER_enable                    FALSE    # Master enable flag
ADL_SCREENER_daily_enable              FALSE    # Daily timeframe enable
ADL_SCREENER_weekly_enable             FALSE    # Weekly timeframe enable
ADL_SCREENER_monthly_enable            FALSE    # Monthly timeframe enable
ADL_SCREENER_lookback_period           50       # Period for ADL calculation
ADL_SCREENER_divergence_period         20       # Divergence analysis period
ADL_SCREENER_breakout_period           30       # Breakout detection period
ADL_SCREENER_min_divergence_strength   0.7      # Minimum divergence threshold
ADL_SCREENER_min_breakout_strength     1.2      # Minimum breakout threshold
ADL_SCREENER_min_volume_avg            100000   # Minimum average volume filter
ADL_SCREENER_min_price                 5        # Minimum price filter
ADL_SCREENER_save_individual_files     TRUE     # Save component result files
```

---

## 4. Proposed New Parameters

### 4.1 Month-over-Month Analysis Parameters (Step 2)
```csv
# MONTH-OVER-MONTH ACCUMULATION ANALYSIS
ADL_SCREENER_mom_analysis_enable           TRUE    # Enable MoM analysis module
ADL_SCREENER_mom_period                    22      # Trading days per month (~22 days)
ADL_SCREENER_mom_min_threshold_pct         15      # Minimum monthly growth (%)
ADL_SCREENER_mom_max_threshold_pct         30      # Maximum monthly growth (%)
ADL_SCREENER_mom_consecutive_months        3       # Min consecutive months meeting criteria
ADL_SCREENER_mom_lookback_months           6       # Total months to analyze
ADL_SCREENER_mom_min_consistency_score     60      # Minimum consistency score (0-100)
```

**Parameter Descriptions**:
- `mom_period`: Number of trading days considered as 1 month (default 22)
- `mom_min_threshold_pct`: Lower bound for acceptable monthly ADL growth
- `mom_max_threshold_pct`: Upper bound for acceptable monthly ADL growth
- `mom_consecutive_months`: Required consecutive months within threshold
- `mom_lookback_months`: Historical period to analyze for consistency
- `mom_min_consistency_score`: Minimum score to flag as strong accumulation

### 4.2 Short-term Momentum Parameters (Step 3)
```csv
# SHORT-TERM MOMENTUM ANALYSIS
ADL_SCREENER_short_term_enable             TRUE        # Enable short-term module
ADL_SCREENER_short_term_periods            5;10;20     # Periods for % change (semicolon separated)
ADL_SCREENER_short_term_momentum_threshold 5           # Minimum % change to signal momentum shift
ADL_SCREENER_short_term_acceleration_detect TRUE       # Detect acceleration patterns
ADL_SCREENER_short_term_min_score          50          # Minimum short-term momentum score (0-100)
```

**Parameter Descriptions**:
- `short_term_periods`: Periods for calculating ADL % change (e.g., "5;10;20")
- `short_term_momentum_threshold`: Minimum % change to be considered significant
- `short_term_acceleration_detect`: Enable acceleration/deceleration pattern detection
- `short_term_min_score`: Minimum momentum score for inclusion in results

### 4.3 Moving Average Analysis Parameters (Step 4)
```csv
# MOVING AVERAGE ANALYSIS
ADL_SCREENER_ma_enable                     TRUE        # Enable MA analysis module
ADL_SCREENER_ma_periods                    20;50;100   # MA periods (semicolon separated)
ADL_SCREENER_ma_type                       SMA         # Type: SMA or EMA
ADL_SCREENER_ma_bullish_alignment_required TRUE        # Require bullish MA alignment
ADL_SCREENER_ma_crossover_detection        TRUE        # Detect MA crossovers
ADL_SCREENER_ma_crossover_lookback         10          # Periods to look back for crossovers
ADL_SCREENER_ma_min_slope_threshold        0.01        # Minimum positive slope for MAs
ADL_SCREENER_ma_min_alignment_score        70          # Minimum MA alignment score (0-100)
```

**Parameter Descriptions**:
- `ma_periods`: Moving average periods to calculate (e.g., "20;50;100")
- `ma_type`: Type of moving average (SMA or EMA)
- `ma_bullish_alignment_required`: If TRUE, require 20 > 50 > 100 alignment
- `ma_crossover_detection`: Enable golden/death cross detection
- `ma_crossover_lookback`: How many periods to check for recent crossovers
- `ma_min_slope_threshold`: Minimum positive slope to confirm upward trend
- `ma_min_alignment_score`: Minimum score for MA alignment quality

### 4.4 Composite Scoring Parameters (Step 5)
```csv
# COMPOSITE SCORING AND RANKING
ADL_SCREENER_composite_scoring_enable      TRUE        # Enable composite scoring system
ADL_SCREENER_composite_weight_longterm     0.4         # Weight for long-term accumulation
ADL_SCREENER_composite_weight_shortterm    0.3         # Weight for short-term momentum
ADL_SCREENER_composite_weight_ma_align     0.3         # Weight for MA alignment
ADL_SCREENER_composite_min_score           70          # Minimum composite score (0-100)
ADL_SCREENER_ranking_method                composite   # Ranking: composite | momentum | accumulation
ADL_SCREENER_output_ranking_file           TRUE        # Create ranked output file
ADL_SCREENER_top_candidates_count          50          # Number of top candidates to highlight
```

**Parameter Descriptions**:
- `composite_weight_*`: Weights for each component (must sum to 1.0)
- `composite_min_score`: Minimum composite score to include in results
- `ranking_method`: Primary ranking methodology to use
- `output_ranking_file`: Generate separate ranked output file
- `top_candidates_count`: Number of top-ranked stocks to highlight

### 4.5 Output Organization Parameters
```csv
# OUTPUT CONFIGURATION
ADL_SCREENER_output_separate_signals       TRUE        # Separate files for each signal type
ADL_SCREENER_output_include_charts         FALSE       # Generate ADL charts (future feature)
ADL_SCREENER_output_summary_stats          TRUE        # Include summary statistics in output
```

---

## 5. Implementation Plan

### Phase 1: Folder Structure Setup
**Estimated Time**: 30 minutes
**Dependencies**: None

**Tasks**:
1. Create `src/screeners/ad_line/` directory
2. Create `__init__.py` with module exports
3. Move existing `adl_screener.py` content to new location
4. Update import statements in `screeners_streaming.py`

**Deliverables**:
- New folder structure
- Updated imports across codebase

### Phase 2: Core ADL Calculator Refactoring
**Estimated Time**: 2 hours
**Dependencies**: Phase 1

**Module**: `src/screeners/ad_line/adl_calculator.py`

**Tasks**:
1. Extract ADL calculation logic from `ADLINE.py` and `adl_screener.py`
2. Create unified `ADLCalculator` class
3. Add method: `calculate_adl(df)` → returns ADL series
4. Add method: `calculate_adl_pct_change(adl_series, period)` → returns % change
5. Add method: `calculate_rolling_adl(df, window)` → returns rolling ADL values
6. Implement comprehensive input validation
7. Add error handling for edge cases (doji bars, zero volume, etc.)
8. Write unit tests for each method

**Deliverables**:
- `adl_calculator.py` module
- Unit test suite
- Documentation

### Phase 3: Month-over-Month Analysis Module (Step 2)
**Estimated Time**: 4 hours
**Dependencies**: Phase 2

**Module**: `src/screeners/ad_line/adl_mom_analysis.py`

**Tasks**:
1. Create `ADLMoMAnalyzer` class
2. Implement `analyze_monthly_accumulation(df, params)` method
   - Calculate ADL at monthly intervals
   - Compute month-over-month % changes
   - Check if changes fall within threshold range
   - Track consecutive months meeting criteria
3. Implement `calculate_consistency_score(monthly_changes, params)` method
   - Score from 0-100 based on accumulation consistency
   - Penalize erratic patterns
   - Reward steady accumulation
4. Implement `detect_accumulation_streaks(monthly_changes, params)` method
   - Identify longest consecutive accumulation streak
   - Flag stocks with qualifying streaks
5. Create output DataFrame structure:
   ```python
   Columns: [
       'ticker',
       'monthly_changes',      # List of monthly % changes
       'avg_monthly_change',   # Average monthly change
       'consistency_score',    # 0-100 score
       'qualifying_streak',    # Longest streak meeting criteria
       'current_streak',       # Current active streak
       'meets_criteria'        # Boolean flag
   ]
   ```
6. Write unit tests
7. Document methodology and formulas

**Deliverables**:
- `adl_mom_analysis.py` module
- Unit test suite
- Documentation

### Phase 4: Short-term Momentum Module (Step 3)
**Estimated Time**: 3 hours
**Dependencies**: Phase 2

**Module**: `src/screeners/ad_line/adl_short_term.py`

**Tasks**:
1. Create `ADLShortTermAnalyzer` class
2. Implement `calculate_short_term_changes(df, periods)` method
   - Calculate 5-day, 10-day, 20-day ADL % changes
   - Return DataFrame with all period changes
3. Implement `detect_momentum_shifts(short_term_df, params)` method
   - Compare current changes to historical averages
   - Identify inflection points
   - Flag significant momentum shifts
4. Implement `detect_acceleration_patterns(short_term_df)` method
   - Check if 5-day > 10-day > 20-day (acceleration)
   - Check if 5-day < 10-day < 20-day (deceleration)
   - Calculate acceleration magnitude
5. Implement `calculate_momentum_score(short_term_df, params)` method
   - Weighted score based on all period changes
   - Higher weight for shorter periods (recency bias)
   - Scale to 0-100
6. Create output DataFrame structure:
   ```python
   Columns: [
       'ticker',
       'adl_5d_pct_change',
       'adl_10d_pct_change',
       'adl_20d_pct_change',
       'momentum_signal',      # 'acceleration' | 'deceleration' | 'neutral'
       'momentum_score',       # 0-100
       'inflection_point'      # Boolean flag
   ]
   ```
7. Write unit tests
8. Document methodology

**Deliverables**:
- `adl_short_term.py` module
- Unit test suite
- Documentation

### Phase 5: Moving Average Analysis Module (Step 4)
**Estimated Time**: 4 hours
**Dependencies**: Phase 2

**Module**: `src/screeners/ad_line/adl_ma_analysis.py`

**Tasks**:
1. Create `ADLMAAnalyzer` class
2. Implement `calculate_adl_mas(adl_series, periods, ma_type)` method
   - Calculate SMA or EMA for each period
   - Return DataFrame with all MA values
3. Implement `check_ma_alignment(ma_df)` method
   - Check bullish alignment (20 > 50 > 100)
   - Check bearish alignment (20 < 50 < 100)
   - Calculate alignment score based on separation
4. Implement `detect_ma_crossovers(ma_df, lookback)` method
   - Detect golden cross (fast MA crosses above slow MA)
   - Detect death cross (fast MA crosses below slow MA)
   - Track crossover dates and strength
5. Implement `calculate_ma_slopes(ma_df)` method
   - Calculate slope for each MA using linear regression
   - Classify as positive/negative/flat
   - Return slope magnitude
6. Implement `calculate_ma_alignment_score(ma_df, params)` method
   - Score from 0-100 based on:
     - Alignment quality (how well separated)
     - Slope direction (all positive = best)
     - Recent crossovers (golden cross = bonus points)
7. Create output DataFrame structure:
   ```python
   Columns: [
       'ticker',
       'adl_ma_20',
       'adl_ma_50',
       'adl_ma_100',
       'ma_alignment',         # 'bullish' | 'bearish' | 'neutral'
       'ma_alignment_score',   # 0-100
       'recent_crossover',     # 'golden' | 'death' | None
       'crossover_date',       # Date of crossover
       'ma_20_slope',
       'ma_50_slope',
       'ma_100_slope'
   ]
   ```
8. Write unit tests
9. Document methodology

**Deliverables**:
- `adl_ma_analysis.py` module
- Unit test suite
- Documentation

### Phase 6: Composite Scoring Module (Step 5)
**Estimated Time**: 3 hours
**Dependencies**: Phases 3, 4, 5

**Module**: `src/screeners/ad_line/adl_composite_scoring.py`

**Tasks**:
1. Create `ADLCompositeScorer` class
2. Implement `calculate_composite_score(mom_score, momentum_score, ma_score, weights)` method
   - Apply configurable weights to each component
   - Normalize to 0-100 scale
   - Return composite score
3. Implement `rank_stocks(composite_df, params)` method
   - Sort by composite score (descending)
   - Add rank column
   - Filter by minimum score threshold
4. Implement `generate_top_candidates(ranked_df, count)` method
   - Extract top N candidates
   - Include all component scores
   - Add summary statistics
5. Implement `generate_score_breakdown(composite_df)` method
   - Show contribution of each component to final score
   - Identify strongest/weakest areas per stock
6. Create output DataFrame structure:
   ```python
   Columns: [
       'rank',
       'ticker',
       'composite_score',
       'longterm_score',
       'shortterm_score',
       'ma_alignment_score',
       'longterm_contribution',   # Weight × Score
       'shortterm_contribution',
       'ma_contribution',
       'meets_threshold'          # Boolean
   ]
   ```
7. Write unit tests
8. Document methodology

**Deliverables**:
- `adl_composite_scoring.py` module
- Unit test suite
- Documentation

### Phase 7: Enhanced Screener Orchestrator
**Estimated Time**: 4 hours
**Dependencies**: All previous phases

**Module**: `src/screeners/ad_line/adl_screener_enhanced.py`

**Tasks**:
1. Create `ADLScreenerEnhanced` class
2. Implement `run_enhanced_screening(batch_data, ticker_info, rs_data, params)` method
   - Orchestrate all analysis modules
   - Aggregate results
   - Handle errors and edge cases
3. Implement module integration:
   ```python
   def process_ticker(ticker, df):
       # Step 1: Calculate ADL (existing)
       adl_series = calculator.calculate_adl(df)

       # Step 2: MoM analysis
       mom_results = mom_analyzer.analyze_monthly_accumulation(df)

       # Step 3: Short-term momentum
       momentum_results = short_term_analyzer.calculate_short_term_changes(df)

       # Step 4: MA analysis
       ma_results = ma_analyzer.calculate_adl_mas(adl_series)

       # Step 5: Composite scoring
       composite_results = scorer.calculate_composite_score(
           mom_results, momentum_results, ma_results
       )

       return composite_results
   ```
4. Implement batch processing logic
5. Implement result aggregation and file writing
6. Add hierarchical configuration handling
7. Add logging and progress tracking
8. Write integration tests

**Deliverables**:
- `adl_screener_enhanced.py` module
- Integration test suite
- Documentation

### Phase 8: Streaming Processor Integration
**Estimated Time**: 3 hours
**Dependencies**: Phase 7

**Module**: `src/screeners_streaming.py`

**Tasks**:
1. Create `ADLEnhancedStreamingProcessor` class extending `StreamingCalculationBase`
2. Implement required abstract methods:
   - `get_calculation_name()` → "adl_enhanced"
   - `get_output_directory()` → screener output directory
   - `calculate_single_ticker()` → placeholder (not used)
3. Implement `process_batch_streaming(batch_data, timeframe)` method
   - Follow PVB/GUPPY pattern for memory management
   - Call enhanced screener for each batch
   - Write results immediately
   - Implement memory cleanup
4. Implement `_write_component_results(component_results, timeframe, date)` method
   - Write separate files for each signal type:
     - `adl_mom_accumulation_{timeframe}_{date}.csv`
     - `adl_short_term_momentum_{timeframe}_{date}.csv`
     - `adl_ma_alignment_{timeframe}_{date}.csv`
     - `adl_composite_ranked_{timeframe}_{date}.csv`
     - `adl_top_candidates_{timeframe}_{date}.csv`
5. Implement hierarchical flag checking
6. Add progress feedback messages
7. Test with small/medium/large batches

**Deliverables**:
- Enhanced streaming processor
- Memory management validation
- Performance benchmarks

### Phase 9: Runner Function Implementation
**Estimated Time**: 2 hours
**Dependencies**: Phase 8

**Module**: `src/screeners_streaming.py`

**Tasks**:
1. Create `run_all_adl_enhanced_streaming(config, user_config, timeframes, clean_file_path)` function
2. Implement hierarchical flag validation:
   - Check master flag (`adl_screener_enable`)
   - Check timeframe flags (`adl_screener_{timeframe}_enable`)
   - Skip if no timeframes enabled
3. Implement DataReader integration for batch loading
4. Implement batch processing loop
5. Add result counting and aggregation
6. Add user feedback messages:
   - Start message with enabled timeframes
   - Progress messages per batch
   - Completion message with result counts
7. Add error handling and logging
8. Test with various flag combinations

**Deliverables**:
- Runner function
- Flag validation logic
- User feedback system

### Phase 10: Configuration Integration
**Estimated Time**: 2 hours
**Dependencies**: None (can run in parallel)

**Files to Update**:
1. `user_data.csv` - Add all new parameters
2. `src/user_defined_data.py` - Update CONFIG_MAPPING and UserConfiguration class
3. Parameter helper function - Create `get_adl_enhanced_params_for_timeframe()`

**Tasks**:
1. Add all new parameters to `user_data.csv` (see Section 4)
2. Map parameters in `CONFIG_MAPPING`:
   ```python
   # ADL Enhanced Parameters
   'ADL_SCREENER_mom_analysis_enable': 'adl_screener_mom_analysis_enable',
   'ADL_SCREENER_mom_period': 'adl_screener_mom_period',
   # ... etc
   ```
3. Add fields to `UserConfiguration` class:
   ```python
   self.adl_screener_mom_analysis_enable = True
   self.adl_screener_mom_period = 22
   # ... etc
   ```
4. Create parameter helper function:
   ```python
   def get_adl_enhanced_params_for_timeframe(user_config, timeframe):
       # Check master flag
       if not getattr(user_config, "adl_screener_enable", False):
           return None

       # Check timeframe flag
       timeframe_flag = f"adl_screener_{timeframe}_enable"
       if not getattr(user_config, timeframe_flag, True):
           return None

       # Return parameter dictionary
       return {
           'enable_adl_enhanced': True,
           'mom_params': { ... },
           'short_term_params': { ... },
           'ma_params': { ... },
           'composite_params': { ... }
       }
   ```
5. Test parameter parsing and validation

**Deliverables**:
- Updated configuration files
- Parameter validation tests

### Phase 11: Main.py Integration
**Estimated Time**: 1 hour
**Dependencies**: Phases 9, 10

**File**: `main.py`

**Tasks**:
1. Add import statement:
   ```python
   from src.screeners_streaming import run_all_adl_enhanced_streaming
   ```
2. Add result variable initialization:
   ```python
   adl_enhanced_results = {}
   ```
3. Add to SCREENERS phase:
   ```python
   # ===============================
   # ADL ENHANCED SCREENER
   # ===============================
   try:
       adl_enhanced_results = run_all_adl_enhanced_streaming(
           config, user_config, timeframes_to_process, clean_file
       )
       logger.info(f"ADL Enhanced Screener completed")
   except Exception as e:
       logger.error(f"Error running ADL Enhanced Screener: {e}")
       adl_enhanced_results = {}
   ```
4. Add to results summary:
   ```python
   adl_enhanced_count = sum(adl_enhanced_results.values()) if adl_enhanced_results else 0
   print(f"🔍 ADL Enhanced Screener Results: {adl_enhanced_count}")
   ```
5. Test integration with main pipeline

**Deliverables**:
- Updated `main.py`
- Integration validation

### Phase 12: Testing and Validation
**Estimated Time**: 4 hours
**Dependencies**: All previous phases

**Testing Scope**:

1. **Unit Tests** (per module):
   - Test ADL calculation with known inputs
   - Test MoM analysis edge cases
   - Test short-term momentum calculations
   - Test MA calculations and crossovers
   - Test composite scoring formulas

2. **Integration Tests**:
   - Test module interactions
   - Test end-to-end screening process
   - Test with various ticker datasets

3. **Batch Processing Tests**:
   - Small batch (10 tickers)
   - Medium batch (100 tickers)
   - Large batch (1000+ tickers)

4. **Memory Tests**:
   - Monitor memory usage during batch processing
   - Verify memory cleanup effectiveness
   - Test with different batch sizes

5. **Output Validation**:
   - Verify CSV file structure
   - Check data integrity
   - Validate composite scores sum correctly
   - Verify ranking order

6. **Performance Tests**:
   - Measure processing time per ticker
   - Test scalability with large datasets
   - Identify bottlenecks

7. **Configuration Tests**:
   - Test all flag combinations
   - Test parameter edge cases
   - Test hierarchical flag logic

**Deliverables**:
- Comprehensive test suite
- Performance benchmarks
- Bug fixes and optimizations

---

## 6. Folder Structure

```
src/screeners/ad_line/
├── __init__.py
│   └── Export main classes and functions
│
├── adl_calculator.py
│   └── Core ADL calculation logic (Step 1)
│       - ADLCalculator class
│       - calculate_adl()
│       - calculate_adl_pct_change()
│       - calculate_rolling_adl()
│
├── adl_mom_analysis.py
│   └── Month-over-month analysis (Step 2)
│       - ADLMoMAnalyzer class
│       - analyze_monthly_accumulation()
│       - calculate_consistency_score()
│       - detect_accumulation_streaks()
│
├── adl_short_term.py
│   └── Short-term momentum analysis (Step 3)
│       - ADLShortTermAnalyzer class
│       - calculate_short_term_changes()
│       - detect_momentum_shifts()
│       - detect_acceleration_patterns()
│       - calculate_momentum_score()
│
├── adl_ma_analysis.py
│   └── Moving average analysis (Step 4)
│       - ADLMAAnalyzer class
│       - calculate_adl_mas()
│       - check_ma_alignment()
│       - detect_ma_crossovers()
│       - calculate_ma_slopes()
│       - calculate_ma_alignment_score()
│
├── adl_composite_scoring.py
│   └── Composite scoring and ranking (Step 5)
│       - ADLCompositeScorer class
│       - calculate_composite_score()
│       - rank_stocks()
│       - generate_top_candidates()
│       - generate_score_breakdown()
│
├── adl_screener_enhanced.py
│   └── Main orchestrator class
│       - ADLScreenerEnhanced class
│       - run_enhanced_screening()
│       - process_single_ticker()
│       - aggregate_results()
│       - _apply_base_filters()
│
└── adl_utils.py
    └── Shared utilities
        - Input validation functions
        - Data transformation helpers
        - Error handling utilities
        - Logging helpers
```

---

## 7. Module Specifications

### 7.1 ADL Calculator (`adl_calculator.py`)

**Purpose**: Core ADL calculation logic, refactored for reusability

**Key Classes**:
```python
class ADLCalculator:
    def __init__(self):
        pass

    def calculate_adl(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Accumulation/Distribution Line"""

    def calculate_adl_pct_change(self, adl_series: pd.Series, period: int) -> pd.Series:
        """Calculate percentage change of ADL over specified period"""

    def calculate_rolling_adl(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Calculate ADL over rolling windows"""

    def validate_input(self, df: pd.DataFrame) -> bool:
        """Validate input DataFrame has required columns"""
```

**Input Requirements**:
- DataFrame with columns: `High`, `Low`, `Close`, `Volume`
- Minimum length: 50 periods (configurable)

**Output**:
- ADL Series with same index as input DataFrame
- Percentage change Series
- Rolling ADL DataFrame

### 7.2 MoM Analyzer (`adl_mom_analysis.py`)

**Purpose**: Month-over-month accumulation trend analysis

**Key Classes**:
```python
class ADLMoMAnalyzer:
    def __init__(self, params: Dict[str, Any]):
        self.period = params['mom_period']  # 22 days
        self.min_threshold = params['mom_min_threshold_pct']  # 15%
        self.max_threshold = params['mom_max_threshold_pct']  # 30%
        self.consecutive_months = params['mom_consecutive_months']  # 3

    def analyze_monthly_accumulation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze monthly accumulation patterns"""

    def calculate_consistency_score(self, monthly_changes: List[float]) -> float:
        """Calculate 0-100 consistency score"""

    def detect_accumulation_streaks(self, monthly_changes: List[float]) -> Dict[str, int]:
        """Detect longest consecutive accumulation streak"""
```

**Output Structure**:
```python
{
    'ticker': str,
    'monthly_changes': List[float],  # List of MoM % changes
    'avg_monthly_change': float,     # Average change
    'consistency_score': float,      # 0-100
    'qualifying_streak': int,        # Longest streak meeting criteria
    'current_streak': int,           # Current active streak
    'meets_criteria': bool           # Passes threshold
}
```

### 7.3 Short-term Analyzer (`adl_short_term.py`)

**Purpose**: Short-term momentum and acceleration detection

**Key Classes**:
```python
class ADLShortTermAnalyzer:
    def __init__(self, params: Dict[str, Any]):
        self.periods = params['short_term_periods']  # [5, 10, 20]
        self.threshold = params['short_term_momentum_threshold']  # 5%

    def calculate_short_term_changes(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate ADL % change for each period"""

    def detect_momentum_shifts(self, short_term_df: pd.DataFrame) -> str:
        """Detect acceleration/deceleration/neutral"""

    def calculate_momentum_score(self, changes: Dict[str, float]) -> float:
        """Calculate weighted 0-100 momentum score"""
```

**Output Structure**:
```python
{
    'ticker': str,
    'adl_5d_pct_change': float,
    'adl_10d_pct_change': float,
    'adl_20d_pct_change': float,
    'momentum_signal': str,        # 'acceleration' | 'deceleration' | 'neutral'
    'momentum_score': float,       # 0-100
    'inflection_point': bool       # True if significant shift detected
}
```

### 7.4 MA Analyzer (`adl_ma_analysis.py`)

**Purpose**: Moving average trend and alignment analysis

**Key Classes**:
```python
class ADLMAAnalyzer:
    def __init__(self, params: Dict[str, Any]):
        self.ma_periods = params['ma_periods']  # [20, 50, 100]
        self.ma_type = params['ma_type']        # 'SMA' or 'EMA'

    def calculate_adl_mas(self, adl_series: pd.Series) -> pd.DataFrame:
        """Calculate all MAs of ADL line"""

    def check_ma_alignment(self, ma_df: pd.DataFrame) -> str:
        """Check bullish/bearish/neutral alignment"""

    def detect_ma_crossovers(self, ma_df: pd.DataFrame, lookback: int) -> Dict:
        """Detect recent MA crossovers"""

    def calculate_ma_alignment_score(self, ma_df: pd.DataFrame) -> float:
        """Calculate 0-100 alignment quality score"""
```

**Output Structure**:
```python
{
    'ticker': str,
    'adl_ma_20': float,
    'adl_ma_50': float,
    'adl_ma_100': float,
    'ma_alignment': str,          # 'bullish' | 'bearish' | 'neutral'
    'ma_alignment_score': float,  # 0-100
    'recent_crossover': str,      # 'golden' | 'death' | None
    'crossover_date': str,        # Date of crossover
    'ma_20_slope': float,
    'ma_50_slope': float,
    'ma_100_slope': float
}
```

### 7.5 Composite Scorer (`adl_composite_scoring.py`)

**Purpose**: Combine all signals into composite score and ranking

**Key Classes**:
```python
class ADLCompositeScorer:
    def __init__(self, params: Dict[str, Any]):
        self.weight_longterm = params['composite_weight_longterm']    # 0.4
        self.weight_shortterm = params['composite_weight_shortterm']  # 0.3
        self.weight_ma = params['composite_weight_ma_align']          # 0.3
        self.min_score = params['composite_min_score']                # 70

    def calculate_composite_score(self, mom_score: float, momentum_score: float,
                                  ma_score: float) -> float:
        """Calculate weighted composite score"""

    def rank_stocks(self, composite_df: pd.DataFrame) -> pd.DataFrame:
        """Rank and filter stocks by composite score"""

    def generate_top_candidates(self, ranked_df: pd.DataFrame, count: int) -> pd.DataFrame:
        """Extract top N candidates"""
```

**Composite Score Formula**:
```
Composite = (0.4 × LongTerm_Score) + (0.3 × ShortTerm_Score) + (0.3 × MA_Score)

Where:
- LongTerm_Score = Consistency score from MoM analysis (0-100)
- ShortTerm_Score = Momentum score from short-term analysis (0-100)
- MA_Score = MA alignment score (0-100)
```

**Output Structure**:
```python
{
    'rank': int,
    'ticker': str,
    'composite_score': float,       # Final weighted score
    'longterm_score': float,        # Component score
    'shortterm_score': float,       # Component score
    'ma_alignment_score': float,    # Component score
    'longterm_contribution': float, # Weight × Score
    'shortterm_contribution': float,
    'ma_contribution': float,
    'meets_threshold': bool         # Passes minimum score
}
```

---

## 8. Integration Points

### 8.1 Data Reader Integration
**Location**: `src/data_reader.py`

**Usage Pattern**:
```python
from src.data_reader import DataReader

# Initialize DataReader for specific timeframe
data_reader = DataReader(config, timeframe='daily', batch_size=100)

# Load tickers from file
data_reader.load_tickers_from_file(clean_file_path)

# Get ticker list
tickers_df = pd.read_csv(clean_file_path)
ticker_list = tickers_df['ticker'].tolist()

# Process batches
for batch_tickers in batch_iterator:
    batch_data = data_reader.read_batch_data(batch_tickers)
    # Process batch...
```

### 8.2 Streaming Processor Integration
**Location**: `src/screeners_streaming.py`

**Base Class**: `StreamingCalculationBase`

**Required Methods**:
```python
class ADLEnhancedStreamingProcessor(StreamingCalculationBase):
    def get_calculation_name(self) -> str:
        return "adl_enhanced"

    def get_output_directory(self) -> Path:
        return self.screener_dir

    def calculate_single_ticker(self, df, ticker, timeframe):
        # Placeholder (not used for screeners)
        return None

    def process_batch_streaming(self, batch_data, timeframe):
        # Main batch processing logic
        pass
```

### 8.3 Configuration Integration
**Location**: `src/user_defined_data.py`

**CONFIG_MAPPING Addition**:
```python
CONFIG_MAPPING = {
    # ... existing mappings ...

    # ADL Enhanced Parameters
    'ADL_SCREENER_mom_analysis_enable': 'adl_screener_mom_analysis_enable',
    'ADL_SCREENER_mom_period': 'adl_screener_mom_period',
    # ... all new parameters ...
}
```

**UserConfiguration Class Addition**:
```python
class UserConfiguration:
    def __init__(self):
        # ... existing fields ...

        # ADL Enhanced Parameters
        self.adl_screener_mom_analysis_enable = True
        self.adl_screener_mom_period = 22
        # ... all new fields ...
```

### 8.4 Main Pipeline Integration
**Location**: `main.py`

**Integration Pattern**:
```python
# Import
from src.screeners_streaming import run_all_adl_enhanced_streaming

# Initialize result variable
adl_enhanced_results = {}

# Execute in SCREENERS phase
if user_config.SCREENERS:
    # ... other screeners ...

    # ADL Enhanced Screener
    try:
        adl_enhanced_results = run_all_adl_enhanced_streaming(
            config, user_config, timeframes_to_process, clean_file
        )
        logger.info("ADL Enhanced Screener completed")
    except Exception as e:
        logger.error(f"Error running ADL Enhanced Screener: {e}")
        adl_enhanced_results = {}

    # ... other screeners ...

# Results summary
adl_enhanced_count = sum(adl_enhanced_results.values()) if adl_enhanced_results else 0
print(f"🔍 ADL Enhanced Screener Results: {adl_enhanced_count}")
```

---

## 9. Output Files

### 9.1 Output Directory Structure
```
results/screeners/adl/
├── adl_mom_accumulation_daily_20250930.csv
├── adl_short_term_momentum_daily_20250930.csv
├── adl_ma_alignment_daily_20250930.csv
├── adl_composite_ranked_daily_20250930.csv
├── adl_top_candidates_daily_20250930.csv
├── adl_enhanced_daily_20250930.csv (consolidated)
└── ... (weekly and monthly variants)
```

### 9.2 File Specifications

#### MoM Accumulation File
**Filename**: `adl_mom_accumulation_{timeframe}_{date}.csv`

**Columns**:
```
ticker, date, price, volume,
monthly_changes, avg_monthly_change, consistency_score,
qualifying_streak, current_streak, meets_criteria
```

**Sample Data**:
```csv
ticker,date,price,volume,monthly_changes,avg_monthly_change,consistency_score,qualifying_streak,current_streak,meets_criteria
AAPL,2025-09-30,175.50,65234567,"[18.2, 22.5, 19.8, 25.1]",21.4,85.5,4,4,TRUE
MSFT,2025-09-30,420.25,23456789,"[12.3, 28.7, 15.9, 20.4]",19.3,72.0,3,3,TRUE
```

#### Short-term Momentum File
**Filename**: `adl_short_term_momentum_{timeframe}_{date}.csv`

**Columns**:
```
ticker, date, price, volume,
adl_5d_pct_change, adl_10d_pct_change, adl_20d_pct_change,
momentum_signal, momentum_score, inflection_point
```

**Sample Data**:
```csv
ticker,date,price,volume,adl_5d_pct_change,adl_10d_pct_change,adl_20d_pct_change,momentum_signal,momentum_score,inflection_point
AAPL,2025-09-30,175.50,65234567,8.5,7.2,6.1,acceleration,82.3,FALSE
NVDA,2025-09-30,485.75,98765432,12.3,10.5,8.7,acceleration,91.5,TRUE
```

#### MA Alignment File
**Filename**: `adl_ma_alignment_{timeframe}_{date}.csv`

**Columns**:
```
ticker, date, price, volume,
adl_ma_20, adl_ma_50, adl_ma_100,
ma_alignment, ma_alignment_score,
recent_crossover, crossover_date,
ma_20_slope, ma_50_slope, ma_100_slope
```

**Sample Data**:
```csv
ticker,date,price,volume,adl_ma_20,adl_ma_50,adl_ma_100,ma_alignment,ma_alignment_score,recent_crossover,crossover_date,ma_20_slope,ma_50_slope,ma_100_slope
AAPL,2025-09-30,175.50,65234567,125000,118000,110000,bullish,88.5,golden,2025-09-15,0.025,0.018,0.012
```

#### Composite Ranked File
**Filename**: `adl_composite_ranked_{timeframe}_{date}.csv`

**Columns**:
```
rank, ticker, date, price, volume,
composite_score, longterm_score, shortterm_score, ma_alignment_score,
longterm_contribution, shortterm_contribution, ma_contribution,
meets_threshold
```

**Sample Data**:
```csv
rank,ticker,date,price,volume,composite_score,longterm_score,shortterm_score,ma_alignment_score,longterm_contribution,shortterm_contribution,ma_contribution,meets_threshold
1,NVDA,2025-09-30,485.75,98765432,89.2,92.5,91.5,84.0,37.0,27.5,25.2,TRUE
2,AAPL,2025-09-30,175.50,65234567,85.7,85.5,82.3,88.5,34.2,24.7,26.6,TRUE
3,MSFT,2025-09-30,420.25,23456789,78.5,72.0,85.0,79.0,28.8,25.5,23.7,TRUE
```

#### Top Candidates File
**Filename**: `adl_top_candidates_{timeframe}_{date}.csv`

**Content**: Top N stocks (default 50) from composite ranked file with all component details

**Additional Columns**:
```
All columns from composite ranked file plus:
- summary_reason: Text description of why stock ranks highly
- key_strengths: Strongest component areas
- key_weaknesses: Areas needing attention
```

### 9.3 Consolidated File
**Filename**: `adl_enhanced_{timeframe}_{date}.csv`

**Content**: All results combined into single file with all metrics from all modules

---

## 10. Testing Strategy

### 10.1 Unit Testing Scope

**Per Module Tests**:

1. **ADL Calculator** (`test_adl_calculator.py`):
   - Test ADL calculation with known inputs
   - Test edge cases (zero volume, doji bars, gaps)
   - Test percentage change calculations
   - Test rolling window calculations
   - Test input validation

2. **MoM Analyzer** (`test_adl_mom_analysis.py`):
   - Test monthly interval extraction
   - Test MoM % change calculation
   - Test consistency score calculation
   - Test streak detection
   - Test threshold filtering

3. **Short-term Analyzer** (`test_adl_short_term.py`):
   - Test period change calculations
   - Test momentum signal detection
   - Test acceleration pattern detection
   - Test momentum score calculation
   - Test inflection point detection

4. **MA Analyzer** (`test_adl_ma_analysis.py`):
   - Test SMA/EMA calculations
   - Test MA alignment detection
   - Test crossover detection
   - Test slope calculations
   - Test alignment score calculation

5. **Composite Scorer** (`test_adl_composite_scoring.py`):
   - Test composite score formula
   - Test weight application
   - Test ranking logic
   - Test filtering by threshold
   - Test top candidate extraction

### 10.2 Integration Testing Scope

**Integration Test Cases**:

1. **End-to-End Screening**:
   - Test full screening process on sample dataset
   - Verify all modules are called in correct order
   - Verify results are aggregated correctly
   - Verify output files are created

2. **Module Interaction**:
   - Test data flow between modules
   - Test error propagation
   - Test result merging

3. **Configuration Integration**:
   - Test parameter parsing from user_data.csv
   - Test hierarchical flag logic
   - Test parameter validation

### 10.3 Performance Testing Scope

**Performance Test Cases**:

1. **Scalability Tests**:
   - Small dataset (10 tickers)
   - Medium dataset (100 tickers)
   - Large dataset (1000 tickers)
   - Very large dataset (5000+ tickers)

2. **Memory Tests**:
   - Monitor memory usage during batch processing
   - Test memory cleanup effectiveness
   - Test for memory leaks

3. **Speed Tests**:
   - Measure processing time per ticker
   - Measure total pipeline time
   - Identify bottlenecks with profiling

**Performance Targets**:
- Processing time: < 1 second per ticker
- Memory growth: < 100MB per 100 tickers
- Memory cleanup: > 90% recovery after batch

### 10.4 Test Data Preparation

**Sample Test Data**:

1. **Known ADL Pattern**: Stock with clear accumulation trend
2. **Volatile Stock**: High volatility with erratic ADL
3. **Flat Stock**: Minimal price/ADL movement
4. **Gapper**: Stock with frequent price gaps
5. **Low Volume**: Stock with sporadic volume
6. **Perfect Candidate**: All signals aligned positively

---

## 11. Performance Considerations

### 11.1 Memory Management

**Strategy**:
1. Process in configurable batches (default 100 tickers)
2. Write results immediately after each batch
3. Delete large objects explicitly before garbage collection
4. Use `gc.collect()` after batch processing
5. Optimize DataFrame dtypes (use int16/float32 where possible)

**Memory Cleanup Pattern**:
```python
try:
    # Process batch
    batch_results = process_batch(batch_data)

    # Write immediately
    write_results_to_csv(batch_results)

    # Explicit cleanup
    del batch_results, batch_data
    gc.collect()

except Exception as e:
    logger.error(f"Batch processing error: {e}")
```

### 11.2 Processing Optimization

**Optimization Techniques**:

1. **Vectorized Operations**:
   - Use pandas vectorized operations instead of loops
   - Avoid `apply()` with lambda functions where possible
   - Use NumPy operations for numerical calculations

2. **Lazy Evaluation**:
   - Only calculate metrics that will be used
   - Skip disabled modules early
   - Filter tickers before expensive calculations

3. **Caching**:
   - Cache ADL calculations when used by multiple modules
   - Cache MA calculations for crossover detection
   - Reuse calculated values across modules

4. **Parallel Processing** (future enhancement):
   - Process batches in parallel using multiprocessing
   - Careful with shared state and memory

### 11.3 Scalability Considerations

**Design for Scale**:

1. **Batch Size Tuning**:
   - Smaller batches (50-100) for limited memory systems
   - Larger batches (200-500) for high-memory systems
   - Auto-tune based on available memory (future feature)

2. **Progressive File Writing**:
   - Append to CSV files instead of accumulating all results
   - Use chunked writing for very large result sets

3. **Database Integration** (future enhancement):
   - Store results in SQLite/PostgreSQL for large datasets
   - Enable querying and filtering without loading all data

### 11.4 Performance Monitoring

**Key Metrics to Track**:

1. **Processing Time**:
   - Time per ticker
   - Time per batch
   - Total pipeline time

2. **Memory Usage**:
   - Peak memory usage
   - Memory growth rate
   - Memory after cleanup

3. **Throughput**:
   - Tickers processed per minute
   - Batches processed per minute

**Logging Pattern**:
```python
logger.info(f"Batch {batch_num} completed: "
           f"{len(batch_data)} tickers processed in {elapsed_time:.2f}s, "
           f"Memory: {current_memory_mb:.1f}MB")
```

---

## 12. Signal Interpretation Guide

### 12.1 Understanding Component Scores

#### Long-term Accumulation Score (0-100)
**Source**: Month-over-month analysis

**Interpretation**:
- **90-100**: Exceptional consistency, 4+ months of steady accumulation within ideal range
- **75-89**: Strong consistency, 3+ months meeting criteria
- **60-74**: Moderate consistency, some months meeting criteria
- **Below 60**: Weak or inconsistent accumulation pattern

**What to Look For**:
- High consistency score + high average monthly change = strong sustained accumulation
- High consistency but low average change = weak accumulation
- Low consistency with high average = erratic/unreliable pattern

#### Short-term Momentum Score (0-100)
**Source**: Short-term percentage change analysis **of the ADL line itself** (NOT price)

**What It Measures**:
- Calculates **5-day, 10-day, and 20-day percentage changes** of the ADL line
- Detects acceleration/deceleration patterns in accumulation
- Example: If ADL was 1,000,000 five days ago and is 1,050,000 today, that's +5% ADL change

**Why This Matters**:
- ADL momentum can lead price momentum
- Accelerating ADL = institutions are increasing their buying pace
- Decelerating ADL = buying pressure weakening (potential warning)
- A score of 100 means the ADL line is growing faster in recent days (5d) than in the medium term (10d, 20d) = **acceleration**

**Interpretation**:
- **90-100**: Very strong ADL momentum, all periods positive and accelerating
  - 5d ADL change > 10d change > 20d change (getting stronger)
  - Momentum signal: "acceleration"
  - Institutional buying is intensifying
- **75-89**: Strong ADL momentum, most periods positive
  - Momentum signal: "momentum" (positive but not accelerating)
  - Steady institutional accumulation
- **60-74**: Moderate ADL momentum, mixed signals
  - Some periods positive, some flat or negative
  - Momentum signal: may be "deceleration" or "neutral"
- **Below 60**: Weak or negative ADL momentum
  - ADL line flat or declining
  - Distribution or consolidation

**What to Look For**:
- High momentum + "acceleration" signal = **strong near-term catalyst**, institutions rapidly accumulating
- High momentum + "deceleration" signal = **potential topping**, buying pressure weakening
- Inflection point flag = significant momentum shift (ADL trend reversal)

**Real Example (IDXX)**:
- Short-term Score: 100 → ADL line growing very rapidly in last 5 days
- Momentum signal: "acceleration" → 5d ADL% > 10d ADL% > 20d ADL%
- Interpretation: Institutions dramatically increased buying pace recently

#### MA Alignment Score (0-100)
**Source**: Moving average analysis

**Interpretation**:
- **90-100**: Perfect bullish alignment, all MAs positive slope, wide separation
- **75-89**: Strong bullish alignment, all MAs trending up
- **60-74**: Moderate bullish alignment or neutral
- **Below 60**: Weak alignment or bearish configuration

**What to Look For**:
- High MA score + recent golden cross = confirming uptrend
- High MA score + tight MA separation = potential breakout setup
- All MA slopes positive = healthy sustained trend

### 12.2 Composite Score Interpretation

**Composite Score Ranges**:

- **90-100**: Elite candidates
  - All components strong
  - High confidence in sustained accumulation
  - Top priority for further research

- **80-89**: Excellent candidates
  - Most components strong
  - One area may need attention
  - Strong consideration for watchlist

- **70-79**: Good candidates
  - Mixed strength across components
  - Closer evaluation needed
  - May have specific catalysts

- **60-69**: Fair candidates (if threshold allows)
  - Some weakness in one or more areas
  - Higher risk
  - May be early-stage accumulation

- **Below 60**: Weak or excluded
  - Multiple areas of concern
  - Not meeting minimum criteria

### 12.3 Combined Signal Analysis

**Ideal Profile** (Top Candidates):
```
Composite Score: 85+
├── Long-term Score: 80+ (consistent accumulation for 3+ months)
├── Short-term Score: 80+ (strong recent momentum with acceleration)
└── MA Alignment: 80+ (bullish MA configuration, all slopes positive)
```

**Example Profiles**:

1. **Growth Momentum Stock**:
   - Composite: 88
   - Long-term: 92 (very consistent accumulation)
   - Short-term: 85 (accelerating)
   - MA: 88 (perfect alignment, recent golden cross)
   - **Interpretation**: Strong sustained accumulation with accelerating momentum. Very high probability of continued strength.

2. **Early Stage Accumulation**:
   - Composite: 72
   - Long-term: 68 (building consistency over 2 months)
   - Short-term: 78 (good recent momentum)
   - MA: 75 (MAs aligning but not perfect yet)
   - **Interpretation**: Potentially catching accumulation early. Higher risk but could be rewarding if trend continues.

3. **Mature Accumulation**:
   - Composite: 82
   - Long-term: 95 (excellent consistency for 5+ months)
   - Short-term: 65 (momentum slowing)
   - MA: 85 (still well-aligned but widening)
   - **Interpretation**: Long-standing accumulation but showing signs of maturity. May be near end of accumulation phase.

4. **Momentum Leader**:
   - Composite: 78
   - Long-term: 65 (moderate consistency)
   - Short-term: 95 (very strong recent momentum)
   - MA: 85 (newly aligned, recent golden cross)
   - **Interpretation**: Shorter accumulation period but very strong recent momentum. Could be breakout candidate or short-term catalyst.

### 12.4 Red Flags to Watch For

1. **High Composite with Declining Short-term**:
   - May indicate accumulation phase ending
   - Consider reducing position or taking profits

2. **High Long-term but Low MA Alignment**:
   - Accumulation without price trend confirmation
   - May be early-stage (good) or distribution masking as accumulation (bad)

3. **High Short-term but Low Long-term**:
   - Short-term pop without sustained accumulation
   - Higher risk of reversal
   - May be news-driven spike

4. **Recent Death Cross Despite High Scores**:
   - Trend may be changing
   - Re-evaluate position or wait for confirmation

### 12.5 Using Scores for Position Sizing

**Suggested Position Sizing Framework**:

- **Composite 90-100**: Maximum position size (e.g., 5-10% portfolio)
- **Composite 80-89**: Standard position size (e.g., 3-7% portfolio)
- **Composite 70-79**: Reduced position size (e.g., 1-4% portfolio)
- **Composite 60-69**: Minimal/speculative position (e.g., 0.5-2% portfolio)

**Adjustments Based on Component Scores**:
- Strong long-term + weak short-term → Patient accumulation, normal size
- Weak long-term + strong short-term → Shorter holding period, reduced size
- All components balanced → Standard approach

### 12.6 Combining with Other Analysis

**ADL Screener + Other Screeners**:

1. **ADL + Relative Strength (RS)**:
   - High ADL composite + High RS = Institutional accumulation of strong stock
   - High ADL + Low RS = Potential turnaround or early-stage accumulation

2. **ADL + Volume Screeners**:
   - High ADL + High Volume Anomaly = Confirming accumulation with institutional interest
   - High ADL + Normal Volume = Steady accumulation without attention (could be opportunity)

3. **ADL + Minervini/Dr. Wish**:
   - High ADL + GLB Breakout = Very strong technical setup
   - High ADL + Stage 2 Uptrend = Confirming trend with accumulation

**Recommended Workflow**:
1. Run ADL Enhanced Screener → Get top candidates
2. Cross-reference with RS Screener → Filter for relative strength
3. Check Volume Screeners → Confirm institutional activity
4. Review Stage Analysis → Confirm trend stage
5. Manual chart review → Final validation

### 12.7 Output File Column Interpretation

**Example Output Line** (from `adl_composite_ranked_daily_20250905.csv`):
```
IDXX,2025-09-05,642.99,370400,70.33,good,33.33,100,90,13.33,30,27,True,shortterm;ma_alignment,longterm,1,77.92,momentum,,bullish,,1
```

**Column-by-Column Breakdown**:

| Column | Value | Meaning |
|--------|-------|---------|
| **ticker** | IDXX | Stock symbol |
| **date** | 2025-09-05 | Analysis date |
| **price** | 642.99 | Closing price on analysis date |
| **volume** | 370400 | Volume on analysis date |
| **composite_score** | 70.33 | **Overall score (0-100)** - Weighted combination of all components |
| **score_category** | good | **Classification**: elite (90+), excellent (80-89), good (70-79), fair (60-69), weak (<60) |
| **longterm_score** | 33.33 | **MoM accumulation consistency (0-100)** - Based on monthly ADL percentage changes |
| **shortterm_score** | 100 | **Short-term ADL momentum (0-100)** - Based on 5d, 10d, 20d ADL% changes |
| **ma_alignment_score** | 90 | **MA alignment quality (0-100)** - Based on 20/50/100 MAs on ADL line |
| **longterm_contribution** | 13.33 | Points contributed by longterm score (weight × score = 0.4 × 33.33) |
| **shortterm_contribution** | 30 | Points contributed by shortterm score (weight × score = 0.3 × 100) |
| **ma_contribution** | 27 | Points contributed by MA score (weight × score = 0.3 × 90) |
| **meets_threshold** | True | Whether composite_score ≥ minimum threshold (default 70) |
| **key_strengths** | shortterm;ma_alignment | Components scoring ≥80 (separated by semicolon) |
| **key_weaknesses** | longterm | Components scoring <60 (separated by semicolon) |
| **rank** | 1 | Global rank by composite score (1 = highest) |
| **mom_avg_monthly_change** | 77.92 | Average monthly ADL percentage change over lookback period |
| **momentum_signal** | momentum | Short-term signal: acceleration/momentum/deceleration/neutral |
| **acceleration_pattern** | (empty) | Specific pattern: strong_acceleration, moderate_acceleration, slowing_momentum, etc. |
| **ma_alignment** | bullish | MA configuration: bullish (MA20>MA50>MA100), bearish, or neutral |
| **recent_crossover** | (empty) | Recent MA crossover: golden (bullish), death (bearish), or empty if none |
| **mom_current_streak** | 1 | Number of consecutive months meeting MoM criteria (15-30% growth) |

**How to Read This Row**:
```
IDXX ranks #1 with a composite score of 70.33 (good category).

Component Breakdown:
- Long-term: 33.33 (WEAK) - Only 1 month streak, not meeting 3+ month consistency
- Short-term: 100 (STRONG) - Perfect momentum with acceleration signal
- MA Alignment: 90 (STRONG) - Near-perfect bullish alignment

The stock shows EXPLOSIVE short-term ADL momentum (100) and excellent
MA alignment (90), despite weak long-term consistency (33.33). This
indicates institutions are rapidly accumulating NOW, but the pattern
is relatively new (1-month streak vs. 3+ required).

Trading Interpretation: Strong momentum breakout candidate. The perfect
short-term score suggests institutional buying is accelerating in the
last 5-20 days. The weakness in long-term suggests this is a newer setup,
not a well-established accumulation base.
```

**Key Insight from This Example**:
- **High short-term + Low long-term** = Recent catalyst or breakout
- **High long-term + Low short-term** = Mature accumulation losing steam
- **High both** = Sustained accumulation with acceleration (ideal)
- **Low both** = Avoid or distribution phase

---

## Appendix A: Configuration Parameter Reference

### Complete Parameter List

```csv
# MASTER CONTROL
ADL_SCREENER_enable                         FALSE       # Master enable flag

# TIMEFRAME CONTROL
ADL_SCREENER_daily_enable                   FALSE       # Daily timeframe
ADL_SCREENER_weekly_enable                  FALSE       # Weekly timeframe
ADL_SCREENER_monthly_enable                 FALSE       # Monthly timeframe

# BASIC FILTERS (existing)
ADL_SCREENER_lookback_period                50          # Base ADL calculation period
ADL_SCREENER_divergence_period              20          # Divergence analysis period
ADL_SCREENER_breakout_period                30          # Breakout detection period
ADL_SCREENER_min_divergence_strength        0.7         # Minimum divergence threshold
ADL_SCREENER_min_breakout_strength          1.2         # Minimum breakout threshold
ADL_SCREENER_min_volume_avg                 100000      # Minimum average volume filter
ADL_SCREENER_min_price                      5           # Minimum price filter
ADL_SCREENER_save_individual_files          TRUE        # Save component files

# MONTH-OVER-MONTH ANALYSIS (new)
ADL_SCREENER_mom_analysis_enable            TRUE        # Enable MoM module
ADL_SCREENER_mom_period                     22          # Trading days per month
ADL_SCREENER_mom_min_threshold_pct          15          # Minimum monthly growth (%)
ADL_SCREENER_mom_max_threshold_pct          30          # Maximum monthly growth (%)
ADL_SCREENER_mom_consecutive_months         3           # Min consecutive months
ADL_SCREENER_mom_lookback_months            6           # Total months to analyze
ADL_SCREENER_mom_min_consistency_score      60          # Minimum consistency score

# SHORT-TERM MOMENTUM (new)
ADL_SCREENER_short_term_enable              TRUE        # Enable short-term module
ADL_SCREENER_short_term_periods             5;10;20     # Periods for % change
ADL_SCREENER_short_term_momentum_threshold  5           # Minimum significant % change
ADL_SCREENER_short_term_acceleration_detect TRUE        # Enable acceleration detection
ADL_SCREENER_short_term_min_score           50          # Minimum momentum score

# MOVING AVERAGE ANALYSIS (new)
ADL_SCREENER_ma_enable                      TRUE        # Enable MA module
ADL_SCREENER_ma_periods                     20;50;100   # MA periods
ADL_SCREENER_ma_type                        SMA         # Type: SMA or EMA
ADL_SCREENER_ma_bullish_alignment_required  TRUE        # Require bullish alignment
ADL_SCREENER_ma_crossover_detection         TRUE        # Detect crossovers
ADL_SCREENER_ma_crossover_lookback          10          # Periods for crossover detection
ADL_SCREENER_ma_min_slope_threshold         0.01        # Minimum positive slope
ADL_SCREENER_ma_min_alignment_score         70          # Minimum alignment score

# COMPOSITE SCORING (new)
ADL_SCREENER_composite_scoring_enable       TRUE        # Enable composite scoring
ADL_SCREENER_composite_weight_longterm      0.4         # Weight for long-term
ADL_SCREENER_composite_weight_shortterm     0.3         # Weight for short-term
ADL_SCREENER_composite_weight_ma_align      0.3         # Weight for MA alignment
ADL_SCREENER_composite_min_score            70          # Minimum composite score
ADL_SCREENER_ranking_method                 composite   # Ranking method
ADL_SCREENER_output_ranking_file            TRUE        # Create ranked file
ADL_SCREENER_top_candidates_count           50          # Number of top candidates

# OUTPUT CONFIGURATION (new)
ADL_SCREENER_output_separate_signals        TRUE        # Separate files per signal type
ADL_SCREENER_output_include_charts          FALSE       # Generate charts (future)
ADL_SCREENER_output_summary_stats           TRUE        # Include summary statistics
```

---

## Appendix B: Example Use Cases

### Use Case 1: Finding Long-term Accumulation Plays

**Goal**: Identify stocks with consistent accumulation over 3+ months

**Configuration**:
```csv
ADL_SCREENER_enable                        TRUE
ADL_SCREENER_daily_enable                  TRUE
ADL_SCREENER_mom_analysis_enable           TRUE
ADL_SCREENER_mom_consecutive_months        3
ADL_SCREENER_mom_min_consistency_score     80
ADL_SCREENER_composite_weight_longterm     0.6    # Higher weight on long-term
ADL_SCREENER_composite_weight_shortterm    0.2
ADL_SCREENER_composite_weight_ma_align     0.2
ADL_SCREENER_composite_min_score           75
```

**Expected Output**:
- Focus on stocks with high long-term accumulation scores
- Less emphasis on short-term momentum
- Top candidates will have sustained accumulation patterns

### Use Case 2: Finding Momentum Breakout Candidates

**Goal**: Identify stocks with strong recent momentum and MA alignment

**Configuration**:
```csv
ADL_SCREENER_enable                        TRUE
ADL_SCREENER_daily_enable                  TRUE
ADL_SCREENER_short_term_enable             TRUE
ADL_SCREENER_short_term_acceleration_detect TRUE
ADL_SCREENER_ma_enable                     TRUE
ADL_SCREENER_ma_crossover_detection        TRUE
ADL_SCREENER_composite_weight_longterm     0.2
ADL_SCREENER_composite_weight_shortterm    0.5    # Higher weight on momentum
ADL_SCREENER_composite_weight_ma_align     0.3
ADL_SCREENER_composite_min_score           75
```

**Expected Output**:
- Focus on stocks with strong recent momentum
- MA crossovers flagged
- Acceleration patterns highlighted
- Higher risk/reward profile

### Use Case 3: Conservative Screening (All Components Strong)

**Goal**: Identify only stocks that meet high standards across all metrics

**Configuration**:
```csv
ADL_SCREENER_enable                         TRUE
ADL_SCREENER_daily_enable                   TRUE
ADL_SCREENER_mom_analysis_enable            TRUE
ADL_SCREENER_short_term_enable              TRUE
ADL_SCREENER_ma_enable                      TRUE
ADL_SCREENER_mom_min_consistency_score      75
ADL_SCREENER_short_term_min_score           75
ADL_SCREENER_ma_min_alignment_score         75
ADL_SCREENER_composite_weight_longterm      0.4    # Balanced weights
ADL_SCREENER_composite_weight_shortterm     0.3
ADL_SCREENER_composite_weight_ma_align      0.3
ADL_SCREENER_composite_min_score            85     # High threshold
```

**Expected Output**:
- Very selective screening
- Fewer results but higher quality
- All components must be strong
- Lower risk profile

---

## Appendix C: Troubleshooting Guide

### Common Issues and Solutions

#### Issue 1: No results returned
**Possible Causes**:
- Threshold scores set too high
- All timeframe flags disabled
- Insufficient historical data

**Solutions**:
- Lower `composite_min_score` threshold
- Check timeframe enable flags
- Verify data files have enough history (6+ months for MoM analysis)

#### Issue 2: Memory errors during processing
**Possible Causes**:
- Batch size too large
- Insufficient RAM
- Memory leak in custom code

**Solutions**:
- Reduce `batch_size` in user_data.csv
- Check memory cleanup implementation
- Monitor memory usage with logging

#### Issue 3: Processing very slow
**Possible Causes**:
- Large dataset
- Inefficient calculations
- Disabled batch processing

**Solutions**:
- Enable batch processing
- Optimize vectorized operations
- Profile code to identify bottlenecks
- Consider parallel processing

#### Issue 4: Inconsistent scores
**Possible Causes**:
- Parameter configuration mismatch
- Data quality issues
- Edge cases not handled

**Solutions**:
- Verify parameter values in user_data.csv
- Check for data gaps or anomalies
- Review edge case handling in code

---

## Appendix D: Future Enhancements

### Potential Future Features

1. **Machine Learning Integration**:
   - Train ML model on historical ADL patterns
   - Predict accumulation strength
   - Anomaly detection

2. **Advanced Charting**:
   - Generate ADL charts with MA overlay
   - Annotate signals and crossovers
   - Export to TradingView format

3. **Real-time Monitoring**:
   - Intraday ADL calculation
   - Alert system for threshold crossings
   - WebSocket integration for live data

4. **Sector/Industry Analysis**:
   - Aggregate ADL by sector
   - Compare individual stocks to sector ADL
   - Sector rotation signals

5. **Backtesting Framework**:
   - Test ADL signals historically
   - Optimize parameters based on returns
   - Risk-adjusted performance metrics

6. **Integration with Other Systems**:
   - Export to portfolio management tools
   - API for external access
   - Database storage for historical tracking

7. **Advanced Filtering**:
   - Market cap filters
   - Liquidity filters
   - Sector/industry filters
   - Custom user-defined filters

8. **Statistical Analysis**:
   - Correlation analysis between components
   - Distribution analysis of scores
   - Success rate tracking

---

**Document Version**: 1.0
**Last Updated**: 2025-09-30
**Status**: Planning Complete - Ready for Implementation

---
