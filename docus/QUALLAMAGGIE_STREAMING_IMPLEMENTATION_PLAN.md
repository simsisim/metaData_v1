# Qullamaggie Screener Streaming Integration - Implementation Plan

**Date**: 2025-10-01
**Status**: 🚧 RESEARCH & PLANNING PHASE
**Reference**: `docus/SCREENER_IMPLEMENTATION_GUIDE.md`

---

## EXECUTIVE SUMMARY

Qullamaggie (Kristjan Kullamägi) is a legendary momentum trader who turned $5K into $100M+ using systematic momentum breakout strategies. This plan outlines integrating the existing Qullamaggie screener into the streaming processing framework.

**Key Challenge**: Unlike other screeners, Qullamaggie requires **external data dependencies**:
- Relative Strength (RS) data across multiple timeframes
- ATR universe rankings for $1B+ market cap stocks
- Market cap and ticker metadata

---

## QULLAMAGGIE STRATEGY OVERVIEW

### Trading Methodology (From Research)

**Core Philosophy**:
- Momentum swing trading with 5-25% position sizes
- Risk 0.3-0.5% per trade (rarely >1%)
- Hold 3-5 days for 1/3 to 1/2 position, trail remainder with 10/20-day MA
- Only trade in uptrending or sideways markets (no downtrends)

**Three Main Setups**:

1. **Continuation Patterns** (Primary)
   - Flags: Swift moves + tight consolidation
   - Triangles: Tightening ranges after rally
   - VCPs (Volatility Contraction Patterns): Progressive tightening
   - Cup-and-Handle formations

2. **Episodic Pivots (EPs)**
   - Gap-up base breakouts
   - News/earnings-driven spikes
   - Fast-moving opportunities

3. **Darvas Boxes**
   - Flat bases with narrow oscillation
   - Sideways consolidation before breakout

**Key Indicators**:
- **Moving Averages**: 10-day (fast), 20-day (intermediate), 50-day (long-term)
- **Entry**: Opening range highs (1m/5m/60m charts post-breakout)
- **Exit**: Trail with 10/20-day MA, close below = exit

---

## CURRENT STATUS

### ✅ Already Complete

1. **Core Qullamaggie Screener** (`src/screeners/qullamaggie_suite.py` - 707 lines)
   - `QullamaggieScreener` class fully implemented
   - `run_qullamaggie_screening()` method exists
   - Comprehensive filtering logic with 5 main criteria
   - IPO stock handling (reduced data requirements)
   - ATR extension calculations and formatting

2. **User Configuration** (`user_data.csv` lines 585-590)
   - Master flag: `QULLAMAGGIE_SUITE_enable,FALSE`
   - **⚠️ MISSING**: Timeframe flags (daily/weekly/monthly)
   - 5 parameters configured:
     - `rs_threshold` (97 = top 3%)
     - `atr_rs_threshold` (50 = top 50%)
     - `range_position_threshold` (0.5 = 50% of range)
     - `extension_warning` (7x from SMA50)
     - `extension_danger` (11x from SMA50)

3. **Output Directory** (line 137)
   - `QULLAMAGGIE_output_dir,results/screeners/qullamaggie`

### ❌ Missing Components

1. **Timeframe Flags** in `user_data.csv`
2. **Timeframe Flag Fields** in `user_defined_data.py` dataclass
3. **Timeframe Flag Mapping** in CSV config mapping
4. **Additional Parameters** (missing from user_data.csv):
   - `min_market_cap` (default: $1B)
   - `min_price` (default: $5)
   - `save_individual_files` flag
5. **Qullamaggie Streaming Processor** in `src/screeners_streaming.py`
6. **Main Runner Function** in `src/screeners_streaming.py`
7. **Main.py Integration**
8. **RS Data Pipeline** (CRITICAL - no current implementation)
9. **ATR Universe Pipeline** (CRITICAL - no current implementation)

---

## SCREENING CRITERIA (IMPLEMENTED IN CORE SCREENER)

The Qullamaggie screener implements these 5 strict filters:

### 1. Market Cap Filter
- **Requirement**: Market cap ≥ $1 billion
- **Rationale**: Ensures liquidity for large position sizes
- **Implementation**: Checks `ticker_info` DataFrame

### 2. Relative Strength (RS) Filter ⚠️ **REQUIRES EXTERNAL DATA**
- **Requirement**: RS ≥ 97 on at least one timeframe (top 3%)
- **Timeframes Checked**: 1-week, 1-month, 3-month, 6-month
- **Implementation**:
  - Requires `rs_data` dictionary with structure:
    ```python
    rs_data = {
        'daily': {
            'period_7': DataFrame,   # 1-week RS
            'period_21': DataFrame,  # 1-month RS
            'period_63': DataFrame,  # 3-month RS
            'period_126': DataFrame  # 6-month RS
        }
    }
    ```
  - Each DataFrame has ticker as index, RS score as value
  - **MISSING**: No RS calculation pipeline exists yet

### 3. Moving Averages Alignment
- **Requirement**: Perfect alignment for established stocks:
  - Price ≥ EMA10 ≥ SMA20 ≥ SMA50 ≥ SMA100 ≥ SMA200
- **For IPOs** (< 250 days data): Uses available MAs only
- **Implementation**: Built into screener, calculates from OHLCV

### 4. ATR Relative Strength ⚠️ **REQUIRES EXTERNAL DATA**
- **Requirement**: ATR RS ≥ 50 vs $1B+ universe (top 50% volatility)
- **Implementation**:
  - Requires `atr_universe_data` dictionary:
    ```python
    atr_universe_data = {
        'AAPL': DataFrame,
        'MSFT': DataFrame,
        # ... all $1B+ stocks
    }
    ```
  - Screener calculates ATR for all universe stocks
  - Ranks ticker's ATR percentile within universe
  - **MISSING**: No ATR universe pipeline exists yet

### 5. Range Position Filter
- **Requirement**: Price in upper 50% of 20-day range
- **Calculation**: `(price - low_20d) / (high_20d - low_20d) ≥ 0.5`
- **Implementation**: Built into screener, calculates from OHLCV

### Output Metrics

Each passing ticker returns:
- Ticker symbol and exchange
- Current price
- Market cap
- RS scores (all qualified timeframes)
- MA alignment status and score
- ATR RS percentile
- Range position percentage
- **ATR extension from SMA50** (primary sort metric)
- Extension formatting (normal / ⚠️ warning 7x+ / 🔴 danger 11x+)

---

## DATA DEPENDENCY ANALYSIS

### Critical Missing Data Pipelines

#### 1. Relative Strength (RS) Calculation Pipeline ⚠️ **CRITICAL**

**What It Is**:
- Rank each stock's price performance vs entire universe
- Calculate for multiple lookback periods (7d, 21d, 63d, 126d)
- Return percentile ranking (0-100, where 97+ = top 3%)

**Current Status**:
- ❌ **NO RS calculation exists in codebase**
- Need to research if RS data is calculated elsewhere
- May require new calculation module

**Complexity**:
- HIGH - requires loading all ticker data simultaneously
- Memory-intensive (all tickers × multiple periods)
- Computationally expensive (rankings for 5000+ stocks)

**Options**:
1. **Pre-calculate during Basic Calculations phase**
   - Add RS calculation to existing calculation pipeline
   - Save RS data to disk for screener to load
   - Most memory-efficient approach

2. **Calculate on-demand in screener**
   - Load all data when screener runs
   - Calculate RS on the fly
   - High memory usage, slower

3. **Use external RS data source**
   - Import from existing RS calculations if available
   - Check if RS already calculated elsewhere

**Recommended**: Option 1 - Pre-calculate and save

#### 2. ATR Universe Data Pipeline ⚠️ **CRITICAL**

**What It Is**:
- Filter all stocks with market cap ≥ $1B
- Calculate 14-period ATR for each
- Provide to screener for percentile ranking

**Current Status**:
- ❌ **NO ATR universe pipeline exists**
- Market cap data exists in `ticker_universe_all.csv`
- ATR calculation likely exists in other modules

**Complexity**:
- MEDIUM - requires market cap filtering + ATR calculation
- Less memory-intensive than RS (only $1B+ stocks)

**Options**:
1. **Pre-filter and calculate in runner function**
   - Load $1B+ tickers before processing batches
   - Calculate ATR once, reuse for all batches
   - Pass to screener as parameter

2. **Calculate per batch**
   - Load $1B+ universe for each batch
   - Inefficient but simpler

**Recommended**: Option 1 - Pre-calculate once per timeframe

#### 3. Ticker Metadata (Market Cap, Exchange)

**What It Is**:
- Market cap for filtering ≥ $1B
- Exchange for display/filtering

**Current Status**:
- ✅ **EXISTS**: `ticker_universe_all.csv` has market cap and exchange
- Already used by other screeners (RTI, GUPPY, etc.)

**Implementation**: ✅ Load once, pass to all batches (standard pattern)

---

## IMPLEMENTATION STRATEGY

### Two-Phase Approach

Given the data dependency complexity, recommend **two-phase implementation**:

### **PHASE A: Basic Streaming Integration** (Estimated: 2-3 hours)
*Implement streaming framework WITHOUT RS/ATR universe data*

**What to Implement**:
- Configuration setup (timeframe flags, additional parameters)
- Streaming processor class
- Main runner function
- Main.py integration

**Limitations**:
- RS filter will be **skipped** (screener has fallback logic)
- ATR RS filter will use **estimation** (existing fallback)
- Only 3 of 5 filters active:
  - ✅ Market cap ≥ $1B
  - ⚠️ RS ≥ 97 (SKIPPED - no data)
  - ✅ MA alignment (Price ≥ EMA10 ≥ SMA20 ≥ SMA50...)
  - ⚠️ ATR RS ≥ 50 (ESTIMATED - imprecise)
  - ✅ Range position ≥ 50%

**Value**:
- Screener functional for 3/5 criteria
- Framework ready for full data integration
- Can test and validate streaming mechanics
- Partial results better than no results

### **PHASE B: Full Data Integration** (Estimated: 6-8 hours)
*Add RS calculation pipeline and ATR universe pipeline*

**What to Implement**:
1. **RS Calculation Module**
   - Calculate RS scores for all tickers
   - Multiple lookback periods (7d, 21d, 63d, 126d)
   - Save to disk for screener loading
   - Integrate into main calculation phase

2. **ATR Universe Pipeline**
   - Filter $1B+ tickers from universe
   - Calculate ATR for universe
   - Pass to screener for ranking

3. **Update Screener Integration**
   - Load RS data in runner function
   - Load ATR universe in runner function
   - Pass to screener for full filtering

**Value**:
- All 5 filters active (100% Qullamaggie criteria)
- Accurate RS rankings
- Accurate ATR RS percentiles
- Production-ready screener

---

## RECOMMENDED IMPLEMENTATION PLAN

### PHASE A: Basic Streaming Integration

#### Phase A.1: Configuration Setup ⏱️ 15 minutes

**1.1 Add Timeframe Flags to `user_data.csv`**

Location: After line 590 (after `QULLAMAGGIE_SUITE_extension_danger`)

```csv
QULLAMAGGIE_SUITE_daily_enable,TRUE,Enable Qullamaggie screening for daily timeframe
QULLAMAGGIE_SUITE_weekly_enable,FALSE,Enable Qullamaggie screening for weekly timeframe
QULLAMAGGIE_SUITE_monthly_enable,FALSE,Enable Qullamaggie screening for monthly timeframe
```

**1.2 Add Missing Parameters to `user_data.csv`**

Location: After new timeframe flags

```csv
QULLAMAGGIE_SUITE_min_market_cap,1000000000,Minimum market cap requirement (default: $1B)
QULLAMAGGIE_SUITE_min_price,5,Minimum stock price requirement
QULLAMAGGIE_SUITE_save_individual_files,FALSE,Save individual signal type files (currently not used)
```

**1.3 Add Dataclass Fields to `user_defined_data.py`**

Location: Find the UserConfiguration dataclass, add after qullamaggie_suite fields:

```python
# Qullamaggie Suite Screener Configuration
qullamaggie_suite_enable: bool = False
qullamaggie_suite_daily_enable: bool = True
qullamaggie_suite_weekly_enable: bool = False
qullamaggie_suite_monthly_enable: bool = False
qullamaggie_suite_rs_threshold: float = 97.0
qullamaggie_suite_atr_rs_threshold: float = 50.0
qullamaggie_suite_range_position_threshold: float = 0.5
qullamaggie_suite_extension_warning: float = 7.0
qullamaggie_suite_extension_danger: float = 11.0
qullamaggie_suite_min_market_cap: float = 1_000_000_000
qullamaggie_suite_min_price: float = 5.0
qullamaggie_suite_save_individual_files: bool = False
```

**1.4 Add CSV Mappings to `user_defined_data.py`**

Location: Find CONFIG_MAPPING dictionary, add:

```python
# Qullamaggie Suite Screener
'QULLAMAGGIE_SUITE_enable': ('qullamaggie_suite_enable', parse_boolean),
'QULLAMAGGIE_SUITE_daily_enable': ('qullamaggie_suite_daily_enable', parse_boolean),
'QULLAMAGGIE_SUITE_weekly_enable': ('qullamaggie_suite_weekly_enable', parse_boolean),
'QULLAMAGGIE_SUITE_monthly_enable': ('qullamaggie_suite_monthly_enable', parse_boolean),
'QULLAMAGGIE_SUITE_rs_threshold': ('qullamaggie_suite_rs_threshold', float),
'QULLAMAGGIE_SUITE_atr_rs_threshold': ('qullamaggie_suite_atr_rs_threshold', float),
'QULLAMAGGIE_SUITE_range_position_threshold': ('qullamaggie_suite_range_position_threshold', float),
'QULLAMAGGIE_SUITE_extension_warning': ('qullamaggie_suite_extension_warning', float),
'QULLAMAGGIE_SUITE_extension_danger': ('qullamaggie_suite_extension_danger', float),
'QULLAMAGGIE_SUITE_min_market_cap': ('qullamaggie_suite_min_market_cap', float),
'QULLAMAGGIE_SUITE_min_price': ('qullamaggie_suite_min_price', float),
'QULLAMAGGIE_SUITE_save_individual_files': ('qullamaggie_suite_save_individual_files', parse_boolean),
```

**1.5 Add Parameter Helper Function**

Location: End of `user_defined_data.py`

```python
def get_qullamaggie_suite_params_for_timeframe(user_config: UserConfiguration, timeframe: str) -> Optional[Dict[str, Any]]:
    """
    Get Qullamaggie Suite screener parameters with hierarchical flag checking.

    Args:
        user_config: User configuration object
        timeframe: Timeframe to check ('daily', 'weekly', 'monthly')

    Returns:
        Dict with parameters if enabled, None if disabled
    """
    # Check master flag
    if not getattr(user_config, "qullamaggie_suite_enable", False):
        return None

    # Check timeframe flag
    timeframe_flag = f"qullamaggie_suite_{timeframe}_enable"
    if not getattr(user_config, timeframe_flag, True):
        return None

    return {
        'enable_qullamaggie_suite': True,
        'timeframe': timeframe,
        'qullamaggie_suite': {
            'rs_threshold': getattr(user_config, 'qullamaggie_suite_rs_threshold', 97.0),
            'atr_rs_threshold': getattr(user_config, 'qullamaggie_suite_atr_rs_threshold', 50.0),
            'range_position_threshold': getattr(user_config, 'qullamaggie_suite_range_position_threshold', 0.5),
            'extension_warning': getattr(user_config, 'qullamaggie_suite_extension_warning', 7.0),
            'extension_danger': getattr(user_config, 'qullamaggie_suite_extension_danger', 11.0),
            'min_market_cap': getattr(user_config, 'qullamaggie_suite_min_market_cap', 1_000_000_000),
            'min_price': getattr(user_config, 'qullamaggie_suite_min_price', 5.0),
            'save_individual_files': getattr(user_config, 'qullamaggie_suite_save_individual_files', False),
        },
        'qullamaggie_output_dir': getattr(user_config, 'qullamaggie_output_dir', 'results/screeners/qullamaggie')
    }
```

#### Phase A.2: Move Core Screener ⏱️ 5 minutes

**2.1 Move `qullamaggie_suite.py` to new directory**

```bash
mv /home/imagda/_invest2024/python/metaData_v1/src/screeners/qullamaggie_suite.py \
   /home/imagda/_invest2024/python/metaData_v1/src/screeners/quallamagie/quallamagie_screener.py
```

**Note**: Update any imports if needed

#### Phase A.3: Streaming Processor ⏱️ 45 minutes

**3.1 Create QullamaggieStreamingProcessor Class**

Location: `src/screeners_streaming.py`, insert after ADL Enhanced, before final runner functions

Template follows GUPPY/RTI pattern but **simpler** (no signal type separation):

```python
class QullamaggieStreamingProcessor(StreamingCalculationBase):
    """
    Streaming processor for Qullamaggie Suite screener.

    Implements Kristjan Kullamägi's momentum screening methodology:
    - High RS (≥97 on 1w/1m/3m/6m)
    - Perfect MA alignment
    - ATR RS ≥50 vs $1B+ universe
    - Price in upper 50% of range
    - Market cap ≥$1B

    Note: PHASE A implementation operates with limited data:
    - RS data: OPTIONAL (skipped if not available)
    - ATR universe: OPTIONAL (estimated if not available)
    - Ticker info: REQUIRED (market cap, exchange)
    """

    def __init__(self, config, user_config):
        """Initialize Qullamaggie streaming processor"""
        super().__init__(config, user_config)

        # Create output directory
        self.qulla_dir = config.directories['RESULTS_DIR'] / 'screeners' / 'qullamaggie'
        self.qulla_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Qullamaggie screener with configuration
        qulla_config = {
            'timeframe': 'daily',  # Will be overridden per timeframe
            'enable_qullamaggie_suite': True,
            'qullamaggie_suite': {
                'rs_threshold': getattr(user_config, 'qullamaggie_suite_rs_threshold', 97.0),
                'atr_rs_threshold': getattr(user_config, 'qullamaggie_suite_atr_rs_threshold', 50.0),
                'range_position_threshold': getattr(user_config, 'qullamaggie_suite_range_position_threshold', 0.5),
                'extension_warning': getattr(user_config, 'qullamaggie_suite_extension_warning', 7.0),
                'extension_danger': getattr(user_config, 'qullamaggie_suite_extension_danger', 11.0),
                'min_market_cap': getattr(user_config, 'qullamaggie_suite_min_market_cap', 1_000_000_000),
                'min_price': getattr(user_config, 'qullamaggie_suite_min_price', 5.0),
                'save_individual_files': getattr(user_config, 'qullamaggie_suite_save_individual_files', False),
            },
            'qullamaggie_output_dir': str(self.qulla_dir)
        }

        from src.screeners.quallamagie.quallamagie_screener import QullamaggieScreener
        self.qulla_screener = QullamaggieScreener(qulla_config)

        logger.info(f"Qullamaggie streaming processor initialized, output dir: {self.qulla_dir}")
        logger.warning("⚠️ PHASE A: Running with limited data (RS and ATR universe may be estimated)")

    def get_calculation_name(self) -> str:
        """Get the name of this calculation for file naming"""
        return "qullamaggie"

    def get_output_directory(self) -> Path:
        """Get the output directory for this calculation"""
        return self.qulla_dir

    def calculate_single_ticker(self, df: pd.DataFrame, ticker: str, timeframe: str) -> Dict[str, Any]:
        """
        Required by StreamingCalculationBase abstract class.
        Not used since Qullamaggie processes batches.
        """
        return None

    def process_batch_streaming(self, batch_data: Dict[str, pd.DataFrame],
                              timeframe: str,
                              ticker_info: Optional[pd.DataFrame] = None,
                              rs_data: Optional[Dict] = None,
                              atr_universe_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process Qullamaggie batch using memory-efficient streaming pattern.

        Args:
            batch_data: Dictionary of ticker -> DataFrame
            timeframe: Processing timeframe
            ticker_info: DataFrame with market cap and exchange data (REQUIRED)
            rs_data: Optional RS data for enhanced screening (PHASE B)
            atr_universe_data: Optional ATR universe for accurate ranking (PHASE B)

        Returns:
            Dictionary with processing results
        """
        if not batch_data:
            logger.warning(f"No batch data provided for {timeframe} Qullamaggie streaming")
            return {}

        logger.debug(f"Processing Qullamaggie batch for {timeframe}: {len(batch_data)} tickers")

        # Get Qullamaggie parameters for this timeframe
        try:
            qulla_params = get_qullamaggie_suite_params_for_timeframe(self.user_config, timeframe)
            if not qulla_params or not qulla_params.get('enable_qullamaggie_suite'):
                logger.debug(f"Qullamaggie disabled for {timeframe}")
                return {}
        except Exception as e:
            logger.error(f"Failed to get Qullamaggie parameters for {timeframe}: {e}")
            return {}

        # Initialize result containers
        all_results = []
        current_date = self.extract_date_from_batch_data(batch_data)
        processed_tickers = 0

        try:
            # Update screener configuration for this timeframe
            self.qulla_screener.config['timeframe'] = timeframe
            self.qulla_screener.timeframe = timeframe

            # Warn if optional data not provided
            if rs_data is None:
                logger.warning(f"⚠️ No RS data provided - RS filter will be SKIPPED")
            if atr_universe_data is None:
                logger.warning(f"⚠️ No ATR universe data provided - ATR RS will be ESTIMATED")
            if ticker_info is None:
                logger.error(f"❌ No ticker_info provided - market cap filter DISABLED")

            # Run Qullamaggie screening for entire batch
            batch_results = self.qulla_screener.run_qullamaggie_screening(
                batch_data,
                ticker_info=ticker_info,
                rs_data=rs_data,
                atr_universe_data=atr_universe_data,
                batch_info={'timeframe': timeframe}
            )

            if batch_results:
                all_results.extend(batch_results)
                processed_tickers = len(batch_results)

            # Memory cleanup after batch processing
            gc.collect()

            # Write consolidated results immediately (with append mode!)
            output_files = []
            if all_results:
                consolidated_filename = f"qullamaggie_consolidated_{timeframe}_{current_date}.csv"
                consolidated_file = self.qulla_dir / consolidated_filename
                self._write_results_to_csv(consolidated_file, all_results)
                output_files.append(str(consolidated_file))
                logger.info(f"Qullamaggie consolidated: {len(all_results)} results saved to {consolidated_file}")

            # Memory cleanup
            self.cleanup_memory(all_results, batch_data)

        except Exception as e:
            logger.error(f"Error in Qullamaggie batch processing: {e}")

        logger.info(f"Qullamaggie batch summary ({timeframe}): {processed_tickers} tickers passed screening, "
                   f"{len(all_results)} total results")

        return {
            "tickers_processed": processed_tickers,
            "total_results": len(all_results),
            "output_files": output_files
        }

    def _write_results_to_csv(self, output_file: Path, results: List[Dict]):
        """Write Qullamaggie results to CSV with memory optimization and append mode for batches"""
        if not results:
            return

        try:
            # Convert to DataFrame with optimized dtypes
            df = pd.DataFrame(results)
            df = self.optimize_dataframe_dtypes(df)

            # CRITICAL: Write to CSV with append mode for streaming batches
            if output_file.exists():
                df.to_csv(output_file, mode='a', header=False, index=False)
            else:
                df.to_csv(output_file, index=False)

            # Memory cleanup
            del df
            gc.collect()

        except Exception as e:
            logger.error(f"Error writing Qullamaggie results to {output_file}: {e}")
```

#### Phase A.4: Main Runner Function ⏱️ 30 minutes

**4.1 Add Runner Function**

Location: After QullamaggieStreamingProcessor class in `src/screeners_streaming.py`

```python
def run_all_qullamaggie_streaming(config, user_config, timeframes: List[str], clean_file_path: str) -> Dict[str, int]:
    """
    Run Qullamaggie Suite screener using streaming processing with hierarchical flag validation.

    PHASE A Implementation Notes:
    - RS data: NOT YET IMPLEMENTED (filter skipped)
    - ATR universe: NOT YET IMPLEMENTED (estimation used)
    - Ticker info: Loaded from ticker_universe_all.csv
    - Results may be less accurate until PHASE B data pipelines added

    Args:
        config: System configuration
        user_config: User configuration
        timeframes: List of timeframes to process
        clean_file_path: Path to ticker list file

    Returns:
        Dictionary with timeframe results
    """
    # Check master flag first
    if not getattr(user_config, "qullamaggie_suite_enable", False):
        print(f"\n⏭️  Qullamaggie Screener disabled - skipping processing")
        logger.info("Qullamaggie Screener disabled (master flag)")
        return {}

    # Check if any timeframe is enabled
    enabled_timeframes = []
    for timeframe in timeframes:
        if getattr(user_config, f"qullamaggie_suite_{timeframe}_enable", False):
            enabled_timeframes.append(timeframe)

    if not enabled_timeframes:
        print(f"\n⚠️  Qullamaggie master enabled but all timeframes disabled - skipping processing")
        logger.warning("Qullamaggie master enabled but all timeframes disabled")
        return {}

    print(f"\n🎯 QULLAMAGGIE SCREENER - Processing timeframes: {', '.join(enabled_timeframes)}")
    print(f"⚠️  PHASE A: Running with limited data (RS filter skipped, ATR RS estimated)")
    logger.info(f"Qullamaggie enabled for: {', '.join(enabled_timeframes)}")

    # Initialize processor
    processor = QullamaggieStreamingProcessor(config, user_config)
    results = {}

    # Load ticker_info once for all timeframes (market cap + exchange)
    ticker_universe_all_path = config.base_dir / 'results' / 'ticker_universes' / 'ticker_universe_all.csv'
    ticker_info = None
    if ticker_universe_all_path.exists():
        try:
            ticker_info = pd.read_csv(ticker_universe_all_path, usecols=['ticker', 'exchange', 'market_cap'])
            logger.info(f"Loaded ticker info for {len(ticker_info)} tickers")
        except Exception as e:
            logger.error(f"Could not load ticker_universe_all.csv: {e}")
            logger.warning("⚠️ Market cap filter will be DISABLED")
    else:
        logger.warning(f"⚠️ ticker_universe_all.csv not found - market cap filter DISABLED")

    # PHASE A: RS data and ATR universe NOT YET IMPLEMENTED
    rs_data = None  # TODO: PHASE B - implement RS calculation pipeline
    atr_universe_data = None  # TODO: PHASE B - implement ATR universe pipeline

    # Process each enabled timeframe
    for timeframe in enabled_timeframes:
        qulla_enabled = getattr(user_config, f'qullamaggie_suite_{timeframe}_enable', False)
        if not qulla_enabled:
            print(f"⏭️  Qullamaggie disabled for {timeframe} timeframe")
            continue

        print(f"\n🎯 Processing Qullamaggie {timeframe.upper()} timeframe...")
        logger.info(f"Starting Qullamaggie for {timeframe} timeframe...")

        # Initialize DataReader for this timeframe
        batch_size = getattr(user_config, 'batch_size', 100)
        from src.data_reader import DataReader
        data_reader = DataReader(config, timeframe, batch_size)

        # Load tickers from file
        data_reader.load_tickers_from_file(clean_file_path)

        # Get ticker list for batch processing
        import pandas as pd
        tickers_df = pd.read_csv(clean_file_path)
        ticker_list = tickers_df['ticker'].tolist()

        # Process all batches with streaming approach
        total_tickers = len(ticker_list)
        import math
        total_batches = math.ceil(total_tickers / batch_size)

        print(f"📦 Processing {total_tickers} tickers in {total_batches} batches of {batch_size}")

        total_results = 0
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, total_tickers)
            batch_tickers = ticker_list[start_idx:end_idx]

            print(f"🔄 Loading batch {batch_num + 1}/{total_batches} ({len(batch_tickers)} tickers) - {((batch_num+1)/total_batches)*100:.1f}%")

            # Get batch data using DataReader
            batch_data = data_reader.read_batch_data(batch_tickers, validate=True)

            if batch_data:
                print(f"✅ Loaded {len(batch_data)} valid tickers from batch {batch_num + 1}")

                # Process batch using Qullamaggie screener
                batch_result = processor.process_batch_streaming(
                    batch_data,
                    timeframe,
                    ticker_info=ticker_info,
                    rs_data=rs_data,  # None in PHASE A
                    atr_universe_data=atr_universe_data  # None in PHASE A
                )
                if batch_result and "total_results" in batch_result:
                    total_results += batch_result["total_results"]
                    logger.info(f"Qullamaggie batch {batch_num + 1} completed: {batch_result['total_results']} results")
            else:
                print(f"⚠️  No valid data in batch {batch_num + 1}")

        results[timeframe] = total_results
        print(f"✅ Qullamaggie completed for {timeframe}: {total_results} stocks passed screening")
        logger.info(f"Qullamaggie completed for {timeframe}: {total_results} total results")

    if results:
        print(f"✅ QULLAMAGGIE SCREENER COMPLETED!")
        print(f"📊 Total results: {sum(results.values())}")
        print(f"🕒 Timeframes processed: {', '.join(results.keys())}")
        print(f"⚠️  NOTE: PHASE A results use limited data (full accuracy requires PHASE B)")
    else:
        print(f"⚠️  Qullamaggie completed with no results")

    return results
```

#### Phase A.5: Main.py Integration ⏱️ 10 minutes

**5.1 Add Import**

Location: ~line 1892, with other screener imports

```python
from src.screeners_streaming import (..., run_all_qullamaggie_streaming, ...)
```

**5.2 Add Execution Block**

Location: After RTI screener (around line 1905), before ADL Enhanced

```python
        # 11. Qullamaggie Suite Screener - All timeframes (PHASE A - limited data)
        try:
            qullamaggie_results = run_all_qullamaggie_streaming(config, user_config, timeframes_to_process, clean_file)
            logger.info(f"Qullamaggie Screener completed")
        except Exception as e:
            logger.error(f"Error running Qullamaggie Screener: {e}")
            qullamaggie_results = {}
```

**5.3 Add Else Block Initialization**

Location: In else block for skipped SCREENERS phase (~line 1925)

```python
        qullamaggie_results = {}
```

**5.4 Add to Results Summary**

Location: Results summary section (~line 1952)

```python
        print(f"🎯 Qullamaggie Screener: {sum(qullamaggie_results.values())} total results (⚠️ PHASE A - limited data)")
```

---

## PHASE A TESTING PLAN

### Test 1: Configuration Loading
```bash
python -c "from src.user_defined_data import load_user_configuration; uc = load_user_configuration('user_data.csv'); print(f'Qullamaggie enabled: {uc.qullamaggie_suite_enable}')"
```

### Test 2: Small Batch Test
```bash
# Enable Qullamaggie in user_data.csv first
# Set batch_size to 10 for quick test
python main.py
```

### Test 3: Verify Output Files
```bash
ls -lh results/screeners/qullamaggie/
# Should see: qullamaggie_consolidated_daily_YYYYMMDD.csv
```

### Test 4: Check Results Quality
- Open CSV file
- Verify all expected columns present
- Check ATR extension sorting (highest = most extended)
- Look for warning/danger formatting flags

### Test 5: Verify Append Mode
- Run again with same settings
- Should APPEND to existing file, not overwrite
- Row count should increase

---

## PHASE A EXPECTED LIMITATIONS

**What Works (3 of 5 filters)**:
- ✅ Market cap ≥ $1B filter
- ✅ MA alignment (Price ≥ EMA10 ≥ SMA20 ≥ SMA50...)
- ✅ Range position ≥ 50% filter

**What's Limited (2 of 5 filters)**:
- ⚠️ **RS ≥ 97 filter**: SKIPPED (no RS data)
  - Screener has fallback logic to skip this filter
  - Results will include stocks that might fail RS requirement

- ⚠️ **ATR RS ≥ 50 filter**: ESTIMATED (no universe data)
  - Screener uses estimation based on historical volatility
  - Less accurate than true percentile ranking
  - May include/exclude stocks incorrectly

**Impact**:
- Expect **~5-10x more results** than full implementation
- Results still valuable (MA alignment is strong filter)
- Can test streaming mechanics and validate framework
- Easy to upgrade to PHASE B later

---

## PHASE B: FULL DATA INTEGRATION (FUTURE)

### Overview

Phase B adds the missing RS calculation and ATR universe pipelines for 100% Qullamaggie criteria compliance.

### B.1: RS Calculation Pipeline ⏱️ 4-5 hours

**What to Build**:
1. RS calculation module (`src/calculations/relative_strength.py`)
2. Calculate RS for all tickers vs universe
3. Multiple lookback periods (7d, 21d, 63d, 126d)
4. Save to disk: `results/relative_strength/rs_daily_YYYYMMDD.csv`
5. Integrate into main calculation phase

**Dependencies**:
- Need all ticker data loaded simultaneously
- Memory-intensive operation
- Should run ONCE per day, save results

**Integration Point**:
- Add to BASIC_CALCULATIONS phase in main.py
- Load RS data in Qullamaggie runner function
- Pass to screener

### B.2: ATR Universe Pipeline ⏱️ 2-3 hours

**What to Build**:
1. Filter ticker_universe_all for market_cap ≥ $1B
2. Load data for all $1B+ tickers
3. Calculate 14-period ATR for each
4. Pass dictionary to screener

**Integration Point**:
- Calculate in Qullamaggie runner function (before batch loop)
- Reuse for all batches in timeframe

### B.3: Update Screener Integration ⏱️ 1 hour

**Changes**:
- Remove warning messages about limited data
- Load RS data from saved files
- Build ATR universe before batch processing
- Update documentation to remove PHASE A limitations

---

## ESTIMATED COMPLETION TIMES

### Phase A (Recommended for Initial Implementation)
- **A.1** Configuration Setup: 15 minutes
- **A.2** Move Core Screener: 5 minutes
- **A.3** Streaming Processor: 45 minutes
- **A.4** Main Runner Function: 30 minutes
- **A.5** Main.py Integration: 10 minutes
- **Testing**: 15 minutes
- **Documentation**: 10 minutes
- **TOTAL PHASE A**: ~2 hours

### Phase B (Future Enhancement)
- **B.1** RS Calculation Pipeline: 4-5 hours
- **B.2** ATR Universe Pipeline: 2-3 hours
- **B.3** Integration Updates: 1 hour
- **Testing**: 1 hour
- **TOTAL PHASE B**: ~6-8 hours

### **GRAND TOTAL**: ~10 hours (2 hours Phase A + 8 hours Phase B)

---

## DEPENDENCIES & RISKS

### Phase A Dependencies
- ✅ Core screener exists (`qullamaggie_suite.py`)
- ✅ Market cap data available (`ticker_universe_all.csv`)
- ✅ OHLCV data available (standard pipeline)
- ✅ Streaming framework established (GUPPY, RTI, etc.)

### Phase A Risks
- **LOW**: All dependencies met, follows established patterns
- **Limitation**: Results less accurate without RS/ATR universe data
- **Mitigation**: Clearly document PHASE A limitations

### Phase B Dependencies
- ⚠️ RS calculation algorithm (need to research/design)
- ⚠️ Memory capacity for loading all ticker data
- ⚠️ Computational time for RS calculations
- ✅ ATR calculation exists (used in other modules)

### Phase B Risks
- **MEDIUM**: RS calculation is complex and memory-intensive
- **HIGH**: May require significant optimization for 5000+ stocks
- **Mitigation**:
  - Pre-calculate and save to disk
  - Run during off-hours if needed
  - Consider chunked processing if memory issues

---

## RECOMMENDATION

### **START WITH PHASE A**

**Why**:
1. ✅ **Quick Implementation**: 2 hours vs 10 hours total
2. ✅ **Low Risk**: Follows established patterns, all dependencies met
3. ✅ **Immediate Value**: 60% of filters working (MA alignment is very strong)
4. ✅ **Framework Ready**: Easy to upgrade to PHASE B later
5. ✅ **No Blockers**: Can implement today without research/design work

**Value Proposition**:
- Get Qullamaggie screener operational quickly
- Test streaming integration and validate results
- Identify any issues before investing 8+ hours in Phase B
- Provide partial results to users immediately
- Phase B can be scheduled as separate project

### **THEN ADD PHASE B**

**When**:
- After Phase A tested and validated
- When time allows for RS pipeline research
- If users request more accurate results
- As planned enhancement (not blocking)

---

## FILES TO CREATE/MODIFY

### Phase A

**New Files**:
- None (screener already exists)

**Modified Files**:
1. `user_data.csv` - Add timeframe flags and parameters
2. `src/user_defined_data.py` - Add dataclass fields, mappings, helper function
3. `src/screeners_streaming.py` - Add QullamaggieStreamingProcessor class and runner
4. `main.py` - Add imports, execution block, results summary
5. `src/screeners/quallamagie/quallamagie_screener.py` - Move from old location

### Phase B

**New Files**:
1. `src/calculations/relative_strength.py` - RS calculation module
2. `results/relative_strength/` - RS data storage directory

**Modified Files**:
1. `src/screeners_streaming.py` - Update runner to load RS data and build ATR universe
2. `main.py` - Add RS calculation to BASIC_CALCULATIONS phase

---

## NEXT STEPS

1. ✅ **COMPLETED**: Research Qullamaggie methodology
2. ✅ **COMPLETED**: Analyze existing screener implementation
3. ✅ **COMPLETED**: Create comprehensive implementation plan
4. ⏳ **PENDING**: User approval to proceed with Phase A
5. ⏳ **PENDING**: Implementation of Phase A (estimated 2 hours)
6. ⏳ **PENDING**: Testing and validation
7. ⏳ **FUTURE**: Schedule Phase B for full data integration

---

**Document Status**: 📋 PLANNING COMPLETE - AWAITING APPROVAL
**Recommended Action**: Proceed with Phase A implementation
**Expected Delivery**: Phase A can be completed in single 2-hour session
