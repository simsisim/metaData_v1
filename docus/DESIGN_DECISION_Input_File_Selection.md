# Input File Selection Strategy - Design Decision

**Date**: 2025-09-30
**Author**: Claude Code
**Decision Required**: How to select input CSV files for TradingView watchlist export

---

## PROBLEM STATEMENT

The TradingView watchlist export needs to read PVB screener CSV files. Two main approaches:

1. **Approach A: Use files from current run** (saved filenames during execution)
2. **Approach B: Find latest files by date/time** (file discovery at export time)

---

## APPROACH A: SAVED FILENAMES (CURRENT RUN)

### How It Works

```python
# In main.py - during PVB screener execution
pvb_output_files = []  # Track files created this run

for timeframe in timeframes:
    # ... process batch ...
    result = processor.process_timeframe_streaming(...)

    if result and 'output_file' in result:
        pvb_output_files.append(result['output_file'])  # Save filename

# After all timeframes complete
if pvb_output_files:
    exporter.export_pvb_screener(
        csv_files=pvb_output_files,  # Use saved filenames
        output_filename=f'pvb_watchlist_{ticker_choice}_{date}.txt'
    )
```

### Files Used
```
pvb_screener_2_daily_20250930.csv      ← Created this run
pvb_screener_2_weekly_20250930.csv     ← Created this run
pvb_screener_2_monthly_20250930.csv    ← Created this run
```

### Pros ✅
1. **Guaranteed Consistency**: Exports exactly what was just calculated
2. **No File Discovery Logic**: Simple, no glob patterns or date parsing
3. **Fast**: No filesystem search required
4. **Predictable**: Always uses current run's data
5. **Clean Integration**: Natural flow in pipeline

### Cons ❌
1. **Tightly Coupled**: Watchlist export only works during screener run
2. **Not Standalone**: Can't generate watchlist from existing files
3. **No Manual Control**: User can't choose which files to use
4. **Requires Screener Run**: Even if data hasn't changed

### Use Cases
- ✅ Automatic export after every screener run
- ❌ Manual watchlist regeneration from old results
- ❌ Export historical data without re-running screener
- ❌ Combine multiple dates (e.g., last 7 days)

---

## APPROACH B: LATEST FILES (FILE DISCOVERY)

### How It Works

```python
# In main.py OR standalone script
def find_latest_pvb_files(screener_dir, ticker_choice, date=None):
    """
    Find latest PVB screener files.

    Args:
        screener_dir: Directory containing PVB CSV files
        ticker_choice: User's ticker choice (e.g., '2')
        date: Optional date (YYYYMMDD), defaults to today

    Returns:
        List of matching CSV file paths
    """
    if date is None:
        date = datetime.now().strftime('%Y%m%d')

    # Strategy 1: Match exact date
    pattern = f'pvb_screener_{ticker_choice}_*_{date}.csv'
    files = list(Path(screener_dir).glob(pattern))

    if not files:
        # Strategy 2: Find most recent files if today's not found
        pattern = f'pvb_screener_{ticker_choice}_*.csv'
        all_files = list(Path(screener_dir).glob(pattern))

        if all_files:
            # Sort by modification time, take most recent
            files = sorted(all_files, key=lambda x: x.stat().st_mtime, reverse=True)

            # Group by date, take latest date's files
            latest_date = extract_date_from_filename(files[0].name)
            files = [f for f in files if extract_date_from_filename(f.name) == latest_date]

    return files

# Usage
csv_files = find_latest_pvb_files(
    screener_dir=config.directories['PVB_SCREENER_DIR'],
    ticker_choice=user_config.ticker_choice,
    date=config.current_date  # Or None for auto-detection
)

if csv_files:
    exporter.export_pvb_screener(csv_files, output_filename)
```

### File Discovery Logic

**Step 1**: Try exact date match
```python
# Look for: pvb_screener_2_daily_20250930.csv
#           pvb_screener_2_weekly_20250930.csv
pattern = f'pvb_screener_{ticker_choice}_*_{date}.csv'
```

**Step 2**: If not found, use latest available
```python
# Find all: pvb_screener_2_*.csv
# Sort by: file modification time
# Take:    most recent date's files
```

### Files Used (Example Scenarios)

**Scenario 1**: Current date files exist
```
pvb_screener_2_daily_20250930.csv      ← Latest, matches date
pvb_screener_2_weekly_20250930.csv     ← Latest, matches date
→ Uses both files
```

**Scenario 2**: Current date missing, older files exist
```
pvb_screener_2_daily_20250929.csv      ← Most recent available
pvb_screener_2_daily_20250928.csv      ← Older
pvb_screener_2_weekly_20250929.csv     ← Most recent available
→ Uses 20250929 files (latest date)
```

**Scenario 3**: Mixed dates (edge case)
```
pvb_screener_2_daily_20250930.csv      ← Newest daily
pvb_screener_2_weekly_20250929.csv     ← Older weekly
→ Problem: Inconsistent dates!
→ Solution: Take latest date group (20250930) → only daily
```

### Pros ✅
1. **Standalone Capability**: Can run export independently
2. **Manual Regeneration**: User can re-export without re-running screener
3. **Flexible**: Can specify date or use latest
4. **Historical Support**: Can export old results
5. **User Control**: Choose which date to export

### Cons ❌
1. **File Discovery Complexity**: Need robust glob/search logic
2. **Date Parsing**: Extract date from filename (error-prone)
3. **Ambiguity**: Which files if multiple dates exist?
4. **Slower**: Filesystem search overhead
5. **Error Cases**: Missing files, date mismatches

### Use Cases
- ✅ Automatic export after every screener run
- ✅ Manual watchlist regeneration from old results
- ✅ Export historical data without re-running screener
- ✅ Standalone script: `export_watchlist.py --date 20250929`
- ⚠️ Combine multiple dates (requires additional logic)

---

## HYBRID APPROACH (RECOMMENDED ✅)

### Strategy: Best of Both Worlds

**Default Mode (Integrated)**: Use saved filenames from current run
**Standalone Mode (Optional)**: Use file discovery for manual export

### Implementation

```python
def export_pvb_watchlist(
    config,
    user_config,
    csv_files: Optional[List[Path]] = None,
    date: Optional[str] = None
):
    """
    Export PVB screener results to TradingView watchlist.

    Mode 1 (Integrated): Pass csv_files from current run
    Mode 2 (Standalone): Pass date, will find latest files

    Args:
        config: Configuration object
        user_config: User configuration object
        csv_files: Optional list of CSV files (Mode 1)
        date: Optional date string YYYYMMDD (Mode 2)

    Returns:
        Path to created watchlist file(s)
    """
    exporter = TradingViewWatchlistExporter(
        output_dir=config.directories['PVB_SCREENER_DIR']
    )

    # MODE 1: Use provided files (integrated with screener run)
    if csv_files:
        logger.info(f"Using provided CSV files: {len(csv_files)} files")
        files_to_process = csv_files

    # MODE 2: Discover files (standalone mode)
    else:
        logger.info("No CSV files provided, discovering latest files")
        files_to_process = find_latest_pvb_files(
            screener_dir=config.directories['PVB_SCREENER_DIR'],
            ticker_choice=user_config.ticker_choice,
            date=date or config.current_date
        )

        if not files_to_process:
            logger.warning("No PVB screener files found for export")
            return None

        logger.info(f"Found {len(files_to_process)} PVB files to export")

    # Export watchlist
    output_filename = f'pvb_watchlist_{user_config.ticker_choice}_{date or config.current_date}.txt'
    return exporter.export_pvb_screener(files_to_process, output_filename)
```

### Usage Patterns

**Pattern 1: Integrated (During Screener Run)**
```python
# In main.py after PVB screener completes
if pvb_enabled and user_config.pvb_TWmodel_export_tradingview:
    pvb_output_files = results_summary.get('output_files', [])

    export_pvb_watchlist(
        config,
        user_config,
        csv_files=pvb_output_files  # Pass saved filenames
    )
```

**Pattern 2: Standalone (Manual Export)**
```python
# New script: export_watchlist_standalone.py
from src.config import Config
from src.user_defined_data import read_user_data
from src.tw_export_watchlist import export_pvb_watchlist

config = Config()
user_config = read_user_data()

# Export latest available
export_pvb_watchlist(config, user_config)

# Or export specific date
export_pvb_watchlist(config, user_config, date='20250929')
```

**Pattern 3: Command-Line Interface**
```bash
# Export latest
python export_watchlist_standalone.py

# Export specific date
python export_watchlist_standalone.py --date 20250929

# Export specific files
python export_watchlist_standalone.py --files pvb_screener_2_daily_20250930.csv
```

### Pros ✅ (Combines All Benefits)
1. **Integrated by Default**: Works seamlessly in pipeline
2. **Standalone Capable**: Can run independently when needed
3. **Flexible**: Supports both automatic and manual use cases
4. **User Choice**: Let user decide via config or CLI
5. **Clean Fallback**: If files not provided, discovers latest
6. **Best Performance**: Fast when integrated, smart when standalone

### Cons ❌ (Minimal)
1. **Slightly More Complex**: Two code paths to maintain
2. **More Testing**: Need to test both modes

---

## COMPARISON MATRIX

| Feature | Approach A (Saved) | Approach B (Discovery) | Hybrid (Recommended) |
|---------|-------------------|------------------------|---------------------|
| **Integration with screener** | ✅ Perfect | ⚠️ Possible | ✅ Perfect |
| **Standalone capability** | ❌ No | ✅ Yes | ✅ Yes |
| **Historical export** | ❌ No | ✅ Yes | ✅ Yes |
| **Manual regeneration** | ❌ No | ✅ Yes | ✅ Yes |
| **Performance** | ✅ Fast | ⚠️ Slower | ✅ Fast (integrated) |
| **Simplicity** | ✅ Simple | ⚠️ Complex | ⚠️ Medium |
| **Reliability** | ✅ High | ⚠️ Medium | ✅ High |
| **User control** | ❌ Low | ✅ High | ✅ High |
| **Error handling** | ✅ Easy | ⚠️ Tricky | ✅ Manageable |
| **Future extensibility** | ❌ Limited | ✅ Good | ✅ Excellent |

---

## RECOMMENDED DECISION: HYBRID APPROACH

### Implementation Priority

**Phase 1 (MVP)**: Integrated mode only
- Use saved filenames from current run
- Simple, reliable, works immediately
- Covers 90% of use cases

**Phase 2 (Enhancement)**: Add standalone mode
- Implement file discovery logic
- Create standalone script
- Add CLI support

### Rationale

1. **Meets Immediate Need**: Automatic export after screener run
2. **Future-Proof**: Easy to add standalone later
3. **Best User Experience**: Works automatically, can be manual
4. **Maintainable**: Clean separation between modes
5. **Extensible**: Can add advanced features (multi-date, filtering)

### Configuration

Add to `user_data.csv`:
```csv
PVB_TWmodel_export_tradingview,TRUE,Export PVB results to TradingView watchlist format
PVB_TWmodel_watchlist_standalone_mode,FALSE,Enable standalone watchlist export (finds latest files)
PVB_TWmodel_watchlist_auto_date,TRUE,Auto-detect latest date if current not found
```

---

## IMPLEMENTATION PLAN

### Phase 1: Integrated Mode (Week 1)

1. **Modify `main.py`** to track output files
   ```python
   pvb_output_files = []
   for timeframe in timeframes:
       result = processor.process_timeframe_streaming(...)
       if result and 'output_file' in result:
           pvb_output_files.append(Path(result['output_file']))
   ```

2. **Call exporter** after PVB completion
   ```python
   if pvb_output_files and user_config.pvb_TWmodel_export_tradingview:
       export_pvb_watchlist(config, user_config, csv_files=pvb_output_files)
   ```

3. **Update `streaming_base.py`** to return output_file
   ```python
   return {
       'tickers_processed': total_processed,
       'output_file': str(output_file),  # Add this
       'memory_saved_mb': memory_saved
   }
   ```

### Phase 2: Standalone Mode (Week 2)

1. **Implement file discovery** in `tw_export_watchlist.py`
   ```python
   def find_latest_pvb_files(screener_dir, ticker_choice, date=None):
       # Implementation with glob, date extraction, sorting
   ```

2. **Create standalone script**: `scripts/export_watchlist_standalone.py`
   ```python
   if __name__ == '__main__':
       import argparse
       parser = argparse.ArgumentParser()
       parser.add_argument('--date', help='Date YYYYMMDD')
       # ...
   ```

3. **Add mode detection** in main export function
   ```python
   if csv_files:
       # Integrated mode
   else:
       # Standalone mode - discover files
   ```

### Phase 3: Advanced Features (Future)

1. **Multi-date export**: Combine last N days
2. **Filtering**: Min score, signal type
3. **Scheduling**: Cron job for daily export
4. **API Integration**: Auto-upload to TradingView (if API exists)

---

## FILE DISCOVERY ALGORITHM (PHASE 2)

```python
def find_latest_pvb_files(screener_dir: Path, ticker_choice: str, date: Optional[str] = None) -> List[Path]:
    """
    Find latest PVB screener files for watchlist export.

    Args:
        screener_dir: Directory containing PVB CSV files
        ticker_choice: User's ticker choice (e.g., '2', '2-5')
        date: Optional target date (YYYYMMDD). If None, uses today.
              If today's files missing, auto-detects latest.

    Returns:
        List of CSV file paths sorted by timeframe (daily, weekly, monthly)

    Examples:
        # Find today's files
        files = find_latest_pvb_files(dir, '2')

        # Find specific date
        files = find_latest_pvb_files(dir, '2', '20250929')

        # Find latest available (any date)
        files = find_latest_pvb_files(dir, '2', date=None)
    """

    # Step 1: Determine target date
    if date is None:
        date = datetime.now().strftime('%Y%m%d')

    # Step 2: Build search pattern
    # Matches: pvb_screener_2_daily_20250930.csv
    #          pvb_screener_2-5_weekly_20250930.csv
    pattern = f'pvb_screener_{ticker_choice}_*_{date}.csv'

    # Step 3: Search for exact date match
    files = list(screener_dir.glob(pattern))

    if files:
        logger.info(f"Found {len(files)} PVB files for date {date}")
        return sorted(files, key=lambda x: ('daily' in x.name, 'weekly' in x.name, 'monthly' in x.name))

    # Step 4: No files for target date - find latest available
    logger.warning(f"No PVB files found for date {date}, searching for latest")

    # Get all PVB files for this ticker_choice
    pattern_all = f'pvb_screener_{ticker_choice}_*.csv'
    all_files = list(screener_dir.glob(pattern_all))

    if not all_files:
        logger.error(f"No PVB screener files found for ticker_choice {ticker_choice}")
        return []

    # Step 5: Extract dates and find latest
    file_dates = {}
    for file in all_files:
        # Extract date from filename: pvb_screener_2_daily_20250930.csv -> 20250930
        match = re.search(r'_(\d{8})\.csv$', file.name)
        if match:
            file_date = match.group(1)
            if file_date not in file_dates:
                file_dates[file_date] = []
            file_dates[file_date].append(file)

    if not file_dates:
        logger.error("Could not parse dates from PVB filenames")
        return []

    # Step 6: Get files from latest date
    latest_date = max(file_dates.keys())
    latest_files = file_dates[latest_date]

    logger.info(f"Using latest available date: {latest_date} ({len(latest_files)} files)")

    # Step 7: Sort by timeframe
    return sorted(latest_files, key=lambda x: ('daily' in x.name, 'weekly' in x.name, 'monthly' in x.name))
```

---

## ERROR HANDLING STRATEGY

### File Not Found Scenarios

| Scenario | Behavior | User Message |
|----------|----------|--------------|
| **No files for today** | Search for latest | ⚠️ "Using latest available: 2025-09-29" |
| **No files at all** | Skip export | ❌ "No PVB files found - run screener first" |
| **Partial files** (only daily, no weekly) | Use what's available | ℹ️ "Exported 1 timeframe (weekly/monthly missing)" |
| **Mixed dates** | Use latest date group | ⚠️ "Found mixed dates, using latest: 2025-09-30" |
| **Invalid filenames** | Skip invalid | ⚠️ "Skipped 2 files with invalid names" |

---

## TESTING STRATEGY

### Test Cases for File Discovery

```python
# Test 1: Exact date match
files = find_latest_pvb_files(dir, '2', '20250930')
assert len(files) == 3  # daily, weekly, monthly
assert '20250930' in files[0].name

# Test 2: Missing date, fallback to latest
files = find_latest_pvb_files(dir, '2', '20251001')  # Future date
assert len(files) > 0  # Falls back to 20250930

# Test 3: No files at all
files = find_latest_pvb_files(dir, '999', '20250930')  # Invalid choice
assert len(files) == 0

# Test 4: Only daily exists
# Setup: Delete weekly/monthly
files = find_latest_pvb_files(dir, '2', '20250930')
assert len(files) == 1
assert 'daily' in files[0].name

# Test 5: Mixed dates
# Setup: daily_20250930.csv, weekly_20250929.csv
files = find_latest_pvb_files(dir, '2')
assert all('20250930' in f.name for f in files)  # Only latest date

# Test 6: Ticker choice with hyphen
files = find_latest_pvb_files(dir, '2-5', '20250930')
assert len(files) > 0
assert '2-5' in files[0].name
```

---

## DECISION SUMMARY

### ✅ FINAL RECOMMENDATION: HYBRID APPROACH

**Phase 1 Implementation** (Immediate):
- Use saved filenames from current screener run
- Call exporter automatically after PVB completes
- Simple, reliable, covers primary use case

**Phase 2 Enhancement** (Future):
- Add file discovery for standalone mode
- Create CLI script for manual export
- Support historical data export

**Rationale**:
1. Delivers value immediately
2. Future-proof for advanced use cases
3. Maintains simplicity while enabling flexibility
4. Best user experience (automatic + manual options)

**Code Location**:
- Phase 1: Modify `main.py` lines 440-460
- Phase 2: Add `find_latest_pvb_files()` to `tw_export_watchlist.py`

---

**Document Status**: DECISION COMPLETE
**Recommended Approach**: Hybrid (Integrated + Standalone)
**Implementation**: Phase 1 first, Phase 2 optional enhancement