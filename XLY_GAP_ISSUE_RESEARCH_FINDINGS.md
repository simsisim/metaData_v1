# XLY_gap Display Issue - Research Findings

## Problem Statement
XLY_gap is configured in MMM Panel_2 but doesn't display gap analysis data in the chart. Instead, it shows regular OHLCV data.

Configuration shows:
```
mmm_QQQ_Analysis_mmm,line,QQQ + EMA(10) + SMA(50),XLY_gap,,,,,A_PPO(12,26,9)_for_(QQQ),B_RSI(14)_for_(QQQ),,,,
```

## Root Cause Analysis

### 1. Data File Investigation
- ✅ **XLY_gap.csv file EXISTS** at `../downloadData_v1/data/market_data/daily/XLY_gap.csv`
- ✅ **Gap data is present**: 420/421 rows have gap values
- ✅ **File contains gap columns**: `['gap', 'AdjustClose_woGap']` plus standard OHLCV

### 2. Data Reader Behavior
- ✅ **DataReader CAN load XLY_gap.csv**
- ❌ **DataReader STRIPS gap columns**: Only returns `['Open', 'High', 'Low', 'Close', 'Volume']`
- 🔍 **Filtering occurs at src/data_reader.py:169-176**:
  ```python
  standard_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
  available_columns = [col for col in standard_columns if col in df.columns]
  return df[available_columns]
  ```

### 3. SR System Data Flow
1. MMM configuration specifies `XLY_gap` as data source
2. SR system calls `DataReader.read_stock_data('XLY_gap')`
3. DataReader loads `XLY_gap.csv` but filters out gap columns
4. Chart receives OHLCV data instead of gap analysis data
5. Chart displays regular price data instead of gap metrics

### 4. Gap Data Content Verification
**File**: `/home/imagda/_invest2024/python/downloadData_v1/data/market_data/daily/XLY_gap.csv`

Sample gap data:
```
Date,gap,AdjustClose_woGap
2024-01-03,-0.607,108.256
2024-01-04,0.196,109.196
2024-01-05,-0.078,109.049
...
Recent: [-0.23, -1.59, -0.04, 0.28, 0.26]
```

## Impact Assessment

### Current State
- Gap files are generated correctly by MMM module
- Gap data contains valid gap calculations
- SR system loads files but loses gap-specific data
- Charts show misleading OHLCV data instead of gap analysis

### Comparison with Panel/Overview
This is the SAME ISSUE as today's Panel/Overview problems:
- Data files exist with expected names
- SR system can locate files
- Column filtering removes analysis-specific data
- Charts display generic OHLCV instead of intended analysis

## Technical Solutions

### Option 1: DataReader Gap-Aware Mode
Modify `DataReader.read_stock_data()` to preserve gap columns for gap files:
```python
def read_stock_data(self, ticker: str, preserve_analysis_columns: bool = False):
    # If preserve_analysis_columns=True, return all columns
    # If ticker ends with '_gap', automatically preserve gap columns
```

### Option 2: SR System Custom Data Access
Add gap-aware data loading in SR calculations:
```python
def load_gap_data(self, ticker: str):
    # Custom method for gap data that preserves analysis columns
```

### Option 3: MMM Module Custom Data Reader
Implement gap-specific data access in MMM module:
```python
def _load_gap_analysis_data(self, ticker: str):
    # Load gap data directly, bypassing SR filtering
```

## Recommended Fix

**Immediate Fix**: Modify DataReader to detect gap files and preserve gap columns

**Implementation**:
1. Detect if ticker contains '_gap' suffix
2. If gap file, preserve gap columns alongside OHLCV
3. Maintain backward compatibility for regular ticker data

This fix would resolve:
- XLY_gap display issue in MMM Panel_2
- Similar issues with other analysis-specific data columns
- Maintain consistency across Panel, Overview, and MMM modules

## Files Requiring Changes
- `src/data_reader.py` - Add gap-aware column preservation
- Test gap data loading across SR modules
- Verify chart generation uses gap columns correctly

---
*Research completed: All gap data exists, DataReader filtering is the root cause*