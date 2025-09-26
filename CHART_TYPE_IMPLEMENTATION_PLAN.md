# Chart Type Implementation Plan
## Adding chart_type Support (line, candles, none) to SR Module

### **🎯 Objective**
Enable flexible ticker display options through `chart_type` column in `user_charts_display.csv`:
- **line**: Standard line chart (current behavior)
- **candles**: OHLC candlestick chart
- **none**: Hide ticker, show only indicators/overlays (e.g., "SPY + EMA(20)" → show only EMA(20))

### **📊 Current State Analysis**

#### **1. CSV Structure (✅ Ready)**
```csv
#file_name_id,chart_type,Panel_1,Panel_2,Panel_3,Panel_4,Panel_5,Panel_6,Panel_1_index,Panel_2_index,Panel_3_index,Panel_4_index,Panel_5_index,Panel_6_index
SPY_vs_IWM,candle,SPY + EMA(20),IWM,SPY:IWM,,,,"A_PPO(12,26,9)_for_(SPY)",B_RSI(14)_for_(IWM),,,,
QQQ_Analysis,line,QQQ + EMA(10) + SMA(50),,,,,,"A_PPO(12,26,9)_for_(QQQ) ",B_RSI(14)_for_(QQQ),,,,
```

#### **2. Current Bundled Format Handling (✅ Available)**
- **Bundled format**: "SPY + EMA(20)" already parsed correctly
- **Base ticker extraction**: `extract_base_ticker_from_bundled()` → "SPY"
- **Overlay extraction**: `extract_overlay_info_from_bundled()` → ["EMA(20)"]
- **Market data**: OHLCV data already available for candlestick charts

#### **3. Current Plot Logic (🔧 Needs Modification)**
- **Main plotting**: `plot_overlay_chart()` at line 593-594 hardcoded as line chart
- **Data access**: `main_series` contains Close price data
- **OHLCV access**: Full OHLCV data available in `result` dict

### **🏗️ Implementation Architecture**

#### **1. Configuration Parsing Enhancement**

**File**: `src/sustainability_ratios/sr_config_reader.py`

**Location**: Around line 580 where CSV columns are parsed

**Modification needed**:
```python
# Current logic processes columns starting from panel_start_idx
# Need to extract chart_type column before panel processing

chart_type = 'line'  # Default
for col_idx, col_name in enumerate(columns):
    col_name_str = str(col_name).strip().lower()
    if col_name_str == 'chart_type':
        chart_type_value = str(row.iloc[col_idx]).strip().lower()
        if chart_type_value in ['line', 'candles', 'candlestick', 'candle', 'none']:
            chart_type = 'candles' if chart_type_value in ['candles', 'candlestick', 'candle'] else chart_type_value
        break
```

**Pass chart_type through**:
- Add `chart_type` to panel configuration metadata
- Propagate through `panel_config[config_key]` dict

#### **2. Chart Plotting Enhancement**

**File**: `src/sustainability_ratios/sr_dashboard_generator.py`

**Function**: `plot_overlay_chart()` (line 517)

**Key Modifications**:

**A. Function signature update**:
```python
def plot_overlay_chart(ax, result: Dict, main_series: pd.Series, data_source: str,
                      indicator: str, is_bundled: bool = False, chart_type: str = 'line'):
```

**B. Chart type routing logic** (replace line 593-594):
```python
# Current:
main_line = ax.plot(x_positions, main_series.values, label=main_label, linewidth=1.5, alpha=0.8, color='blue')

# New:
if chart_type == 'none':
    # Skip plotting main series entirely
    logger.info(f"🚫 Skipping main series plot (chart_type='none')")

elif chart_type == 'candles':
    # Plot candlestick chart
    plot_candlestick_chart(ax, result, x_positions, main_label)

else:  # chart_type == 'line' (default)
    # Current line chart logic
    main_line = ax.plot(x_positions, main_series.values, label=main_label, linewidth=1.5, alpha=0.8, color='blue')
```

#### **3. Candlestick Chart Implementation**

**New Function**: `plot_candlestick_chart()`

**Dependencies**: mplfinance (✅ available)

**OHLCV Data Access**:
```python
def plot_candlestick_chart(ax, result: Dict, x_positions: range, main_label: str):
    # Extract OHLCV data from result dict
    ohlc_data = {
        'Open': result.get('Open'),
        'High': result.get('High'),
        'Low': result.get('Low'),
        'Close': result.get('Close'),
        'Volume': result.get('Volume')
    }

    # Create candlestick chart using mplfinance or custom implementation
    # Handle weekend gap removal with x_positions indexing
```

#### **4. Data Flow Integration**

**Path**: CSV → Config → Data Loading → Chart Generation

1. **CSV Parsing**: `sr_config_reader.py` extracts `chart_type`
2. **Config Storage**: Add `chart_type` to panel metadata
3. **Data Loading**: `sr_market_data.py` loads OHLCV (already implemented)
4. **Chart Generation**: `plot_panel()` passes `chart_type` to `plot_overlay_chart()`

### **📋 Implementation Steps**

#### **Phase 1: Configuration Enhancement**
1. **Modify `sr_config_reader.py`**:
   - Extract `chart_type` column during CSV parsing
   - Add `chart_type` to panel configuration metadata
   - Handle backward compatibility (default to 'line')

2. **Update panel processing functions**:
   - `_process_bundled_panel_entry()`
   - `_process_enhanced_panel_entry()`
   - Add `chart_type` parameter propagation

#### **Phase 2: Chart Plotting Enhancement**
1. **Modify `plot_panel()` function**:
   - Extract `chart_type` from panel metadata
   - Pass to `plot_overlay_chart()`

2. **Enhance `plot_overlay_chart()` function**:
   - Add `chart_type` parameter
   - Implement chart type routing logic
   - Handle 'none' type (skip main series)

#### **Phase 3: Candlestick Implementation**
1. **Create `plot_candlestick_chart()` function**:
   - Use mplfinance for professional candlestick charts
   - Handle weekend gap removal
   - Maintain consistency with line chart styling

2. **OHLCV Data Integration**:
   - Ensure OHLCV data availability in `result` dict
   - Handle missing data gracefully

#### **Phase 4: Testing & Validation**
1. **Test Cases**:
   - `chart_type=line`: Standard behavior (baseline)
   - `chart_type=candles`: OHLC candlestick with overlays
   - `chart_type=none`: Only overlays visible (SPY + EMA → show only EMA)

### **🧪 Test Scenarios**

#### **Scenario 1: Line Chart (Baseline)**
```csv
QQQ_Line,line,QQQ + EMA(10),,,,,,"A_PPO(12,26,9)_for_(QQQ)",,,,
```
**Expected**: Standard line chart with EMA overlay (current behavior)

#### **Scenario 2: Candlestick Chart**
```csv
SPY_Candles,candles,SPY + EMA(20) + SMA(50),,,,,,"A_PPO(12,26,9)_for_(SPY)",,,,
```
**Expected**: OHLC candlesticks for SPY + EMA(20) + SMA(50) overlays + A_PPO above

#### **Scenario 3: Indicators Only**
```csv
QQQ_IndicatorsOnly,none,QQQ + EMA(10) + SMA(50),,,,,,"A_PPO(12,26,9)_for_(QQQ)",,,,
```
**Expected**: Only EMA(10) + SMA(50) lines visible, no QQQ price data + A_PPO above

#### **Scenario 4: Mixed Chart Types**
```csv
Mixed_Dashboard,line,SPY + EMA(20),QQQ + SMA(50),IWM,,,,"A_PPO(12,26,9)_for_(SPY)","B_RSI(14)_for_(QQQ)",,,,
```
**Expected**: All panels use same chart_type (row-level setting)

### **⚠️ Technical Considerations**

#### **1. OHLCV Data Availability**
- **Current**: SR module loads Close price data
- **Required**: Ensure Open, High, Low, Volume data loaded for candlestick charts
- **Solution**: Market data loading already comprehensive

#### **2. Weekend Gap Handling**
- **Current**: Uses `x_positions = range(len(main_series))` for no gaps
- **Required**: Apply same logic to candlestick charts
- **Solution**: Custom mplfinance integration with index positions

#### **3. Overlay Consistency**
- **Current**: Overlays plotted as lines with index positions
- **Required**: Maintain same positioning for all chart types
- **Solution**: Consistent x_positions usage across all plot types

#### **4. Volume Display Integration**
- **Current**: Optional volume subplot for line charts
- **Required**: Enhanced volume display for candlestick charts
- **Solution**: Extend existing volume logic for candlesticks

### **📁 Files to Modify**

#### **Primary Files**:
1. **`sr_config_reader.py`**: Chart type parsing and configuration
2. **`sr_dashboard_generator.py`**: Chart plotting logic and candlestick implementation

#### **Secondary Files** (if needed):
3. **`sr_market_data.py`**: Ensure OHLCV data availability
4. **`sr_calculations.py`**: Pass chart_type through pipeline

### **🔄 Backward Compatibility**

#### **Default Behavior**:
- Missing `chart_type` column → default to 'line'
- Invalid `chart_type` values → fallback to 'line'
- Existing configurations → unchanged behavior

#### **Migration Path**:
- Existing CSV files work without modification
- New CSV files can add `chart_type` column progressively
- No breaking changes to current functionality

### **🎯 Success Criteria**

#### **Functional Requirements**:
- ✅ `chart_type=line`: Current line chart behavior maintained
- ✅ `chart_type=candles`: Professional OHLC candlestick charts
- ✅ `chart_type=none`: Only indicators/overlays visible
- ✅ Bundled format support: "SPY + EMA(20)" works with all chart types
- ✅ Overlay consistency: EMA/SMA overlays work with all chart types

#### **Technical Requirements**:
- ✅ No weekend gaps in all chart types
- ✅ Volume subplot integration for candlesticks
- ✅ Backward compatibility with existing configurations
- ✅ Error handling for missing OHLCV data
- ✅ Performance: No significant impact on chart generation speed

This implementation plan provides a comprehensive roadmap for adding flexible chart type support while maintaining the robust architecture and features of the existing SR module.