# MMM (Market Maker Manipulation) Submodule Implementation Plan

## 🎯 **Overview**

The MMM submodule will analyze daily price gaps (yesterday's close - today's open) to identify potential market maker manipulation patterns. It integrates with the existing SR (Sustainability Ratios) module architecture and provides both CSV outputs and chart visualizations.

## 📊 **Configuration Analysis**

Based on `user_data.csv` configuration:

```csv
#MMM SUBMODULE
SR_MMM_enable,TRUE,
SR_MMM_daily_enable,FALSE,
SR_MMM_weekly_enable,FALSE,
SR_MMM_monthly_enable,FALSE,
SR_MMM_gaps_values,TRUE,
SR_MMM_gaps_tickers,XLY;XLC,list or tickers
SR_MMM_gaps_values_input_folder,../downloadData_v1/data/market_data/,
SR_MMM_gaps_chart_enable,
SR_MMM_gaps_charts_display_panel,gaps_display.csv,Configure chart panels with indicators
SR_MMM_gaps_charts_display_history,30,unit time frame
SR_MMM_output_dir,results/sustainability_ratios/MMM,
```

**Configuration Issues Found:**
- Multiple empty input/output folder entries (lines 198-202)
- Missing filename suffix configuration
- Incomplete chart enable flag

**Corrected Configuration Should Be:**
```csv
SR_MMM_gaps_values_input_folder_daily,../downloadData_v1/data/market_data/daily/
SR_MMM_gaps_values_input_folder_weekly,../downloadData_v1/data/market_data/weekly/
SR_MMM_gaps_values_input_folder_monthly,../downloadData_v1/data/market_data/monthly/
SR_MMM_gaps_values_output_folder_daily,../downloadData_v1/data/market_data/daily/
SR_MMM_gaps_values_output_folder_weekly,../downloadData_v1/data/market_data/weekly/
SR_MMM_gaps_values_output_folder_monthly,../downloadData_v1/data/market_data/monthly/
SR_MMM_gaps_values_filename_suffix,_gap
SR_MMM_gaps_chart_enable,TRUE
```

## 🏗️ **Architecture Design**

### **Existing SR Architecture Pattern:**
```
sustainability_ratios/
├── sr_calculations.py        # Main orchestrator
├── sr_config_reader.py      # Configuration management
├── sr_dashboard_generator.py # Chart generation
├── sr_market_data.py        # Market data loading
├── sr_output_manager.py     # Output organization
├── overview/                # Overview submodule
│   ├── __init__.py         # OverviewProcessor
│   ├── overview_values.py  # Values calculation
│   └── overview_charts.py  # Chart generation
└── mmm/                     # MMM submodule (to implement)
    ├── __init__.py         # MmmProcessor
    ├── mmm_gaps.py         # Gap calculations
    └── mmm_charts.py       # Gap chart generation
```

### **MMM Integration Points:**

1. **sr_calculations.py**: Add MMM submodule execution
2. **sr_output_manager.py**: Add MMM output directory management
3. **mmm/__init__.py**: Create MmmProcessor main controller
4. **mmm/mmm_gaps.py**: Gap calculation and data processing
5. **mmm/mmm_charts.py**: Chart generation using existing chart display system

## 📈 **Gap Calculation Logic**

### **Core Algorithm:**
```python
def calculate_daily_gap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily price gaps and gap-adjusted price movements

    Formulas:
    1. opening_gap = day_(i)[open] - day_(i-1)[close]
    2. price_without_opening_gap = day(i)[close] - day(i)[open]

    opening_gap:
    - Positive = Gap up (opening above previous close)
    - Negative = Gap down (opening below previous close)

    price_without_opening_gap:
    - Intraday price movement excluding the opening gap
    - Pure trading session performance without gap influence
    """
    # Shift previous close by 1 day
    prev_close = df['Close'].shift(1)
    current_open = df['Open']
    current_close = df['Close']

    # Calculate opening gap (user specification)
    opening_gap = current_open - prev_close

    # Calculate opening gap percentage
    opening_gap_pct = (opening_gap / prev_close) * 100

    # Calculate price movement without opening gap (user specification)
    price_without_opening_gap = current_close - current_open

    # Calculate price without gap percentage
    price_without_gap_pct = (price_without_opening_gap / current_open) * 100

    return pd.DataFrame({
        'Date': df.index,
        'Open': current_open,
        'Close': current_close,
        'Previous_Close': prev_close,
        'opening_gap': opening_gap,                           # day_(i)[open] - day_(i-1)[close]
        'opening_gap_pct': opening_gap_pct,                   # opening_gap as percentage
        'price_without_opening_gap': price_without_opening_gap, # day(i)[close] - day(i)[open]
        'price_without_gap_pct': price_without_gap_pct,       # price_without_opening_gap as percentage
        'High': df['High'],
        'Low': df['Low'],
        'Volume': df['Volume']
    })
```

### **Data Processing Workflow:**

1. **Input**: Read OHLCV data from configured input folders
2. **Gap Calculation**: Apply gap calculation formula
3. **Gap Enrichment**: Add statistical metrics (rolling averages, percentiles)
4. **Output**: Save gap data with `_gap` suffix
5. **Chart Generation**: Create visualizations using panel system

## 📁 **Output File Structure**

### **Gap Data Files:**
- **Input**: `XLY.csv` (OHLCV data)
- **Output**: `XLY_gap.csv` (Gap analysis data)

### **Gap File Format:**
```csv
Date,Open,Close,High,Low,Volume,Previous_Close,opening_gap,opening_gap_pct,price_without_opening_gap,price_without_gap_pct,gap_5MA,gap_20MA,gap_percentile
2024-09-24,150.25,149.80,150.75,149.20,1250000,152.00,-1.75,-1.15,-0.45,-0.30,0.23,0.45,25.5
2024-09-23,152.00,153.25,153.50,151.75,980000,151.50,0.50,0.33,1.25,0.82,0.18,0.42,65.2
2024-09-22,151.50,151.85,152.10,150.90,1100000,150.75,0.75,0.50,0.35,0.23,-0.12,0.38,72.1
```

**Key Columns Explained:**
- **opening_gap**: `day_(i)[open] - day_(i-1)[close]` - Gap between today's open and yesterday's close
- **price_without_opening_gap**: `day(i)[close] - day(i)[open]` - Intraday price movement excluding gap
- **opening_gap_pct**: Opening gap as percentage of previous close
- **price_without_gap_pct**: Intraday movement as percentage of opening price
- **Statistical columns**: Moving averages and percentiles for gap analysis

### **Output Directory Structure:**
```
results/sustainability_ratios/MMM/
├── gaps/           # Gap calculation CSV files
│   ├── XLY_gap.csv
│   └── XLC_gap.csv
├── charts/         # Gap visualization PNG files
│   ├── XLY_gaps_chart.png
│   └── XLC_gaps_chart.png
└── summary/        # Summary analysis files
    └── mmm_gaps_summary.csv
```

## 📊 **Chart Integration Design**

### **Using Existing Panel System:**

The MMM submodule will leverage the existing chart display system used by the overview and panel submodules.

### **gaps_display.csv Configuration:**
```csv
#file_name_id,chart_type,Panel_1,Panel_2,Panel_1_index,Panel_2_index
XLY_gaps,line,XLY_gap,,A_MA(XLY_gap,5),B_PERCENTILE(XLY_gap,20)
XLC_gaps,candle,XLC_gap + MA(5),,A_HISTOGRAM(XLC_gap),B_STATS(XLC_gap)
```

### **Chart Types for Gap Analysis:**
- **Line Charts**: Gap trend over time
- **Histogram**: Gap distribution analysis
- **Candlestick**: Price action with gap overlay
- **Statistical Overlays**: Moving averages, percentiles, standard deviations

## 🔧 **Implementation Steps**

### **Phase 1: Core MMM Module Structure**

1. **Create MmmProcessor class** (mmm/__init__.py)
   - Main controller following OverviewProcessor pattern
   - Integration with sr_calculations.py
   - Configuration handling

2. **Implement Gap Calculator** (mmm/mmm_gaps.py)
   - Daily gap calculation logic
   - Statistical enrichment (moving averages, percentiles)
   - Multi-timeframe support (daily/weekly/monthly)
   - File I/O with _gap suffix

### **Phase 2: Chart Integration**

3. **Create Gap Chart Generator** (mmm/mmm_charts.py)
   - Leverage existing sr_dashboard_generator.py
   - Support gaps_display.csv configuration
   - Generate gap-specific visualizations
   - Integration with chart_type system (line/candle/no_drawing)

### **Phase 3: Configuration & Integration**

4. **Update Configuration Management**
   - Extend user_data.csv reading in sr_config_reader.py
   - Add MMM-specific configuration validation
   - Handle ticker list parsing (XLY;XLC format)

5. **Update Output Management**
   - Extend sr_output_manager.py with MMM directory structure
   - File migration and organization
   - Gap file naming conventions

### **Phase 4: Main Integration**

6. **Integrate with sr_calculations.py**
   - Add MMM submodule execution block
   - Follow overview submodule pattern
   - Error handling and logging
   - Results aggregation

## 📋 **Technical Specifications**

### **Gap Calculation Requirements:**
- **opening_gap Formula**: `day_(i)[open] - day_(i-1)[close]`
- **price_without_opening_gap Formula**: `day(i)[close] - day(i)[open]`
- **Gap Types**: Absolute points and percentage for both metrics
- **Statistical Metrics**: 5-day MA, 20-day MA, percentile ranking
- **Data Preservation**: Include all original OHLCV columns
- **Missing Data Handling**: Skip days with insufficient data

### **File Processing Requirements:**
- **Input Format**: Standard OHLCV CSV files
- **Output Format**: Gap analysis CSV with additional columns
- **Filename Convention**: {ticker}_gap.csv
- **Directory Structure**: Organized by timeframe (daily/weekly/monthly)

### **Chart Generation Requirements:**
- **Integration**: Use existing panel chart system
- **Configuration**: gaps_display.csv for panel definitions
- **Chart Types**: Support all existing chart_type options
- **History**: Configurable display period (SR_MMM_gaps_charts_display_history)

### **Performance Considerations:**
- **Batch Processing**: Process multiple tickers efficiently
- **Memory Management**: Handle large datasets appropriately
- **Error Recovery**: Graceful handling of missing or corrupted data
- **Logging**: Comprehensive progress and error logging

## 🎯 **Success Criteria**

### **Functional Requirements:**
- ✅ Parse MMM configuration from user_data.csv
- ✅ Calculate daily gaps: yesterday close - today open
- ✅ Save gap data with _gap filename suffix
- ✅ Generate gap charts using panel display system
- ✅ Support configurable ticker lists (XLY;XLC format)
- ✅ Integrate with existing SR module workflow

### **Technical Requirements:**
- ✅ Follow SR submodule architecture pattern
- ✅ Use organized output directory structure (MMM/)
- ✅ Support multi-timeframe processing
- ✅ Maintain backward compatibility with existing SR features
- ✅ Provide comprehensive error handling and logging

### **Integration Requirements:**
- ✅ Chart system compatibility with chart_type support
- ✅ Configuration system integration
- ✅ Output manager integration
- ✅ Main calculation pipeline integration

## 🚀 **Next Steps**

1. **Fix user_data.csv configuration** (missing values and structure)
2. **Implement MmmProcessor** (main controller class)
3. **Create gap calculation logic** (core algorithm)
4. **Develop chart generation** (using existing panel system)
5. **Integrate with sr_calculations.py** (main pipeline)
6. **Test with XLY and XLC tickers** (validation)

This implementation plan provides a comprehensive roadmap for adding MMM (Market Maker Manipulation) gap analysis as a fully integrated submodule within the existing SR architecture.