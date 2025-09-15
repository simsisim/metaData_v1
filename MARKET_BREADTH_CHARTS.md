# Market Breadth Visualization Implementation

## ✅ COMPLETED: TradingView-Style Market Breadth Charts

The system now automatically generates **PNG charts** for every market breadth analysis file, creating comprehensive 3-layer stacked visualizations.

## Chart Structure

### **TradingView-Style 3-Layer Layout**:
```
┌─────────────────────────────────────────────┐
│          CANDLESTICK CHART (60%)            │  ← Price + Volume
│     SPY/QQQ/IWM Index Data + Volume         │
├─────────────────────────────────────────────┤
│      BREADTH INDICATORS (25%)               │  ← MA Breadth %
│   % Above MA20/50/200 + Breadth Score       │
├─────────────────────────────────────────────┤
│     NEW HIGHS/LOWS (15%)                    │  ← Histogram
│  52W New Highs vs Lows + Net Highs Line     │
└─────────────────────────────────────────────┘
```

## Key Features

### **1. Synchronized Date Axis**
- All 3 subplots share the same time axis
- Perfect alignment for visual comparison
- Professional TradingView-like presentation

### **2. Comprehensive Data Integration**
- **Price Data**: Automatic index mapping (SP500→SPY, NASDAQ100→QQQ, etc.)
- **Breadth Data**: All historical breadth metrics from analysis files
- **Volume Data**: Scaled volume bars in candlestick chart
- **Synthetic Data**: Fallback when real price data unavailable

### **3. Professional Styling**
- **Dark Theme**: Matches modern trading platforms
- **Color Coding**: 
  - 🟢 Bullish/Highs: Green (`#4caf50`)
  - 🔴 Bearish/Lows: Red (`#f44336`) 
  - 🔵 MA50: Blue (`#2196f3`)
  - 🟠 MA20/Breadth Score: Orange (`#ffa726`)
  - 🟣 MA200: Purple (`#ab47bc`)

### **4. Automatic Chart Generation**
Charts are automatically generated for **every CSV file** created by the breadth analyzer:

## File Output Examples

### **For Semicolon Separator** (`SP500;NASDAQ100`):
**CSV Files**:
- `ba_historical_SP500_0-5_daily_20250905.csv` 
- `ba_historical_NASDAQ100_0-5_daily_20250905.csv`

**PNG Charts** (auto-generated):
- ✅ `ba_historical_SP500_0-5_daily_20250905.png`
- ✅ `ba_historical_NASDAQ100_0-5_daily_20250905.png`

### **For Plus Separator** (`SP500+NASDAQ100`):
**CSV Files**:
- `ba_historical_SP500+NASDAQ100_0-5_daily_20250905.csv`

**PNG Charts** (auto-generated):
- ✅ `ba_historical_SP500+NASDAQ100_0-5_daily_20250905.png`

## Technical Implementation

### **Chart Components**:

#### **Top Subplot: Candlestick Chart**
- **Data**: Index OHLCV (SPY for SP500, QQQ for NASDAQ100, IWM for Russell)
- **Features**: Professional candlestick rendering with volume bars
- **Fallback**: Synthetic price data based on breadth score when real data unavailable

#### **Middle Subplot: Breadth Indicators**
- **Lines**: % Above MA20, MA50, MA200 with area fills
- **Score**: Overall breadth score as bold line
- **References**: Horizontal lines at 80% (strong), 50% (neutral), 20% (weak)

#### **Bottom Subplot: New Highs/Lows**
- **Histogram**: 52-week new highs (green bars above zero) vs new lows (red bars below zero)  
- **Line**: Net new highs trend line overlay
- **Reference**: Zero line for balance point

### **Advanced Features**:

#### **Smart Index Mapping**:
```python
index_mapping = {
    'SP500': 'SPY',      # S&P 500 → SPDR S&P 500 ETF
    'NASDAQ100': 'QQQ',  # NASDAQ 100 → Invesco QQQ ETF  
    'RUSSELL1000': 'IWM', # Russell 1000 → iShares Russell 2000 ETF
    'ALL': 'SPY'         # Combined → Default to SPY
}
```

#### **Multi-Path Data Loading**:
1. `DAILY_DATA_DIR` (primary)
2. `MARKET_DATA_DIR/daily` (fallback)
3. Synthetic generation (final fallback)

#### **Shared Date Axis**:
```python
fig.ax1.sharex(fig.ax3)  # Link all subplots
fig.ax2.sharex(fig.ax3)  # Perfect time alignment
```

## Usage

The chart generation is **fully automatic**. When the breadth analyzer runs:

1. ✅ **CSV Analysis Files Created**
2. ✅ **PNG Charts Auto-Generated** (same filename, .png extension)
3. ✅ **Results Logged**: `Generated X breadth chart(s): [file_paths]`

## Dependencies

- **matplotlib**: Professional charting library
- **pandas**: Data manipulation and analysis  
- **numpy**: Numerical computations

## Chart Quality

- **High DPI**: 150 DPI for crisp, professional output
- **Wide Format**: 16:12 aspect ratio (TradingView style)
- **Dark Theme**: Modern trading platform aesthetic
- **Optimized**: Memory-efficient with automatic cleanup

The market breadth analysis now provides both **comprehensive data analysis** (CSV) and **professional visualizations** (PNG) for complete market insight!