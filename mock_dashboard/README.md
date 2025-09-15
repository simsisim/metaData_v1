# Mock Trading Dashboard Project
*Independent design prototype using randomly generated data*

## Overview

This is an independent project within your trading system designed to prototype and design an Excel-based market overview dashboard. Uses mock data to focus purely on design, layout, and user experience before integrating with real trading system data.

## Design Philosophy

**Inspired by professional trading dashboards**:
- **Information Hierarchy**: Market health → Opportunities → Execution details
- **Visual Design**: Institutional color scheme (Green/Red/Yellow traffic lights)
- **Actionable Intelligence**: Clear entry/exit signals with risk/reward ratios
- **Professional Layout**: Clean, data-dense presentation

## Project Structure

```
mock_dashboard/
├── README.md                    # This file
├── mock_data_generator.py       # Generates realistic random trading data
├── dashboard_builder.py         # Creates Excel dashboard with formatting
├── sample_data/                 # Generated mock datasets
│   ├── bullish_market/         # Bullish scenario data
│   ├── bearish_market/         # Bearish scenario data  
│   ├── neutral_market/         # Neutral scenario data
│   └── volatile_market/        # High volatility scenario data
├── output/                     # Generated Excel dashboards
└── config/                     # Configuration files
```

## Dashboard Tabs

### **Tab 1: Market Pulse** 📊
Primary market overview with:
- **GMI Analysis**: Signals for SPY, QQQ, IWM, DJI
- **Risk Monitor**: Distribution Days warnings, Follow-Through Day confirmations
- **Market Breadth**: New highs/lows across universe
- **Sector Performance**: ETF performance with trend indicators

### **Tab 2: Screener Heatmap** 🔥
Opportunity identification with:
- **Top Opportunities**: Best stocks from screener suites
- **Signal Strength**: STRONG/MODERATE/WEAK classification
- **Screener Summary**: Hit counts from all 12+ screener suites
- **Setup Stages**: Entry/Watch/Breakout classifications

### **Tab 3: Sector Analysis** 📈
Sector rotation monitoring:
- **Performance Rankings**: Daily/Weekly/Monthly changes
- **Trend Status**: Chillax MA color coding
- **Rotation Signals**: Sector strength/weakness patterns
- **Visual Charts**: Performance bar charts

### **Tab 4: Alerts & Actions** ⚡
Execution focus:
- **High Priority Alerts**: Distribution Days, breakouts, volume spikes
- **Action Items**: Entry levels, stop levels, target prices
- **Risk Management**: Position sizing recommendations
- **Watchlist Updates**: Key levels to monitor

## Usage

### Generate Mock Data
```bash
cd mock_dashboard
python mock_data_generator.py
```

### Build Dashboard
```bash
python dashboard_builder.py
```

### View Results
Check `output/` directory for generated Excel files:
- `trading_dashboard_mock_default_YYYYMMDD_HHMMSS.xlsx`
- `trading_dashboard_mock_bullish_market_YYYYMMDD_HHMMSS.xlsx`
- etc.

## Design Features

### Color Scheme (Institutional Trading)
- **🟢 Green**: Bullish signals, positive performance
- **🔴 Red**: Bearish signals, negative performance, warnings
- **🟡 Yellow**: Neutral signals, caution areas
- **📊 Blue Headers**: Professional section headers
- **⚡ Emojis**: Quick visual scanning (inspired by modern trading analysis)

### Professional Formatting
- **Conditional Formatting**: Automatic color coding based on values
- **Data Validation**: Dropdown filters for different views
- **Clean Typography**: Calibri font family, proper sizing hierarchy
- **Optimal Layout**: Information density without clutter

## Integration Path

Once design is approved:
1. **Real Data Connector**: Replace mock data with actual screener results
2. **Pipeline Integration**: Connect to your existing `run_screeners()` output
3. **Market Pulse Integration**: Use real GMI, FTD/DD analysis
4. **Automation**: Daily refresh from your trading system pipeline

## Dependencies

```bash
pip install pandas numpy openpyxl
```

## Mock Data Scenarios

The system generates multiple market scenarios:
- **Bullish Market**: Strong GMI signals, low distribution days, high new highs
- **Bearish Market**: Weak GMI signals, high distribution days, high new lows  
- **Neutral Market**: Mixed signals, balanced breadth
- **Volatile Market**: Rapid signal changes, high ATR readings

Perfect for testing dashboard design across different market conditions.

---
*This is a design prototype using mock data. Real trading system integration available after design approval.*