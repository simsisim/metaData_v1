# Market Pulse Analysis Module Documentation

## Overview

The Market Pulse Analysis module provides comprehensive market timing and trend analysis for major market indexes (SPY, QQQ, IWM, ^DJI). It implements professional-grade institutional indicators to assess current market conditions and generate actionable trading signals.

## Indicators Implemented

### 1. GMI (General Market Index) - Dr. Eric Wish Methodology
- **Purpose**: 4-component market timing system 
- **Components**: P1 (Short-term trend), P2 (Trend momentum), P3 (Long-term trend), P5 (Price momentum)
- **Signal Logic**: GREEN when GMI > 3 for 2+ days, RED when < 3 for 2+ days
- **Implementation**: `src/gmi_calculator.py`

### 2. FTD & DD Analyzer - William J. O'Neil Methodology  
- **Purpose**: Institutional-grade market timing based on Follow-Through Days and Distribution Days
- **Follow-Through Days**: Price gain ≥1.5% with higher volume, 4-13 days after market bottom
- **Distribution Days**: Price decline ≥0.2% with higher volume (institutional selling)
- **Signal Logic**: 4+ DDs = caution, 6+ DDs = severe warning, FTD in 4-7 day window = optimal

### 3. Net New Highs/Lows - Market Breadth Analysis
- **Purpose**: Universe-wide breadth analysis across multiple timeframes  
- **Timeframes**: 52-week, 3-month, 1-month
- **Signal Logic**: >2% net new highs = healthy, >2% net new lows = unhealthy
- **Scalable**: Current 757 tickers → expandable to 5,000+ tickers

### 4. Chillax Moving Averages - Qullamaggie Color Coding
- **Purpose**: Trend analysis with visual color-coded signals
- **Parameters**: 10-day MA vs 20-day MA with trend confirmation
- **Colors**: DARK_GREEN (strongest), LIGHT_GREEN (moderate), YELLOW (weakening), RED (bearish)

### 5. Moving Average Cycles - Cycle Identification
- **Purpose**: Identify bullish/bearish cycles relative to 50-day MA
- **Analysis**: Current cycle type, cycle length, distance from MA
- **Use Case**: Timing entries/exits based on cycle position

## Architecture

### Main Class: `MarketPulseAnalyzer`
```python
from src.market_pulse import MarketPulseAnalyzer, run_market_pulse_analysis

# Initialize
analyzer = MarketPulseAnalyzer(config, user_config, data_reader)

# Run complete analysis
results = analyzer.run_complete_market_analysis()

# Get human-readable summary
summary = analyzer.get_market_pulse_summary()
```

### Standalone Functions
```python
# GMI analysis for single index
from src.market_pulse import run_gmi_analysis_for_index
gmi_results = run_gmi_analysis_for_index('QQQ', config, threshold=3)

# Complete market pulse analysis
from src.market_pulse import run_market_pulse_analysis
results = run_market_pulse_analysis(config, user_config, data_reader)
```

## Configuration

Market Pulse is fully configurable via `user_data.csv`:

### Main Controls
- `MARKET_PULSE_enable`: Enable/disable entire Market Pulse system
- `MARKET_PULSE_gmi_enable`: Enable GMI analysis
- `MARKET_PULSE_ftd_dd_enable`: Enable FTD & DD analysis
- `MARKET_PULSE_net_highs_lows_enable`: Enable Net Highs/Lows analysis
- `MARKET_PULSE_chillax_ma_enable`: Enable Chillax MA analysis  
- `MARKET_PULSE_ma_cycles_enable`: Enable MA Cycles analysis

### GMI Configuration
- `MARKET_PULSE_gmi_threshold`: Score threshold for signals (default: 3)
- `MARKET_PULSE_gmi_confirmation_days`: Days required for confirmation (default: 2)

### FTD & DD Configuration
- `MARKET_PULSE_dd_threshold`: Minimum decline for Distribution Day (default: 0.2%)
- `MARKET_PULSE_ftd_threshold`: Minimum gain for Follow-Through Day (default: 1.5%)
- `MARKET_PULSE_ftd_optimal_days_min/max`: Optimal FTD timing window (default: 4-7 days)
- `MARKET_PULSE_dd_lookback_period`: DD counting period (default: 25 days)

### Market Breadth Configuration
- `MARKET_PULSE_breadth_threshold_healthy`: Healthy breadth threshold (default: 2.0%)
- `MARKET_PULSE_breadth_threshold_unhealthy`: Unhealthy breadth threshold (default: -2.0%)

### Output Configuration
- `MARKET_PULSE_output_dir`: Output directory (default: "results/market_pulse")
- `MARKET_PULSE_save_detailed_results`: Save detailed JSON results
- `MARKET_PULSE_generate_alerts`: Generate actionable alerts

## Integration with Main Pipeline

Market Pulse is integrated into the main screening pipeline via `src/run_screeners.py`. When enabled, it runs automatically during the screener phase and generates:

1. **JSON Results**: Detailed analysis data in `results/market_pulse/market_pulse_daily_YYYYMMDD_HHMMSS.json`
2. **Summary Report**: Human-readable summary in `results/market_pulse/market_pulse_summary_daily_YYYYMMDD_HHMMSS.txt`
3. **Console Output**: Real-time market pulse summary displayed during pipeline execution

## Data Requirements

### Index Data Required
- SPY.csv, QQQ.csv, IWM.csv, ^DJI.csv in `data/market_data/daily/`
- Minimum 150 days of OHLCV data for GMI analysis
- Date column format: 'YYYY-MM-DD' (auto-parsed)

### Universe Data (for Net Highs/Lows)
- All ticker CSV files in `data/market_data/daily/` (excludes indexes starting with ^)
- Minimum 252 days (1 year) of data per ticker
- Automatically scales from current 757 tickers to 5,000+ future expansion

## Output Format

### JSON Results Structure
```json
{
  "timestamp": "2025-09-01T...",
  "success": true,
  "indexes": {
    "SPY": {
      "gmi": {"current_signal": "GREEN", "current_score": 4},
      "ftd_dd": {"market_health": {"assessment": "BEARISH"}},
      "chillax_ma": {"color_signal": "DARK_GREEN"},
      "ma_cycles": {"current_cycle_type": "BULLISH"},
      "new_highs_lows": {...}
    },
    ...
  },
  "market_summary": {
    "overall_signal": "BULLISH",
    "confidence": "HIGH"
  },
  "alerts": [...]
}
```

### Summary Report Format
```
Market Pulse Summary (2025-09-01):
═══════════════════════════════════════════════════

Overall Signal: 🟢 BULLISH (Confidence: HIGH)

Index Analysis:
• SPY: GMI=GREEN, FTD/DD=BEARISH, Trend=DARK_GREEN
• QQQ: GMI=GREEN, FTD/DD=BEARISH, Trend=RED
• IWM: GMI=GREEN, FTD/DD=BEARISH, Trend=DARK_GREEN
• ^DJI: GMI=GREEN, FTD/DD=BEARISH, Trend=DARK_GREEN

Active Alerts (7):
🚨 CAUTION: 4 distribution days detected
⚠️ Follow-Through Day detected on 2025-08-15
```

## Usage Examples

### Basic Usage
```python
# Run Market Pulse as part of main pipeline
python main.py  # Market Pulse runs automatically if enabled

# Standalone Market Pulse analysis
from src.market_pulse import run_market_pulse_analysis
from src.config import Config
from src.user_defined_data import read_user_data

config = Config()
user_config = read_user_data()
results = run_market_pulse_analysis(config, user_config)
```

### Testing
```bash
# Run comprehensive test suite
python test_market_pulse.py

# Test specific functionality
python -c "from src.market_pulse import run_gmi_analysis_for_index; from src.config import Config; print(run_gmi_analysis_for_index('QQQ', Config()))"
```

## Performance and Scalability

- **Current Capacity**: Handles 757 tickers efficiently
- **Future Scaling**: Designed for 5,000+ ticker universe expansion
- **Runtime**: ~2-5 seconds for complete 4-index analysis
- **Memory**: Optimized for batch processing with minimal memory footprint
- **Data Loading**: Cached index data, lazy-loaded universe data

## Alert System

Market Pulse generates actionable alerts based on institutional criteria:

### High Priority Alerts
- **Severe Distribution Days**: 6+ DDs detected (reduce long exposure)
- **Follow-Through Days**: FTD in optimal 4-7 day window (increase long exposure)

### Medium Priority Alerts  
- **Caution Distribution Days**: 4-5 DDs detected (monitor position sizing)
- **Late Follow-Through Days**: FTD outside optimal window (moderate signal)

### Signal Integration
The overall market signal aggregates all indicator signals:
- **BULLISH**: Majority of indicators positive, high confidence
- **BEARISH**: Majority of indicators negative, high confidence  
- **NEUTRAL**: Mixed signals or insufficient confirmation

## References and Methodology

### Academic and Professional Sources
1. **Dr. Eric Wish GMI**: TradingView implementation, 6-component market timing
2. **William J. O'Neil FTD & DD**: "How to Make Money in Stocks" methodology
3. **Daryl Guppy GMMA**: Multiple Moving Average analysis (future enhancement)
4. **Qullamaggie**: Color-coded moving average trend analysis
5. **Market Breadth Theory**: Net new highs/lows institutional analysis

### Implementation Status
- ✅ **Phase 1 Complete**: GMI (4/6 components), FTD & DD, Net Highs/Lows, Chillax MA, MA Cycles
- 🔄 **Future Enhancements**: GMI P4 (breadth) and P6 (distribution), NASI/NASI+ indicators
- 📈 **Integration**: Fully integrated into main pipeline with comprehensive configuration

This module provides institutional-quality market analysis previously only available in premium platforms, now integrated into the comprehensive stock analysis system.