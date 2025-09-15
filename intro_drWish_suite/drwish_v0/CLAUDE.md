# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python implementation of Dr. Eric Wish's trading system methodology, specifically his hierarchical market-first approach to stock screening and selection. The system implements:

1. **GMI (General Market Index)** - Market regime filter that determines when to take positions
2. **Green Line Breakout** - Momentum breakout screener for individual stocks
3. **Future implementations**: Blue Dot and Black Dot oversold bounce signals

## Architecture

### Core Philosophy
The system follows Dr. Wish's hierarchical approach: individual stock signals are ONLY generated when the overall market conditions (GMI) are favorable. This prevents taking long positions during unfavorable market regimes.

### Main Entry Points
- `main.py` - Primary Dr. Wish trading system implementation

### Key Modules

#### Market Analysis (`src/gmi_calculator.py`)
- Implements 6-component GMI proxy model (GMI-P)
- Generates daily market regime signals (GREEN/RED) based on market index analysis
- Default market index: QQQ (configurable to SPY or others)
- Signal logic: GMI > 3 for 2+ days = Green, GMI < 3 for 2+ days = Red

#### Stock Screening (Accurate GLB Implementation)
- **Accurate GLB Calculator** - Implements exact TradingView PineScript logic
- Proper pivot high detection with configurable strength (not simple ATH proximity)
- Exact confirmation periods before GLB is valid
- Multi-timeframe support (Daily/Weekly/Monthly)
- Volume confirmation for breakout signals

#### Data Management
- `src/data_reader.py` - Handles batch processing of stock data
- `src/config.py` - Path management and configuration
- `src/combined_tickers.py` - Ticker universe management
- `src/tickers_choice.py` - User selection of ticker universes

### Data Dependencies

The system requires external market data located at:
- `../downloadData_v1/data/market_data/daily/` - Daily stock price data
- `../downloadData_v1/data/tickers/` - Ticker lists and metadata

Supported ticker universes:
- Portfolio tickers (option 0)
- S&P 500 (option 1)  
- NASDAQ 100 (option 2)
- NASDAQ All (option 3)
- Russell 1000 (option 4)
- Various combinations (options 5-15)
- Index tickers (option 16)

## Common Development Commands

### Running the System
```bash
# Run the complete Dr. Wish trading system
python main.py
```

### Development Environment
- Python 3.11.7
- Key dependencies: pandas, numpy
- No requirements.txt file - dependencies are imported as needed

### Configuration
The system uses:
- **Modern Configuration**: Clean parameter management for accurate GLB implementation
- **Accurate GLB Parameters**: Pivot strength, timeframes, lookback periods, confirmation periods
- **GMI Settings**: Market index, thresholds, confirmation days

### Output Structure
Results are saved to:
- `src/scanners/results/` - Analysis reports and signal files
- GLB database for persistent tracking

## System Workflow

1. **Initialization**: Load ticker universe and configuration
2. **Market Regime Analysis**: Calculate GMI signal for market index
3. **GLB Screening**: Run accurate GLB screening using PineScript-matching logic
4. **Report Generation**: Create comprehensive daily trading report with recommendations
5. **File Output**: Save detailed results and summary reports

## Key Design Patterns

- **Market-first approach**: GMI provides context but doesn't prevent screening
- **Accurate GLB detection**: Exact TradingView PineScript logic matching
- **Configurable parameters**: All thresholds and periods are user-adjustable
- **Comprehensive reporting**: Both summary and detailed output files
- **Error handling**: Graceful handling of missing data files

## Important Notes

- System generates GLB signals regardless of market conditions (GMI provides warnings)
- Accurate GLB implementation uses proper pivot detection, not simple ATH proximity
- Confirmation periods ensure GLB validity before signals are generated
- Future expansion planned for Blue Dot and Black Dot strategies