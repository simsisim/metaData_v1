
# QUICK USAGE GUIDE
==================

## 1. Basic Usage
```python
from stock_analysis_framework import StockAnalyzer

# Create analyzer and run full analysis
analyzer = StockAnalyzer()
results = analyzer.run_full_analysis()
```

## 2. Advanced Usage
```python
# Load multiple files
results = analyzer.run_full_analysis(latest_only=False)

# Access specific results
sector_performance = results['sector_performance']
merged_data = results['merged_data']

# Custom analysis
top_gainers = merged_data.nlargest(10, 'daily_daily_yearly_252d_pct_change')
```

## 3. Adding New Data
- Place new files as: basic_calculation_daily_YYYYMMDD.csv
- Update tradingview_universe.csv with new stocks
- Run analyzer.run_full_analysis()

## 4. Customization
- Edit analysis_config.json to modify:
  - Performance metrics to track
  - Technical indicators to analyze  
  - Indices to monitor
  - Output file names

## 5. Visualization
- Use generated CSV files for custom charts
- Import results['merged_data'] for detailed analysis
- Extend the framework with your own visualization functions
