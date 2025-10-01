# TradingView Watchlist Export - Quick Reference

**Last Updated**: 2025-09-30
**Status**: ✅ Production Ready

---

## 🚀 QUICK START

### Automatic Export
```bash
python main.py
```
Output: `results/screeners/pvbTW/pvb_watchlist_2_20250930.txt`

### Manual Export
```bash
# Latest files
python scripts/export_watchlist_standalone.py

# Specific date
python scripts/export_watchlist_standalone.py --date 20250905

# With ticker choice
python scripts/export_watchlist_standalone.py --choice 2-5 --date 20250905
```

---

## 📋 OUTPUT FORMAT

File contains 4 sections per timeframe (sorted by days_since_signal):

```
###PVB_Daily_Buy
NASDAQ:GOOG, NASDAQ:MSFT, NASDAQ:AAPL, NASDAQ:META

###PVB_Daily_Sell
NASDAQ:LULU, NASDAQ:ISRG, NASDAQ:KHC, NASDAQ:MRVL

###PVB_Daily_Close_Buy
NASDAQ:COST, NASDAQ:AXON, NASDAQ:DDOG

###PVB_Daily_Close_Sell
NASDAQ:TTWO, NASDAQ:PANW, NASDAQ:TXN
```

---

## ⚙️ CONFIGURATION

`user_data.csv` lines 496-500:

```csv
PVB_TWmodel_export_tradingview,TRUE,Enable export
PVB_TWmodel_watchlist_max_symbols,1000,Max symbols per file
PVB_TWmodel_watchlist_include_buy,TRUE,Include Buy + Close_Buy
PVB_TWmodel_watchlist_include_sell,TRUE,Include Sell + Close_Sell
```

---

## 🔧 TROUBLESHOOTING

### No file created?
```bash
# Check if enabled
grep "export_tradingview" user_data.csv

# Verify config loaded
python3 -c "from src.user_defined_data import read_user_data; \
print(read_user_data().pvb_TWmodel_export_tradingview)"
```

### Missing Close_Buy/Close_Sell?
Check `tw_export_watchlist.py` line 113-116:
```python
if include_buy:
    signal_filter.extend(['Buy', 'Close_Buy'])  # Must include both
```

### Wrong sort order?
Check `tw_export_watchlist.py` line 248:
```python
unique_group = unique_group.sort_values('days_since_signal', ascending=True)
```

### Exchange shows N/A?
```bash
# Verify ticker_universe_all.csv exists
ls -lh results/ticker_universes/ticker_universe_all.csv

# Check exchange column in PVB output
head -3 results/screeners/pvbTW/pvb_screener_*.csv | grep exchange
```

---

## 📁 KEY FILES

| File | Purpose |
|------|---------|
| `src/tw_export_watchlist.py` | Main exporter (620 lines) |
| `scripts/export_watchlist_standalone.py` | CLI tool (166 lines) |
| `main.py` lines 390-497 | Integration points |
| `user_data.csv` lines 496-500 | Configuration |
| `src/user_defined_data.py` lines 454-458 | Config fields |

---

## 🎯 SIGNAL TYPES

| Signal | Meaning |
|--------|---------|
| **Buy** | New long position |
| **Sell** | New short position |
| **Close_Buy** | Close short (buy to cover) |
| **Close_Sell** | Close long (sell position) |

**Note**: `include_buy=TRUE` includes both Buy and Close_Buy. Same for sell.

---

## 📊 IMPORT TO TRADINGVIEW

1. Open TradingView
2. Click watchlist name (right toolbar)
3. Select "Import list..."
4. Choose `.txt` file
5. Symbols imported with sections ✓

---

## 🔍 VALIDATION

```bash
# Count sections
grep "^###" results/screeners/pvbTW/pvb_watchlist_*.txt

# View full file
cat results/screeners/pvbTW/pvb_watchlist_2_20250905.txt

# Check sorting
python3 -c "
import pandas as pd
df = pd.read_csv('results/screeners/pvbTW/pvb_screener_2_daily_20250905.csv')
df['signal_type'] = df['signal_type'].str.replace(' ', '_')
print(df[df['signal_type']=='Close_Buy'][['ticker','days_since_signal']].sort_values('days_since_signal'))
"
```

---

## 📚 FULL DOCUMENTATION

See: `docus/RESEARCH_TradingView_Watchlist_Export.md` (1,237 lines)

Sections include:
- Complete troubleshooting guide (15.1-15.8)
- Performance metrics (19.1-19.3)
- Implementation challenges (18.2)
- Future enhancements (16.1-16.2)
- Testing checklist (17.1-17.4)

---

## 🆘 COMMON ISSUES

| Issue | Quick Fix |
|-------|-----------|
| No file created | Set `PVB_TWmodel_export_tradingview,TRUE` |
| Missing Close signals | Check filter includes `Close_Buy`, `Close_Sell` |
| Wrong sort order | Verify `sort_values('days_since_signal')` |
| All exchanges "N/A" | Verify `ticker_universe_all.csv` exists |
| UnboundLocalError Path | Remove duplicate `from pathlib import Path` |

---

**For detailed troubleshooting**: See section 15 in full documentation
