# Stockbee vs Qullamaggie - Data Dependencies Comparison

**Date**: 2025-10-01
**Purpose**: Compare external data requirements for streaming implementation

---

## QUICK ANSWER: Does Stockbee Need External Data?

**YES** - Stockbee has similar but slightly simpler data dependencies than Qullamaggie.

---

## DATA DEPENDENCIES COMPARISON

### Qullamaggie Suite

**External Data Required**:

1. ✅ **ticker_info** (market_cap, exchange) - AVAILABLE
   - Source: `ticker_universe_all.csv`
   - Used for: Market cap ≥ $1B filter

2. ⚠️ **rs_data** (Relative Strength) - NOT AVAILABLE
   - Timeframes: 1w, 1m, 3m, 6m (daily/period_7, daily/period_21, etc.)
   - Used for: RS ≥ 97 filter (top 3%)
   - Complexity: HIGH - requires ranking all stocks
   - **Status**: Must be calculated or estimated

3. ⚠️ **atr_universe_data** - NOT AVAILABLE
   - Universe: All $1B+ stocks
   - Used for: ATR RS ≥ 50 filter (top 50% volatility)
   - Complexity: MEDIUM - requires $1B+ universe filtering
   - **Status**: Must be calculated or estimated

**Summary**: 3 dependencies, 1 available, 2 missing (critical)

---

### Stockbee Suite

**External Data Required**:

1. ✅ **ticker_info** (market_cap, exchange, **industry**) - PARTIALLY AVAILABLE
   - Source: `ticker_universe_all.csv`
   - Used for: Market cap filter, Industry Leaders screener
   - **CRITICAL**: Needs `industry` column for Industry Leaders
   - **Status**: Check if industry field exists in ticker_universe_all.csv

2. ⚠️ **rs_data** (Relative Strength) - NOT AVAILABLE
   - Timeframes: daily/1d, weekly/4w, monthly/3m
   - Used for: Ranking stocks within each screener, Industry Leaders
   - Complexity: HIGH - same as Qullamaggie
   - **Optional**: Screeners can work without RS, but less accurate
   - **Status**: Must be calculated or estimated

**Summary**: 2 dependencies, 1 partially available, 1 missing (optional)

---

## DETAILED BREAKDOWN BY SCREENER

### Stockbee: Four Screening Strategies

#### 1. 9M Movers (Volume-Based)
**Data Required**:
- ✅ OHLCV data (available)
- ✅ ticker_info (market cap, exchange) - available
- ⚠️ rs_data - optional (used for additional scoring)

**Can Run Without External Data**: ✅ YES (uses volume + momentum from OHLCV)

#### 2. 20% Weekly Movers (Momentum)
**Data Required**:
- ✅ OHLCV data (available)
- ✅ ticker_info (market cap, exchange) - available
- ⚠️ rs_data - optional (used for additional scoring)

**Can Run Without External Data**: ✅ YES (calculates weekly gain % from OHLCV)

#### 3. 4% Daily Gainers (Momentum)
**Data Required**:
- ✅ OHLCV data (available)
- ✅ ticker_info (market cap, exchange) - available
- ⚠️ rs_data - optional (used for additional scoring)

**Can Run Without External Data**: ✅ YES (calculates daily gain % from OHLCV)

#### 4. Industry Leaders (Sector Rotation)
**Data Required**:
- ✅ OHLCV data (available)
- ⚠️ ticker_info (**industry** field) - MUST VERIFY
- ⚠️ rs_data - IMPORTANT (calculates industry RS rankings)

**Can Run Without External Data**: ⚠️ **PARTIAL**
- Needs `industry` field in ticker_info
- Can estimate industry RS without formal RS data
- Less accurate without proper RS calculations

---

## CRITICAL DEPENDENCY: Industry Field

### Check Required

```python
# Need to verify if industry field exists
ticker_universe_all = pd.read_csv('results/ticker_universes/ticker_universe_all.csv')
print(ticker_universe_all.columns)
# Expected: ['ticker', 'exchange', 'market_cap', 'sector', 'industry', ...]
```

**If Industry Field Missing**:
- Options:
  1. Add industry data to ticker_universe_all.csv generation
  2. Use sector instead of industry (less granular)
  3. Disable Industry Leaders screener (run only 3 of 4 screeners)

**If Industry Field Exists**: ✅ Stockbee can run with 4/4 screeners

---

## IMPLEMENTATION COMPLEXITY COMPARISON

### Qullamaggie Implementation

**Phase A** (Basic - 3 of 5 filters):
- ✅ Market cap filter
- ❌ RS filter (skipped)
- ✅ MA alignment
- ❌ ATR RS filter (estimated - inaccurate)
- ✅ Range position

**Phase B** (Full - 5 of 5 filters):
- Must build RS calculation pipeline
- Must build ATR universe pipeline
- HIGH complexity

**Estimated Time**:
- Phase A: 2 hours
- Phase B: 6-8 hours
- **Total**: 8-10 hours

---

### Stockbee Implementation

**Phase A** (Basic - without RS data):
- ✅ 9M Movers (100% functional)
- ✅ 20% Weekly Movers (100% functional)
- ✅ 4% Daily Gainers (100% functional)
- ⚠️ Industry Leaders (needs industry field, limited without RS)

**Phase B** (Full - with RS data):
- Add RS calculation pipeline (same as Qullamaggie Phase B)
- Enhance Industry Leaders with proper RS
- All 4 screeners at 100% accuracy

**Estimated Time**:
- Phase A: 2 hours (same as Qullamaggie)
- Phase B: 4-6 hours (simpler than Qullamaggie - no ATR universe needed)
- **Total**: 6-8 hours

---

## COMPARISON TABLE

| Aspect | Qullamaggie | Stockbee |
|--------|-------------|----------|
| **Core Dependencies** | 3 (ticker_info, rs_data, atr_universe) | 2 (ticker_info+industry, rs_data) |
| **Available Now** | 1 of 3 | 1-2 of 2 (depends on industry field) |
| **Can Run Without RS** | ⚠️ Partial (3/5 filters) | ✅ Mostly (3/4 screeners 100%, 1/4 limited) |
| **Phase A Functionality** | 60% (3 of 5 filters) | 75-85% (3 of 4 screeners full, 1 limited) |
| **Phase B Complexity** | HIGH (RS + ATR universe) | MEDIUM (RS only) |
| **Implementation Time** | 8-10 hours | 6-8 hours |
| **Recommended Start** | Phase A | Phase A |

---

## RECOMMENDATIONS

### For Qullamaggie
1. ✅ **Start with Phase A** (2 hours)
   - 3 of 5 filters working
   - MA alignment is strong filter
   - Good enough for initial screening
2. ⏳ **Add Phase B later** (6-8 hours)
   - Build RS pipeline
   - Build ATR universe
   - Full 5/5 filter accuracy

### For Stockbee
1. ✅ **Start with Phase A** (2 hours)
   - **FIRST**: Verify industry field in ticker_universe_all.csv
   - If industry exists: 3/4 screeners at 100%, 1/4 at ~60%
   - If industry missing: 3/4 screeners at 100%, disable Industry Leaders
   - Still high value (momentum-based screeners work fully)
2. ⏳ **Add Phase B later** (4-6 hours)
   - Build RS pipeline (shared with Qullamaggie)
   - Enhance Industry Leaders with proper RS
   - Full 4/4 screener accuracy

---

## SHARED INFRASTRUCTURE OPPORTUNITY

### RS Calculation Pipeline

**Key Insight**: Both screeners need the same RS data!

**If Building RS Pipeline**:
- ✅ Build once, use for both screeners
- ✅ Calculate RS for all tickers, all periods
- ✅ Save to disk: `results/relative_strength/rs_daily_YYYYMMDD.csv`
- ✅ Both screeners load from same files
- ✅ Qullamaggie uses for RS ≥ 97 filter
- ✅ Stockbee uses for industry rankings and stock scoring

**Benefits**:
- Single implementation (6-8 hours)
- Both screeners get full accuracy
- No duplicate work
- Consistent RS calculations across all screeners

**Recommendation**: If implementing Phase B for either screener, build RS pipeline to benefit both!

---

## ACTION ITEMS

### Immediate (Before Phase A)
1. ✅ **Check industry field**:
   ```bash
   head -1 results/ticker_universes/ticker_universe_all.csv
   ```
   - If has `industry` column: ✅ proceed
   - If missing: decide on Industry Leaders screener strategy

### Phase A Implementation Priority
1. **Option 1**: Implement Stockbee first
   - Simpler (3 screeners work 100% without external data)
   - Faster validation
   - Less dependencies

2. **Option 2**: Implement Qullamaggie first
   - More sophisticated filtering (MA alignment is powerful)
   - Better for quality over quantity
   - Good test case for RS pipeline

3. **Option 3**: Implement both Phase A in parallel
   - Same timeframe (2 hours each = 4 hours total)
   - Both provide value immediately
   - Can test RS pipeline design with both use cases

### Phase B (Future)
1. Build shared RS calculation pipeline
2. Update both screeners to use RS data
3. Add ATR universe for Qullamaggie
4. Test and validate both at full accuracy

---

## FINAL ANSWER

**Does Stockbee need external data?**

**YES, but less critical than Qullamaggie**:

✅ **Can run 75% functionality WITHOUT external RS data**:
- 9M Movers: 100% functional
- 20% Weekly Movers: 100% functional
- 4% Daily Gainers: 100% functional
- Industry Leaders: ~60% functional (needs industry field, works better with RS)

⚠️ **Qullamaggie only 60% functional WITHOUT external data**:
- 3 of 5 filters work (market cap, MA alignment, range)
- 2 of 5 filters skipped/estimated (RS, ATR RS)

**Recommendation**: Stockbee is better candidate for Phase A implementation due to higher functionality without external data dependencies.

---

**Document Status**: ✅ ANALYSIS COMPLETE
**Date**: 2025-10-01
**Next Step**: Verify industry field, then choose implementation priority
