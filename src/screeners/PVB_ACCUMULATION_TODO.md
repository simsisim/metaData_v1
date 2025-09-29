# PVB Screener Accumulation Pattern TODO

## Current Implementation Status

**PVB Screener currently uses STREAMING pattern, NOT true accumulation.**

### Current Pattern (Streaming - Potential Issues)
```python
# In PVBScreenerStreamingProcessor.process_batch_streaming():
batch_results = pvb_screener(batch_data, pvb_params)
formatted_results = [convert results]
self.append_results_to_csv(output_file, formatted_results)  # ❌ IMMEDIATE WRITE per batch
```

### Issues with Current Streaming Approach
1. **Potential Overwriting Risk**: If multiple parameter sets were added (like DRWISH semicolon approach)
2. **Memory Inefficiency**: No batch-wide memory optimization
3. **File Locking Dependencies**: Relies on `append_results_to_csv()` file locking
4. **No True Accumulation**: Results written immediately instead of accumulated

### Working Currently Because
- ✅ **Single parameter set** - no multiple sets competing for same file
- ✅ **Sequential processing** - batches processed one after another
- ✅ **Simple append logic** - basic file locking handles concurrency
- ✅ **No complex parameter variations** - straightforward single-output pattern

## Future Implementation Needed: True Accumulation Pattern

### Recommended Changes (When Time Permits)

**1. Create PVBScreenerProcessor (Accumulation-Based)**
```python
# Similar to DRWISHScreenerProcessor and ATR1ScreenerProcessor
class PVBScreenerProcessor:
    def process_pvb_batch(self, batch_data, timeframe, ticker_choice):
        # Accumulate results across batches
        self.all_results[timeframe][ticker] = result_row  # ✅ ACCUMULATE

    def save_pvb_matrix(self, ticker_choice):
        # Save accumulated results at end
        results_df.to_csv(output_path)  # ✅ BATCH WRITE
```

**2. Replace Streaming with Accumulation**
```python
# In main.py, replace:
pvb_screener_results = run_all_pvb_screener(config, user_config, timeframes, clean_file)

# With accumulation version:
pvb_screener_results = run_all_pvb_screener_accumulation(config, user_config, timeframes, clean_file)
```

**3. Benefits of Future Accumulation Implementation**
- **Multiple Parameter Sets Support**: Could add semicolon-separated parameters like DRWISH
- **Better Memory Management**: Comprehensive cleanup after processing
- **No Overwriting Issues**: True accumulation prevents file conflicts
- **Consistent Architecture**: Matches ATR1/DRWISH/Stage Analysis patterns

## Implementation Priority
**LOW PRIORITY** - Current streaming implementation works adequately for single parameter sets.

**IMPLEMENT WHEN:**
- Multiple PVB parameter sets are needed
- Performance optimization required
- Consistency with other accumulation processors desired
- File overwriting issues observed

## Reference Implementation
See `src/drwish_screener_processor.py` and `src/atr1_screener_processor.py` for proven accumulation pattern examples.

---
**Created**: 2025-09-29
**Status**: TODO - Future Enhancement
**Impact**: Low (current implementation works, but not optimal)