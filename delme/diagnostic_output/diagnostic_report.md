# EMA Overlay Diagnostic Test Report

Generated: 2025-09-22 12:15:14

## Test Objective
Reproduce and diagnose the user's reported issue: 'only a red line, a red label that says QQQ' with no EMA overlay.

## Test Scenarios

### Scenario 1
**Status**: success
**Description**: Working case with proper EMA_ema column naming
**Chart Path**: diagnostic_output/scenario_1_working_ema.png
**Data Keys**: ['Close', 'EMA_ema', 'metadata']
**Has EMA Column**: True

### Scenario 2
**Status**: success
**Description**: Failing case - missing EMA_ema column (reproduces user issue)
**Chart Path**: 
**Data Keys**: ['Close', 'metadata']
**Has EMA Column**: False
**⚠️ This reproduces the user's issue!**

### Scenario 3
**Status**: success
**Description**: Wrong column naming case
**Chart Path**: diagnostic_output/scenario_3_wrong_naming.png
**Data Keys**: ['Close', 'EMA', 'ema_10', 'moving_average', 'metadata']
**Has EMA Column**: False

### Scenario 4
**Status**: success
**Description**: Real SR system processing test
**Chart Path**: diagnostic_output/scenario_4_real_system.png
**Data Keys**: ['Close', 'EMA_ema', 'EMA_price', 'EMA_signals', 'metadata']
**Has EMA Column**: False

### Scenario 5
**Status**: success
**Description**: Enhanced overlay detection with multiple column patterns
**Chart Path**: diagnostic_output/scenario_5_enhanced_detection.png
**Data Keys**: ['Close', 'EMA_ema', 'ema_value', 'ma_10', 'indicator_ema', 'metadata']
**Has EMA Column**: False

## Key Findings

1. **Root Cause**: Missing `EMA_ema` column in panel data
2. **Expected Column**: The chart generation expects `EMA_ema` for overlays
3. **Failure Mode**: When `calculate_bundled_indicator()` fails, no overlay data is generated
4. **Result**: Only base ticker data is plotted (single red line with ticker label)

## Recommendations

1. Debug `calculate_bundled_indicator()` function in sr_market_data.py
2. Check EMA calculation in indicators/MAs.py
3. Verify indicator parsing in enhanced_panel_parser.py
4. Add fallback error handling for missing overlay data
5. Implement more robust column name detection
