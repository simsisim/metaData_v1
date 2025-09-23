# SR CSV Parser Fix Implementation Plan

## Problem Summary

The `user_data_panel.csv` timeframe column was correctly removed as per the filename improvement plan, but the CSV parser in `sr_config_reader.py` was not updated to handle the new structure. This causes complete parsing failure and prevents chart generation.

## Current Issues

### **1. Header Detection Failure**
- **Problem**: Parser only looks for `#timeframe` or `timeframe` in header
- **Reality**: New header starts with `#file_name_id`
- **Result**: "No header line found in CSV" error

### **2. Column Misalignment**
- **Expected by Parser**: `[timeframe, file_name_id, Panel_1, Panel_2, ...]`
- **Actual CSV Structure**: `[file_name_id, Panel_1, Panel_2, Panel_3, ...]`
- **Result**: All data maps to wrong columns

### **3. Data Corruption**
- Parser gets `'QQQ_vs_SPY'` as timeframe (should be file_name_id)
- Panel_1 data gets assigned as file_name_id
- All subsequent panels shift incorrectly

## Required Changes

### **Change 1: Header Detection (`_read_csv_with_comment_headers`)**

**Current Logic (Line 54-58):**
```python
# Header line (starts with #timeframe or timeframe)
if line.startswith('#timeframe') or line.startswith('timeframe'):
    if header_line is None:  # Take first header found
        header_line = line.lstrip('#')  # Remove leading #
    continue
```

**New Logic:**
```python
# Header line detection - support both old and new formats
if (line.startswith('#timeframe') or line.startswith('timeframe') or
    line.startswith('#file_name_id') or line.startswith('file_name_id')):
    if header_line is None:  # Take first header found
        header_line = line.lstrip('#')  # Remove leading #
    continue
```

### **Change 2: Format Detection Enhancement**

Add format detection to determine if CSV uses old or new structure:

```python
def detect_csv_format(header_line: str) -> str:
    """
    Detect CSV format based on header structure.

    Returns:
        'old_format': timeframe,file_name_id,Panel_1,...
        'new_format': file_name_id,Panel_1,Panel_2,...
    """
    if header_line.startswith('timeframe') or 'timeframe,' in header_line:
        return 'old_format'
    elif header_line.startswith('file_name_id') or header_line.startswith('#file_name_id'):
        return 'new_format'
    else:
        # Fallback detection
        if 'timeframe' in header_line and 'file_name_id' in header_line:
            return 'old_format'
        else:
            return 'new_format'
```

### **Change 3: Single Row Parser Update (`_parse_single_row`)**

**Current Logic (Line 516-518):**
```python
# Get timeframe (usually first column)
timeframe = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else 'daily'
```

**New Logic:**
```python
# Determine format and extract data accordingly
csv_format = detect_csv_format(','.join(columns))

if csv_format == 'old_format':
    # Old format: timeframe, file_name_id, Panel_1, Panel_2, ...
    timeframe = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else 'daily'
    file_name_id = str(row.iloc[1]).strip() if len(row) > 1 and pd.notna(row.iloc[1]) else ''
    panel_start_idx = 2
else:
    # New format: file_name_id, Panel_1, Panel_2, Panel_3, ...
    file_name_id = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ''
    timeframe = get_global_timeframe()  # From user_data.csv settings
    panel_start_idx = 1
```

### **Change 4: Global Timeframe Detection**

Add function to get timeframe from global settings:

```python
def get_global_timeframe(user_config=None) -> str:
    """
    Get timeframe from global SR_timeframe_* settings in user_data.csv.

    Args:
        user_config: User configuration object (optional)

    Returns:
        Active timeframe string ('daily', 'weekly', 'monthly')
    """
    if user_config:
        if hasattr(user_config, 'sr_timeframe_daily') and user_config.sr_timeframe_daily:
            return 'daily'
        elif hasattr(user_config, 'sr_timeframe_weekly') and user_config.sr_timeframe_weekly:
            return 'weekly'
        elif hasattr(user_config, 'sr_timeframe_monthly') and user_config.sr_timeframe_monthly:
            return 'monthly'

    # Default fallback
    return 'daily'
```

### **Change 5: Panel Data Extraction Update**

Update panel processing to handle the column offset:

```python
for col_idx, col_name in enumerate(columns[panel_start_idx:], panel_start_idx):
    col_name_str = str(col_name).strip()

    # Panel data sources (Panel_1, Panel_2, etc.)
    if col_name_str.startswith('Panel_') and not col_name_str.endswith('_index'):
        if pd.notna(row.iloc[col_idx]) and str(row.iloc[col_idx]).strip():
            data_source = str(row.iloc[col_idx]).strip()
            if data_source not in ['', 'nan']:
                panel_data_sources[col_name_str] = data_source
```

## Implementation Steps

### **Step 1: Update Header Detection**
- Modify `_read_csv_with_comment_headers()` function
- Add support for `#file_name_id` header pattern
- Test with both old and new CSV formats

### **Step 2: Add Format Detection**
- Create `detect_csv_format()` helper function
- Integrate format detection into parsing flow
- Add comprehensive format validation

### **Step 3: Update Single Row Parser**
- Modify `_parse_single_row()` to handle both formats
- Add column offset logic based on detected format
- Integrate global timeframe detection

### **Step 4: Add Global Timeframe Support**
- Create `get_global_timeframe()` function
- Integrate with user configuration system
- Add fallback to 'daily' if no settings found

### **Step 5: Update Panel Processing**
- Adjust column indexing based on format
- Ensure panel data extraction works correctly
- Maintain indicator parsing functionality

## Testing Strategy

### **Test Case 1: New Format Compatibility**
```csv
#file_name_id,Panel_1,Panel_2,Panel_3,Panel_4,Panel_5,Panel_6,Panel_1_index,Panel_2_index,Panel_3_index,Panel_4_index,Panel_5_index,Panel_6_index
QQQ_vs_SPY,"QQQ + EMA(QQQ, 10)",SPY,SPY:QQQ,,,,"A_PPO(12,26,9)_for_(QQQ)",,,,,
```

**Expected Result:**
- Format detected: 'new_format'
- file_name_id: 'QQQ_vs_SPY'
- timeframe: 'daily' (from global config)
- Panel_1: 'QQQ + EMA(QQQ, 10)'
- Panel_2: 'SPY'
- Panel_3: 'SPY:QQQ'

### **Test Case 2: Old Format Backward Compatibility**
```csv
#timeframe,file_name_id,Panel_1,Panel_2,Panel_3,Panel_4,Panel_5,Panel_6,Panel_1_index,Panel_2_index,Panel_3_index,Panel_4_index,Panel_5_index,Panel_6_index
daily,QQQ_vs_SPY,"QQQ + EMA(QQQ, 10)",SPY,SPY:QQQ,,,,"A_PPO(12,26,9)_for_(QQQ)",,,,,
```

**Expected Result:**
- Format detected: 'old_format'
- timeframe: 'daily' (from CSV)
- file_name_id: 'QQQ_vs_SPY'
- Panel_1: 'QQQ + EMA(QQQ, 10)'
- Panel_2: 'SPY'
- Panel_3: 'SPY:QQQ'

### **Test Case 3: Multiple Rows**
Test both formats with multiple data rows to ensure row-by-row processing works correctly.

### **Test Case 4: Error Handling**
- Empty file_name_id (should generate default)
- Missing global timeframe settings (should default to 'daily')
- Malformed CSV structure (should provide clear error messages)

## Benefits After Implementation

### **Immediate Fixes:**
- ✅ CSV parsing works with new format
- ✅ Chart generation resumes functionality
- ✅ Proper panel configuration parsing
- ✅ file_name_id available for improved filenames

### **Enhanced Features:**
- ✅ Backward compatibility with old format
- ✅ Global timeframe configuration (cleaner design)
- ✅ Improved error messages and validation
- ✅ Support for filename improvements

### **Future-Proofing:**
- ✅ Flexible format detection system
- ✅ Easy to add new CSV structure variations
- ✅ Maintainable and extensible code

## Risk Mitigation

### **Backward Compatibility:**
- Old CSV format continues to work
- Gradual migration path available
- No breaking changes for existing users

### **Error Handling:**
- Clear error messages for format issues
- Graceful fallbacks for missing data
- Validation at multiple levels

### **Testing Coverage:**
- Comprehensive test cases for both formats
- Edge case validation
- Integration testing with full SR pipeline

## Conclusion

The CSV parser fix requires targeted updates to handle the new timeframe-less format while maintaining backward compatibility. The changes are focused and minimize risk while enabling the improved filename functionality.

**Priority: CRITICAL** - This fix is required for SR module functionality to resume.