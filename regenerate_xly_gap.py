#!/usr/bin/env python3
"""
Regenerate XLY_gap.csv with Column Renaming Fix
==============================================

Apply the modified gap calculation to create new XLY_gap.csv with:
- Close column containing gap-adjusted values
- Close_original preserving original Close values
- Gap column for reference
"""

import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def regenerate_xly_gap():
    """Regenerate XLY_gap.csv with the column renaming fix."""

    print("=" * 70)
    print("REGENERATING XLY_GAP.CSV WITH COLUMN RENAMING FIX")
    print("=" * 70)

    # File paths
    data_dir = Path("../downloadData_v1/data/market_data/daily")
    original_xly = data_dir / "XLY.csv"
    gap_file = data_dir / "XLY_gap.csv"
    backup_file = data_dir / "XLY_gap_backup.csv"

    # Step 1: Backup existing gap file
    print("\n1. Backing up existing XLY_gap.csv...")
    if gap_file.exists():
        shutil.copy2(gap_file, backup_file)
        print(f"   ✅ Backup created: {backup_file}")
    else:
        print("   ⚠️  No existing gap file to backup")

    # Step 2: Load original XLY data
    print("\n2. Loading original XLY data...")
    if not original_xly.exists():
        print(f"   ❌ Original XLY file not found: {original_xly}")
        return False

    try:
        # Load using same format as DataReader
        df = pd.read_csv(original_xly, index_col='Date', parse_dates=False)
        df.index = df.index.str.split(' ').str[0]
        df.index = pd.to_datetime(df.index)

        # Ensure timezone-naive datetime index
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Remove rows with invalid dates
        df = df[df.index.notna()]

        # Sort by date
        df = df.sort_index()

        # Filter to business days only
        df = df[df.index.weekday < 5]

        print(f"   ✅ Loaded {len(df)} rows from original XLY data")
        print(f"   Date range: {df.index.min()} to {df.index.max()}")

    except Exception as e:
        print(f"   ❌ Error loading XLY data: {e}")
        return False

    # Step 3: Apply modified gap calculation
    print("\n3. Applying modified gap calculation with column renaming...")

    def calculate_gaps_with_column_renaming(df):
        """
        Apply the new gap calculation logic from modified mmm_gaps.py
        """
        try:
            # Create copy to avoid modifying original data
            result_df = df.copy()

            # Ensure data is sorted by date
            result_df = result_df.sort_index()

            # Get previous close by shifting 1 day
            prev_close = result_df['Close'].shift(1)
            current_open = result_df['Open']
            current_close = result_df['Close']

            # Calculate gap: Open[i] - Close[i-1]
            gap = current_open - prev_close

            # Calculate AdjustClose_woGap: Close[i] - (Close[i-1] - Open[i])
            adjust_close_wo_gap = current_close - (prev_close - current_open)

            # COLUMN RENAMING FOR GAP ANALYSIS:
            # Preserve original Close for reference
            result_df['Close_original'] = result_df['Close']

            # Replace Close column with gap-adjusted values for analysis
            result_df['Close'] = adjust_close_wo_gap

            # Add gap column for reference
            result_df['gap'] = gap

            logger.info(f"Gap calculations completed: {len(result_df)} rows processed")
            logger.info("Applied column renaming: Close -> Close_original, AdjustClose_woGap -> Close")

            return result_df

        except Exception as e:
            logger.error(f"Error in gap calculations: {e}")
            raise e

    try:
        gap_df = calculate_gaps_with_column_renaming(df)
        print(f"   ✅ Gap calculation successful: {len(gap_df)} rows")

        # Verify the calculation
        non_null_gaps = gap_df['gap'].notna().sum()
        print(f"   Gap data: {non_null_gaps}/{len(gap_df)} rows have valid gaps")

    except Exception as e:
        print(f"   ❌ Error in gap calculation: {e}")
        return False

    # Step 4: Show transformation results
    print("\n4. Verification of column renaming...")

    print("   Recent data transformation (last 5 rows):")
    print("   Date                Close_original    Gap        New_Close")
    print("   " + "-" * 70)

    recent_data = gap_df.tail(5)
    for date, row in recent_data.iterrows():
        orig_close = row['Close_original']
        gap_val = row['gap'] if pd.notna(row['gap']) else 0.0
        new_close = row['Close']
        print(f"   {date.strftime('%Y-%m-%d')}    {orig_close:>10.2f}    {gap_val:>6.2f}    {new_close:>10.2f}")

    # Verify mathematical relationship
    print("\n   Mathematical verification (last 3 rows):")
    test_data = gap_df.dropna(subset=['gap']).tail(3)
    all_correct = True

    for date, row in test_data.iterrows():
        orig_close = row['Close_original']
        gap = row['gap']
        new_close = row['Close']
        expected = orig_close + gap
        diff = abs(new_close - expected)

        status = "✅" if diff < 1e-6 else "❌"
        if diff >= 1e-6:
            all_correct = False

        print(f"   {status} {orig_close:.2f} + {gap:.2f} = {new_close:.2f} (diff: {diff:.8f})")

    if not all_correct:
        print("   ❌ Mathematical verification failed!")
        return False

    print("   ✅ Mathematical verification passed!")

    # Step 5: Save the new gap file
    print("\n5. Saving new XLY_gap.csv...")

    try:
        gap_df.to_csv(gap_file, index=True)
        print(f"   ✅ New XLY_gap.csv saved: {gap_file}")

        # Verify file was saved correctly
        test_read = pd.read_csv(gap_file, index_col='Date', parse_dates=True)
        print(f"   ✅ File verification: {len(test_read)} rows, {len(test_read.columns)} columns")
        print(f"   New file columns: {list(test_read.columns)}")

    except Exception as e:
        print(f"   ❌ Error saving gap file: {e}")
        return False

    # Step 6: Compare with backup
    print("\n6. Comparing with backup file...")

    if backup_file.exists():
        try:
            backup_df = pd.read_csv(backup_file, index_col='Date', parse_dates=True)

            print("   Comparison summary:")
            print(f"   Backup file:  {len(backup_df)} rows, columns: {list(backup_df.columns)}")
            print(f"   New file:     {len(gap_df)} rows, columns: {list(gap_df.columns)}")

            # Check if Close values changed
            if 'Close' in backup_df.columns:
                common_dates = backup_df.index.intersection(gap_df.index)[-3:]

                print("\n   Close column comparison (last 3 common dates):")
                print("   Date                Backup_Close      New_Close")
                print("   " + "-" * 55)

                for date in common_dates:
                    backup_close = backup_df.loc[date, 'Close']
                    new_close = gap_df.loc[date, 'Close']
                    print(f"   {date.strftime('%Y-%m-%d')}    {backup_close:>10.2f}    {new_close:>10.2f}")

                if not np.allclose(backup_df.loc[common_dates, 'Close'].values,
                                 gap_df.loc[common_dates, 'Close'].values, rtol=1e-3):
                    print("   ✅ Close values successfully changed to gap-adjusted data!")
                else:
                    print("   ⚠️  Close values appear unchanged")

        except Exception as e:
            print(f"   ⚠️  Could not compare with backup: {e}")

    print("\n" + "=" * 70)
    print("SUCCESS! XLY_GAP.CSV REGENERATED WITH COLUMN RENAMING")
    print("=" * 70)

    print("🎯 CHANGES APPLIED:")
    print("   ✅ Close column now contains gap-adjusted values")
    print("   ✅ Close_original preserves original closing prices")
    print("   ✅ Gap column available for reference")
    print("   ✅ File structure compatible with DataReader")

    print("\n🚀 EXPECTED RESULT:")
    print("   When MMM Panel_2 displays XLY_gap:")
    print("   - Charts will show gap-adjusted closing prices")
    print("   - Gap analysis data will be properly visualized")
    print("   - Original data remains available in Close_original column")

    print("\n📝 NEXT STEPS:")
    print("   1. Test MMM chart generation with new XLY_gap.csv")
    print("   2. Verify Panel_2 displays gap analysis correctly")
    print("   3. Apply same fix to other gap analysis tickers if needed")

    return True

if __name__ == "__main__":
    success = regenerate_xly_gap()
    if success:
        print("\n🎉 XLY_gap regeneration completed successfully!")
    else:
        print("\n❌ XLY_gap regeneration failed!")