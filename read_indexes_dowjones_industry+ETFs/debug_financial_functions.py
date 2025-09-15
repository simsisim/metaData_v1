import pandas as pd
import os
from src.config import user_choice, PARAMS_DIR

def debug_financial_functions():
    """Debug the exact issue with financial data functions"""
    
    print("🔧 DEBUGGING FINANCIAL FUNCTIONS")
    
    # Load the actual financial data file
    financial_data_file = os.path.join(PARAMS_DIR["TICKERS_DIR"], f'financial_data_{user_choice}.csv')
    
    if not os.path.exists(financial_data_file):
        print(f"❌ Financial data file not found: {financial_data_file}")
        return
    
    print(f"✅ Loading financial data from: {financial_data_file}")
    
    try:
        financial_df = pd.read_csv(financial_data_file)
        print(f"✅ Loaded DataFrame with shape: {financial_df.shape}")
        print(f"✅ Columns: {len(financial_df.columns)}")
        print(f"✅ First few columns: {list(financial_df.columns)[:10]}")
        
    except Exception as e:
        print(f"❌ Error loading financial data: {e}")
        return
    
    # Test summary function logic step by step
    print("\n" + "="*50)
    print("TESTING SUMMARY FUNCTION LOGIC")
    print("="*50)
    
    try:
        print("STEP 1: Testing basic operations...")
        
        # Test the first operations that happen in create_financial_data_summary
        print(f"len(financial_df): {len(financial_df)}")
        print(f"list(financial_df.columns): Working...")  # Don't print full list, just test
        
        print("STEP 2: Testing summary columns...")
        summary_columns = [
            'ticker', 'sector', 'industry', 'marketCap', 'earningsGrowth', 
            'revenueGrowth', 'earningsQuarterlyGrowth', 'revenueQuarterlyGrowth',
            'trailingPE', 'pegRatio', 'returnOnEquity', 'profitMargins',
            'shortPercentOfFloat', 'heldPercentInstitutions', 'canslim_score',
            'canslim_score_percentage', 'earnings_acceleration', 'revenue_acceleration'
        ]
        
        print("STEP 3: Testing column filtering...")
        existing_columns = [col for col in summary_columns if col in financial_df.columns]
        missing_columns = [col for col in summary_columns if col not in financial_df.columns]
        
        print(f"Found {len(existing_columns)} existing columns")
        print(f"Missing {len(missing_columns)} columns")
        print(f"Existing: {existing_columns}")
        print(f"Missing: {missing_columns}")
        
        print("STEP 4: Testing DataFrame creation...")
        if existing_columns:
            summary_df = financial_df[existing_columns].copy()
            print(f"✅ Created summary DataFrame with shape: {summary_df.shape}")
        else:
            summary_df = financial_df.copy()
            print(f"✅ Using full DataFrame with shape: {summary_df.shape}")
        
        print("STEP 5: Testing file save...")
        summary_file = os.path.join(PARAMS_DIR["TICKERS_DIR"], f'debug_summary_{user_choice}.csv')
        summary_df.to_csv(summary_file, index=False)
        
        if os.path.exists(summary_file):
            file_size = os.path.getsize(summary_file)
            print(f"✅ Debug summary file created: {summary_file} (Size: {file_size} bytes)")
        else:
            print(f"❌ Debug summary file NOT created")
            
    except Exception as e:
        print(f"❌ Error in summary logic: {e}")
        import traceback
        traceback.print_exc()
    
    # Test screened function logic step by step
    print("\n" + "="*50)
    print("TESTING SCREENED FUNCTION LOGIC")
    print("="*50)
    
    try:
        print("STEP 1: Testing DataFrame copy...")
        screened_df = financial_df.copy()
        original_count = len(screened_df)
        print(f"✅ Copied DataFrame, original count: {original_count}")
        
        print("STEP 2: Testing required columns check...")
        required_columns = ['earningsQuarterlyGrowth', 'earningsGrowth', 'shortPercentOfFloat', 'marketCap']
        
        for col in required_columns:
            if col in screened_df.columns:
                print(f"  ✅ Found column: {col}")
                # Test numeric conversion
                screened_df[col] = pd.to_numeric(screened_df[col], errors='coerce')
                valid_count = screened_df[col].notna().sum()
                print(f"     Valid numeric values: {valid_count}/{len(screened_df)}")
            else:
                print(f"  ❌ Missing column: {col}")
        
        print("STEP 3: Testing simple filtering...")
        # Simple filter test
        if 'marketCap' in screened_df.columns:
            screened_df['marketCap'] = pd.to_numeric(screened_df['marketCap'], errors='coerce')
            large_cap = screened_df[screened_df['marketCap'] > 300000000]
            print(f"  Stocks with market cap > $300M: {len(large_cap)}/{len(screened_df)}")
            screened_df = large_cap  # Use this as our screened result
        
        print("STEP 4: Testing screened file save...")
        screened_file = os.path.join(PARAMS_DIR["TICKERS_DIR"], f'debug_screened_{user_choice}.csv')
        screened_df.to_csv(screened_file, index=False)
        
        if os.path.exists(screened_file):
            file_size = os.path.getsize(screened_file)
            print(f"✅ Debug screened file created: {screened_file} (Size: {file_size} bytes)")
            print(f"✅ Screened {original_count} → {len(screened_df)} stocks")
        else:
            print(f"❌ Debug screened file NOT created")
            
    except Exception as e:
        print(f"❌ Error in screened logic: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🔧 DEBUGGING COMPLETED")

if __name__ == "__main__":
    debug_financial_functions()
