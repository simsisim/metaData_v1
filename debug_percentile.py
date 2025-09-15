#!/usr/bin/env python3

import pandas as pd
import numpy as np

def debug_percentile_calculation():
    """Debug why different RS values produce identical percentiles."""
    
    print("=== PERCENTILE CALCULATION DEBUG ===\n")
    
    # Let's first examine what the issue might be by looking at both RS files
    print("1. Reading RS data from both benchmark tests...")
    
    try:
        # Read QQQ benchmark RS data
        qqq_rs_file = "/home/imagda/_invest2024/python/metaData_v1/results_BM=QQQ_universe=NASDAQ100/rs/rs_ibd_stocks_daily_0-5_20250909.csv"
        qqq_rs_df = pd.read_csv(qqq_rs_file)
        qqq_rs_df.set_index('ticker', inplace=True)
        
        # Read SPY benchmark RS data  
        spy_rs_file = "/home/imagda/_invest2024/python/metaData_v1/results_BM=SPY_universe=NASDAQ100/rs/rs_ibd_stocks_daily_0-5_20250909.csv"
        spy_rs_df = pd.read_csv(spy_rs_file)
        spy_rs_df.set_index('ticker', inplace=True)
        
        print(f"QQQ RS data shape: {qqq_rs_df.shape}")
        print(f"SPY RS data shape: {spy_rs_df.shape}")
        
        # Focus on 1-day RS values
        qqq_1d_rs = qqq_rs_df['daily_daily_daily_1d_rs_vs_QQQ'].dropna()
        spy_1d_rs = spy_rs_df['daily_daily_daily_1d_rs_vs_SPY'].dropna()
        
        print(f"\nQQQ 1d RS values count: {len(qqq_1d_rs)}")
        print(f"SPY 1d RS values count: {len(spy_1d_rs)}")
        
        # Check if we have the same tickers
        common_tickers = set(qqq_1d_rs.index) & set(spy_1d_rs.index)
        print(f"Common tickers: {len(common_tickers)}")
        
        print("\n2. Examining AAPL specifically:")
        if 'AAPL' in qqq_1d_rs.index and 'AAPL' in spy_1d_rs.index:
            aapl_qqq_rs = qqq_1d_rs['AAPL']
            aapl_spy_rs = spy_1d_rs['AAPL']
            print(f"AAPL QQQ RS: {aapl_qqq_rs}")
            print(f"AAPL SPY RS: {aapl_spy_rs}")
            print(f"Different values: {aapl_qqq_rs != aapl_spy_rs}")
        
        print("\n3. Testing percentile calculation with both datasets:")
        
        # Calculate percentiles using the same method as per_processor.py
        def calculate_percentiles(rs_values, label):
            print(f"\n--- {label} ---")
            print(f"Input RS values (first 10): {rs_values.head(10).values}")
            
            # This is the exact same logic from per_processor.py line 322-324
            percentiles = rs_values.rank(pct=True, na_option='keep')
            percentiles = (percentiles * 98) + 1
            percentiles = percentiles.round().astype('Int64')
            
            print(f"Calculated percentiles (first 10): {percentiles.head(10).values}")
            
            if 'AAPL' in percentiles.index:
                aapl_rank = rs_values.rank(pct=True, na_option='keep')['AAPL']
                aapl_percentile = percentiles['AAPL'] 
                aapl_rs = rs_values['AAPL']
                print(f"AAPL: RS={aapl_rs:.4f}, Rank={aapl_rank:.4f}, Percentile={aapl_percentile}")
                
                # Show AAPL's position in sorted data
                sorted_rs = rs_values.sort_values()
                aapl_position = list(sorted_rs.index).index('AAPL') + 1
                total_count = len(sorted_rs)
                print(f"AAPL position in sorted order: {aapl_position}/{total_count}")
            
            return percentiles
        
        qqq_percentiles = calculate_percentiles(qqq_1d_rs, "QQQ Benchmark")
        spy_percentiles = calculate_percentiles(spy_1d_rs, "SPY Benchmark")
        
        print("\n4. Comparing percentile results:")
        if 'AAPL' in qqq_percentiles.index and 'AAPL' in spy_percentiles.index:
            print(f"AAPL QQQ percentile: {qqq_percentiles['AAPL']}")
            print(f"AAPL SPY percentile: {spy_percentiles['AAPL']}")
            print(f"Identical: {qqq_percentiles['AAPL'] == spy_percentiles['AAPL']}")
        
        # Compare all common tickers
        print(f"\n5. Checking all common tickers for identical percentiles:")
        identical_count = 0
        total_compared = 0
        
        for ticker in common_tickers:
            if ticker in qqq_percentiles.index and ticker in spy_percentiles.index:
                qqq_per = qqq_percentiles[ticker]
                spy_per = spy_percentiles[ticker]
                if pd.notna(qqq_per) and pd.notna(spy_per):
                    total_compared += 1
                    if qqq_per == spy_per:
                        identical_count += 1
                        if total_compared <= 10:  # Show first 10 examples
                            print(f"{ticker}: QQQ={qqq_per}, SPY={spy_per} *** IDENTICAL")
                    else:
                        if total_compared <= 10:  # Show first 10 examples  
                            print(f"{ticker}: QQQ={qqq_per}, SPY={spy_per}")
        
        print(f"\nSUMMARY: {identical_count}/{total_compared} tickers have identical percentiles ({100*identical_count/total_compared:.1f}%)")
        
        # Let's check if the RS values themselves have identical ordering
        print(f"\n6. Testing if RS value ordering is identical:")
        
        # Get RS values for common tickers
        common_qqq = qqq_1d_rs[qqq_1d_rs.index.isin(common_tickers)].sort_index()
        common_spy = spy_1d_rs[spy_1d_rs.index.isin(common_tickers)].sort_index()
        
        # Sort by RS values and check if ordering is identical
        qqq_sorted_tickers = common_qqq.sort_values().index.tolist()
        spy_sorted_tickers = common_spy.sort_values().index.tolist()
        
        print(f"QQQ sorted order (first 10): {qqq_sorted_tickers[:10]}")
        print(f"SPY sorted order (first 10): {spy_sorted_tickers[:10]}")
        print(f"Identical ordering: {qqq_sorted_tickers == spy_sorted_tickers}")
        
        if qqq_sorted_tickers == spy_sorted_tickers:
            print("*** ROOT CAUSE: RS values have identical relative ordering despite different absolute values!")
            print("This means rank() produces identical results even though input values differ.")
        else:
            print(f"Orderings differ in {sum(a != b for a, b in zip(qqq_sorted_tickers, spy_sorted_tickers))} positions")
        
        # Calculate correlation between QQQ and SPY RS values
        print(f"\n7. Correlation analysis:")
        correlation = common_qqq.corr(common_spy)
        print(f"Correlation between QQQ and SPY RS values: {correlation:.6f}")
        
        # Check if high correlation explains the identical rankings
        if correlation > 0.99:
            print("*** HIGH CORRELATION: QQQ and SPY RS values are extremely correlated")
            print("This could explain why percentile rankings are nearly identical")
        elif correlation > 0.95:
            print("*** MODERATE CORRELATION: This could partially explain similar rankings")
        else:
            print("*** LOW CORRELATION: This suggests a potential bug in calculation")
        
        # Show some specific examples where percentiles differ
        print(f"\n8. Examples where percentiles DO differ:")
        differ_count = 0
        for ticker in common_tickers:
            if ticker in qqq_percentiles.index and ticker in spy_percentiles.index:
                qqq_per = qqq_percentiles[ticker]
                spy_per = spy_percentiles[ticker]
                if pd.notna(qqq_per) and pd.notna(spy_per) and qqq_per != spy_per:
                    differ_count += 1
                    if differ_count <= 5:  # Show first 5 examples
                        qqq_rs = qqq_1d_rs[ticker]
                        spy_rs = spy_1d_rs[ticker] 
                        print(f"{ticker}: QQQ RS={qqq_rs:.4f} (per={qqq_per}), SPY RS={spy_rs:.4f} (per={spy_per})")
        
        print(f"Total tickers with different percentiles: {differ_count}")
        
    except Exception as e:
        print(f"Error during debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_percentile_calculation()