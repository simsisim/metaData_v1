#!/usr/bin/env python3

import pandas as pd
import numpy as np

def manual_percentile_calculation():
    """Manually calculate percentiles for NASDAQ100 vs SPY to verify system."""
    
    print("=== MANUAL PERCENTILE CALCULATION (NASDAQ100 vs SPY) ===\n")
    
    try:
        print("1. Loading RS data...")
        rs_file = "/home/imagda/_invest2024/python/metaData_v1/results/rs/rs_ibd_stocks_daily_0-5_20250909.csv"
        rs_data = pd.read_csv(rs_file)
        rs_data.set_index('ticker', inplace=True)
        
        print("2. Loading NASDAQ100 universe...")
        nasdaq100_file = "/home/imagda/_invest2024/python/metaData_v1/results/ticker_universes/ticker_universe_NASDAQ100.csv"
        nasdaq100_data = pd.read_csv(nasdaq100_file)
        nasdaq100_tickers = nasdaq100_data['ticker'].tolist()
        
        print(f"NASDAQ100 universe: {len(nasdaq100_tickers)} stocks")
        
        print("3. Finding SPY RS column...")
        spy_rs_col = None
        for col in rs_data.columns:
            if 'daily_1d_rs_vs_SPY' in col:
                spy_rs_col = col
                break
        
        if not spy_rs_col:
            print("ERROR: Could not find SPY RS column!")
            return
            
        print(f"Found SPY RS column: {spy_rs_col}")
        
        print("4. Filtering to NASDAQ100 universe...")
        nasdaq100_mask = rs_data.index.isin(nasdaq100_tickers)
        nasdaq100_rs = rs_data[nasdaq100_mask]
        
        # Get RS values for SPY
        spy_rs_values = nasdaq100_rs[spy_rs_col].dropna()
        
        print(f"NASDAQ100 stocks with SPY RS data: {len(spy_rs_values)}")
        
        print("5. Manual percentile calculation...")
        
        # Sort RS values to see ranking
        sorted_rs = spy_rs_values.sort_values()
        
        print("Bottom 10 stocks (lowest RS):")
        for i, (ticker, rs_val) in enumerate(sorted_rs.head(10).items()):
            print(f"  {i+1:2d}. {ticker}: {rs_val:.6f}")
        
        print("\nTop 10 stocks (highest RS):")
        for i, (ticker, rs_val) in enumerate(sorted_rs.tail(10).items()):
            rank = len(sorted_rs) - 9 + i
            print(f"  {rank:2d}. {ticker}: {rs_val:.6f}")
        
        # Manual percentile calculation using pandas rank
        print(f"\n6. Calculating percentiles using pandas rank()...")
        
        # This is exactly the same as the system uses
        percentiles = spy_rs_values.rank(pct=True, na_option='keep')
        percentiles_ibd = (percentiles * 98) + 1
        percentiles_final = percentiles_ibd.round().astype('Int64')
        
        print("7. Results for key stocks:")
        
        key_stocks = ['AAPL', 'AMZN', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        
        for ticker in key_stocks:
            if ticker in spy_rs_values.index:
                rs_val = spy_rs_values[ticker]
                rank_pct = percentiles[ticker]
                percentile = percentiles_final[ticker]
                
                # Find position in sorted list
                position = list(sorted_rs.index).index(ticker) + 1
                total = len(sorted_rs)
                
                print(f"{ticker:5s}: RS={rs_val:.6f}, Position={position:2d}/{total}, Rank%={rank_pct:.3f}, Percentile={percentile:2d}")
        
        print(f"\n8. Comparing with system percentiles...")
        
        # Load system percentile results
        per_file = "/home/imagda/_invest2024/python/metaData_v1/results/per/per_ibd_stocks_daily_0-5_20250909.csv"
        per_data = pd.read_csv(per_file)
        per_data.set_index('ticker', inplace=True)
        
        # Find SPY percentile column
        spy_per_col = None
        for col in per_data.columns:
            if 'daily_1d_percentile_NASDAQ100_vs_SPY' in col:
                spy_per_col = col
                break
        
        if spy_per_col:
            print(f"Found system percentile column: {spy_per_col}")
            
            print("\nComparison (Manual vs System):")
            
            discrepancies = 0
            total_compared = 0
            
            for ticker in key_stocks:
                if ticker in percentiles_final.index and ticker in per_data.index:
                    manual_per = percentiles_final[ticker]
                    system_per = per_data.loc[ticker, spy_per_col]
                    
                    match = "✓" if manual_per == system_per else "✗"
                    if manual_per != system_per:
                        discrepancies += 1
                    total_compared += 1
                    
                    print(f"{ticker:5s}: Manual={manual_per:2d}, System={system_per:2.0f}, {match}")
            
            # Check all stocks for discrepancies
            print(f"\nFull comparison across all NASDAQ100 stocks:")
            
            all_discrepancies = 0
            all_compared = 0
            
            for ticker in spy_rs_values.index:
                if ticker in per_data.index:
                    manual_per = percentiles_final[ticker]
                    system_per = per_data.loc[ticker, spy_per_col]
                    
                    all_compared += 1
                    if pd.notna(system_per):
                        if manual_per != int(system_per):
                            all_discrepancies += 1
                            if all_discrepancies <= 5:  # Show first 5 examples
                                print(f"  DISCREPANCY: {ticker} Manual={manual_per}, System={system_per}")
            
            print(f"Total discrepancies: {all_discrepancies}/{all_compared} stocks")
            
            if all_discrepancies == 0:
                print("✓ MANUAL CALCULATION MATCHES SYSTEM - No bug in percentile calculation")
            else:
                print("✗ DISCREPANCIES FOUND - Possible bug in system")
        
        else:
            print("Could not find system percentile column")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    manual_percentile_calculation()