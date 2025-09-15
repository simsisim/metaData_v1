#!/usr/bin/env python3

import pandas as pd
import numpy as np

def manual_rs_calculation():
    """Manually calculate RS for a stock vs SPY and GLD to verify the calculation."""
    
    print("=== MANUAL RS CALCULATION ===\n")
    
    # Let's pick AAPL and manually calculate its RS vs SPY and GLD
    ticker = "AAPL"
    
    try:
        # Read price data
        print("1. Loading price data...")
        aapl_data = pd.read_csv(f"/home/imagda/_invest2024/python/metaData_v1/data/market_data/daily/{ticker}.csv")
        spy_data = pd.read_csv("/home/imagda/_invest2024/python/metaData_v1/data/market_data/daily/SPY.csv")
        gld_data = pd.read_csv("/home/imagda/_invest2024/python/metaData_v1/data/market_data/daily/GLD.csv")
        
        # Convert date columns
        aapl_data['Date'] = pd.to_datetime(aapl_data['Date'])
        spy_data['Date'] = pd.to_datetime(spy_data['Date'])  
        gld_data['Date'] = pd.to_datetime(gld_data['Date'])
        
        # Set date as index
        aapl_data.set_index('Date', inplace=True)
        spy_data.set_index('Date', inplace=True)
        gld_data.set_index('Date', inplace=True)
        
        print(f"AAPL data: {len(aapl_data)} rows")
        print(f"SPY data: {len(spy_data)} rows")
        print(f"GLD data: {len(gld_data)} rows")
        
        # Get most recent date (should be 2025-09-05 based on the PER file)
        target_date = pd.to_datetime('2025-09-05')
        
        print(f"\n2. Calculating 1-day returns for {target_date}...")
        
        # Get 1-day returns
        def calculate_return(data, target_date, days=1):
            if target_date not in data.index:
                print(f"Target date {target_date} not found in data")
                return None
            
            # Get the date 'days' before target_date (trading days)
            dates_before_target = data.index[data.index < target_date]
            if len(dates_before_target) < days:
                print(f"Not enough historical data for {days}-day return")
                return None
                
            start_date = dates_before_target[-days]
            
            start_price = data.loc[start_date, 'Close']
            end_price = data.loc[target_date, 'Close'] 
            
            return_pct = (end_price - start_price) / start_price
            
            print(f"  {data.index.name or 'Data'}: {start_date.date()} to {target_date.date()}")
            print(f"  Price: {start_price:.2f} -> {end_price:.2f}")
            print(f"  Return: {return_pct:.4f} ({return_pct*100:.2f}%)")
            
            return return_pct
        
        aapl_return = calculate_return(aapl_data, target_date, 1)
        spy_return = calculate_return(spy_data, target_date, 1)
        gld_return = calculate_return(gld_data, target_date, 1)
        
        if None in [aapl_return, spy_return, gld_return]:
            print("Could not calculate returns")
            return
        
        print(f"\n3. Calculating Relative Strength...")
        
        # RS = (1 + stock_return) / (1 + benchmark_return)
        rs_vs_spy = (1 + aapl_return) / (1 + spy_return)
        rs_vs_gld = (1 + aapl_return) / (1 + gld_return)
        
        print(f"AAPL RS vs SPY: {rs_vs_spy:.6f}")
        print(f"AAPL RS vs GLD: {rs_vs_gld:.6f}")
        print(f"Difference: {abs(rs_vs_spy - rs_vs_gld):.6f}")
        
        # Check what the system calculated
        print(f"\n4. Checking system RS calculations...")
        
        rs_file = "/home/imagda/_invest2024/python/metaData_v1/results/rs/rs_ibd_stocks_daily_0-5_20250909.csv"
        try:
            rs_data = pd.read_csv(rs_file)
            rs_data.set_index('ticker', inplace=True)
            
            if ticker in rs_data.index:
                # Look for 1-day RS columns
                spy_col = None
                gld_col = None
                
                for col in rs_data.columns:
                    if 'daily_1d_rs_vs_SPY' in col:
                        spy_col = col
                    elif 'daily_1d_rs_vs_GLD' in col:
                        gld_col = col
                
                if spy_col and gld_col:
                    system_rs_spy = rs_data.loc[ticker, spy_col]
                    system_rs_gld = rs_data.loc[ticker, gld_col]
                    
                    print(f"System RS vs SPY: {system_rs_spy:.6f}")
                    print(f"System RS vs GLD: {system_rs_gld:.6f}")
                    print(f"System difference: {abs(system_rs_spy - system_rs_gld):.6f}")
                    
                    print(f"\nManual vs System comparison:")
                    print(f"SPY - Manual: {rs_vs_spy:.6f}, System: {system_rs_spy:.6f}, Diff: {abs(rs_vs_spy - system_rs_spy):.6f}")
                    print(f"GLD - Manual: {rs_vs_gld:.6f}, System: {system_rs_gld:.6f}, Diff: {abs(rs_vs_gld - system_rs_gld):.6f}")
                    
                    # If RS values are different but percentiles are same, investigate percentile calc
                    if abs(system_rs_spy - system_rs_gld) > 0.001:  # Significant difference
                        print(f"\n5. Investigating percentile calculation with different RS values...")
                        
                        # Load all RS data for NASDAQ100 universe
                        spy_rs_col = spy_col
                        gld_rs_col = gld_col
                        
                        # Filter to NASDAQ100 universe
                        nasdaq100_file = "/home/imagda/_invest2024/python/metaData_v1/results/ticker_universes/ticker_universe_NASDAQ100.csv"
                        nasdaq100_data = pd.read_csv(nasdaq100_file)
                        nasdaq100_tickers = nasdaq100_data['ticker'].tolist()
                        
                        # Get RS values for NASDAQ100 universe
                        nasdaq100_mask = rs_data.index.isin(nasdaq100_tickers)
                        nasdaq100_rs = rs_data[nasdaq100_mask]
                        
                        spy_rs_values = nasdaq100_rs[spy_rs_col].dropna()
                        gld_rs_values = nasdaq100_rs[gld_rs_col].dropna()
                        
                        print(f"NASDAQ100 universe size: {len(spy_rs_values)} stocks with SPY RS, {len(gld_rs_values)} with GLD RS")
                        
                        # Calculate percentiles manually
                        spy_percentiles = spy_rs_values.rank(pct=True, na_option='keep')
                        spy_percentiles = (spy_percentiles * 98) + 1
                        spy_percentiles = spy_percentiles.round().astype('Int64')
                        
                        gld_percentiles = gld_rs_values.rank(pct=True, na_option='keep')
                        gld_percentiles = (gld_percentiles * 98) + 1
                        gld_percentiles = gld_percentiles.round().astype('Int64')
                        
                        if ticker in spy_percentiles.index and ticker in gld_percentiles.index:
                            aapl_spy_per = spy_percentiles[ticker]
                            aapl_gld_per = gld_percentiles[ticker]
                            
                            print(f"Manual percentile calculation:")
                            print(f"AAPL SPY percentile: {aapl_spy_per}")
                            print(f"AAPL GLD percentile: {aapl_gld_per}")
                            print(f"Identical percentiles: {aapl_spy_per == aapl_gld_per}")
                            
                            # Show AAPL's rank position
                            aapl_spy_rank_pos = list(spy_rs_values.sort_values().index).index(ticker) + 1
                            aapl_gld_rank_pos = list(gld_rs_values.sort_values().index).index(ticker) + 1
                            
                            print(f"AAPL rank position in SPY ranking: {aapl_spy_rank_pos}/{len(spy_rs_values)}")
                            print(f"AAPL rank position in GLD ranking: {aapl_gld_rank_pos}/{len(gld_rs_values)}")
                        
                else:
                    print("Could not find RS columns in system data")
            else:
                print(f"{ticker} not found in system RS data")
                
        except Exception as e:
            print(f"Could not load system RS data: {e}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    manual_rs_calculation()