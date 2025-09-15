# atr_cloud.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import os


def calculate_true_range(df):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range

def calculate_atr(df, length=14, smoothing='RMA'):
    true_range = calculate_true_range(df)
    
    if smoothing == 'RMA':
        atr = true_range.ewm(alpha=1/length, min_periods=length).mean()
    elif smoothing == 'SMA':
        atr = true_range.rolling(window=length).mean()
    elif smoothing == 'EMA':
        atr = true_range.ewm(span=length, adjust=False).mean()
    elif smoothing == 'WMA':
        weights = np.arange(1, length + 1)
        atr = true_range.rolling(window=length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    else:
        raise ValueError("Invalid smoothing method. Choose 'RMA', 'SMA', 'EMA', or 'WMA'.")
    
    return atr

def vol_stop(df, src_column, length, factor):
    df['TR'] = calculate_true_range(df)
    df['ATR'] = calculate_atr(df, length)
    df['src'] = df[src_column]
    df['max'] = df['src'].copy()
    df['min'] = df['src'].copy()
    df['stop'] = 0.0
    df['uptrend'] = True

    for i in range(1, len(df)):
        if np.isnan(df['ATR'].iloc[i]):
            df.loc[df.index[i], 'atr_m'] = df['TR'].iloc[i]  # Fallback to True Range
        else:
            df.loc[df.index[i], 'atr_m'] = df['ATR'].iloc[i] * factor
        # Reset max and min
        # Update max and min
        df.loc[df.index[i], 'max'] = max(df['max'].iloc[i-1], df['src'].iloc[i])
        df.loc[df.index[i], 'min'] = min(df['min'].iloc[i-1], df['src'].iloc[i])
        # Pine Script:
        # stop := nz(uptrend ? math.max(stop, max - atrM) : math.min(stop, min + atrM), src)
        # Calculate new stop
        if df['uptrend'].iloc[i-1]:
            new_stop = max(df['stop'].iloc[i-1], df['max'].iloc[i] - df['atr_m'].iloc[i])
        else:
            new_stop = min(df['stop'].iloc[i-1], df['min'].iloc[i] + df['atr_m'].iloc[i])

        if np.isnan(new_stop):
            df.loc[df.index[i], 'stop'] = df['src'].iloc[i]
        else:
            df.loc[df.index[i], 'stop'] = int(round(new_stop))
        # Calculate current uptrend
        uptrend = df['src'].iloc[i] - df['stop'].iloc[i-1] >= 0.0
        # Get previous uptrend (use True for the first row)
        # Update uptrend
        df.loc[df.index[i], 'uptrend'] = df['src'].iloc[i] - df['stop'].iloc[i] >= 0.0

        # Check for trend reversal
        if df['uptrend'].iloc[i] != df['uptrend'].iloc[i-1]:
            df.loc[df.index[i], 'max'] = df['src'].iloc[i]
            df.loc[df.index[i], 'min'] = df['src'].iloc[i]
            df.loc[df.index[i], 'stop'] = df['src'].iloc[i] - df['atr_m'].iloc[i] if df['uptrend'].iloc[i] else df['src'].iloc[i] + df['atr_m'].iloc[i]

    return df[['stop', 'uptrend']]

def calculate_atr_cloud(df, length=20, factor=3.0, length2=20, factor2=1.5, src='Close', src2='Close'):
    v_stop = vol_stop(df, src, length, factor)
    v_stop2 = vol_stop(df, src2, length2, factor2)
    
    df['vStop'] = v_stop['stop']
    df['uptrend'] = v_stop['uptrend']
    df['vStop2'] = v_stop2['stop']
    df['uptrend2'] = v_stop2['uptrend']
    
    df['vstopseries'] = (df['vStop'] + df['vStop2']) / 2
    
    df['crossUp'] = (df['vStop2'] > df['vStop']) & (df['vStop2'].shift(1) <= df['vStop'].shift(1))
    df['crossDn'] = (df['vStop2'] < df['vStop']) & (df['vStop2'].shift(1) >= df['vStop'].shift(1))
    
    return df

def apply_atr_cloud(df, length=20, factor=3.0, length2=20, factor2=1.5, src='Close', src2='Close'):
    return calculate_atr_cloud(df, length, factor, length2, factor2, src, src2)

def generate_alerts(df):
    long_alerts = df[df['crossUp']].index
    short_alerts = df[df['crossDn']].index
    return long_alerts, short_alerts

def plot_atr_cloud(df, symbol, output_folder):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Price', color='black')
    plt.plot(df.index, df['vStop'], label='VStop', color='blue')
    plt.plot(df.index, df['vStop2'], label='VStop2', color='red')
    plt.plot(df.index, df['vstopseries'], label='VStop Series', color='green', alpha=0.7)
    
    plt.title(f'{symbol} - Price and ATR Cloud')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    
    plt.savefig(f"{output_folder}/{symbol}_atr_cloud.png")
    plt.close()

def run_atr_cloud_strategy(portfolio_data, atr_params):
    results = []
    length = atr_params['vstop_length']
    factor = atr_params['vstop_factor']
    length2 = atr_params['vstop_length2']
    factor2 = atr_params['vstop_factor2']
    src = atr_params['src']
    src2 = atr_params['src2']
    ATR_filename = atr_params['ATR_filename']
    ATR_dir = atr_params['ATR_dir']
    results = []
    
    for symbol, df in portfolio_data.items():
        df = calculate_atr_cloud(df, length, factor, length2, factor2, src, src2)
        
        # Print out lines where VStop values exceed 30
        high_vstop_mask = (df['vStop'] > 30) | (df['vStop2'] > 30)
        #if high_vstop_mask.any():
        #    print(f"\nHigh VStop values for {symbol}:")
        #    print(df.loc[high_vstop_mask, ['Close', 'vStop', 'vStop2', 'ATR']].tail(50))
        
        long_alerts, short_alerts = generate_alerts(df)
        
        # Plot the chart
        plot_atr_cloud(df, symbol, ATR_dir)
        
        # Get the last signal
        last_signal = df[df['crossUp'] | df['crossDn']].iloc[-1] if not df[df['crossUp'] | df['crossDn']].empty else None
        
        if last_signal is not None:
            current_price = df['Close'].iloc[-1]
            price_change = ((current_price - last_signal['Close']) / last_signal['Close']) * 100
            days_since_signal = (df.index[-1] - last_signal.name).days
            
            results.append({
                'Symbol': symbol,
                'Signal Date': last_signal.name.strftime('%Y-%m-%d'),
                'Current Price': round(current_price, 2),
                'Signal': 'Long' if last_signal['crossUp'] else 'Short',
                'vStop': round(last_signal['vStop'], 2),
                'vStop2': round(last_signal['vStop2'], 2),
                'vstopseries': round(last_signal['vstopseries'], 2),
                'Price Change %': round(price_change, 2),
                'Days Since Signal': days_since_signal
            })
        else:
            results.append({
                'Symbol': symbol,
                'Signal Date': 'N/A',
                'Current Price': round(df['Close'].iloc[-1], 2),
                'Signal': 'No Signal',
                'vStop': round(df['vStop'].iloc[-1], 2),
                'vStop2': round(df['vStop2'].iloc[-1], 2),
                'vstopseries': round(df['vstopseries'].iloc[-1], 2),
                'Price Change %': 'N/A',
                'Days Since Signal': 'N/A'
            })
    
    results_df = pd.DataFrame(results)
    #results_df.to_csv(ATR_filename, index=False)
    
    # If the file exists, append without header. If not, create with header.
    if os.path.exists(ATR_filename):
        results_df.to_csv(ATR_filename, mode='a', header=False, index=False)
    else:
        results_df.to_csv(ATR_filename, index=False)
    
    print(f"ATR Cloud results saved to {ATR_filename}")
    
    return results_df

