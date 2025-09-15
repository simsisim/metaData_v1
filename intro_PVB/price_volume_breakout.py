import pandas as pd
import os

def calculate_indicators(data, price_breakout_period, volume_breakout_period, trendline_length):
    data['Price_Highest'] = data['High'].rolling(window=price_breakout_period).max()# data['Close'].rolling(window=price_breakout_period).max()
    data['Price_Lowest'] = data['Low'].rolling(window=price_breakout_period).min()#    data['Price_Lowest'] = data['Close'].rolling(window=price_breakout_period).min()
    data['Volume_Highest'] = data['Volume'].rolling(window=volume_breakout_period).max()
    data['SMA'] = data['Close'].rolling(window=trendline_length).mean()
    return data

def generate_signals(data, order_direction):
    signals = []
    current_signal = "No Signal"
    signal_date = None
    consecutive_days = 0

    for i in range(1, len(data)):
        current_row = data.iloc[i]
        prev_row = data.iloc[i-1]

        # Long Signal
        if (order_direction in ["Long", "Long and Short"] and
            current_row['Close'] > prev_row['Price_Highest'] and
            current_row['Volume'] > prev_row['Volume_Highest'] and
            current_row['Close'] > current_row['SMA'] and
            current_signal not in ["Buy"]):
            current_signal = "Buy"
            signal_date = data.index[i]
            consecutive_days = 0

        # Short Signal
        elif (order_direction in ["Short", "Long and Short"] and
              current_row['Close'] < prev_row['Price_Lowest'] and
              current_row['Volume'] > prev_row['Volume_Highest'] and
              current_row['Close'] < current_row['SMA'] and
              current_signal not in ["Sell"]):
            current_signal = "Sell"
            signal_date = data.index[i]
            consecutive_days = 0

        # Check for closing conditions
        if current_signal == "Buy":
            if current_row['Close'] < current_row['SMA']:
                consecutive_days += 1
            else:
                consecutive_days = 0
            if consecutive_days >= 5:
                current_signal = "Close Buy"
                signal_date = data.index[i]
                consecutive_days = 0
        elif current_signal == "Sell":
            if current_row['Close'] > current_row['SMA']:
                consecutive_days += 1
            else:
                consecutive_days = 0
            if consecutive_days >= 5:
                current_signal = "Close Sell"
                signal_date = data.index[i]
                consecutive_days = 0

        # Add signal to the list if it's a new signal
        if current_signal != "No Signal" and (not signals or signals[-1]['Signal'] != current_signal):
            signals.append({
                "Date": signal_date or data.index[i],
                "Signal": current_signal,
                "Close": current_row['Close'],
                "Volume": current_row['Volume'],
                "SMA": current_row['SMA']
            })

    return signals

def run_PVBstrategy(batch_data, params):
    results = []

    for symbol, data in batch_data.items():
        try:
            # Ensure data has the required columns
            required_columns = ['Close', 'High', 'Low', 'Volume']
            if not all(col in data.columns for col in required_columns):
                print(f"Skipping {symbol}: Missing required columns")
                continue

            data = calculate_indicators(data, params['price_breakout_period'], params['volume_breakout_period'], params['trendline_length'])

            signals = generate_signals(data, params['order_direction'])

            if signals:
                latest_signal = signals[-1]
                results.append({
                    "Symbol": symbol,
                    "Signal Date": latest_signal['Date'].strftime('%Y-%m-%d'),
                    "Current Price": round(data['Close'].iloc[-1], 2),
                    "Signal": latest_signal['Signal'],
                    "SMA": round(latest_signal['SMA'], 2),
                    "Volume": round(latest_signal['Volume'], 2),
                    "Volume Highest": round(data['Volume_Highest'].iloc[-1], 2),
                    "Price Change %": round(((data['Close'].iloc[-1] - latest_signal['SMA']) / latest_signal['SMA']) * 100, 2),
                    "Volume Change %": round(((data['Volume'].iloc[-1] - data['Volume_Highest'].iloc[-1]) / data['Volume_Highest'].iloc[-1]) * 100, 2),
                    "Days Since Signal": (data.index[-1] - latest_signal['Date']).days
                })
            else:
                results.append({
                    "Symbol": symbol,
                    "Signal Date": "N/A",
                    "Current Price": round(data['Close'].iloc[-1], 2),
                    "Signal": "No Signal",
                    "SMA": round(data['SMA'].iloc[-1], 2),
                    "Volume": round(data['Volume'].iloc[-1], 2),
                    "Volume Highest": round(data['Volume_Highest'].iloc[-1], 2),
                    "Price Change %": "N/A",
                    "Volume Change %": "N/A",
                    "Days Since Signal": "N/A"
                })
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")

    df_results = pd.DataFrame(results)
    csv_filename = f"{params['PVB_filename']}"
    
    # If the file exists, append without header. If not, create with header.
    if os.path.exists(csv_filename):
        df_results.to_csv(csv_filename, mode='a', header=False, index=False)
    else:
        df_results.to_csv(csv_filename, index=False)
    
    print(f"Results appended to {csv_filename}")
    return df_results


