import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from data_fetcher import download_ticker_data

def pct_to_ma(stock, ma_period):
    """Calculate percentage change relative to moving averages."""
    ma = stock.rolling(window=ma_period).mean()
    return (stock - ma) / ma * 100

def create_rrg(data, stocks, tail_length=10):
    """Create Relative Rotation Graph data."""
    rrg_data = pd.DataFrame()

    for stock in stocks:
        # Check if stock exists in data
        if stock not in data.columns:
            print(f"{stock} not found in data. Skipping.")
            continue
        
        # Calculate percentage change relative to moving averages
        pct_change_20MA = pct_to_ma(data[stock], 20)
        pct_change_50MA = pct_to_ma(data[stock], 50)

        # Store the percentage change relative to MAs
        rrg_data[f'{stock}_20MA'] = pct_change_20MA
        rrg_data[f'{stock}_50MA'] = pct_change_50MA

        # Debugging output
        #print(stock)
        #print(rrg_data[f'{stock}_20MA'].tail())
        #print(rrg_data[f'{stock}_50MA'].tail())
        
        # Get the last `tail_length` values for plotting
        if len(pct_change_20MA) >= tail_length:
            # Initialize tail columns if they don't exist yet
            rrg_data[f'{stock}_20MA_tail'] = np.nan  # Create column with NaN values
            rrg_data[f'{stock}_50MA_tail'] = np.nan  # Create column with NaN values
            
            # Assign last tail_length values directly
            rrg_data[f'{stock}_20MA_tail'].iloc[-tail_length:] = pct_change_20MA.iloc[-tail_length:].values
            rrg_data[f'{stock}_50MA_tail'].iloc[-tail_length:] = pct_change_50MA.iloc[-tail_length:].values

    return rrg_data




def plot_rrg(rrg_data, stocks, plot_title, output_folder, tail_length=10):
    """Plot the Relative Rotation Graph with tails."""
    plt.figure(figsize=(10, 10))
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)

    for stock in stocks:
        # Get the last values for 20MA and 50MA
        x = rrg_data[f'{stock}_20MA'].iloc[-1]
        y = rrg_data[f'{stock}_50MA'].iloc[-1]

        # Debugging output to check values
        #print(f"{stock}: x = {x}, y = {y}")

        # Check if x and y are valid
        if not np.isnan(x) and not np.isnan(y):
            # Plot current point with increased size and opacity
            plt.scatter(x, y, s=150, alpha=0.8)  # Increased size and opacity
            
            # Plot tail (last `tail_length` points)
            if f'{stock}_20MA_tail' in rrg_data.columns and f'{stock}_50MA_tail' in rrg_data.columns:
                tail_x = rrg_data[f'{stock}_20MA_tail']
                tail_y = rrg_data[f'{stock}_50MA_tail']
                
                # Check if tails have valid data
                #print(f"Tails for {stock}:")
                #print(tail_x)
                #print(tail_y)

                # Ensure that tails are not all NaN before plotting
                if not tail_x.isnull().all() and not tail_y.isnull().all():
                    plt.plot(tail_x.values, tail_y.values, marker='o', alpha=0.5)  # Soft line with points
            
            # Annotate stock name with adjusted position
            plt.annotate(stock, (x, y), xytext=(10, 10), textcoords='offset points', fontsize=9)

    plt.xlabel('Percentage Change Relative to 20-day MA')
    plt.ylabel('Percentage Change Relative to 50-day MA')
    plt.title(plot_title)

    plt.text(0.95, 0.95, 'Leading', transform=plt.gca().transAxes, ha='right', va='top')
    plt.text(0.05, 0.95, 'Improving', transform=plt.gca().transAxes, ha='left', va='top')
    plt.text(0.05, 0.05, 'Lagging', transform=plt.gca().transAxes, ha='left', va='bottom')
    plt.text(0.95, 0.05, 'Weakening', transform=plt.gca().transAxes, ha='right', va='bottom')

    plt.grid(True, alpha=0.3)
    
    # Ensure output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the plot as a PNG file
    plot_file_path = os.path.join(output_folder, f"{plot_title}.png")
    plt.savefig(plot_file_path)
    
    # Show the plot
    #plt.show()



def save_to_csv(rrg_data, output_folder):
    """Save RRG data to a CSV file."""
    csv_file_path = os.path.join(output_folder, 'rrg_data.csv')
    rrg_data.to_csv(csv_file_path)

def run_rrg_analysis_wTail(data, params):
    """Main function to generate RRG and save results."""
    
    # Initialize an empty DataFrame for adjusted close prices
    all_data = pd.DataFrame()

    # Download data using provided function one by one
    for _, row in data.iterrows():
        ticker = row['Ticker']
        try:
            ticker_data = download_ticker_data(ticker, params['start_date'], params['end_date'], interval=params['interval'])
            if ticker_data is not None and 'Close' in ticker_data.columns:
                all_data[ticker] = ticker_data['Close']  # Store adjusted close prices
            else:
                print(f"No valid data available for {ticker}. Skipping.")
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")

    if all_data.empty:  # Check if any valid data was collected
        print("No valid data available for any tickers.")
        return None
    #print('haaaaaaaaaaaaaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA', all_data)
    # Create RRG data using only the stocks with valid data
    rrg_data = create_rrg(all_data, all_data.columns.tolist())
    #print(rrg_data[['MMM_20MA','LNT_50MA','GOOGL_20MA']])
    # Save data to CSV and plot
    save_to_csv(rrg_data, params['output_folder'])  
    plot_title = f"{params['file_name_prefix']}_{params['start_date']}_{params['end_date']}"
    
    plot_rrg(rrg_data, all_data.columns.tolist(), plot_title, params['output_folder'])

