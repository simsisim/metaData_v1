import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from data_fetcher import download_ticker_data

def pct_to_ma(stock, ma_period):
    """Calculate percentage change relative to moving averages."""
    ma = stock.rolling(window=ma_period).mean()
    return (stock - ma) / ma * 100

def create_rrg(data, stocks, params):
    """Create Relative Rotation Graph data."""
    rrg_data = pd.DataFrame()
    
    for stock in stocks:
        rrg_data[f'{stock}_{params["short_ma"]}MA'] = pct_to_ma(data[stock], params['short_ma'])
        rrg_data[f'{stock}_{params["long_ma"]}MA'] = pct_to_ma(data[stock], params['long_ma'])

    return rrg_data
def plot_rrg(rrg_data, stocks, plot_title, output_folder, params):
    """Plot the Relative Rotation Graph."""
    plt.figure(figsize=(15, 15))
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)

    for stock in stocks:
        x = rrg_data[f'{stock}_{params["short_ma"]}MA'].iloc[-1]
        y = rrg_data[f'{stock}_{params["long_ma"]}MA'].iloc[-1]
        plt.scatter(x, y, s=100)
        plt.annotate(stock, (x, y), xytext=(5, 5), textcoords='offset points')

    plt.xlabel(f'Percentage Change Relative to {params["short_ma"]}-day MA')
    plt.ylabel(f'Percentage Change Relative to {params["long_ma"]}-day MA')
    plt.title(f'{plot_title}\nShort MA: {params["short_ma"]}, Long MA: {params["long_ma"]}')



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

def run_rrg_analysis_noTail(data, params):
    """Main function to generate RRG and save results."""
    
    # Initialize an empty DataFrame for adjusted close prices
    all_data = pd.DataFrame()

    # Download data using provided function
    for _, row in data.iterrows():
        ticker = row['Ticker']
        try:
            ticker_data = download_ticker_data(ticker, params['start_date'], params['end_date'], interval=params['interval'])
            #print(ticker_data)
            if ticker_data is not None and 'Close' in ticker_data.columns:
                all_data[ticker] = ticker_data['Close']  # Store adjusted close prices
            else:
                print(f"No valid data available for {ticker}. Skipping.")
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")

    if all_data.empty:  # Check if any valid data was collected
        print("No valid data available for any tickers.")
        return None

    # Create RRG data
    rrg_data = create_rrg(all_data, all_data.columns.tolist(), params)
    
    # Save data to CSV and plot
    save_to_csv(rrg_data, params['output_folder'])  
    plot_title = f"{params['file_name_prefix']}_{params['start_date']}_{params['end_date']}"
    
    plot_rrg(rrg_data, all_data.columns.tolist(), plot_title, params['output_folder'], params)


