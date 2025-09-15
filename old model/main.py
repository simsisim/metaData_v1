import pandas as pd
from src.data_reader import DataReader
from src.tickers_choice import get_ticker_files, user_choice
from src.combined_tickers import combine_tickers 
#from src.basic_calculation import basic_calculation



from src.config import Config
import os 
import logging

#logging.basicConfig(level=logging.INFO)

def main():
    batch_size = 100

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger().setLevel(logging.CRITICAL)

    ticker_files, user_choice = get_ticker_files()
    #simplified = True
    # Always include indexes tickers (but skip simplified mode for choice 17)
    #if (simplified):
    #    print('simplified is activated, only indexes calculated')
    #    ticker_files.clear()
    #    if 'indexes_tickers.csv' not in ticker_files:
    #        ticker_files.append('indexes_tickers.csv')
    # Always include portfolio tickers
    #    if 'portofolio_tickers.csv' not in ticker_files:
    #        ticker_files.append('portofolio_tickers.csv')
            
    config = Config(user_choice)
    config.update_params(user_choice)


    #config = Config(user_choice if not onlyPorto else 'portfolio')

    combined_file, info_tickers_file_path, industry_df, sector_df, industry_names, sector_names = combine_tickers(ticker_files, config.paths)
    #print(combined_file, info_tickers_file_path)
    
        # Create output directory for basic calculations
    output_path = os.path.join(config.paths['dest_tickers_data'], 'basic_calculations')
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize DataReader
    data_reader = DataReader(config.paths, combined_file, batch_size)

    # Process batches
    for batch in data_reader.get_batches():
        # Read CSV data for the batch
        batch_data = {ticker: data_reader.read_stock_data(ticker) for ticker in batch}
        # Perform calculations on the batch
        #basic_calculation(batch_data, output_path)
        # Perform PriceVolumeBreakout Strategy
        print('done')
        
    #print_breakout_results(breakout_results)
##########################MODIFY    
#    params = {'lookback': 20}   # Example parameter for lookback period if neede}
## Run the breakout/retest strategy
  #  results = run_supp_resist_breakout_strategy(batch_data, config.BOSR_params)params, output_folder)
   # #print_supp_resist_breakout_results(results)
## Print the results

    print("All tickers processed & calculation completed.")



if __name__ == "__main__":
    main()
#### CALL FUNCTION TO PERFORM CALCULATION

