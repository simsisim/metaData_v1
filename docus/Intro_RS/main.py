import pandas as pd
from src.data_reader import DataReader
from src.giusti_screener import GiustiScreener
from src.minervini_screener import MinerviniScreener
from src.rs import RelativeStrengthCalculator
from src.tickers_choice import get_ticker_files, user_choice
from src.combined_tickers import combine_tickers
from src.extended_results import extend_rs_results 
from src.config import Config
import os 
import logging
from src.percentCalc import PercentageCalculator



#logging.basicConfig(level=logging.INFO)

def main():
    config = Config()
    ticker_files, user_choice = get_ticker_files()
    config = Config(user_choice)
    config.update_params(user_choice)
    # Always include the indexes file
    if 'indexes_tickers.csv' not in ticker_files:
        ticker_files.append('indexes_tickers.csv')
    combined_file, info_tickers_file_path, industry_df, sector_df, industry_names, sector_names = combine_tickers(ticker_files, config.paths)
    #print(info_tickers_file_path)
    # Read market data peviously saved on hard disk
    data_reader = DataReader(config.paths, combined_file)
    info_tickers = pd.read_csv(info_tickers_file_path)
    # Create combined DataFrame according to user_choice choice
    df = data_reader.create_combined_df() # from local data previously saved
    # Filter sector_df and industry_df to only include tickers present in df
    valid_tickers = set(df.columns)
    sector_df = sector_df[sector_df.index.get_level_values('ticker').isin(valid_tickers)]
    industry_df = industry_df[industry_df.index.get_level_values('ticker').isin(valid_tickers)]
    # Update sector_names and industry_names
    sector_names = sector_df.index.get_level_values('sector').unique().tolist()
    industry_names = industry_df.index.get_level_values('industry').unique().tolist()
# Calculate RELATIVE STRENGTH in raport to benchmark_ticker
    rs_calculator = RelativeStrengthCalculator(df, config.model_params)
    rs_results = rs_calculator.run_analysis(config.model_params)
    rs_results.to_csv(config.results_files['rs_stocks'])
    rs_results = rs_results.reset_index().rename(columns={'index': 'Symbol'})
    #print(rs_results)
    # add info about ticker
    #print(info_tickers)
    rs_results_extended = extend_rs_results(rs_results, info_tickers)
    

        # Calculate percentage changes
    calculator = PercentageCalculator()
    percentage_changes = calculator.calculate_percentage_changes(df)
    
    # Reset index of percentage_changes to align with rs_results_extended
    percentage_changes = percentage_changes.reset_index().rename(columns={'index': 'Symbol'})
    #print(percentage_changes)
    # Merge rs_results_extended with percentage_changes
    merged_extended = pd.merge(rs_results_extended, percentage_changes, on='Symbol', how='outer')
    #print(merged_extended.columns.tolist())
    # Merge rs_results_extended with percentage_changes
    
    # Save merged results to CSV
    merged_extended.to_csv(config.results_files['rs_stocks'], index=False)
    print(f"Results saved to {config.results_files['rs_stocks']}")

    # Create a new column 'exchange1_short' that contains only the first word of 'exchange1'
    merged_extended['exchange2_short'] = merged_extended['exchange2'].str.split().str[0].str.upper()
    
    # Create new columns for the desired output
    merged_extended['exchange2short:Symbol'] = merged_extended['exchange2_short'] + ':' + merged_extended['Symbol']
    
    # Select only the required columns for the output
    output_df = merged_extended[['exchange2short:Symbol']]
    
    # Write the output to a CSV file
    #print(config.results_files['rs_stocks'])
    new_filename = config.results_files['rs_stocks'] + '_TW.csv'
    output_df.to_csv(new_filename, index=False)
    
    print(f"Output saved to: {new_filename}")
    

    
    #rs_results_extended.to_csv(config.results_files['rs_stocks'], index=False)
    #print(f"Results saved to {config.results_files['rs_stocks']}")
    # Generate heatmap: it works, but commented as i find the cvs file better
    heatmap = False
    #if heatmap: 
    #    rs_calculator.create_rs_rating_heatmap(rs_results, config.model_params, config.directories['results'], index_column='Symbol' )#rs_calculator.create_rs_rating_heatmap(rs_results, config.model_params, config.directories['results'], index_column='Symbol' )
# Calculate RELATIVE STRENGTH OF SECTORS AND INDUSTRIES in raport to benchmark_ticker market: NOT SURE IS RIGHT
    #industry_rs, sector_rs = rs_calculator.calculate_industry_sector_rs(rs_results, info_tickers_file_path)
    #industry_rs, sector_rs = industry_rs.reset_index(), sector_rs.reset_index()
    #industry_rs.to_csv(config.results_files['rs_industries'], index=False)
    #sector_rs.to_csv(config.results_files['rs_sectors'], index=False)
    #print(f"Results saved to {config.results_files['rs_sectors']}. Warrning: this is not a weighted average calculation.")
    #print(f"Results saved to {config.results_files['rs_industries']}.Warrning: this is not a weighted average calculation.")
    #if heatmap: 
    #    rs_calculator.create_rs_rating_heatmap(industry_rs, config.model_params, config.directories['sectors'], index_column='industry')
    #    rs_calculator.create_rs_rating_heatmap(sector_rs, config.model_params, config.directories['industries'], index_column='sector')
    #
    # Calculate industry and sector RS
    #industry_rs, sector_rs = rs_calculator.calculate_industry_sector_rs(rs_results, info_tickers_file_path)
    #industry_rs, sector_rs = rs_calculator.calculate_industry_sector_rs(rs_results, info_tickers_file_path)
    # Run Minervini scanner
    minervini_screener = MinerviniScreener(df, rs_results, config.model_params)
    minervini_results = minervini_screener.run_screener()
    minervini_results_extended = extend_rs_results(minervini_results, info_tickers)
    minervini_results_extended.to_csv(config.results_files['minervini'], index=False)
    print(f"Results saved to {config.results_files['minervini']}")

    # Create a new column 'exchange1_short' that contains only the first word of 'exchange1'- TRadingview
    minervini_results_extended['exchange2_short'] = minervini_results_extended['exchange2'].str.split().str[0].str.upper()
    # Create new columns for the desired output
    minervini_results_extended['exchange2short:Symbol'] = minervini_results_extended['exchange2_short'] + ':' + minervini_results_extended['Symbol']
    # Select only the required columns for the output
    output_df = minervini_results_extended[['exchange2short:Symbol']]
    # Write the output to a CSV file
    print(config.results_files['rs_stocks'])
    new_filename = config.results_files['minervini'] + '_TW.csv'
    output_df.to_csv(new_filename, index=False)
    print(f"Output saved to: {new_filename}")
    
    # Run Giusti scanner
    giusti_screener = GiustiScreener(df)
    giusti_results = giusti_screener.run_screener()
    giusti_results.to_csv(config.results_files['giusti'], index=False)
    print(f"Results saved to {config.results_files['giusti']}")
    
#################################################################################
    benchmark_ticker = config.model_params['benchmark_ticker']
    # Calculate RS for each sector
    for sector in sector_names:

        sector_tickers = sector_df.loc[sector].index.tolist()
        sector_tickers_with_benchmark = list(set(sector_tickers + [config.model_params['benchmark_ticker']]))
        sector_df_filtered = df[sector_tickers_with_benchmark]

        sector_rs_calculator = RelativeStrengthCalculator(sector_df_filtered, config.model_params)
        sector_rs_results = sector_rs_calculator.run_analysis(config.model_params).reset_index().rename(columns={'index': 'Symbol'})

        sector_rs_results_extended = extend_rs_results(sector_rs_results, info_tickers)
        sector_rs_results_extended.to_csv(os.path.join(config.directories['sectors'], f'rs_{sector}_{benchmark_ticker}_{config.user_choice}.csv'), index=False)


        # Run Minervini scanner for the sector
        sector_minervini_screener = MinerviniScreener(sector_df_filtered, sector_rs_results, config.model_params)
        sector_minervini_results = sector_minervini_screener.run_screener()

        sector_minervini_results_extended = extend_rs_results(sector_minervini_results, info_tickers)
        sector_minervini_results_extended.to_csv(os.path.join(config.directories['sectors'], f'minervini_{sector}_{benchmark_ticker}_{config.user_choice}.csv'))
        


        # Run Giusti scanner for the sector
        sector_giusti_screener = GiustiScreener(sector_df_filtered)
        sector_giusti_results = sector_giusti_screener.run_screener()
        sector_giusti_results.to_csv(os.path.join(config.directories['sectors'], f'giusti_{sector}_{benchmark_ticker}_{config.user_choice}.csv'), index=False)

    # Calculate RS for each industry
    for industry in industry_names:
        industry_tickers = industry_df.loc[industry].index.tolist()
        industry_tickers_with_benchmark = list(set(industry_tickers + [config.model_params['benchmark_ticker']]))
        industry_df_filtered = df[industry_tickers_with_benchmark]
        industry_rs_calculator = RelativeStrengthCalculator(industry_df_filtered, config.model_params)
        industry_rs_results = industry_rs_calculator.run_analysis(config.model_params).reset_index().rename(columns={'index': 'Symbol'})
        industry_rs_results_extended = extend_rs_results(industry_rs_results, info_tickers)
        industry_rs_results_extended.to_csv(os.path.join(config.directories['industries'], f'rs_{industry}_{benchmark_ticker}_{config.user_choice}.csv'), index=False)
        

        # Run Minervini scanner for the industry
        industry_minervini_screener = MinerviniScreener(industry_df_filtered, industry_rs_results, config.model_params)
        industry_minervini_results = industry_minervini_screener.run_screener()
        industry_minervini_results_extended = extend_rs_results(industry_minervini_results, info_tickers)
        industry_minervini_results_extended.to_csv(os.path.join(config.directories['industries'], f'minervini_{industry}_{benchmark_ticker}_{config.user_choice}.csv'), index=False)

    # Run Giusti scanner for the industry
        industry_giusti_screener = GiustiScreener(industry_df_filtered)
        industry_giusti_results = industry_giusti_screener.run_screener()
        industry_giusti_results.to_csv(os.path.join(config.directories['industries'], f'giusti_{industry}_{config.user_choice}.csv'), index=False)

    
if __name__ == "__main__":
    main()
