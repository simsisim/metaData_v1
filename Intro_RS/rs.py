import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math 
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
import os
from datetime import datetime
"""
RS_rating.py

This file implements the Relative Strength (RS) Rating calculation based on the methodology
used by Investor's Business Daily (IBD). The RS Rating is a measure of a stock's price
performance relative to all other stocks in a given universe over a specified time period.

Key features:
1. Calculates RS Rating for multiple time frames (fractals)
2. Uses a percentile rank to convert relative performance to a 1-99 scale
3. Generates a report with RS Ratings for all stocks in the S&P 500

Formula:
1. Calculate price change: (Current Price / Price at start of period) - 1
2. Calculate relative performance/Relative Strength: (1 + Stock Return) / (1 + Benchmark Return)
3. Convert to percentile rank and scale to 1-99

References:
- Investor's Business Daily: https://www.investors.com/ibd-university/relative-strength-rating/
- O'Neil, William J. "How to Make Money in Stocks" (Chapter 7)

Note: While IBD typically uses a 12-month (252 trading days) period, this implementation
allows for variable time frames, specified as 'fractals' in the params dictionary.
"""

class RelativeStrengthCalculator:
    def __init__(self, df, model_params):
        self.df = df
        self.benchmark_ticker = model_params['benchmark_ticker']
        self.fractals = None


    #def create_directory_structure(self):
    #    os.makedirs('./results', exist_ok=True)
    #    os.makedirs('./results/sectors', exist_ok=True)
    #    os.makedirs('./results/industries', exist_ok=True)
        

    def calculate_rs_line(self, fractals):
        rs_lines = {}
        benchmark_data = self.df[self.benchmark_ticker]
        
        for ticker in self.df.columns:
            if ticker == self.benchmark_ticker:
                continue
            
            stock_data = self.df[ticker]
            ticker_rs = {}
            
            for fractal in fractals:
                stock_return = (stock_data.iloc[-1] / stock_data.iloc[-fractal-1]) - 1
                benchmark_return = (benchmark_data.iloc[-1] / benchmark_data.iloc[-fractal-1]) - 1
                ticker_rs[f'RS_{fractal}'] = (1 + stock_return) / (1 + benchmark_return)
            
            rs_lines[ticker] = ticker_rs
        
        return pd.DataFrame.from_dict(rs_lines, orient='index')

    def calculate_rs_rating(self, rs_lines):
        rs_ratings = {}
        for column in rs_lines.columns:
            percentiles = rs_lines[column].rank(pct=True)
            rs_ratings[f'RS_Rating_{column[3:]}'] = (percentiles * 99 + 1).round(2)
        return pd.DataFrame(rs_ratings, index=rs_lines.index)

    def calculate_weighted_average(self, rs_ratings, weights):
        rs_rating_columns = [f"RS_Rating_{fractal}" for fractal in self.fractals]
        weighted_avg = np.sum(rs_ratings[rs_rating_columns].values * weights, axis=1)
        return pd.Series(weighted_avg, index=rs_ratings.index, name='RS_Rating_WA')

    def create_rs_rating_heatmap(self, results, model_params, file_path, index_column='Symbol'):
        fractals = model_params['fractals']
        benchmark_ticker = model_params['benchmark_ticker']
        #file_path = file_path_params['results']
        rs_rating_columns = [f"RS_Rating_{fractal}" for fractal in fractals]
        heatmap_data = results.set_index(index_column)[rs_rating_columns]
        
        tickers_per_page = 20
        num_pages = math.ceil(len(heatmap_data) / tickers_per_page)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"RSRating_heatmap_{benchmark_ticker}_{timestamp}.pdf"
        pdf_path = os.path.join(file_path, filename)
        
        with PdfPages(pdf_path) as pdf:
            for page in range(num_pages):
                start_idx = page * tickers_per_page
                end_idx = min((page + 1) * tickers_per_page, len(heatmap_data))
                page_data = heatmap_data.iloc[start_idx:end_idx]
                
                fig, ax = plt.subplots(figsize=(12, 8))
                colors = ['red', 'white', 'green']
                cmap = LinearSegmentedColormap.from_list('custom', colors, N=100)
                sns.heatmap(page_data, annot=True, cmap=cmap, center=50,
                            vmin=1, vmax=99, fmt='.2f', linewidths=0.5, ax=ax)
                
                plt.title(f'Relative Strength Rating Heatmap (Page {page+1}/{num_pages})', fontsize=16)
                plt.xlabel('Time Intervals (days)')
                plt.ylabel('Symbols')
                plt.tight_layout()
                
                pdf.savefig(fig)
                plt.close(fig)
        
        print(f"RS rating heatmap generated: {pdf_path}")
        
    def calculate_industry_sector_rs(self, rs_results, info_tickers_file_path):
    # Read the info file containing sector and industry data
        info_df = pd.read_csv(info_tickers_file_path)

    # Merge RS results with sector and industry data
        merged_data = rs_results.merge(info_df[['ticker', 'sector', 'industry']], left_on='Symbol', right_on='ticker')

    # Calculate and save RS for each sector
        for sector in merged_data['sector'].unique():
            sector_data = merged_data[merged_data['sector'] == sector]
            #sector_file_name = f"./results/sectors/{sector}_rs.csv"
            #sector_data.to_csv(sector_file_name, index=False)
            #print(f"Generated file: {sector_file_name}")

    # Calculate and save RS for each industry
        for industry in merged_data['industry'].unique():
            industry_data = merged_data[merged_data['industry'] == industry]
            #industry_file_name = f"./results/industries/{industry}_rs.csv"
            #industry_data.to_csv(industry_file_name, index=False)
            #print(f"Generated file: {industry_file_name}")

    # Calculate average RS for industries and sectors
        industry_rs = merged_data.groupby('industry').mean()
        sector_rs = merged_data.groupby('sector').mean()

        return industry_rs, sector_rs


       
        
        
    def run_analysis(self, model_params):
        #self.create_directory_structure()
        self.fractals = model_params['fractals']
        self.weights = model_params['rs_wa_weights']
        rs_lines = self.calculate_rs_line(self.fractals)
        rs_ratings = self.calculate_rs_rating(rs_lines)
        # Calculate weighted average
        rs_rating_wa = self.calculate_weighted_average(rs_ratings, self.weights)
        # Combine all results
        #results = pd.concat([rs_lines, rs_ratings, rs_rating_wa], axis=1)
        results = pd.concat([rs_ratings, rs_rating_wa], axis=1)
        return results

        
    #def calculate_industry_sector_rs(self, rs_results):
    # Read the info file containing sector and industry data
        #info_df = pd.read_csv('/home/imagda/_invest2024/python/downloadData/data/info.csv')
    
    # Merge RS results with sector and industry data
        #merged_data = rs_results.merge(info_df[['ticker', 'sector', 'industry']], left_on='Symbol', right_on='ticker')
    
    # Calculate average RS for industries
        #industry_rs = merged_data.groupby('industry').mean()
    
    # Calculate average RS for sectors
        #sector_rs = merged_data.groupby('sector').mean()
    
        #return industry_rs, sector_rs


