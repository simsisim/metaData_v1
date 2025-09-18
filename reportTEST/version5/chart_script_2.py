import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Load the data files
basic_df = pd.read_csv("basic_calculation_daily_0_20250829.csv")
trading_df = pd.read_csv("tradingview_universe.csv")

# Create mapping from ticker to 1-year return
returns_data = basic_df[['ticker', 'daily_year_pct_change']].copy()
returns_data = returns_data.dropna()

# Create mapping from symbol to index membership
index_data = trading_df[['Symbol', 'Index']].copy()
index_data = index_data.dropna()

# Merge the datasets
merged_df = returns_data.merge(index_data, left_on='ticker', right_on='Symbol', how='inner')

# Define major indices to analyze
major_indices = ['S&P 500', 'NASDAQ 100', 'NASDAQ Composite', 'Russell 1000', 'Dow Jones Industrial Average']

# Create analysis for each major index
results = []

for index_name in major_indices:
    # Stocks in this index
    in_index = merged_df[merged_df['Index'].str.contains(index_name, na=False, case=False)]
    
    # Stocks not in this index (from our dataset)
    not_in_index = merged_df[~merged_df['Index'].str.contains(index_name, na=False, case=False)]
    
    if len(in_index) > 0 and len(not_in_index) > 0:
        in_index_avg = in_index['daily_year_pct_change'].mean()
        not_in_index_avg = not_in_index['daily_year_pct_change'].mean()
        
        results.append({
            'Index': index_name.replace('S&P 500', 'SP500').replace('NASDAQ 100', 'NASDAQ100').replace('NASDAQ Composite', 'NASDAQ').replace('Russell 1000', 'Russell1000').replace('Dow Jones Industrial Average', 'DJIA'),
            'In_Index_Return': in_index_avg,
            'Not_In_Index_Return': not_in_index_avg
        })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Create grouped bar chart
fig = go.Figure()

# Add bars for "In Index"
fig.add_trace(go.Bar(
    x=results_df['Index'],
    y=results_df['In_Index_Return'],
    name='In Index',
    marker_color='#1FB8CD',
    cliponaxis=False
))

# Add bars for "Not In Index"  
fig.add_trace(go.Bar(
    x=results_df['Index'],
    y=results_df['Not_In_Index_Return'],
    name='Not In Index',
    marker_color='#DB4545',
    cliponaxis=False
))

# Update layout
fig.update_layout(
    title='Index Membership vs 1-Year Performance',
    barmode='group',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

# Update axes
fig.update_xaxes(title='Index')
fig.update_yaxes(title='Avg 1-Year Ret (%)')

# Save the chart
fig.write_image('index_performance_comparison.png')

print("Chart saved successfully!")
print("\nData summary:")
print(results_df)