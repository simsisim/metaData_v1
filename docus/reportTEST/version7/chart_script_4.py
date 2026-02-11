import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Load the data from the provided JSON
data = {
    "ticker": ["NVDA","MSFT","AAPL","GOOGL","GOOG","AMZN","META","AVGO","TSLA","JPM","WMT","LLY","V","ORCL","MA","NFLX","XOM","JNJ","COST","HD","PLTR","BAC","ABBV","PG","CVX","KO","ASML","GE","TMUS","UNH","CSCO","AMD","PM","AZN","C","PDD","QCOM","GEV","ISRG","ANET","SPGI","ACN","APP","BSX","AMGN","HPQ","LULU","VRSN","NTRS","FE","XLF","XLE","XLK","XLV","XLI","XLB","XLRE","XLP","XLY","XLU","XLC","VGT","SPY","QQQ"],
    "current_price": [174.18,506.69,232.14,212.91,213.53,229.0,738.7,297.39,333.87,301.42,96.98,732.58,351.78,226.13,595.29,1208.25,114.29,177.17,943.32,406.77,156.71,50.74,210.4,157.04,160.6,68.99,742.62,275.2,251.99,309.87,69.09,162.63,167.13,79.9,96.57,120.22,160.73,612.97,473.3,136.55,548.44,259.97,478.59,105.5,287.71,28.54,202.2,273.37,131.28,43.62,53.99,90.39,262.45,137.43,152.01,92.28,42.31,80.78,231.74,84.32,111.39,697.02,645.05,570.4],
    "daily_year_pct_change": [38.71,24.12,2.97,31.37,30.43,34.07,43.43,90.15,62.27,39.27,28.71,-22.21,31.6,65.72,26.9,76.69,1.63,11.58,6.77,11.78,416.17,30.12,11.53,-4.76,14.99,-1.03,-15.77,61.15,26.86,-46.36,42.4,11.12,41.71,-6.59,61.64,34.82,-4.42,233.79,-1.85,60.1,9.32,-21.53,443.85,32.67,-10.49,-14.7,-21.88,51.72,52.62,3.86,22.03,4.01,20.43,-10.28,19.39,1.45,1.0,0.67,26.98,15.26,29.57,23.0,16.98,21.7],
    "MarketCap_Category": ["Mid Cap","Large Cap","Mid Cap","Mid Cap","Mid Cap","Mid Cap","Mega Cap","Mid Cap","Mid Cap","Mid Cap","Small Cap","Mega Cap","Mid Cap","Mid Cap","Large Cap","Mega Cap","Small Cap","Mid Cap","Mega Cap","Large Cap","Mid Cap","Small Cap","Mid Cap","Mid Cap","Mid Cap","Small Cap","Mega Cap","Mid Cap","Mid Cap","Mid Cap","Small Cap","Mid Cap","Mid Cap","Small Cap","Small Cap","Mid Cap","Mid Cap","Large Cap","Large Cap","Mid Cap","Large Cap","Mid Cap","Large Cap","Mid Cap","Mid Cap","Small Cap","Mid Cap","Mid Cap","Mid Cap","Small Cap","Small Cap","Small Cap","Mid Cap","Mid Cap","Mid Cap","Small Cap","Small Cap","Small Cap","Mid Cap","Small Cap","Mid Cap","Large Cap","Large Cap","Large Cap"]
}

df = pd.DataFrame(data)

# Define the color palette
colors = ["#1FB8CD", "#DB4545", "#2E8B57", "#5D878F"]

# Create the violin plot
fig = go.Figure()

# Get unique categories and sort them for consistent ordering
categories = ["Small Cap", "Mid Cap", "Large Cap", "Mega Cap"]

for i, category in enumerate(categories):
    category_data = df[df['MarketCap_Category'] == category]['daily_year_pct_change']
    
    # Add violin plot
    fig.add_trace(go.Violin(
        y=category_data,
        x=[category] * len(category_data),
        name=category,
        box_visible=True,
        meanline_visible=True,
        points='all',
        pointpos=0,
        jitter=0.3,
        fillcolor=colors[i],
        line_color=colors[i],
        opacity=0.7,
        showlegend=True
    ))

# Update layout
fig.update_layout(
    title="Annual Returns by Market Cap",
    yaxis_title="Annual Return %",
    xaxis_title="Market Cap",
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5),
    showlegend=True
)

# Save the chart
fig.write_image("violin_plot_returns.png")