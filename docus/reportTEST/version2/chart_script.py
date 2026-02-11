import plotly.graph_objects as go
import plotly.io as pio

# Data
data = [
    {"ticker": "APP", "ytd_return": 443.85}, 
    {"ticker": "PLTR", "ytd_return": 416.17}, 
    {"ticker": "GEV", "ytd_return": 233.79}, 
    {"ticker": "AVGO", "ytd_return": 90.15}, 
    {"ticker": "NFLX", "ytd_return": 76.69}, 
    {"ticker": "ORCL", "ytd_return": 65.72}, 
    {"ticker": "TSLA", "ytd_return": 62.27}, 
    {"ticker": "C", "ytd_return": 61.64}, 
    {"ticker": "GE", "ytd_return": 61.15}, 
    {"ticker": "ANET", "ytd_return": 60.10}, 
    {"ticker": "META", "ytd_return": 43.43}, 
    {"ticker": "JPM", "ytd_return": 39.27}, 
    {"ticker": "NVDA", "ytd_return": 38.71}, 
    {"ticker": "AMZN", "ytd_return": 34.07}, 
    {"ticker": "GOOGL", "ytd_return": 31.37}
]

# Extract tickers and returns
tickers = [item["ticker"] for item in data]
returns = [item["ytd_return"] for item in data]

# Create horizontal bar chart
fig = go.Figure(data=go.Bar(
    x=returns,
    y=tickers,
    orientation='h',
    marker_color='#2E8B57',  # Sea green color
    hovertemplate='<b>%{y}</b><br>YTD Return: %{x:.2f}%<extra></extra>'
))

# Update layout
fig.update_layout(
    title="Top 15 Stock Performers - YTD (%)",
    xaxis_title="YTD Return (%)",
    yaxis_title="Ticker"
)

# Format x-axis to show percentage values
fig.update_xaxes(ticksuffix="%")

# Reverse y-axis to show highest performer at top
fig.update_yaxes(categoryorder="total ascending")

# Update traces
fig.update_traces(cliponaxis=False)

# Save the chart
fig.write_image("stock_ytd_performance.png")