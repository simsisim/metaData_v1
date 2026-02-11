import pandas as pd
import plotly.graph_objects as go

# Create DataFrame from the provided data
data = [
    {"Ticker": "WMT", "1D_RS%": 99, "7D_RS%": 7, "22D_RS%": 44, "66D_RS%": 19, "252D_RS%": 32},
    {"Ticker": "MA", "1D_RS%": 93, "7D_RS%": 44, "22D_RS%": 81, "66D_RS%": 38, "252D_RS%": 26},
    {"Ticker": "GOOGL", "1D_RS%": 87, "7D_RS%": 99, "22D_RS%": 93, "66D_RS%": 81, "252D_RS%": 44},
    {"Ticker": "V", "1D_RS%": 81, "7D_RS%": 56, "22D_RS%": 50, "66D_RS%": 13, "252D_RS%": 50},
    {"Ticker": "GOOG", "1D_RS%": 74, "7D_RS%": 93, "22D_RS%": 87, "66D_RS%": 74, "252D_RS%": 38},
    {"Ticker": "JPM", "1D_RS%": 68, "7D_RS%": 81, "22D_RS%": 56, "66D_RS%": 56, "252D_RS%": 68},
    {"Ticker": "LLY", "1D_RS%": 62, "7D_RS%": 87, "22D_RS%": 13, "66D_RS%": 32, "252D_RS%": 7},
    {"Ticker": "AAPL", "1D_RS%": 56, "7D_RS%": 68, "22D_RS%": 99, "66D_RS%": 68, "252D_RS%": 13},
    {"Ticker": "MSFT", "1D_RS%": 50, "7D_RS%": 38, "22D_RS%": 32, "66D_RS%": 44, "252D_RS%": 19},
    {"Ticker": "AMZN", "1D_RS%": 44, "7D_RS%": 62, "22D_RS%": 38, "66D_RS%": 50, "252D_RS%": 56},
    {"Ticker": "META", "1D_RS%": 38, "7D_RS%": 19, "22D_RS%": 74, "66D_RS%": 62, "252D_RS%": 74},
    {"Ticker": "NFLX", "1D_RS%": 32, "7D_RS%": 32, "22D_RS%": 62, "66D_RS%": 26, "252D_RS%": 93},
    {"Ticker": "NVDA", "1D_RS%": 26, "7D_RS%": 26, "22D_RS%": 19, "66D_RS%": 93, "252D_RS%": 62},
    {"Ticker": "TSLA", "1D_RS%": 19, "7D_RS%": 74, "22D_RS%": 68, "66D_RS%": 7, "252D_RS%": 81},
    {"Ticker": "AVGO", "1D_RS%": 13, "7D_RS%": 50, "22D_RS%": 26, "66D_RS%": 87, "252D_RS%": 99},
    {"Ticker": "ORCL", "1D_RS%": 7, "7D_RS%": 13, "22D_RS%": 7, "66D_RS%": 99, "252D_RS%": 87}
]

df = pd.DataFrame(data)

# Set Ticker as index
df = df.set_index('Ticker')

# Create shortened column names under 15 characters
timeframe_labels = ['1D RS%', '7D RS%', '22D RS%', '66D RS%', '252D RS%']
df.columns = timeframe_labels

# Create heatmap
fig = go.Figure(data=go.Heatmap(
    z=df.values,
    x=df.columns,
    y=df.index,
    colorscale=[[0, '#DB4545'], [0.5, '#D2BA4C'], [1, '#2E8B57']],  # Red to yellow to green
    hovertemplate='<b>%{y}</b><br>%{x}: %{z}%<extra></extra>',
    showscale=True
))

# Update layout
fig.update_layout(
    title='RS Percentiles Across Timeframes',
    xaxis_title='Timeframe',
    yaxis_title='Ticker'
)

# Save the chart
fig.write_image('rs_heatmap.png')