import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Create DataFrame from the provided data
data = [
    {"Ticker": "LLY", "Short_Term_Avg": 74.5, "Long_Term": 7},
    {"Ticker": "AAPL", "Short_Term_Avg": 62.0, "Long_Term": 13},
    {"Ticker": "GOOGL", "Short_Term_Avg": 93.0, "Long_Term": 44},
    {"Ticker": "GOOG", "Short_Term_Avg": 83.5, "Long_Term": 38},
    {"Ticker": "MA", "Short_Term_Avg": 68.5, "Long_Term": 26},
    {"Ticker": "MSFT", "Short_Term_Avg": 44.0, "Long_Term": 19},
    {"Ticker": "WMT", "Short_Term_Avg": 53.0, "Long_Term": 32},
    {"Ticker": "V", "Short_Term_Avg": 68.5, "Long_Term": 50},
    {"Ticker": "JPM", "Short_Term_Avg": 74.5, "Long_Term": 68},
    {"Ticker": "AMZN", "Short_Term_Avg": 53.0, "Long_Term": 56},
    {"Ticker": "TSLA", "Short_Term_Avg": 46.5, "Long_Term": 81},
    {"Ticker": "NVDA", "Short_Term_Avg": 26.0, "Long_Term": 62},
    {"Ticker": "META", "Short_Term_Avg": 28.5, "Long_Term": 74},
    {"Ticker": "NFLX", "Short_Term_Avg": 32.0, "Long_Term": 93},
    {"Ticker": "AVGO", "Short_Term_Avg": 31.5, "Long_Term": 99},
    {"Ticker": "ORCL", "Short_Term_Avg": 10.0, "Long_Term": 87}
]

df = pd.DataFrame(data)

# Create scatter plot
fig = go.Figure()

# Add scatter points
fig.add_trace(go.Scatter(
    x=df['Short_Term_Avg'],
    y=df['Long_Term'],
    mode='markers+text',
    text=df['Ticker'],
    textposition="middle right",
    marker=dict(
        size=8,
        color='#1FB8CD'
    ),
    name='Stocks',
    showlegend=False
))

# Add diagonal line (equal performance line)
max_val = max(df['Short_Term_Avg'].max(), df['Long_Term'].max())
min_val = min(df['Short_Term_Avg'].min(), df['Long_Term'].min())

fig.add_trace(go.Scatter(
    x=[min_val, max_val],
    y=[min_val, max_val],
    mode='lines',
    line=dict(
        color='#DB4545',
        width=2,
        dash='dash'
    ),
    name='Equal Perf',
    showlegend=True
))

# Update layout
fig.update_layout(
    title='Short vs Long RS Percentiles',
    xaxis_title='Short-term %',
    yaxis_title='Long-term %',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

# Update traces
fig.update_traces(cliponaxis=False)

# Save the chart
fig.write_image('rs_percentiles_scatter.png')