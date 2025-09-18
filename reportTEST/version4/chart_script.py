import plotly.graph_objects as go

# Data
data = [
    {"ticker": "APP", "yearly_return": 443.85},
    {"ticker": "PLTR", "yearly_return": 416.17},
    {"ticker": "GEV", "yearly_return": 233.79},
    {"ticker": "AVGO", "yearly_return": 90.15},
    {"ticker": "NFLX", "yearly_return": 76.69},
    {"ticker": "ORCL", "yearly_return": 65.72},
    {"ticker": "TSLA", "yearly_return": 62.27},
    {"ticker": "C", "yearly_return": 61.64},
    {"ticker": "GE", "yearly_return": 61.15},
    {"ticker": "ANET", "yearly_return": 60.10}
]

# Extract tickers and returns
tickers = [item["ticker"] for item in data]
returns = [item["yearly_return"] for item in data]

# Create horizontal bar chart with green gradient
fig = go.Figure(data=go.Bar(
    x=returns,
    y=tickers,
    orientation='h',
    marker=dict(
        color=returns,
        colorscale=[[0, '#90EE90'], [1, '#2E8B57']],  # Light green to sea green gradient
        showscale=False
    ),
    text=[f'{ret:.1f}%' for ret in returns],
    textposition='inside',
    textfont=dict(color='white', size=12)
))

# Update layout
fig.update_layout(
    title='Top 10 Yearly Returns',
    xaxis_title='Return (%)',
    yaxis_title='Ticker'
)

# Update traces
fig.update_traces(cliponaxis=False)

# Reverse y-axis to show highest values at top
fig.update_yaxes(autorange='reversed')

# Save the chart
fig.write_image('top_stocks_yearly_returns.png')