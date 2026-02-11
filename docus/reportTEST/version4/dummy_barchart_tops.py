import plotly.graph_objects as go

# Example dummy data
ranks = list(range(1, 11))
index1_returns = [443.9, 416.2, 233.8, 90.2, 76.7, 65.7, 62.3, 61.6, 61.1, 60.1]
index2_returns = [234.4, 200.1, 180.5, 95.5, 79.1, 70.0, 66.3, 65.0, 63.0, 62.2]
index3_returns = [150.2, 143.5, 130.4, 98.2, 85.6, 74.3, 70.7, 69.0, 68.8, 67.3]

index1_tickers = ['APP', 'PLTR', 'GEV', 'AVGO', 'NFLX', 'ORCL', 'TSLA', 'C', 'GE', 'ANET']
index2_tickers = ['NVDA', 'MSFT', 'AAPL', 'GOOG', 'JNJ', 'V', 'BAC', 'JPM', 'KO', 'PG']
index3_tickers = ['SPY1', 'SPY2', 'SPY3', 'SPY4', 'SPY5', 'SPY6', 'SPY7', 'SPY8', 'SPY9', 'SPY10']

fig = go.Figure()

fig.add_trace(go.Bar(
    x=ranks,
    y=index1_returns,
    name='Index 1',
    marker_color='blue',
    text=index1_tickers,
    textposition='outside'
))

fig.add_trace(go.Bar(
    x=ranks,
    y=index2_returns,
    name='Index 2',
    marker_color='red',
    text=index2_tickers,
    textposition='outside'
))

fig.add_trace(go.Bar(
    x=ranks,
    y=index3_returns,
    name='Index 3',
    marker_color='green',
    text=index3_tickers,
    textposition='outside'
))

fig.update_layout(
    barmode='group',
    title='Top 10 Performers by Index (With Stock Names)',
    xaxis_title='Rank',
    yaxis_title='Yearly Return (%)',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
)

fig.show()

