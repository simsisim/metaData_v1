import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Create DataFrame from the data
data = [
    {"sector": "Information Technology", "avg_yearly_return": 58.66, "stock_count": 13},
    {"sector": "Financials", "avg_yearly_return": 35.92, "stock_count": 7},
    {"sector": "Health Care", "avg_yearly_return": -3.59, "stock_count": 7},
    {"sector": "Consumer Discretionary", "avg_yearly_return": 21.56, "stock_count": 4},
    {"sector": "Energy", "avg_yearly_return": 8.31, "stock_count": 2},
    {"sector": "Industrials", "avg_yearly_return": 147.47, "stock_count": 2},
    {"sector": "Utilities", "avg_yearly_return": 3.86, "stock_count": 1}
]

df = pd.DataFrame(data)

# Abbreviate sector names to fit 15 character limit
df['sector_short'] = df['sector'].replace({
    'Information Technology': 'Info Tech',
    'Financials': 'Financials',
    'Health Care': 'Health Care',
    'Consumer Discretionary': 'Consumer Disc',
    'Energy': 'Energy',
    'Industrials': 'Industrials',
    'Utilities': 'Utilities'
})

# Create colors based on positive/negative returns
colors = ['#DB4545' if ret < 0 else '#2E8B57' for ret in df['avg_yearly_return']]

# Create the figure
fig = go.Figure()

# Add bar chart for returns
fig.add_trace(go.Bar(
    x=df['sector_short'],
    y=df['avg_yearly_return'],
    name='Yearly Return',
    marker_color=colors,
    hovertemplate='<b>%{x}</b><br>Return: %{y:.1f}%<extra></extra>'
))

# Add bubble chart for stock count (scaled appropriately for visibility)
fig.add_trace(go.Scatter(
    x=df['sector_short'],
    y=df['avg_yearly_return'],
    mode='markers',
    marker=dict(
        size=df['stock_count'] * 3,  # Scale bubble size
        sizemode='diameter',
        sizeref=1,
        color='rgba(255,255,255,0.7)',
        line=dict(width=2, color='rgba(0,0,0,0.8)')
    ),
    name='Stock Count',
    hovertemplate='<b>%{x}</b><br>Stocks: %{text}<br>Return: %{y:.1f}%<extra></extra>',
    text=df['stock_count']
))

# Update layout
fig.update_layout(
    title='Sector Performance Analysis',
    xaxis_title='Sector',
    yaxis_title='Return (%)',
    showlegend=True,
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

# Update traces
fig.update_traces(cliponaxis=False)

# Save the chart
fig.write_image('sector_performance_chart.png')