import plotly.graph_objects as go
import pandas as pd

# Data for the top 6 performing stocks
data = [
    {"ticker": "APP", "1W": 14.29, "1M": 22.50, "3M": 22.63, "6M": 85.67, "1Y": 443.85},
    {"ticker": "PLTR", "1W": 0.34, "1M": -1.04, "3M": 26.62, "6M": 100.78, "1Y": 416.17},
    {"ticker": "GEV", "1W": 1.15, "1M": -7.17, "3M": 26.44, "6M": 115.93, "1Y": 233.79},
    {"ticker": "AVGO", "1W": 2.69, "1M": 1.26, "3M": 24.50, "6M": 57.29, "1Y": 90.15},
    {"ticker": "NFLX", "1W": 0.17, "1M": 4.21, "3M": -0.02, "6M": 34.98, "1Y": 76.69},
    {"ticker": "ORCL", "1W": -3.02, "1M": -10.89, "3M": 38.30, "6M": 57.74, "1Y": 65.72}
]

# Create DataFrame
df = pd.DataFrame(data)

# Define timeframes and colors
timeframes = ['1W', '1M', '3M', '6M', '1Y']
colors = ['#1FB8CD', '#DB4545', '#2E8B57', '#5D878F', '#D2BA4C', '#B4413C']

# Create figure
fig = go.Figure()

# Add a line for each stock
for i, row in df.iterrows():
    returns = [row[tf] for tf in timeframes]
    
    fig.add_trace(go.Scatter(
        x=timeframes,
        y=returns,
        mode='lines+markers',
        name=row['ticker'],
        line=dict(color=colors[i], width=3),
        marker=dict(size=8, color=colors[i]),
        hovertemplate='<b>%{fullData.name}</b><br>%{x}: %{y:.1f}%<extra></extra>'
    ))

# Update layout
fig.update_layout(
    title='Top 6 Stock Performance by Timeframe',
    xaxis_title='Timeframe',
    yaxis_title='Return (%)',
    showlegend=True,
    xaxis=dict(showgrid=True),
    yaxis=dict(
        showgrid=True, 
        tickformat='.0f', 
        ticksuffix='%',
        dtick=50,  # Set tick spacing to every 50%
        tickmode='linear'
    )
)

# Update traces for cliponaxis
fig.update_traces(cliponaxis=False)

# Save the chart
fig.write_image('stock_performance_comparison.png')