import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Create DataFrame from the volatility_returns data
volatility_data = {
    'atr_pct': [2.84,1.52,1.84,2.07,2.04,2.0,2.27,3.18,3.76,1.56,1.67,2.99,1.47,3.39,1.37,2.22,1.53,1.31,1.54,1.82,4.79,1.59,1.72,1.37,1.58,1.31,2.14,2.08,1.69,3.28,1.76,4.05,2.08,1.53,1.89,3.47,2.06,3.63,2.01,3.27,1.42,2.16,4.73,1.6,1.9,2.43,3.03,1.89,1.79,1.22,0.98,1.43,1.37,1.18,1.04,1.17,1.14,1.02,1.21,1.13,1.01,1.41,0.8,1.1],
    'daily_year_pct_change': [38.71,24.12,2.97,31.37,30.43,34.07,43.43,90.15,62.27,39.27,28.71,-22.21,31.6,65.72,26.9,76.69,1.63,11.58,6.77,11.78,416.17,30.12,11.53,-4.76,14.99,-1.03,-15.77,61.15,26.86,-46.36,42.4,11.12,41.71,-6.59,61.64,34.82,-4.42,233.79,-1.85,60.1,9.32,-21.53,443.85,32.67,-10.49,-14.7,-21.88,51.72,52.62,3.86,22.03,4.01,20.43,-10.28,19.39,1.45,1.0,0.67,26.98,15.26,29.57,23.0,16.98,21.7]
}

df = pd.DataFrame(volatility_data)

# Create scatter plot
fig = go.Figure()

# Add scatter points
fig.add_trace(go.Scatter(
    x=df['atr_pct'],
    y=df['daily_year_pct_change'],
    mode='markers',
    marker=dict(
        color='#1FB8CD',
        size=8,
        opacity=0.7
    ),
    name='Stocks',
    hovertemplate='Volatility: %{x:.2f}%<br>Return: %{y:.0f}%<extra></extra>'
))

# Add trend line
z = np.polyfit(df['atr_pct'], df['daily_year_pct_change'], 1)
p = np.poly1d(z)
trend_x = np.linspace(df['atr_pct'].min(), df['atr_pct'].max(), 100)
trend_y = p(trend_x)

fig.add_trace(go.Scatter(
    x=trend_x,
    y=trend_y,
    mode='lines',
    line=dict(color='#DB4545', width=2, dash='dash'),
    name='Trend',
    hoverinfo='skip'
))

# Update traces
fig.update_traces(cliponaxis=False)

# Update layout
fig.update_layout(
    title='Risk vs Return Analysis',
    xaxis_title='Volatility %',
    yaxis_title='Annual Return %',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5),
    showlegend=True
)

# Update axes
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

# Save the chart
fig.write_image('risk_return_scatter.png')