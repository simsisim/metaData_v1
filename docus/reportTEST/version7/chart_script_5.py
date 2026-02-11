import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Create the dataframe from the provided data
data = {
    "ticker": ["GOOGL","GOOG","AVGO","JPM","WMT","LLY","V","ORCL"],
    "current_price": [212.91,213.53,297.39,301.42,96.98,732.58,351.78,226.13],
    "daily_year_pct_change": [31.37,30.43,90.15,39.27,28.71,-22.21,31.6,65.72],
    "daily_rsi_14": [72.45,72.87,45.12,69.9,25.53,83.83,79.3,27.19],
    "daily_macd": [5.53,5.47,-1.43,2.77,-0.51,-4.85,0.98,-1.43],
    "daily_momentum_20": [6.5,6.47,-0.17,2.93,-5.24,2.54,3.96,-7.29],
    "daily_price_position_52w": [1.0,1.0,0.91,1.0,0.75,0.33,0.97,0.77],
    "atr_pct": [2.07,2.04,3.18,1.56,1.67,2.99,1.47,3.39],
    "directional_strength": [0.54,0.52,0.72,0.19,0.72,0.03,0.36,0.76]
}

df = pd.DataFrame(data)

# Scale metrics to 0-1 range
def min_max_scale(series):
    return (series - series.min()) / (series.max() - series.min())

# Apply scaling
df['rsi_scaled'] = df['daily_rsi_14'] / 100
df['momentum_scaled'] = min_max_scale(df['daily_momentum_20'])
df['position_scaled'] = df['daily_price_position_52w']  # Already 0-1
df['atr_scaled'] = min_max_scale(df['atr_pct'])
df['dir_scaled'] = df['directional_strength']  # Already 0-1
df['ytd_scaled'] = min_max_scale(df['daily_year_pct_change'])

# Define the metrics and their abbreviated names
metrics = ['rsi_scaled', 'momentum_scaled', 'position_scaled', 'atr_scaled', 'dir_scaled', 'ytd_scaled']
metric_labels = ['RSI 14', 'Momentum 20', '52W Position', 'ATR %', 'Dir Strength', 'YTD Change']

# Colors for each stock
colors = ['#1FB8CD', '#DB4545', '#2E8B57', '#5D878F', '#D2BA4C', '#B4413C', '#964325', '#944454']

# Create radar chart
fig = go.Figure()

# Add a trace for each stock
for i, ticker in enumerate(df['ticker']):
    values = df.loc[i, metrics].tolist()
    values.append(values[0])  # Close the radar chart
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=metric_labels + [metric_labels[0]],
        fill='none',
        name=ticker,
        line=dict(color=colors[i], width=2)
    ))

# Update layout
fig.update_layout(
    title='Technical Analysis Radar Chart',
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1],
            tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            ticktext=['0', '0.2', '0.4', '0.6', '0.8', '1.0']
        ),
        angularaxis=dict(
            tickfont=dict(size=12)
        )
    ),
    showlegend=True,
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

# Save the chart
fig.write_image('radar_chart.png', width=800, height=600, scale=2)
print("Chart saved as radar_chart.png")