import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Create DataFrame from the provided data
data = [
    {"ticker": "APP", "rs_percentile": 99},
    {"ticker": "PLTR", "rs_percentile": 97},
    {"ticker": "GEV", "rs_percentile": 96},
    {"ticker": "AVGO", "rs_percentile": 94},
    {"ticker": "NFLX", "rs_percentile": 93},
    {"ticker": "ORCL", "rs_percentile": 91},
    {"ticker": "GE", "rs_percentile": 90},
    {"ticker": "C", "rs_percentile": 88},
    {"ticker": "TSLA", "rs_percentile": 87},
    {"ticker": "ANET", "rs_percentile": 85},
    {"ticker": "NTRS", "rs_percentile": 84},
    {"ticker": "VRSN", "rs_percentile": 82},
    {"ticker": "META", "rs_percentile": 81},
    {"ticker": "PM", "rs_percentile": 79},
    {"ticker": "CSCO", "rs_percentile": 78}
]

df = pd.DataFrame(data)

# Create horizontal bar chart with gradient green colors
fig = px.bar(df, 
             x='rs_percentile', 
             y='ticker',
             orientation='h',
             color='rs_percentile',
             color_continuous_scale=['#90EE90', '#006400'],  # Light green to dark green
             title='Top 15 Stocks by RS Percentile')

# Add percentile values as text labels on bars
fig.update_traces(
    text=df['rs_percentile'],
    texttemplate='%{text}',
    textposition='inside',
    cliponaxis=False
)

# Update layout for clean appearance
fig.update_layout(
    xaxis_title='RS Percentile',
    yaxis_title='Ticker',
    coloraxis_showscale=False,  # Hide color scale
    yaxis={'categoryorder': 'total ascending'}  # Keep order from data
)

# Update axes
fig.update_xaxes(range=[0, 100])

# Save the chart
fig.write_image('rs_percentile_chart.png')