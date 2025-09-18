import plotly.express as px
import pandas as pd

# Data from the provided JSON
data = [
    {"stage": "Bullish Trend", "count": 30},
    {"stage": "Bullish Fade", "count": 8},
    {"stage": "Bearish Trend", "count": 8},
    {"stage": "Pullback", "count": 4},
    {"stage": "Mean Reversion", "count": 4},
    {"stage": "Launch Pad", "count": 4},
    {"stage": "Breakout Confirmation", "count": 2},
    {"stage": "Undefined", "count": 2},
    {"stage": "Volatile Distribution", "count": 2}
]

# Create DataFrame
df = pd.DataFrame(data)

# Define colors based on stage type - using requested colors
color_map = {
    'Bullish Trend': '#2E8B57',  # Green for Bullish Trend
    'Bearish Trend': '#DB4545',  # Red for Bearish Trend  
    'Bullish Fade': '#FF8C00',   # Orange for Bullish Fade
    'Pullback': '#D2BA4C',       # Yellow for other categories
    'Mean Reversion': '#D2BA4C', # Yellow for other categories
    'Launch Pad': '#D2BA4C',     # Yellow for other categories
    'Breakout Confirmation': '#D2BA4C',  # Yellow for other categories
    'Undefined': '#D2BA4C',      # Yellow for other categories
    'Volatile Distribution': '#D2BA4C'   # Yellow for other categories
}

# Create the pie chart
fig = px.pie(df, 
             values='count', 
             names='stage',
             title='Market Stage Analysis Distribution',
             color='stage',
             color_discrete_map=color_map)

# Update traces for better appearance - remove cliponaxis for pie charts
fig.update_traces(
    textposition='inside',
    textinfo='label+percent+value',
    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
)

# Update layout for pie chart specific formatting
fig.update_layout(
    uniformtext_minsize=14, 
    uniformtext_mode='hide'
)

# Save the chart
fig.write_image('market_stage_pie_chart.png')