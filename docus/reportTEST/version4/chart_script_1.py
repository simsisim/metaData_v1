import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Data provided
data = [
    {"stage": "Bullish Trend", "count": 30},
    {"stage": "Bullish Fade", "count": 9},
    {"stage": "Bearish Trend", "count": 8},
    {"stage": "Launch Pad", "count": 6},
    {"stage": "Pullback", "count": 4},
    {"stage": "Mean Reversion", "count": 3},
    {"stage": "Undefined", "count": 2},
    {"stage": "Breakout Confirmation", "count": 1},
    {"stage": "Volatile Distribution", "count": 1}
]

df = pd.DataFrame(data)

# Calculate total and percentages
total = df['count'].sum()
df['percentage'] = (df['count'] / total * 100).round(1)

# Create labels with abbreviated stage names (15 char limit) and show count + percentage
df['label'] = df['stage'].str[:15]  # Truncate to 15 characters
df['display_text'] = df['label'] + '<br>' + df['count'].astype(str) + ' (' + df['percentage'].astype(str) + '%)'

# Define colors based on stage type
# Bullish stages: shades of green
# Bearish stages: shades of red
# Neutral/undefined: gray/yellow tones
color_map = {
    "Bullish Trend": "#2E8B57",      # Sea green (bullish)
    "Bullish Fade": "#5D878F",       # Cyan (bullish)  
    "Launch Pad": "#1FB8CD",         # Strong cyan (bullish)
    "Breakout Confirmation": "#13343B", # Dark cyan (bullish)
    "Bearish Trend": "#DB4545",      # Bright red (bearish)
    "Pullback": "#B4413C",           # Moderate red (bearish)
    "Mean Reversion": "#D2BA4C",     # Moderate yellow (neutral)
    "Undefined": "#964325",          # Dark orange (neutral)
    "Volatile Distribution": "#944454" # Pink-red (neutral)
}

colors = [color_map[stage] for stage in df['stage']]

# Create donut chart
fig = go.Figure(data=[go.Pie(
    labels=df['label'],
    values=df['count'],
    hole=0.4,  # Makes it a donut chart
    textinfo='label+percent',
    textposition='inside',
    marker=dict(colors=colors),
    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
)])

# Update layout
fig.update_layout(
    title="Stock Stage Distribution",
    uniformtext_minsize=14, 
    uniformtext_mode='hide'
)

# Save the chart
fig.write_image("stock_stage_donut_chart.png")