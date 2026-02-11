import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Create the data
data = [
    {"category": "Intermarket Ratios", "indicator": "QQQ:SPY", "current_value": 1.25, "signal": "Tech Strength", "color": "green"},
    {"category": "Intermarket Ratios", "indicator": "XLY:XLP", "current_value": 1.18, "signal": "Risk On", "color": "green"},
    {"category": "Intermarket Ratios", "indicator": "IWF:IWD", "current_value": 1.32, "signal": "Growth Leading", "color": "green"},
    {"category": "Intermarket Ratios", "indicator": "TRAN:UTIL", "current_value": 2.45, "signal": "Economic Strength", "color": "green"},
    
    {"category": "Sentiment Indicators", "indicator": "VIX", "current_value": 16.5, "signal": "Normal", "color": "yellow"},
    {"category": "Sentiment Indicators", "indicator": "Put/Call Ratio", "current_value": 0.82, "signal": "Neutral", "color": "yellow"},
    
    {"category": "Market Breadth", "indicator": "% Above 50-day MA", "current_value": 68, "signal": "Healthy", "color": "green"},
    {"category": "Market Breadth", "indicator": "% Above 200-day MA", "current_value": 72, "signal": "Bullish", "color": "green"},
    {"category": "Market Breadth", "indicator": "Advance/Decline", "current_value": 1250, "signal": "Positive", "color": "green"},
    {"category": "Market Breadth", "indicator": "New Hi/Lo Ratio", "current_value": 3.2, "signal": "Strong", "color": "green"}
]

df = pd.DataFrame(data)

# Color mapping
color_map = {
    'green': '#2E8B57',   # Sea green - bullish
    'yellow': '#D2BA4C',  # Moderate yellow - neutral  
    'red': '#DB4545'      # Bright red - bearish
}

# Create subplots with 3 rows
fig = make_subplots(
    rows=3, cols=1,
    row_heights=[0.4, 0.25, 0.35],
    subplot_titles=['Intermarket Ratios', 'Sentiment Indicators', 'Market Breadth'],
    vertical_spacing=0.08
)

# 1. Intermarket Ratios (Top)
intermarket_data = df[df['category'] == 'Intermarket Ratios'].copy()
fig.add_trace(
    go.Bar(
        x=intermarket_data['current_value'],
        y=intermarket_data['indicator'],
        orientation='h',
        marker=dict(color=[color_map[c] for c in intermarket_data['color']]),
        text=[f"{val:.2f}" for val in intermarket_data['current_value']],
        textposition='inside',
        textfont=dict(color='white', size=11),
        hovertemplate='<b>%{y}</b><br>Value: %{x:.2f}<br>Signal: %{customdata}<extra></extra>',
        customdata=intermarket_data['signal'],
        showlegend=False,
        name="Intermarket"
    ), row=1, col=1
)

# 2. Sentiment Indicators (Middle)
sentiment_data = df[df['category'] == 'Sentiment Indicators'].copy()
fig.add_trace(
    go.Bar(
        x=sentiment_data['current_value'],
        y=sentiment_data['indicator'],
        orientation='h',
        marker=dict(color=[color_map[c] for c in sentiment_data['color']]),
        text=[f"{val:.1f}" if val < 10 else f"{val:.0f}" for val in sentiment_data['current_value']],
        textposition='inside',
        textfont=dict(color='white', size=11),
        hovertemplate='<b>%{y}</b><br>Value: %{x:.2f}<br>Signal: %{customdata}<extra></extra>',
        customdata=sentiment_data['signal'],
        showlegend=False,
        name="Sentiment"
    ), row=2, col=1
)

# Add VIX threshold reference lines
fig.add_shape(
    type="line", x0=15, x1=15, y0=-0.5, y1=1.5,
    line=dict(color="gray", width=1, dash="dash"),
    row=2, col=1
)
fig.add_shape(
    type="line", x0=20, x1=20, y0=-0.5, y1=1.5,
    line=dict(color="gray", width=1, dash="dash"),
    row=2, col=1
)
fig.add_shape(
    type="line", x0=30, x1=30, y0=-0.5, y1=1.5,
    line=dict(color="orange", width=1, dash="dash"),
    row=2, col=1
)

# Add Put/Call threshold reference lines
fig.add_shape(
    type="line", x0=0.7, x1=0.7, y0=-0.5, y1=1.5,
    line=dict(color="gray", width=1, dash="dash"),
    row=2, col=1
)
fig.add_shape(
    type="line", x0=1.0, x1=1.0, y0=-0.5, y1=1.5,
    line=dict(color="gray", width=1, dash="dash"),
    row=2, col=1
)

# 3. Market Breadth (Bottom)
breadth_data = df[df['category'] == 'Market Breadth'].copy()
fig.add_trace(
    go.Bar(
        x=breadth_data['current_value'],
        y=breadth_data['indicator'],
        orientation='h',
        marker=dict(color=[color_map[c] for c in breadth_data['color']]),
        text=[f"{val:.0f}" if val >= 10 else f"{val:.1f}" for val in breadth_data['current_value']],
        textposition='inside',
        textfont=dict(color='white', size=11),
        hovertemplate='<b>%{y}</b><br>Value: %{x}<br>Signal: %{customdata}<extra></extra>',
        customdata=breadth_data['signal'],
        showlegend=False,
        name="Breadth"
    ), row=3, col=1
)

# Add reference lines for breadth indicators
fig.add_shape(
    type="line", x0=50, x1=50, y0=-0.5, y1=3.5,
    line=dict(color="gray", width=1, dash="dash"),
    row=3, col=1
)

# Update layout
fig.update_layout(
    title="Market Timing Dashboard",
    showlegend=False,
    plot_bgcolor='white'
)

# Update x-axes for each subplot
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=1, col=1, title="Ratio")
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=2, col=1, title="Level")
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=3, col=1, title="Value")

# Update y-axes
fig.update_yaxes(showgrid=False, row=1, col=1)
fig.update_yaxes(showgrid=False, row=2, col=1)  
fig.update_yaxes(showgrid=False, row=3, col=1)

# Add annotations for threshold ranges
fig.add_annotation(
    x=12.5, y=1.8, text="Complacency", showarrow=False,
    font=dict(size=9, color="gray"), row=2, col=1
)
fig.add_annotation(
    x=17.5, y=1.8, text="Normal", showarrow=False,
    font=dict(size=9, color="gray"), row=2, col=1
)
fig.add_annotation(
    x=25, y=1.8, text="Elevated", showarrow=False,
    font=dict(size=9, color="gray"), row=2, col=1
)

fig.add_annotation(
    x=0.6, y=0.2, text="Bullish", showarrow=False,
    font=dict(size=9, color="gray"), row=2, col=1
)
fig.add_annotation(
    x=0.85, y=0.2, text="Neutral", showarrow=False,
    font=dict(size=9, color="gray"), row=2, col=1
)
fig.add_annotation(
    x=1.1, y=0.2, text="Bearish", showarrow=False,
    font=dict(size=9, color="gray"), row=2, col=1
)

fig.update_traces(cliponaxis=False)

# Save the chart
fig.write_image("dashboard.png")
fig.write_image("dashboard.svg", format="svg")