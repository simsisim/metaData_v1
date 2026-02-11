import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# Create the data
data = [
  {"date": "2008-01-01", "VIX": 25, "Consumer_Sentiment": 78, "event": "Pre-Crisis"},
  {"date": "2008-10-01", "VIX": 45, "Consumer_Sentiment": 55, "event": "Financial Crisis Peak"},
  {"date": "2009-03-01", "VIX": 52, "Consumer_Sentiment": 52, "event": "Market Bottom"},
  {"date": "2010-01-01", "VIX": 18, "Consumer_Sentiment": 72, "event": "Recovery"},
  {"date": "2015-08-01", "VIX": 28, "Consumer_Sentiment": 92, "event": "China Devaluation"},
  {"date": "2016-01-01", "VIX": 26, "Consumer_Sentiment": 90, "event": "Oil Crash"},
  {"date": "2018-12-01", "VIX": 32, "Consumer_Sentiment": 95, "event": "Rate Fears"},
  {"date": "2019-12-01", "VIX": 12, "Consumer_Sentiment": 99, "event": "Pre-COVID High"},
  {"date": "2020-03-01", "VIX": 75, "Consumer_Sentiment": 71, "event": "COVID Crash"},
  {"date": "2020-04-01", "VIX": 35, "Consumer_Sentiment": 68, "event": "COVID Bottom"},
  {"date": "2021-11-01", "VIX": 15, "Consumer_Sentiment": 108, "event": "Market Peak"},
  {"date": "2022-06-01", "VIX": 31, "Consumer_Sentiment": 58, "event": "Bear Market Low"},
  {"date": "2023-10-01", "VIX": 22, "Consumer_Sentiment": 63, "event": "Recovery"},
  {"date": "2024-08-01", "VIX": 28, "Consumer_Sentiment": 65, "event": "AI Correction"},
  {"date": "2025-09-01", "VIX": 16, "Consumer_Sentiment": 55, "event": "Current"}
]

df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'])

# Create figure with secondary y-axis
fig = go.Figure()

# Add VIX line (primary y-axis)
fig.add_trace(go.Scatter(
    x=df['date'],
    y=df['VIX'],
    mode='lines+markers',
    name='VIX',
    line=dict(color='#1FB8CD', width=3),
    marker=dict(size=6),
    yaxis='y',
    hovertemplate='VIX: %{y}<br>%{customdata}<extra></extra>',
    customdata=df['event']
))

# Add Consumer Sentiment line (secondary y-axis)
fig.add_trace(go.Scatter(
    x=df['date'],
    y=df['Consumer_Sentiment'],
    mode='lines+markers',
    name='Sentiment',
    line=dict(color='#DB4545', width=3),
    marker=dict(size=6),
    yaxis='y2',
    hovertemplate='Sentiment: %{y}<br>%{customdata}<extra></extra>',
    customdata=df['event']
))

# Mark VIX fear spikes (>30)
fear_spikes = df[df['VIX'] > 30]
fig.add_trace(go.Scatter(
    x=fear_spikes['date'],
    y=fear_spikes['VIX'],
    mode='markers',
    name='Fear >30',
    marker=dict(color='#2E8B57', size=10, symbol='triangle-up'),
    yaxis='y',
    hovertemplate='Fear Spike<br>VIX: %{y}<extra></extra>'
))

# Mark VIX complacency (<15)
complacency = df[df['VIX'] < 15]
fig.add_trace(go.Scatter(
    x=complacency['date'],
    y=complacency['VIX'],
    mode='markers',
    name='Calm <15',
    marker=dict(color='#5D878F', size=10, symbol='triangle-down'),
    yaxis='y',
    hovertemplate='Complacency<br>VIX: %{y}<extra></extra>'
))

# Mark extreme sentiment readings (<60 or >100)
extreme_low = df[df['Consumer_Sentiment'] < 60]
extreme_high = df[df['Consumer_Sentiment'] > 100]

if not extreme_low.empty:
    fig.add_trace(go.Scatter(
        x=extreme_low['date'],
        y=extreme_low['Consumer_Sentiment'],
        mode='markers',
        name='Low Sentiment',
        marker=dict(color='#D2BA4C', size=10, symbol='diamond'),
        yaxis='y2',
        hovertemplate='Low Sentiment<br>%{y}<extra></extra>'
    ))

if not extreme_high.empty:
    fig.add_trace(go.Scatter(
        x=extreme_high['date'],
        y=extreme_high['Consumer_Sentiment'],
        mode='markers',
        name='High Sentiment',
        marker=dict(color='#B4413C', size=10, symbol='diamond'),
        yaxis='y2',
        hovertemplate='High Sentiment<br>%{y}<extra></extra>'
    ))

# Mark major market events
major_events = df[df['event'].isin(['Financial Crisis Peak', 'COVID Crash', 'Bear Market Low'])]
fig.add_trace(go.Scatter(
    x=major_events['date'],
    y=major_events['VIX'],
    mode='markers',
    name='Major Events',
    marker=dict(color='#964325', size=12, symbol='star'),
    yaxis='y',
    hovertemplate='%{customdata}<br>VIX: %{y}<extra></extra>',
    customdata=major_events['event']
))

# Update layout with dual y-axes
fig.update_layout(
    title='VIX vs Consumer Sentiment Timeline',
    xaxis_title='Date',
    yaxis=dict(
        title='VIX Level',
        side='left',
        range=[0, 80],
        color='#1FB8CD'
    ),
    yaxis2=dict(
        title='Sentiment',
        side='right',
        range=[40, 120],
        overlaying='y',
        color='#DB4545'
    ),
    showlegend=True
)

fig.update_traces(cliponaxis=False)

# Save both PNG and SVG
fig.write_image("vix_sentiment_timeline.png")
fig.write_image("vix_sentiment_timeline.svg", format="svg")

fig.show()