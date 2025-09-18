import plotly.graph_objects as go

# Data
categories = ['Oversold', 'Neutral', 'Overbought']
counts = [4, 42, 18]
colors = ['#DB4545', '#1FB8CD', '#2E8B57']  # Red, Cyan, Green from brand colors

# Calculate percentages
total = sum(counts)
percentages = [count/total * 100 for count in counts]

# Create bar chart
fig = go.Figure()

fig.add_trace(go.Bar(
    x=categories,
    y=counts,
    marker_color=colors,
    text=[f'{count}<br>{pct:.1f}%' for count, pct in zip(counts, percentages)],
    textposition='outside'
))

fig.update_layout(
    title='RSI Momentum Distribution',
    xaxis_title='RSI Category', 
    yaxis_title='Stock Count'
)

fig.update_traces(cliponaxis=False)

fig.write_image('rsi_momentum_distribution.png')