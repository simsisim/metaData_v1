import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load the data
df = pd.read_csv("sector_performance.csv")

# Sort by performance (highest to lowest)
df_sorted = df.sort_values('daily_daily_yearly_252d_pct_change', ascending=True)  # ascending=True for horizontal bar chart

# Create color mapping for positive/negative returns
colors = []
for value in df_sorted['daily_daily_yearly_252d_pct_change']:
    if value >= 0:
        colors.append('#1FB8CD')  # Strong cyan for positive
    else:
        colors.append('#DB4545')  # Bright red for negative

# Create horizontal bar chart
fig = go.Figure(data=[
    go.Bar(
        y=df_sorted['Sector'],
        x=df_sorted['daily_daily_yearly_252d_pct_change'],
        orientation='h',
        marker_color=colors,
        showlegend=False
    )
])

# Add vertical line at 0%
fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)

# Update layout
fig.update_layout(
    title="1-Year Performance by Sector",
    xaxis_title="1-Yr Return (%)",
    yaxis_title="Sector"
)

# Update traces
fig.update_traces(cliponaxis=False)

# Save the chart
fig.write_image("sector_performance_chart.png")
print("Chart saved successfully!")