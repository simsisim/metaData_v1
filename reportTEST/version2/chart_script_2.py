import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Create dataframe from the provided data
data = [
    {"sector": "Energy minerals", "rs_percentile": 99},
    {"sector": "Consumer non-durables", "rs_percentile": 85},
    {"sector": "Finance", "rs_percentile": 71},
    {"sector": "Health technology", "rs_percentile": 57},
    {"sector": "Retail trade", "rs_percentile": 43},
    {"sector": "Technology services", "rs_percentile": 29},
    {"sector": "Electronic technology", "rs_percentile": 15}
]

df = pd.DataFrame(data)

# Define colors based on performance ranges
colors = []
for percentile in df['rs_percentile']:
    if percentile > 70:
        colors.append('#2E8B57')  # Sea green
    elif percentile >= 40:
        colors.append('#D2BA4C')  # Moderate yellow
    else:
        colors.append('#DB4545')  # Bright red

# Create bar chart
fig = go.Figure()

fig.add_trace(go.Bar(
    x=df['sector'],
    y=df['rs_percentile'],
    marker_color=colors,
    text=df['rs_percentile'],
    textposition='outside'
))

# Update layout
fig.update_layout(
    title="Sector RS Rankings",
    xaxis_title="Sector",
    yaxis_title="RS Percentile",
    showlegend=False
)

# Update traces
fig.update_traces(cliponaxis=False)

# Update x-axis to show abbreviated sector names
fig.update_xaxes(tickangle=45)

# Save the chart
fig.write_image("sector_rs_rankings.png")