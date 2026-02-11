import plotly.graph_objects as go
import pandas as pd

# Data
data = [
    {"Sector": "Finance", "1D_RS%": 99, "7D_RS%": 99, "22D_RS%": 99, "66D_RS%": 26},
    {"Sector": "Retail trade", "1D_RS%": 74, "7D_RS%": 74, "22D_RS%": 26, "66D_RS%": 50},
    {"Sector": "Technology services", "1D_RS%": 50, "7D_RS%": 26, "22D_RS%": 74, "66D_RS%": 74},
    {"Sector": "Electronic technology", "1D_RS%": 26, "7D_RS%": 50, "22D_RS%": 50, "66D_RS%": 99}
]

df = pd.DataFrame(data)

# Timeframe columns and their abbreviated labels
timeframes = ["1D_RS%", "7D_RS%", "22D_RS%", "66D_RS%"]
timeframe_labels = ["1D", "7D", "22D", "66D"]

# Brand colors
colors = ['#1FB8CD', '#DB4545', '#2E8B57', '#5D878F']

# Abbreviated sector names (under 15 chars)
sector_names = ["Finance", "Retail", "Tech Svc", "Electronics"]

fig = go.Figure()

# Add each sector as a separate trace
for i, (_, row) in enumerate(df.iterrows()):
    values = [row[tf] for tf in timeframes]
    # Close the polygon by adding the first value at the end
    values_closed = values + [values[0]]
    theta_closed = timeframe_labels + [timeframe_labels[0]]
    
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=theta_closed,
        fill='toself',
        name=sector_names[i],
        line_color=colors[i],
        fillcolor=colors[i],
        opacity=0.4
    ))

fig.update_layout(
    title="Sector RS Across Timeframes",
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 100],
            ticksuffix="%"
        )
    )
)

# Center legend under title since we have 4 items (≤5)
fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5))

fig.write_image("sector_radar_chart.png")