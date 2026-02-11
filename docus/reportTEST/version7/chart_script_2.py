import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

# Create DataFrame from the provided data
data = {
    "Sector": ["Technology","Healthcare","Financials","Consumer_Disc","Energy","Industrials","Consumer_Staples","Utilities","Materials","Real_Estate","Communication"],
    "Avg_Annual_Return": [34.24,2.13,37.74,28.1,6.88,11.39,5.9,9.56,1.45,1.0,28.22],
    "Avg_Volatility": [2.44,1.78,1.52,2.23,1.51,2.13,1.34,1.17,1.17,1.14,1.35],
    "Stock_Count": [13,6,7,7,3,3,4,2,1,1,2]
}

df = pd.DataFrame(data)

# Abbreviate sector names to fit 15 character limit
df['Sector_Short'] = df['Sector'].replace({
    'Consumer_Staples': 'Cons Staples',
    'Consumer_Disc': 'Consumer Disc',
    'Real_Estate': 'Real Estate'
})

# Define colors in order
colors = ['#1FB8CD', '#DB4545', '#2E8B57', '#5D878F', '#D2BA4C', 
          '#B4413C', '#964325', '#944454', '#13343B', '#DB4545', '#1FB8CD']

# Create bubble chart
fig = go.Figure()

for i, sector in enumerate(df['Sector_Short'].unique()):
    sector_data = df[df['Sector_Short'] == sector]
    fig.add_trace(go.Scatter(
        x=sector_data['Avg_Annual_Return'],
        y=sector_data['Avg_Volatility'],
        mode='markers+text',
        marker=dict(
            size=sector_data['Stock_Count'] * 8,  # Scale bubble size
            color=colors[i % len(colors)],
            line=dict(width=2, color='white')
        ),
        text=sector_data['Sector_Short'],
        textposition='middle center',
        textfont=dict(size=10, color='white'),
        name=sector,
        hovertemplate='<b>%{text}</b><br>Return: %{x:.1f}%<br>Volatility: %{y:.2f}<br>Stocks: %{marker.size}<extra></extra>'
    ))

# Add reference lines
fig.add_hline(y=2.0, line_dash="dash", line_color="gray", opacity=0.7)
fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.7)

# Update layout
fig.update_layout(
    title='Sector Performance Analysis',
    xaxis_title='Avg Ann Return',
    yaxis_title='Avg Volatility',
    showlegend=True,
    xaxis=dict(showgrid=True, gridcolor='lightgray', gridwidth=0.5),
    yaxis=dict(showgrid=True, gridcolor='lightgray', gridwidth=0.5)
)

# Update traces
fig.update_traces(cliponaxis=False)

# Save the chart
fig.write_image("sector_performance_bubble_chart.png")